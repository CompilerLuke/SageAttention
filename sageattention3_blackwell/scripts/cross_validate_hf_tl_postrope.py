#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_query_mean_cosim_models import modeling_module_for, torch_dtype


DEFAULT_TEXT = (
    "A clinic stores vaccine shipments in two rooms. Room A receives 18 boxes on "
    "Monday and 7 boxes on Tuesday. Room B receives 11 boxes on Monday and 13 "
    "boxes on Tuesday. Which room received more boxes in total?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    return parser.parse_args()


def repeated_input_ids(tokenizer, text: str, seq_len: int, device: str) -> torch.Tensor:
    expanded = text
    while True:
        input_ids = tokenizer(expanded, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if input_ids.size(-1) >= seq_len:
            return input_ids[:, :seq_len].to(device)
        expanded = expanded + "\n" + text


def attention_shape(config) -> tuple[int, int, int]:
    n_heads = int(config.num_attention_heads)
    n_kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // n_heads))
    return n_heads, n_kv_heads, head_dim


@contextlib.contextmanager
def capture_hf_postrope_qkv(model, layer: int, n_kv_heads: int, head_dim: int):
    module = modeling_module_for(model.config.model_type)
    original_apply_rope = module.apply_rotary_pos_emb
    captures: dict[str, torch.Tensor] = {}
    call_index = 0

    def wrapped_apply_rope(q, k, cos, sin, *args, **kwargs):
        nonlocal call_index
        q_rot, k_rot = original_apply_rope(q, k, cos, sin, *args, **kwargs)
        if call_index == layer:
            captures["q"] = q_rot.detach().cpu()
            captures["k"] = k_rot.detach().cpu()
        call_index += 1
        return q_rot, k_rot

    def save_v(_module, _inputs, output):
        batch, seq, _ = output.shape
        captures["v"] = (
            output.detach()
            .view(batch, seq, n_kv_heads, head_dim)
            .transpose(1, 2)
            .contiguous()
            .cpu()
        )

    hook = model.model.layers[layer].self_attn.v_proj.register_forward_hook(save_v)
    module.apply_rotary_pos_emb = wrapped_apply_rope
    try:
        yield captures
    finally:
        module.apply_rotary_pos_emb = original_apply_rope
        hook.remove()


def tl_cache_to_bhld(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"expected a 4D TL activation, got {tuple(x.shape)}")
    return x.permute(0, 2, 1, 3).contiguous().cpu()


def summarize(name: str, hf: torch.Tensor, tl: torch.Tensor) -> None:
    if hf.shape != tl.shape:
        print(f"{name}: shape mismatch hf={tuple(hf.shape)} tl={tuple(tl.shape)}")
        return
    hf_f = hf.float()
    tl_f = tl.float()
    diff = tl_f - hf_f
    denom = hf_f.pow(2).mean().sqrt().clamp_min(1e-30)
    flat_hf = hf_f.reshape(-1)
    flat_tl = tl_f.reshape(-1)
    cos = F.cosine_similarity(flat_hf, flat_tl, dim=0).item()
    print(
        f"{name}: shape={tuple(hf.shape)} "
        f"mae={diff.abs().mean().item():.8g} "
        f"rmse={diff.pow(2).mean().sqrt().item():.8g} "
        f"rel_rmse={(diff.pow(2).mean().sqrt() / denom).item():.8g} "
        f"max_abs={diff.abs().max().item():.8g} "
        f"cos={cos:.10f}"
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    dtype = torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = repeated_input_ids(tokenizer, args.text, args.seq_len, args.device)
    input_ids_cpu = input_ids.cpu()
    print(f"tokens: shape={tuple(input_ids.shape)} first8={input_ids_cpu[0, :8].tolist()}")

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    hf_model.eval()
    _, n_kv_heads, head_dim = attention_shape(hf_model.config)
    with capture_hf_postrope_qkv(hf_model, args.layer, n_kv_heads, head_dim) as hf_caps:
        _ = hf_model(input_ids=input_ids, use_cache=False)
    hf_caps = {name: tensor.contiguous() for name, tensor in hf_caps.items()}
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    tl_model = HookedTransformer.from_pretrained_no_processing(
        args.model,
        device=args.device,
        dtype=dtype,
        n_ctx=args.seq_len,
    )
    tl_model.eval()
    hook_names = {
        "q": f"blocks.{args.layer}.attn.hook_rot_q",
        "k": f"blocks.{args.layer}.attn.hook_rot_k",
        "v": f"blocks.{args.layer}.attn.hook_v",
    }
    _, cache = tl_model.run_with_cache(
        input_ids_cpu.to(args.device),
        return_type=None,
        stop_at_layer=args.layer + 1,
        names_filter=lambda name: name in set(hook_names.values()),
    )
    tl_caps = {name: tl_cache_to_bhld(cache[hook]) for name, hook in hook_names.items()}

    print(f"model={args.model} layer={args.layer} dtype={args.dtype} seq_len={args.seq_len}")
    for name in ["q", "k", "v"]:
        summarize(name, hf_caps[name], tl_caps[name])


if __name__ == "__main__":
    main()
