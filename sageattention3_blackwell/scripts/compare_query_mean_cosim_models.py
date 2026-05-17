#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F


MODEL_SPECS = {
    "llama31_8b_instruct": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "layers": [16, 30],
    },
    "llama3_8b_instruct": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "layers": [16, 30],
    },
    "qwen25_7b_instruct": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "layers": [14, 26],
    },
    "mistral_7b_instruct_v03": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "layers": [16, 30],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=Path, default=Path("tests/prompts/clear_question_clinic.txt"))
    parser.add_argument("--eval-marker", default="ANSWER SECTION\n")
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS.keys(), default=list(MODEL_SPECS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--out-json", type=Path, default=Path("reports/query_mean_cosim_models_clinic.json"))
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def pad_to_block(x: torch.Tensor, block: int) -> torch.Tensor:
    pad_len = (block - x.size(2) % block) % block
    if pad_len == 0:
        return x.contiguous()
    return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.float().reshape(-1)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "q50": float(torch.quantile(values, 0.50).item()),
        "q95": float(torch.quantile(values, 0.95).item()),
    }


def token_count(tokenizer, text: str, add_special_tokens: bool) -> int:
    return len(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


def eval_token_range(tokenizer, text: str, marker: str, seq_len: int) -> tuple[int, int]:
    marker_idx = text.index(marker)
    eval_char_start = marker_idx + len(marker)
    start = token_count(tokenizer, text[:eval_char_start], add_special_tokens=True)
    count = token_count(tokenizer, text[eval_char_start:], add_special_tokens=False)
    start = max(0, min(start, seq_len))
    end = max(start, min(start + count, seq_len))
    if end <= start:
        raise ValueError(f"empty eval token range: start={start}, end={end}, seq_len={seq_len}")
    return start, end


def q_cosim_to_means(q: torch.Tensor, token_range: tuple[int, int]) -> dict:
    q = q.float().contiguous()
    start, end = token_range
    q_eval = q[:, :, start:end, :]

    full_mean = q.mean(dim=-2, keepdim=True)
    full_cosim = F.cosine_similarity(
        q_eval,
        full_mean.expand(-1, -1, q.size(2), -1)[:, :, start:end, :],
        dim=-1,
    )

    q_padded = pad_to_block(q, 128)
    bsz, heads, _, dim = q.shape
    _, _, padded_seq_len, _ = q_padded.shape
    q_blocks = q_padded.reshape(bsz, heads, padded_seq_len // 128, 128, dim)
    block_mean = q_blocks.mean(dim=-2, keepdim=True).expand_as(q_blocks).reshape_as(q_padded)
    block_cosim = F.cosine_similarity(q_eval, block_mean[:, :, start:end, :], dim=-1)

    return {
        "full_mean": summarize(full_cosim.cpu()),
        "block128_mean": summarize(block_cosim.cpu()),
    }


def q_mean_bias_cosim(q: torch.Tensor, raw_bias: torch.Tensor | None, rope_mean_bias: torch.Tensor | None) -> dict | None:
    if raw_bias is None:
        return None

    q_mean = q.float().mean(dim=-2)
    raw = raw_bias.float().unsqueeze(0).to(q_mean.device)
    out = {
        "raw_q_proj_bias": summarize(F.cosine_similarity(q_mean, raw, dim=-1).cpu()),
    }
    if rope_mean_bias is not None:
        rope = rope_mean_bias.float().squeeze(0).to(q_mean.device)
        out["rope_mean_q_proj_bias"] = summarize(F.cosine_similarity(q_mean, rope, dim=-1).cpu())
    return out


def model_attention_shape(config) -> tuple[int, int]:
    n_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // n_heads))
    return n_heads, head_dim


def modeling_module_for(model_type: str):
    if model_type == "llama":
        import transformers.models.llama.modeling_llama as module
    elif model_type == "qwen2":
        import transformers.models.qwen2.modeling_qwen2 as module
    elif model_type == "mistral":
        import transformers.models.mistral.modeling_mistral as module
    else:
        raise ValueError(f"unsupported model_type for post-RoPE capture: {model_type}")
    return module


@contextlib.contextmanager
def capture_post_rope_queries(model, layers: list[int], n_heads: int, head_dim: int):
    module = modeling_module_for(model.config.model_type)
    original_apply_rope = module.apply_rotary_pos_emb
    layer_set = set(layers)
    call_index = 0
    captures: dict[int, torch.Tensor] = {}
    rope_mean_biases: dict[int, torch.Tensor] = {}
    raw_biases: dict[int, torch.Tensor | None] = {}

    for layer in layers:
        bias = model.model.layers[layer].self_attn.q_proj.bias
        raw_biases[layer] = None if bias is None else bias.detach().view(n_heads, head_dim).cpu()

    def wrapped_apply_rope(q, k, cos, sin, *args, **kwargs):
        nonlocal call_index
        q_rot, k_rot = original_apply_rope(q, k, cos, sin, *args, **kwargs)
        layer = call_index
        call_index += 1
        if layer in layer_set:
            captures[layer] = q_rot.detach().cpu()
            raw_bias = raw_biases[layer]
            if raw_bias is not None:
                bias_states = raw_bias.to(device=q.device, dtype=q.dtype).view(1, n_heads, 1, head_dim).expand_as(q)
                # The post-RoPE bias contribution is position-dependent, so compare
                # against its document mean in the same rotated coordinate frame.
                bias_rot, _ = original_apply_rope(bias_states, bias_states, cos, sin, *args, **kwargs)
                rope_mean_biases[layer] = bias_rot.detach().float().mean(dim=-2).cpu()
        return q_rot, k_rot

    module.apply_rotary_pos_emb = wrapped_apply_rope
    try:
        yield captures, raw_biases, rope_mean_biases
    finally:
        module.apply_rotary_pos_emb = original_apply_rope


@torch.no_grad()
def measure_model(label: str, spec: dict, args: argparse.Namespace, text: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = spec["name"]
    print(f"loading {label}: {name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens["input_ids"].to(args.device)
    attention_mask = tokens.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)
    token_range = eval_token_range(tokenizer, text, args.eval_marker, int(input_ids.shape[-1]))

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    n_heads, head_dim = model_attention_shape(model.config)
    with capture_post_rope_queries(model, spec["layers"], n_heads, head_dim) as (
        captures,
        raw_biases,
        rope_mean_biases,
    ):
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    layers = {}
    for layer in spec["layers"]:
        layer_metrics = q_cosim_to_means(captures[layer], token_range)
        layer_metrics["q_mean_vs_q_proj_bias"] = q_mean_bias_cosim(
            captures[layer],
            raw_biases[layer],
            rope_mean_biases.get(layer),
        )
        layers[str(layer)] = layer_metrics

    del captures, model, input_ids, attention_mask, tokens
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": name,
        "layers": layers,
        "token_count": int(tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[-1]),
        "eval_range": [int(token_range[0]), int(token_range[1])],
        "eval_token_count": int(token_range[1] - token_range[0]),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "activation_space": "post_rope_q",
    }


def main() -> None:
    args = parse_args()
    text = args.text_file.read_text().strip()
    output = {
        "text_file": str(args.text_file),
        "eval_marker": args.eval_marker,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "results": {},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    for label in args.models:
        output["results"][label] = measure_model(label, MODEL_SPECS[label], args, text)
        args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True))
        print(f"wrote {args.out_json}", flush=True)

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
