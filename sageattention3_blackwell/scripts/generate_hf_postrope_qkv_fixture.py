#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
from pathlib import Path

import torch

from scripts.compare_query_mean_cosim_models import modeling_module_for, torch_dtype, token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 30])
    parser.add_argument("--text-file", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--prefix", default="llama3p1_8b_instruct_postrope")
    parser.add_argument("--question-marker", default="QUESTION SECTION\n")
    parser.add_argument("--eval-marker", default="ANSWER SECTION\n")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def marker_token_meta(tokenizer, text: str, question_marker: str, eval_marker: str, seq_len: int) -> dict:
    meta = {}
    if question_marker and question_marker in text:
        marker_idx = text.index(question_marker)
        question_char_start = marker_idx + len(question_marker)
        question_char_end = text.index(eval_marker) if eval_marker and eval_marker in text else len(text)
        prefix_count = token_count(tokenizer, text[:question_char_start], add_special_tokens=True)
        question_count = token_count(tokenizer, text[question_char_start:question_char_end], add_special_tokens=False)
        question_token_start = min(prefix_count, seq_len)
        question_token_end = min(question_token_start + question_count, seq_len)
        meta.update(
            {
                "question_marker": question_marker,
                "question_char_start": question_char_start,
                "question_char_end": question_char_end,
                "question_token_start": question_token_start,
                "question_token_end": question_token_end,
                "question_token_count": max(0, question_token_end - question_token_start),
                "question_text_preview": text[question_char_start : question_char_start + 200],
            }
        )
    if eval_marker and eval_marker in text:
        marker_idx = text.index(eval_marker)
        eval_char_start = marker_idx + len(eval_marker)
        prefix_count = token_count(tokenizer, text[:eval_char_start], add_special_tokens=True)
        eval_count = token_count(tokenizer, text[eval_char_start:], add_special_tokens=False)
        eval_token_start = min(prefix_count, seq_len)
        eval_token_end = min(eval_token_start + eval_count, seq_len)
        meta.update(
            {
                "eval_marker": eval_marker,
                "eval_char_start": eval_char_start,
                "eval_char_end": len(text),
                "eval_token_start": eval_token_start,
                "eval_token_end": eval_token_end,
                "eval_token_count": max(0, eval_token_end - eval_token_start),
                "eval_text_preview": text[eval_char_start : eval_char_start + 200],
            }
        )
    return meta


def attention_shape(config) -> tuple[int, int, int]:
    n_heads = int(config.num_attention_heads)
    n_kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // n_heads))
    return n_heads, n_kv_heads, head_dim


@contextlib.contextmanager
def capture_post_rope_qkv(model, layers: list[int], n_heads: int, n_kv_heads: int, head_dim: int):
    module = modeling_module_for(model.config.model_type)
    original_apply_rope = module.apply_rotary_pos_emb
    layer_set = set(layers)
    call_index = 0
    q_captures: dict[int, torch.Tensor] = {}
    k_captures: dict[int, torch.Tensor] = {}
    v_captures: dict[int, torch.Tensor] = {}
    hooks = []

    def wrapped_apply_rope(q, k, cos, sin, *args, **kwargs):
        nonlocal call_index
        q_rot, k_rot = original_apply_rope(q, k, cos, sin, *args, **kwargs)
        layer = call_index
        call_index += 1
        if layer in layer_set:
            q_captures[layer] = q_rot.detach().cpu()
            k_captures[layer] = k_rot.detach().cpu()
        return q_rot, k_rot

    def make_v_hook(layer: int):
        def save_v(_module, _inputs, output):
            batch, seq, _ = output.shape
            v_captures[layer] = (
                output.detach()
                .view(batch, seq, n_kv_heads, head_dim)
                .transpose(1, 2)
                .contiguous()
                .cpu()
            )

        return save_v

    for layer in layers:
        hooks.append(model.model.layers[layer].self_attn.v_proj.register_forward_hook(make_v_hook(layer)))

    module.apply_rotary_pos_emb = wrapped_apply_rope
    try:
        yield q_captures, k_captures, v_captures
    finally:
        module.apply_rotary_pos_emb = original_apply_rope
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    n_heads, n_kv_heads, head_dim = attention_shape(model.config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for text_file in args.text_file:
        text = text_file.read_text().strip()
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = tokens["input_ids"].to(args.device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)
        seq_len = int(input_ids.shape[-1])
        text_sha256 = hashlib.sha256(text.encode()).hexdigest()
        marker_meta = marker_token_meta(tokenizer, text, args.question_marker, args.eval_marker, seq_len)

        with capture_post_rope_qkv(model, args.layers, n_heads, n_kv_heads, head_dim) as (q_caps, k_caps, v_caps):
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        for layer in args.layers:
            payload = {
                "q": q_caps[layer].contiguous(),
                "k": k_caps[layer].contiguous(),
                "v": v_caps[layer].contiguous(),
                "meta": {
                    "source": "huggingface_post_rope",
                    "activation_space": "post_rope_qk_pre_attn_v",
                    "model": args.model,
                    "layer": layer,
                    "seq_len": seq_len,
                    "dtype": args.dtype,
                    "q_shape": tuple(q_caps[layer].shape),
                    "kv_shape": tuple(k_caps[layer].shape),
                    "text_file": str(text_file),
                    "text_sha256": text_sha256,
                    "text_char_count": len(text),
                    "text_preview": text[:200],
                    "text_tokens_with_bos": seq_len,
                    "repeated_text": False,
                    "n_heads": n_heads,
                    "n_key_value_heads": n_kv_heads,
                    "gqa_group_size": n_heads // n_kv_heads,
                    **marker_meta,
                },
            }
            out = args.out_dir / f"{args.prefix}_l{layer}_{text_file.stem}_qkv.pt"
            torch.save(payload, out)
            print(f"saved {out}")
            print({k: v for k, v in payload["meta"].items() if k != "text_preview"})

        del input_ids, attention_mask, tokens, q_caps, k_caps, v_caps
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
