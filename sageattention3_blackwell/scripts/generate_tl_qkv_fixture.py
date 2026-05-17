#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


DEFAULT_TEXT = """SageAttention trades small numerical approximations for large bandwidth wins in
attention kernels. A useful quality test should use activations from real model
layers, because token statistics, head specialization, and projection geometry
are not well represented by independent Gaussian tensors."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=4096, help="Target token count; use 0 for the full non-repeated prompt.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--no-repeat", action="store_true")
    parser.add_argument(
        "--question-marker",
        help="If present in the source text, record the token span after this marker as the question section.",
    )
    parser.add_argument(
        "--eval-marker",
        help="If present in the source text, record the token span after this marker as the evaluation section.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    from transformer_lens import HookedTransformer

    dtype = torch_dtype(args.dtype)
    load_seq_len = args.seq_len if args.seq_len > 0 else 4096
    model = HookedTransformer.from_pretrained_no_processing(
        args.model,
        device=args.device,
        dtype=dtype,
        n_ctx=load_seq_len,
    )
    model.eval()

    base_text = args.text_file.read_text().strip() if args.text_file else args.text.strip()
    text_sha256 = hashlib.sha256(base_text.encode()).hexdigest()
    base_tokens = model.to_tokens(base_text, prepend_bos=False)
    seq_len = args.seq_len if args.seq_len > 0 else int(base_tokens.shape[-1]) + 1
    if seq_len > load_seq_len:
        raise SystemExit(f"text provides {seq_len} tokens including BOS, above loaded context {load_seq_len}")
    if args.no_repeat:
        if base_tokens.shape[-1] + 1 < seq_len:
            raise SystemExit(
                f"text only provides {base_tokens.shape[-1] + 1} tokens including BOS; "
                f"need at least {seq_len}"
            )
        text = base_text
    else:
        repeats = max(1, seq_len // max(1, base_tokens.shape[-1]) + 2)
        text = "\n\n".join([base_text] * repeats)
    tokens = model.to_tokens(text, prepend_bos=True)[:, :seq_len].to(args.device)
    question_meta = {}
    if args.question_marker:
        marker_idx = text.index(args.question_marker)
        question_char_start = marker_idx + len(args.question_marker)
        question_char_end = text.index(args.eval_marker) if args.eval_marker else len(text)
        prefix_tokens = model.to_tokens(text[:question_char_start], prepend_bos=True)
        question_tokens = model.to_tokens(text[question_char_start:question_char_end], prepend_bos=False)
        question_token_start = min(int(prefix_tokens.shape[-1]), seq_len)
        question_token_end = min(question_token_start + int(question_tokens.shape[-1]), seq_len)
        question_meta = {
            "question_marker": args.question_marker,
            "question_char_start": question_char_start,
            "question_char_end": question_char_end,
            "question_token_start": question_token_start,
            "question_token_end": question_token_end,
            "question_token_count": max(0, question_token_end - question_token_start),
            "question_text_preview": text[question_char_start : question_char_start + 200],
        }
    eval_meta = {}
    if args.eval_marker:
        marker_idx = text.index(args.eval_marker)
        eval_char_start = marker_idx + len(args.eval_marker)
        prefix_tokens = model.to_tokens(text[:eval_char_start], prepend_bos=True)
        eval_tokens = model.to_tokens(text[eval_char_start:], prepend_bos=False)
        eval_token_start = min(int(prefix_tokens.shape[-1]), seq_len)
        eval_token_end = min(eval_token_start + int(eval_tokens.shape[-1]), seq_len)
        eval_meta = {
            "eval_marker": args.eval_marker,
            "eval_char_start": eval_char_start,
            "eval_char_end": len(text),
            "eval_token_start": eval_token_start,
            "eval_token_end": eval_token_end,
            "eval_token_count": max(0, eval_token_end - eval_token_start),
            "eval_text_preview": text[eval_char_start : eval_char_start + 200],
        }

    hook_names = {
        "q": f"blocks.{args.layer}.attn.hook_q",
        "k": f"blocks.{args.layer}.attn.hook_k",
        "v": f"blocks.{args.layer}.attn.hook_v",
    }
    wanted = set(hook_names.values())
    _, cache = model.run_with_cache(
        tokens,
        return_type=None,
        stop_at_layer=args.layer + 1,
        names_filter=lambda name: name in wanted,
    )

    q = cache[hook_names["q"]].permute(0, 2, 1, 3).contiguous().cpu()
    k = cache[hook_names["k"]].permute(0, 2, 1, 3).contiguous().cpu()
    v = cache[hook_names["v"]].permute(0, 2, 1, 3).contiguous().cpu()
    n_heads = int(q.shape[1])
    n_kv_heads = int(k.shape[1])
    payload = {
        "q": q,
        "k": k,
        "v": v,
        "meta": {
            "source": "transformer_lens",
            "model": args.model,
            "layer": args.layer,
            "seq_len": seq_len,
            "dtype": args.dtype,
            "q_shape": tuple(q.shape),
            "kv_shape": tuple(k.shape),
            "text_file": str(args.text_file) if args.text_file else None,
            "text_sha256": text_sha256,
            "text_char_count": len(base_text),
            "text_preview": base_text[:200],
            "text_tokens_with_bos": int(tokens.shape[-1]),
            "repeated_text": not args.no_repeat,
            "n_heads": n_heads,
            "n_key_value_heads": n_kv_heads,
            "gqa_group_size": n_heads // n_kv_heads,
            "names_filter": sorted(wanted),
            **question_meta,
            **eval_meta,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"saved {args.out}")
    print({k: v for k, v in payload["meta"].items() if k != "text_preview"})


if __name__ == "__main__":
    main()
