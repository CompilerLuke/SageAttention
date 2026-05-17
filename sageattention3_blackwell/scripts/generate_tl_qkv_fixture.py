#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
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
    model = HookedTransformer.from_pretrained_no_processing(
        args.model,
        device=args.device,
        dtype=dtype,
        n_ctx=args.seq_len,
    )
    model.eval()

    base_text = args.text.strip()
    base_tokens = model.to_tokens(base_text, prepend_bos=False)
    repeats = max(1, args.seq_len // max(1, base_tokens.shape[-1]) + 2)
    text = "\n\n".join([base_text] * repeats)
    tokens = model.to_tokens(text, prepend_bos=True)[:, : args.seq_len].to(args.device)

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
            "seq_len": args.seq_len,
            "dtype": args.dtype,
            "q_shape": tuple(q.shape),
            "kv_shape": tuple(k.shape),
            "text": base_text,
            "n_heads": n_heads,
            "n_key_value_heads": n_kv_heads,
            "gqa_group_size": n_heads // n_kv_heads,
            "names_filter": sorted(wanted),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"saved {args.out}")
    print(payload["meta"])


if __name__ == "__main__":
    main()
