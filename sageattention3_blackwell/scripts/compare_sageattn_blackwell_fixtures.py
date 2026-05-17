#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--max-heads", type=int, default=0)
    parser.add_argument("--truncate-seq", type=int, action="append", default=[])
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def logical_x_block_mean_cosine(k: torch.Tensor) -> torch.Tensor:
    from sageattn4.api import round_to_blockscaled_fp4

    quant_block = 8
    q_group = quant_block - 1
    bsz, n_heads, k_len, dim = k.shape
    k = k - k.mean(dim=-2, keepdim=True)
    pad_len = (q_group - k_len % q_group) % q_group
    if pad_len:
        k_work = F.pad(k, (0, 0, 0, pad_len), value=0).contiguous()
    else:
        k_work = k.contiguous()
    total_len = k_work.shape[-2]
    blocks = k_work.reshape(bsz, n_heads, total_len // q_group, q_group, dim)
    mean = round_to_blockscaled_fp4(blocks.mean(dim=-2))
    cos = F.cosine_similarity(blocks.float(), mean.float().unsqueeze(-2), dim=-1)

    group_idx = torch.arange(cos.shape[2], device=k.device).unsqueeze(-1)
    offset = torch.arange(q_group, device=k.device).unsqueeze(0)
    token_idx = group_idx * q_group + offset
    valid = token_idx < k_len
    return cos[..., valid].float().reshape(-1)


def call_silently(fn, *args, **kwargs) -> torch.Tensor:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out


def per_head_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    out_f = out.float().reshape(-1)
    ref_f = ref.float().reshape(-1)
    err = (out_f - ref_f).abs()
    return {
        "cos": float(F.cosine_similarity(out_f, ref_f, dim=0).item()),
        "mae": float(err.mean().item()),
        "max_abs": float(err.max().item()),
    }


@torch.no_grad()
def compare_fixture(path: Path, device: str, max_heads: int, truncate_seq: int) -> dict[str, Any]:
    import sageattn3
    import sageattn4

    payload = torch.load(path, map_location="cpu")
    q_cpu = payload["q"].contiguous()
    k_cpu = payload["k"].contiguous()
    v_cpu = payload["v"].contiguous()
    if truncate_seq > 0:
        q_cpu = q_cpu[:, :, :truncate_seq, :].contiguous()
        k_cpu = k_cpu[:, :, :truncate_seq, :].contiguous()
        v_cpu = v_cpu[:, :, :truncate_seq, :].contiguous()
    meta = payload.get("meta", {})
    n_q_heads = q_cpu.shape[1]
    n_kv_heads = k_cpu.shape[1]
    group_size = n_q_heads // n_kv_heads
    if max_heads > 0:
        n_q_heads = min(n_q_heads, max_heads)

    rows = {"sageattn3": [], "sageattn4": []}
    errors = {"sageattn3": [], "sageattn4": []}
    x_block_mean_cosine_chunks = []

    for q_head in range(n_q_heads):
        kv_head = q_head // group_size
        q = q_cpu[:, q_head : q_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        k = k_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        v = v_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()

        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=False)
        out3 = call_silently(sageattn3.sageattn3_blackwell, q.clone(), k.clone(), v.clone(), is_causal=False, per_block_mean=True)
        out4 = call_silently(sageattn4.sageattn4_blackwell, q.clone(), k.clone(), v.clone(), is_causal=False, per_block_mean=True)

        rows["sageattn3"].append({"q_head": q_head, "kv_head": kv_head, **per_head_metrics(out3, ref)})
        rows["sageattn4"].append({"q_head": q_head, "kv_head": kv_head, **per_head_metrics(out4, ref)})
        errors["sageattn3"].append((out3.float() - ref).abs().reshape(-1).cpu())
        errors["sageattn4"].append((out4.float() - ref).abs().reshape(-1).cpu())
        x_block_mean_cosine_chunks.append(logical_x_block_mean_cosine(k).cpu())

        del q, k, v, ref, out3, out4

    summary: dict[str, Any] = {
        "fixture": str(path),
        "truncate_seq": truncate_seq,
        "meta": meta,
        "shape": {
            "q": tuple(q_cpu.shape),
            "k": tuple(k_cpu.shape),
            "v": tuple(v_cpu.shape),
        },
        "heads_compared": n_q_heads,
        "blackwell_paths": {
            "sageattn3": str(Path(sageattn3.__file__).resolve()),
            "sageattn4": str(Path(sageattn4.__file__).resolve()),
        },
        "variants": {},
    }

    for name in ("sageattn3", "sageattn4"):
        per_head = rows[name]
        err = torch.cat(errors[name])
        summary["variants"][name] = {
            "mean_cos": float(sum(row["cos"] for row in per_head) / len(per_head)),
            "min_cos": float(min(row["cos"] for row in per_head)),
            "mae": float(err.mean().item()),
            "max_abs": float(err.max().item()),
            "abs_error_quantiles": {
                "0.50": float(torch.quantile(err, 0.50).item()),
                "0.80": float(torch.quantile(err, 0.80).item()),
                "0.95": float(torch.quantile(err, 0.95).item()),
            },
            "per_head": per_head,
        }

    cos = torch.cat(x_block_mean_cosine_chunks)
    summary["x_block_mean_cosine"] = {
        "mean": float(cos.mean().item()),
        "std": float(cos.std(unbiased=False).item()),
        "q50": float(torch.quantile(cos, 0.50).item()),
        "q80": float(torch.quantile(cos, 0.80).item()),
        "q95": float(torch.quantile(cos, 0.95).item()),
        "count": int(cos.numel()),
    }
    return summary


def print_summary(results: list[dict[str, Any]]) -> None:
    for result in results:
        seq_len = result["shape"]["q"][2]
        print(f"\nfixture={result['fixture']} seq={seq_len} heads={result['heads_compared']}")
        print("paths:", json.dumps(result["blackwell_paths"], sort_keys=True))
        cos = result["x_block_mean_cosine"]
        print(
            f"x_block/x_mean cos mean={cos['mean']:.8f} std={cos['std']:.8f} "
            f"q50={cos['q50']:.8f} q80={cos['q80']:.8f} q95={cos['q95']:.8f} "
            f"count={cos['count']}"
        )
        print("variant       mean_cos     min_cos          MAE          q50          q80          q95      max_abs")
        for name in ("sageattn3", "sageattn4"):
            row = result["variants"][name]
            q = row["abs_error_quantiles"]
            print(
                f"{name:<10} {row['mean_cos']:12.8f} {row['min_cos']:11.8f} "
                f"{row['mae']:12.8f} {q['0.50']:12.8f} {q['0.80']:12.8f} "
                f"{q['0.95']:12.8f} {row['max_abs']:12.8f}"
            )
        s3 = result["variants"]["sageattn3"]
        s4 = result["variants"]["sageattn4"]
        print(
            "s4/s3 ratios: "
            f"MAE={s4['mae'] / s3['mae']:.6f} "
            f"q95={s4['abs_error_quantiles']['0.95'] / s3['abs_error_quantiles']['0.95']:.6f}"
        )


def main() -> None:
    args = parse_args()
    truncate_seqs = args.truncate_seq or [0]
    results = [
        compare_fixture(path, args.device, args.max_heads, truncate_seq)
        for path in args.fixtures
        for truncate_seq in truncate_seqs
    ]
    print_summary(results)
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
