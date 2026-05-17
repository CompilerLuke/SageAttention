#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--truncate-seq", type=int, action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def packed_logical_len(k_len: int, q_group: int = 7) -> int:
    full = k_len // q_group
    rem = k_len % q_group
    if rem == 0:
        return full * (q_group + 1)
    return full * (q_group + 1) + 1 + rem


def pad_to_block(x: torch.Tensor, n: int) -> torch.Tensor:
    pad_len = (n - x.size(2) % n) % n
    if pad_len == 0:
        return x.contiguous()
    return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()


def call_silently(fn, *args, **kwargs) -> torch.Tensor:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out


def preprocess_lambda_one(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    per_block_mean: bool = True,
    quant_block_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from sageattn4 import api

    bsz, n_heads, _, dim = q.shape
    q_group = quant_block_size - 1
    device = q.device

    k = k - k.mean(dim=-2, keepdim=True)

    q = pad_to_block(q, 128)
    if per_block_mean:
        q, qm = api.triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm

    k_work = pad_to_block(k, q_group)
    total_k = k_work.size(-2)
    k_blocks = k_work.reshape(bsz, n_heads, total_k // q_group, q_group, dim)
    k_mean = api.round_to_blockscaled_fp4(k_blocks.mean(dim=-2))
    k_res = (k_blocks.float() - k_mean.float().unsqueeze(-2)).to(k.dtype)
    k_packed = torch.cat([k_mean.unsqueeze(-2), k_res], dim=-2)
    k_packed = k_packed.reshape(bsz, n_heads, total_k // q_group * quant_block_size, dim)

    lamb = torch.cat(
        [
            torch.zeros((bsz, n_heads, total_k // q_group, 1), device=device, dtype=torch.float32),
            torch.ones((bsz, n_heads, total_k // q_group, q_group), device=device, dtype=torch.float32),
        ],
        dim=-1,
    ).reshape(bsz, n_heads, total_k // q_group * quant_block_size)

    v_work = pad_to_block(v, q_group)
    total_v = v_work.size(-2)
    v_blocks = v_work.reshape(bsz, n_heads, total_v // q_group, q_group, dim)
    v_mean = torch.zeros((bsz, n_heads, total_v // q_group, dim), device=device, dtype=v.dtype)
    v_packed = torch.cat([v_mean.unsqueeze(-2), v_blocks], dim=-2)
    v_packed = v_packed.reshape(bsz, n_heads, total_v // q_group * quant_block_size, dim)

    k_packed = pad_to_block(k_packed, 128)
    lamb = pad_to_block(lamb.unsqueeze(-1), 128).squeeze(-1)
    v_packed = pad_to_block(v_packed, 128)
    delta_s = torch.matmul(qm, k_packed.transpose(-2, -1)).to(torch.float32).contiguous()
    return q, k_packed, v_packed, delta_s, lamb


def sageattn4_lambda_one(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from sageattn4 import api

    q_len = q.size(2)
    k_len = k.size(2)
    q_p, k_p, v_p, delta_s, lamb = preprocess_lambda_one(q, k, v)
    qlist = api.scale_and_quant_fp4(q_p)
    klist = api.scale_and_quant_fp4_permute(k_p)
    vlist = api.scale_and_quant_fp4_transpose(v_p)
    out = api.blockscaled_fp4_attn(
        qlist,
        klist,
        vlist,
        delta_s,
        lamb,
        packed_logical_len(k_len),
        False,
        True,
        True,
    )[0]
    torch.cuda.synchronize()
    return out[:, :, :q_len, :].contiguous()


def metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    err = (out.float() - ref.float()).abs().reshape(-1).cpu()
    outf = out.float().reshape(-1)
    reff = ref.float().reshape(-1)
    return {
        "mean_cos": float(F.cosine_similarity(outf, reff, dim=0).item()),
        "mae": float(err.mean().item()),
        "q50": float(torch.quantile(err, 0.50).item()),
        "q80": float(torch.quantile(err, 0.80).item()),
        "q95": float(torch.quantile(err, 0.95).item()),
        "max_abs": float(err.max().item()),
    }


@torch.no_grad()
def compare(path: Path, seq_len: int, device: str) -> dict:
    import sageattn3
    import sageattn4

    payload = torch.load(path, map_location="cpu")
    q_cpu = payload["q"][:, :, :seq_len, :].contiguous()
    k_cpu = payload["k"][:, :, :seq_len, :].contiguous()
    v_cpu = payload["v"][:, :, :seq_len, :].contiguous()
    group_size = q_cpu.shape[1] // k_cpu.shape[1]

    per_variant = {"sageattn3": [], "sageattn4": [], "sageattn4_lamb1": []}
    err_chunks = {name: [] for name in per_variant}

    for q_head in range(q_cpu.shape[1]):
        kv_head = q_head // group_size
        q = q_cpu[:, q_head : q_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        k = k_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        v = v_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=False)

        outs = {
            "sageattn3": call_silently(sageattn3.sageattn3_blackwell, q.clone(), k.clone(), v.clone(), is_causal=False, per_block_mean=True),
            "sageattn4": call_silently(sageattn4.sageattn4_blackwell, q.clone(), k.clone(), v.clone(), is_causal=False, per_block_mean=True),
            "sageattn4_lamb1": call_silently(sageattn4_lambda_one, q.clone(), k.clone(), v.clone()),
        }
        for name, out in outs.items():
            err_chunks[name].append((out.float() - ref).abs().reshape(-1).cpu())
            per_variant[name].append(metrics(out, ref))

    summary = {
        "fixture": str(path),
        "seq": seq_len,
        "variants": {},
    }
    for name in per_variant:
        err = torch.cat(err_chunks[name])
        rows = per_variant[name]
        summary["variants"][name] = {
            "mean_cos": float(sum(row["mean_cos"] for row in rows) / len(rows)),
            "mae": float(err.mean().item()),
            "q50": float(torch.quantile(err, 0.50).item()),
            "q80": float(torch.quantile(err, 0.80).item()),
            "q95": float(torch.quantile(err, 0.95).item()),
            "max_abs": float(err.max().item()),
        }
    return summary


def print_summary(results: list[dict]) -> None:
    for result in results:
        print(f"\nseq={result['seq']}")
        print("variant             mean_cos          MAE          q50          q80          q95      max_abs")
        for name in ("sageattn3", "sageattn4", "sageattn4_lamb1"):
            row = result["variants"][name]
            print(
                f"{name:<16} {row['mean_cos']:12.8f} {row['mae']:12.8f} "
                f"{row['q50']:12.8f} {row['q80']:12.8f} {row['q95']:12.8f} {row['max_abs']:12.8f}"
            )
        s4 = result["variants"]["sageattn4"]
        l1 = result["variants"]["sageattn4_lamb1"]
        print(
            "lambda1/computed ratios: "
            f"MAE={l1['mae'] / s4['mae']:.6f} "
            f"q95={l1['q95'] / s4['q95']:.6f}"
        )


def main() -> None:
    args = parse_args()
    seqs = args.truncate_seq or [4096]
    results = [compare(args.fixture, seq, args.device) for seq in seqs]
    print_summary(results)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
