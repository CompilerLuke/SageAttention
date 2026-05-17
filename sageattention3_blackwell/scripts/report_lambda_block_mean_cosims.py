#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.compare_decode_constraint_variants import eval_token_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def round_to_blockscaled_fp4(x: torch.Tensor) -> torch.Tensor:
    assert x.size(-1) % 16 == 0
    orig_dtype = x.dtype
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], xf.shape[-1] // 16, 16)

    scale = (blocks.abs().amax(dim=-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn).float()
    scale_inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    scaled = blocks * scale_inv
    scaled_abs = scaled.abs().clamp(max=6.0)

    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x.device,
        dtype=torch.float32,
    )
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=x.device,
        dtype=torch.float32,
    )
    rounded_code = torch.bucketize(scaled_abs, boundaries)
    tie_round_up = (scaled_abs == 0.75) | (scaled_abs == 1.75) | (scaled_abs == 3.5)
    rounded_code = torch.where(tie_round_up, rounded_code + 1, rounded_code).clamp(max=7)
    rounded_abs = levels[rounded_code]
    rounded = rounded_abs.copysign(scaled)
    return (rounded * scale).reshape_as(xf).to(orig_dtype)


def pad_to_block(x: torch.Tensor, block: int) -> torch.Tensor:
    pad_len = (block - x.size(2) % block) % block
    if pad_len == 0:
        return x.contiguous()
    return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.float().reshape(-1)
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "q05": float(torch.quantile(values, 0.05).item()),
        "q50": float(torch.quantile(values, 0.50).item()),
        "q80": float(torch.quantile(values, 0.80).item()),
        "q95": float(torch.quantile(values, 0.95).item()),
        "q99": float(torch.quantile(values, 0.99).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def fixture_label(path: Path, meta: dict) -> str:
    layer = meta.get("layer")
    if layer is not None:
        return f"L{layer}"
    return path.stem


def valid_group_mask(seq_len: int, groups: int, group_size: int, device: torch.device) -> torch.Tensor:
    group_idx = torch.arange(groups, device=device).unsqueeze(-1)
    offset = torch.arange(group_size, device=device).unsqueeze(0)
    return group_idx * group_size + offset < seq_len


@torch.no_grad()
def q_mean_cosims(q: torch.Tensor, token_slice: slice) -> dict[str, torch.Tensor]:
    q = q.float()
    q_eval = q[:, :, token_slice, :]

    full_mean = q.mean(dim=-2, keepdim=True)
    full_cos = F.cosine_similarity(
        q_eval,
        full_mean.expand(-1, -1, q.size(2), -1)[:, :, token_slice, :],
        dim=-1,
    )

    q_padded = pad_to_block(q, 128)
    bsz, n_heads, seq, dim = q_padded.shape
    blocks = q_padded.reshape(bsz, n_heads, seq // 128, 128, dim)
    block_mean = blocks.mean(dim=-2, keepdim=True).expand_as(blocks).reshape_as(q_padded)
    block_cos = F.cosine_similarity(q_eval, block_mean[:, :, token_slice, :], dim=-1)

    return {
        "q_answer_to_full_mean": full_cos.reshape(-1).cpu(),
        "q_answer_to_block128_mean": block_cos.reshape(-1).cpu(),
    }


@torch.no_grad()
def k_lambda_and_block_cosims(k: torch.Tensor) -> dict[str, torch.Tensor]:
    group_size = 7
    k_len = k.size(2)

    # Match sageattn4 preprocessing: global K centering happens before 7-token packing.
    k = k - k.mean(dim=-2, keepdim=True)
    k_work = pad_to_block(k, group_size)
    bsz, n_heads, total_len, dim = k_work.shape
    blocks = k_work.reshape(bsz, n_heads, total_len // group_size, group_size, dim)

    mean_exact = blocks.mean(dim=-2)
    mean_rounded = round_to_blockscaled_fp4(mean_exact)
    mean_rounded_f = mean_rounded.float()
    norm_sq = (mean_rounded_f * mean_rounded_f).sum(dim=-1, keepdim=True)
    mean_dir = torch.where(norm_sq > 0, mean_rounded_f / norm_sq, torch.zeros_like(mean_rounded_f))
    lamb = torch.einsum("b h n m d, b h n d -> b h n m", blocks.float(), mean_dir)

    token_valid = valid_group_mask(k_len, blocks.size(2), group_size, k.device)
    block_valid = token_valid.any(dim=-1)
    full_group = token_valid.all(dim=-1)
    full_group_token_valid = token_valid & full_group.unsqueeze(-1)

    token_to_exact = F.cosine_similarity(blocks.float(), mean_exact.float().unsqueeze(-2), dim=-1)
    token_to_rounded = F.cosine_similarity(blocks.float(), mean_rounded_f.unsqueeze(-2), dim=-1)
    exact_to_rounded = F.cosine_similarity(mean_exact.float(), mean_rounded_f, dim=-1)

    return {
        "lambda": lamb[..., token_valid].reshape(-1).cpu(),
        "abs_lambda": lamb[..., token_valid].abs().reshape(-1).cpu(),
        "lambda_full_groups": lamb[..., full_group_token_valid].reshape(-1).cpu(),
        "abs_lambda_full_groups": lamb[..., full_group_token_valid].abs().reshape(-1).cpu(),
        "k_token_to_exact_block7_mean": token_to_exact[..., token_valid].reshape(-1).cpu(),
        "k_token_to_rounded_block7_mean": token_to_rounded[..., token_valid].reshape(-1).cpu(),
        "k_token_to_exact_block7_mean_full_groups": token_to_exact[..., full_group_token_valid].reshape(-1).cpu(),
        "k_token_to_rounded_block7_mean_full_groups": token_to_rounded[..., full_group_token_valid].reshape(-1).cpu(),
        "k_exact_block7_mean_to_rounded": exact_to_rounded[..., block_valid].reshape(-1).cpu(),
        "k_exact_block7_mean_to_rounded_full_groups": exact_to_rounded[..., full_group].reshape(-1).cpu(),
    }


@torch.no_grad()
def collect_fixture(path: Path, device: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    q_cpu = payload["q"].contiguous()
    k_cpu = payload["k"].contiguous()
    meta = payload.get("meta", {})
    token_slice = eval_token_slice(meta, q_cpu.shape[2], True)
    kv_end = token_slice.start

    q = q_cpu.to(device=device, dtype=torch.bfloat16).contiguous()
    k = k_cpu[:, :, :kv_end, :].to(device=device, dtype=torch.bfloat16).contiguous()

    values = {}
    values.update(q_mean_cosims(q, token_slice))
    values.update(k_lambda_and_block_cosims(k))

    return {
        "label": fixture_label(path, meta),
        "source": Path(meta["text_file"]).name if meta.get("text_file") else path.name,
        "q_seq": int(q_cpu.shape[2]),
        "kv_seq": int(k.shape[2]),
        "eval_range": [int(token_slice.start), int(token_slice.stop)],
        "values": values,
    }


def aggregate(rows: list[dict]) -> dict:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    out = {}
    for label, layer_rows in grouped.items():
        metrics = sorted({metric for row in layer_rows for metric in row["values"]})
        out[label] = {
            "sources": [row["source"] for row in layer_rows],
            "q_seqs": [row["q_seq"] for row in layer_rows],
            "kv_seqs": [row["kv_seq"] for row in layer_rows],
            "eval_ranges": [row["eval_range"] for row in layer_rows],
            "summary": {},
        }
        for metric in metrics:
            values = torch.cat([row["values"][metric] for row in layer_rows])
            out[label]["summary"][metric] = summarize(values)
    return out


def print_summary(summary: dict) -> None:
    for layer, payload in summary.items():
        print(f"\n{layer} sources={','.join(payload['sources'])}")
        print(f"q_seq={payload['q_seqs']} kv_seq={payload['kv_seqs']} eval_ranges={payload['eval_ranges']}")
        print("metric                              mean        std        q05        q50        q80        q95        q99        min        max      count")
        for metric in (
            "lambda",
            "abs_lambda",
            "lambda_full_groups",
            "abs_lambda_full_groups",
            "k_token_to_exact_block7_mean",
            "k_token_to_rounded_block7_mean",
            "k_token_to_exact_block7_mean_full_groups",
            "k_token_to_rounded_block7_mean_full_groups",
            "k_exact_block7_mean_to_rounded",
            "k_exact_block7_mean_to_rounded_full_groups",
            "q_answer_to_block128_mean",
            "q_answer_to_full_mean",
        ):
            row = payload["summary"][metric]
            print(
                f"{metric:<34} "
                f"{row['mean']:10.6f} {row['std']:10.6f} {row['q05']:10.6f} "
                f"{row['q50']:10.6f} {row['q80']:10.6f} {row['q95']:10.6f} "
                f"{row['q99']:10.6f} {row['min']:10.6f} {row['max']:10.6f} {row['count']:10d}"
            )


def main() -> None:
    args = parse_args()
    rows = [collect_fixture(path, args.device) for path in args.fixtures]
    summary = aggregate(rows)
    print_summary(summary)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(args.out_json)


if __name__ == "__main__":
    main()
