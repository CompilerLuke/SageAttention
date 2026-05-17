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
    parser.add_argument(
        "--q-per-block",
        dest="per_block_mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-128-token Q mean subtraction. Disable for a single global Q mean.",
    )
    parser.add_argument(
        "--question-section",
        action="store_true",
        help="Restrict output error metrics to the fixture's recorded question token span.",
    )
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


def _error_summary(err: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(err.mean().item()),
        "mae": float(err.mean().item()),
        "q50": float(torch.quantile(err, 0.50).item()),
        "q80": float(torch.quantile(err, 0.80).item()),
        "q95": float(torch.quantile(err, 0.95).item()),
        "max_abs": float(err.max().item()),
    }


def _vector_metrics(approx: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    approx_f = approx.float()
    target_f = target.float()
    diff = approx_f - target_f
    return {
        "abs_error": diff.abs().reshape(-1),
        "vector_mae": diff.abs().mean(dim=-1).reshape(-1),
        "rmse": torch.sqrt((diff * diff).mean(dim=-1)).reshape(-1),
        "one_minus_cos": (1.0 - F.cosine_similarity(approx_f, target_f, dim=-1)).clamp_min(0).reshape(-1),
    }


def logical_k_rounding_errors(k: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    from sageattn4.api import round_to_blockscaled_fp4

    quant_block = 8
    q_group = quant_block - 1
    bsz, n_heads, k_len, dim = k.shape
    k = k - k.mean(dim=-2, keepdim=True)

    direct = round_to_blockscaled_fp4(k)
    direct_metrics = _vector_metrics(direct, k)

    pad_len = (q_group - k_len % q_group) % q_group
    if pad_len:
        k_work = F.pad(k, (0, 0, 0, pad_len), value=0).contiguous()
    else:
        k_work = k.contiguous()
    total_len = k_work.shape[-2]
    blocks = k_work.reshape(bsz, n_heads, total_len // q_group, q_group, dim)

    mean = round_to_blockscaled_fp4(blocks.mean(dim=-2))
    mean_f = mean.float()
    norm_sq = (mean_f * mean_f).sum(dim=-1, keepdim=True)
    mean_dir = torch.where(norm_sq > 0, mean_f / norm_sq, torch.zeros_like(mean_f))
    lamb = torch.einsum("b h n m d, b h n d -> b h n m", blocks.float(), mean_dir)
    residual = (blocks.float() - lamb.unsqueeze(-1) * mean_f.unsqueeze(-2)).to(k.dtype)
    residual_rounded = round_to_blockscaled_fp4(residual)
    reconstructed = residual_rounded.float() + lamb.unsqueeze(-1) * mean_f.unsqueeze(-2)
    reconstructed = reconstructed.reshape(bsz, n_heads, total_len, dim)[:, :, :k_len, :]
    return {
        "direct": direct_metrics,
        "residual_reconstruct": _vector_metrics(reconstructed, k),
    }


def call_silently(fn, *args, **kwargs) -> torch.Tensor:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out


def per_head_metrics(out: torch.Tensor, ref: torch.Tensor, token_slice: slice | None = None) -> dict[str, float]:
    if token_slice is not None:
        out = out[:, :, token_slice, :]
        ref = ref[:, :, token_slice, :]
    out_f = out.float().reshape(-1)
    ref_f = ref.float().reshape(-1)
    err = (out_f - ref_f).abs()
    return {
        "cos": float(F.cosine_similarity(out_f, ref_f, dim=0).item()),
        "mae": float(err.mean().item()),
        "max_abs": float(err.max().item()),
    }


def output_distribution_metrics(out: torch.Tensor, ref: torch.Tensor, token_slice: slice | None = None) -> dict[str, torch.Tensor]:
    if token_slice is not None:
        out = out[:, :, token_slice, :]
        ref = ref[:, :, token_slice, :]
    out_f = out.float()
    ref_f = ref.float()
    diff = out_f - ref_f
    vector_mae = diff.abs().mean(dim=-1).reshape(-1)
    rmse = torch.sqrt((diff * diff).mean(dim=-1)).reshape(-1)
    one_minus_cos = (1.0 - F.cosine_similarity(out_f, ref_f, dim=-1)).clamp_min(0).reshape(-1)
    return {
        "vector_mae": vector_mae,
        "rmse": rmse,
        "one_minus_cos": one_minus_cos,
    }


def question_token_slice(meta: dict[str, Any], seq_len: int, enabled: bool) -> slice | None:
    if not enabled:
        return None
    start = int(meta["question_token_start"])
    end = int(meta["question_token_end"])
    start = max(0, min(start, seq_len))
    end = max(start, min(end, seq_len))
    if end <= start:
        raise ValueError(f"empty question token span after truncation: start={start}, end={end}, seq_len={seq_len}")
    return slice(start, end)


@torch.no_grad()
def compare_fixture(
    path: Path,
    device: str,
    max_heads: int,
    truncate_seq: int,
    per_block_mean: bool,
    use_question_section: bool,
) -> dict[str, Any]:
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
    output_token_slice = question_token_slice(meta, q_cpu.shape[2], use_question_section)
    n_q_heads = q_cpu.shape[1]
    n_kv_heads = k_cpu.shape[1]
    group_size = n_q_heads // n_kv_heads
    if max_heads > 0:
        n_q_heads = min(n_q_heads, max_heads)

    rows = {"sageattn3": [], "sageattn4": []}
    errors = {"sageattn3": [], "sageattn4": []}
    output_distributions = {
        "sageattn3": {"vector_mae": [], "rmse": [], "one_minus_cos": []},
        "sageattn4": {"vector_mae": [], "rmse": [], "one_minus_cos": []},
    }
    x_block_mean_cosine_chunks = []
    k_rounding_metric_chunks = {
        "direct": {"abs_error": [], "vector_mae": [], "rmse": [], "one_minus_cos": []},
        "residual_reconstruct": {"abs_error": [], "vector_mae": [], "rmse": [], "one_minus_cos": []},
    }

    for q_head in range(n_q_heads):
        kv_head = q_head // group_size
        q = q_cpu[:, q_head : q_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        k = k_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        v = v_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()

        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=False)
        out3 = call_silently(sageattn3.sageattn3_blackwell, q.clone(), k.clone(), v.clone(), is_causal=False, per_block_mean=per_block_mean)
        out4 = call_silently(sageattn4.sageattn4_blackwell, q.clone(), k.clone(), v.clone(), is_causal=False, per_block_mean=per_block_mean)

        rows["sageattn3"].append({"q_head": q_head, "kv_head": kv_head, **per_head_metrics(out3, ref, output_token_slice)})
        rows["sageattn4"].append({"q_head": q_head, "kv_head": kv_head, **per_head_metrics(out4, ref, output_token_slice)})
        out3_for_error = out3[:, :, output_token_slice, :] if output_token_slice is not None else out3
        out4_for_error = out4[:, :, output_token_slice, :] if output_token_slice is not None else out4
        ref_for_error = ref[:, :, output_token_slice, :] if output_token_slice is not None else ref
        errors["sageattn3"].append((out3_for_error.float() - ref_for_error).abs().reshape(-1).cpu())
        errors["sageattn4"].append((out4_for_error.float() - ref_for_error).abs().reshape(-1).cpu())
        for name, out in (("sageattn3", out3), ("sageattn4", out4)):
            for metric, values in output_distribution_metrics(out, ref, output_token_slice).items():
                output_distributions[name][metric].append(values.cpu())
        x_block_mean_cosine_chunks.append(logical_x_block_mean_cosine(k).cpu())
        for name, metric_values in logical_k_rounding_errors(k).items():
            for metric, values in metric_values.items():
                k_rounding_metric_chunks[name][metric].append(values.cpu())

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
        "per_block_mean": per_block_mean,
        "output_token_range": (
            [output_token_slice.start, output_token_slice.stop]
            if output_token_slice is not None
            else None
        ),
        "blackwell_paths": {
            "sageattn3": str(Path(sageattn3.__file__).resolve()),
            "sageattn4": str(Path(sageattn4.__file__).resolve()),
        },
        "variants": {},
    }

    for name in ("sageattn3", "sageattn4"):
        per_head = rows[name]
        err = torch.cat(errors[name])
        vector_mae = torch.cat(output_distributions[name]["vector_mae"])
        rmse = torch.cat(output_distributions[name]["rmse"])
        one_minus_cos = torch.cat(output_distributions[name]["one_minus_cos"])
        summary["variants"][name] = {
            "mean_cos": float(sum(row["cos"] for row in per_head) / len(per_head)),
            "min_cos": float(min(row["cos"] for row in per_head)),
            "mean_1_minus_cos": float(sum(1.0 - row["cos"] for row in per_head) / len(per_head)),
            "mae": float(err.mean().item()),
            "max_abs": float(err.max().item()),
            "abs_error_quantiles": {
                "0.50": float(torch.quantile(err, 0.50).item()),
                "0.80": float(torch.quantile(err, 0.80).item()),
                "0.95": float(torch.quantile(err, 0.95).item()),
            },
            "vector_mae_quantiles": {
                "mean": float(vector_mae.mean().item()),
                "0.50": float(torch.quantile(vector_mae, 0.50).item()),
                "0.80": float(torch.quantile(vector_mae, 0.80).item()),
                "0.95": float(torch.quantile(vector_mae, 0.95).item()),
                "max": float(vector_mae.max().item()),
            },
            "rmse_quantiles": {
                "mean": float(rmse.mean().item()),
                "0.50": float(torch.quantile(rmse, 0.50).item()),
                "0.80": float(torch.quantile(rmse, 0.80).item()),
                "0.95": float(torch.quantile(rmse, 0.95).item()),
                "max": float(rmse.max().item()),
            },
            "one_minus_cos_quantiles": {
                "mean": float(one_minus_cos.mean().item()),
                "0.50": float(torch.quantile(one_minus_cos, 0.50).item()),
                "0.80": float(torch.quantile(one_minus_cos, 0.80).item()),
                "0.95": float(torch.quantile(one_minus_cos, 0.95).item()),
                "max": float(one_minus_cos.max().item()),
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
    summary["k_rounding_error"] = {
        name: {
            metric: _error_summary(torch.cat(chunks))
            for metric, chunks in metric_chunks.items()
        }
        for name, metric_chunks in k_rounding_metric_chunks.items()
    }
    return summary


def print_summary(results: list[dict[str, Any]]) -> None:
    for result in results:
        seq_len = result["shape"]["q"][2]
        print(f"\nfixture={result['fixture']} seq={seq_len} heads={result['heads_compared']}")
        print(f"q_per_block={result['per_block_mean']}")
        if result["output_token_range"] is not None:
            start, end = result["output_token_range"]
            print(f"output_error_token_range=[{start}, {end}) count={end - start}")
        print("paths:", json.dumps(result["blackwell_paths"], sort_keys=True))
        cos = result["x_block_mean_cosine"]
        print(
            f"x_block/x_mean cos mean={cos['mean']:.8f} std={cos['std']:.8f} "
            f"q50={cos['q50']:.8f} q80={cos['q80']:.8f} q95={cos['q95']:.8f} "
            f"count={cos['count']}"
        )
        print("K rounding uses global K centering; padding tokens are excluded.")
        print("k_quant       RMSE_mean    RMSE_q50    RMSE_q80    RMSE_q95    RMSE_max")
        for name in ("direct", "residual_reconstruct"):
            row = result["k_rounding_error"][name]["rmse"]
            print(
                f"{name:<12} {row['mean']:12.8f} {row['q50']:12.8f} "
                f"{row['q80']:12.8f} {row['q95']:12.8f} {row['max_abs']:12.8f}"
            )
        direct = result["k_rounding_error"]["direct"]["rmse"]
        residual = result["k_rounding_error"]["residual_reconstruct"]["rmse"]
        print(
            "residual/direct K-error ratios: "
            f"RMSE_mean={residual['mean'] / direct['mean']:.6f} "
            f"q95={residual['q95'] / direct['q95']:.6f}"
        )
        print("k-vector distributions:")
        print("k_quant        RMSE_mean    RMSE_q50    RMSE_q80    RMSE_q95    1-cos_mean     1-cos_q50     1-cos_q80     1-cos_q95")
        for name in ("direct", "residual_reconstruct"):
            rm = result["k_rounding_error"][name]["rmse"]
            oc = result["k_rounding_error"][name]["one_minus_cos"]
            print(
                f"{name:<12} {rm['mean']:12.8f} {rm['q50']:12.8f} {rm['q80']:12.8f} "
                f"{rm['q95']:12.8f} {oc['mean']:14.10f} {oc['q50']:14.10f} "
                f"{oc['q80']:14.10f} {oc['q95']:14.10f}"
            )
        direct_vec = result["k_rounding_error"]["direct"]
        residual_vec = result["k_rounding_error"]["residual_reconstruct"]
        print(
            "residual/direct K-vector ratios: "
            f"RMSE_mean={residual_vec['rmse']['mean'] / direct_vec['rmse']['mean']:.6f} "
            f"RMSE_q95={residual_vec['rmse']['q95'] / direct_vec['rmse']['q95']:.6f} "
            f"1-cos_mean={residual_vec['one_minus_cos']['mean'] / direct_vec['one_minus_cos']['mean']:.6f} "
            f"1-cos_q95={residual_vec['one_minus_cos']['q95'] / direct_vec['one_minus_cos']['q95']:.6f}"
        )
        print("variant       mean_cos  mean_1-cos    RMSE_mean    RMSE_q50    RMSE_q80    RMSE_q95    RMSE_max")
        for name in ("sageattn3", "sageattn4"):
            row = result["variants"][name]
            q = row["rmse_quantiles"]
            print(
                f"{name:<10} {row['mean_cos']:12.8f} {row['mean_1_minus_cos']:12.8f} "
                f"{q['mean']:12.8f} {q['0.50']:12.8f} {q['0.80']:12.8f} "
                f"{q['0.95']:12.8f} {q['max']:12.8f}"
            )
        s3 = result["variants"]["sageattn3"]
        s4 = result["variants"]["sageattn4"]
        print(
            "s4/s3 ratios: "
            f"RMSE_mean={s4['rmse_quantiles']['mean'] / s3['rmse_quantiles']['mean']:.6f} "
            f"RMSE_q95={s4['rmse_quantiles']['0.95'] / s3['rmse_quantiles']['0.95']:.6f}"
        )
        print("token-vector output distributions:")
        print("variant         RMSE_mean    RMSE_q50    RMSE_q80    RMSE_q95    1-cos_mean     1-cos_q50     1-cos_q80     1-cos_q95")
        for name in ("sageattn3", "sageattn4"):
            row = result["variants"][name]
            rm = row["rmse_quantiles"]
            oc = row["one_minus_cos_quantiles"]
            print(
                f"{name:<10} {rm['mean']:12.8f} {rm['0.50']:12.8f} {rm['0.80']:12.8f} "
                f"{rm['0.95']:12.8f} {oc['mean']:14.10f} {oc['0.50']:14.10f} "
                f"{oc['0.80']:14.10f} {oc['0.95']:14.10f}"
            )
        print(
            "s4/s3 token-vector ratios: "
            f"RMSE_mean={s4['rmse_quantiles']['mean'] / s3['rmse_quantiles']['mean']:.6f} "
            f"RMSE_q95={s4['rmse_quantiles']['0.95'] / s3['rmse_quantiles']['0.95']:.6f} "
            f"1-cos_mean={s4['one_minus_cos_quantiles']['mean'] / s3['one_minus_cos_quantiles']['mean']:.6f} "
            f"1-cos_q95={s4['one_minus_cos_quantiles']['0.95'] / s3['one_minus_cos_quantiles']['0.95']:.6f}"
        )


def main() -> None:
    args = parse_args()
    truncate_seqs = args.truncate_seq or [0]
    results = [
        compare_fixture(path, args.device, args.max_heads, truncate_seq, args.per_block_mean, args.question_section)
        for path in args.fixtures
        for truncate_seq in truncate_seqs
    ]
    print_summary(results)
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
