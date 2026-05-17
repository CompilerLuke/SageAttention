#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from scripts.compare_sageattn_blackwell_fixtures import (
    logical_k_rounding_errors,
    output_distribution_metrics,
    question_token_slice,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--max-heads", type=int, default=0)
    parser.add_argument("--question-section", action="store_true")
    parser.add_argument("--prefix", default="sageattn_qft_4k")
    parser.add_argument("--hist-bins", type=int, default=40)
    parser.add_argument("--hist-cap-stds", type=float, default=1.0)
    parser.add_argument("--log-hist-cap-quantile", type=float, default=0.995)
    parser.add_argument("--aggregate-by-label", action="store_true", help="Pool distributions from fixtures with the same plot label.")
    return parser.parse_args()


def call_silently(fn, *args, **kwargs) -> torch.Tensor:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out


def fixture_label(path: Path, meta: dict[str, Any]) -> str:
    layer = meta.get("layer")
    if layer is not None:
        return f"L{layer}"
    stem = path.stem
    for part in stem.split("_"):
        if part.startswith("l") and part[1:].isdigit():
            return f"L{part[1:]}"
    return stem


def collect_distributions(path: Path, device: str, max_heads: int, use_question_section: bool) -> dict[str, Any]:
    import sageattn3
    import sageattn4

    payload = torch.load(path, map_location="cpu")
    q_cpu = payload["q"].contiguous()
    k_cpu = payload["k"].contiguous()
    v_cpu = payload["v"].contiguous()
    meta = payload.get("meta", {})
    token_slice = question_token_slice(meta, q_cpu.shape[2], use_question_section)
    n_q_heads = q_cpu.shape[1]
    n_kv_heads = k_cpu.shape[1]
    group_size = n_q_heads // n_kv_heads
    if max_heads > 0:
        n_q_heads = min(n_q_heads, max_heads)

    output = {
        "sageattn3": {"rmse": [], "one_minus_cos": []},
        "sageattn4": {"rmse": [], "one_minus_cos": []},
    }
    k_metrics = {
        "direct": {"rmse": [], "one_minus_cos": []},
        "residual_reconstruct": {"rmse": [], "one_minus_cos": []},
    }

    for q_head in range(n_q_heads):
        kv_head = q_head // group_size
        q = q_cpu[:, q_head : q_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        k = k_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        v = v_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()

        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=False)
        out3 = call_silently(sageattn3.sageattn3_blackwell, q.clone(), k.clone(), v.clone(), is_causal=False, per_block_mean=True)
        out4 = call_silently(sageattn4.sageattn4_blackwell, q.clone(), k.clone(), v.clone(), is_causal=False, per_block_mean=True)

        for name, out in (("sageattn3", out3), ("sageattn4", out4)):
            output_metrics = output_distribution_metrics(out, ref, token_slice)
            output[name]["rmse"].append(output_metrics["rmse"].cpu())
            output[name]["one_minus_cos"].append(output_metrics["one_minus_cos"].cpu())

        for name, metric_values in logical_k_rounding_errors(k).items():
            k_metrics[name]["rmse"].append(metric_values["rmse"].cpu())
            k_metrics[name]["one_minus_cos"].append(metric_values["one_minus_cos"].cpu())

        del q, k, v, ref, out3, out4

    return {
        "label": fixture_label(path, meta),
        "seq": q_cpu.shape[2],
        "source": Path(meta["text_file"]).name if meta.get("text_file") else meta.get("text_sha256", path.name),
        "output_token_range": [token_slice.start, token_slice.stop] if token_slice is not None else None,
        "output": {
            name: {metric: torch.cat(chunks) for metric, chunks in metrics.items()}
            for name, metrics in output.items()
        },
        "k": {
            name: {metric: torch.cat(chunks) for metric, chunks in metrics.items()}
            for name, metrics in k_metrics.items()
        },
    }


def aggregate_by_label(collected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for item in collected:
        grouped.setdefault(item["label"], []).append(item)

    aggregated = []
    for label, items in grouped.items():
        seqs = [int(item["seq"]) for item in items]
        seq = seqs[0] if len(set(seqs)) == 1 else f"{min(seqs)}-{max(seqs)}"
        output_ranges = [item["output_token_range"] for item in items]
        output_token_range = output_ranges[0] if all(r == output_ranges[0] for r in output_ranges) else None
        out_item = {
            "label": f"{label} n={len(items)}",
            "seq": seq,
            "source_count": len(items),
            "sources": [item["source"] for item in items],
            "output_token_range": output_token_range,
            "output": {},
            "k": {},
        }
        for group in ("output", "k"):
            variants = items[0][group].keys()
            for variant in variants:
                out_item[group][variant] = {}
                metrics = items[0][group][variant].keys()
                for metric in metrics:
                    chunks = [item[group][variant][metric] for item in items]
                    out_item[group][variant][metric] = torch.cat(chunks)
                    out_item[group][variant][f"{metric}_chunks"] = chunks
        aggregated.append(out_item)
    return aggregated


def plot_ecdf(ax, values: torch.Tensor, label: str) -> None:
    probs = torch.cat(
        [
            torch.linspace(0.0, 0.99, 300),
            torch.linspace(0.991, 0.999, 80),
        ]
    )
    q = torch.quantile(values.float(), probs)
    ax.plot(q.numpy(), probs.numpy(), label=label, linewidth=1.8)


def make_plot(
    collected: list[dict[str, Any]],
    group: str,
    variants: tuple[str, str],
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(len(collected), 2, figsize=(12, 4.8 * len(collected)), squeeze=False)
    metric_titles = {"rmse": "token/vector RMSE", "one_minus_cos": "token/vector 1 - cosine"}

    for row, item in enumerate(collected):
        for col, metric in enumerate(("rmse", "one_minus_cos")):
            ax = axes[row][col]
            for variant in variants:
                plot_ecdf(ax, item[group][variant][metric], variant)
            suffix = ""
            if group == "output" and item["output_token_range"] is not None:
                start, end = item["output_token_range"]
                suffix = f" tokens [{start}, {end})"
            ax.set_title(f"{item['label']} seq={item['seq']}{suffix} {metric_titles[metric]}")
            ax.set_xlabel(metric_titles[metric])
            ax.set_ylabel("CDF")
            ax.grid(True, alpha=0.25)
            ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_hist_plot(
    collected: list[dict[str, Any]],
    group: str,
    variants: tuple[str, str],
    title: str,
    out_path: Path,
    hist_bins: int,
    cap_stds: float,
) -> None:
    fig, axes = plt.subplots(len(collected), 2, figsize=(12, 4.8 * len(collected)), squeeze=False)
    metric_titles = {"rmse": "token/vector RMSE", "one_minus_cos": "token/vector 1 - cosine"}

    for row, item in enumerate(collected):
        for col, metric in enumerate(("rmse", "one_minus_cos")):
            ax = axes[row][col]
            values = [item[group][variant][metric].float() for variant in variants]
            combined = torch.cat(values).float()
            x_max = float((combined.mean() + cap_stds * combined.std(unbiased=False)).item())
            if x_max <= 0:
                x_max = max(v.max().item() for v in values)
            bins = torch.linspace(0.0, x_max, hist_bins + 1).numpy()
            for variant, vals in zip(variants, values):
                chunks = item[group][variant].get(f"{metric}_chunks")
                if chunks is None:
                    visible = vals[vals <= x_max]
                    density, edges = np.histogram(visible.numpy(), bins=bins, density=True)
                else:
                    chunk_densities = []
                    edges = bins
                    for chunk in chunks:
                        visible = chunk.float()[chunk.float() <= x_max]
                        if visible.numel() == 0:
                            chunk_densities.append(np.zeros(hist_bins))
                            continue
                        density, edges = np.histogram(visible.numpy(), bins=bins, density=True)
                        chunk_densities.append(density)
                    density = np.mean(chunk_densities, axis=0)
                centers = 0.5 * (edges[:-1] + edges[1:])
                (line,) = ax.plot(centers, density, linewidth=1.8, label=variant)
                median = float(torch.quantile(vals, 0.50).item())
                if median <= x_max:
                    ax.axvline(
                        median,
                        color=line.get_color(),
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.8,
                        label=f"{variant} median",
                    )
            suffix = ""
            if group == "output" and item["output_token_range"] is not None:
                start, end = item["output_token_range"]
                suffix = f" tokens [{start}, {end})"
            ax.set_title(f"{item['label']} seq={item['seq']}{suffix} {metric_titles[metric]}")
            ax.set_xlabel(f"{metric_titles[metric]} (clipped at mean+{cap_stds:g}std)")
            ax.set_ylabel("density")
            ax.grid(True, alpha=0.25)
            ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_log_hist_plot(
    collected: list[dict[str, Any]],
    group: str,
    variants: tuple[str, str],
    title: str,
    out_path: Path,
    hist_bins: int,
    cap_quantile: float,
) -> None:
    fig, axes = plt.subplots(len(collected), 2, figsize=(12, 4.8 * len(collected)), squeeze=False)
    metric_titles = {"rmse": "token/vector RMSE", "one_minus_cos": "token/vector 1 - cosine"}

    for row, item in enumerate(collected):
        for col, metric in enumerate(("rmse", "one_minus_cos")):
            ax = axes[row][col]
            raw_values = [item[group][variant][metric].float() for variant in variants]
            combined = torch.cat(raw_values)
            positive = combined[combined > 0]
            eps = float(positive.min().item()) if positive.numel() else 1e-12
            log_values = [np.log10(np.maximum(vals.numpy(), eps)) for vals in raw_values]
            lower = min(float(np.quantile(vals, 0.001)) for vals in log_values)
            upper = max(float(np.quantile(vals, cap_quantile)) for vals in log_values)
            if lower == upper:
                lower -= 0.5
                upper += 0.5
            bins = np.linspace(lower, upper, hist_bins + 1)
            for variant, vals in zip(variants, log_values):
                chunks = item[group][variant].get(f"{metric}_chunks")
                if chunks is None:
                    visible = vals[(vals >= lower) & (vals <= upper)]
                    density, edges = np.histogram(visible, bins=bins, density=True)
                else:
                    chunk_densities = []
                    edges = bins
                    for chunk in chunks:
                        chunk_vals = np.log10(np.maximum(chunk.float().numpy(), eps))
                        visible = chunk_vals[(chunk_vals >= lower) & (chunk_vals <= upper)]
                        if visible.size == 0:
                            chunk_densities.append(np.zeros(hist_bins))
                            continue
                        density, edges = np.histogram(visible, bins=bins, density=True)
                        chunk_densities.append(density)
                    density = np.mean(chunk_densities, axis=0)
                centers = 0.5 * (edges[:-1] + edges[1:])
                (line,) = ax.plot(centers, density, linewidth=1.8, label=variant)
                median = float(np.quantile(vals, 0.50))
                if lower <= median <= upper:
                    ax.axvline(
                        median,
                        color=line.get_color(),
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.8,
                        label=f"{variant} median",
                    )
            suffix = ""
            if group == "output" and item["output_token_range"] is not None:
                start, end = item["output_token_range"]
                suffix = f" tokens [{start}, {end})"
            ax.set_title(f"{item['label']} seq={item['seq']}{suffix} {metric_titles[metric]}")
            ax.set_xlabel(f"log10({metric_titles[metric]}) (upper clipped at p{cap_quantile * 100:g})")
            ax.set_ylabel("density")
            ax.grid(True, alpha=0.25)
            ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    collected = [collect_distributions(path, args.device, args.max_heads, args.question_section) for path in args.fixtures]
    if args.aggregate_by_label:
        collected = aggregate_by_label(collected)

    output_path = args.out_dir / f"{args.prefix}_output_error_ecdf.png"
    k_path = args.out_dir / f"{args.prefix}_k_error_ecdf.png"
    output_hist_path = args.out_dir / f"{args.prefix}_output_error_hist.png"
    k_hist_path = args.out_dir / f"{args.prefix}_k_error_hist.png"
    output_log_hist_path = args.out_dir / f"{args.prefix}_output_error_loghist.png"
    k_log_hist_path = args.out_dir / f"{args.prefix}_k_error_loghist.png"
    title_prefix = "question-section" if args.question_section else "full-sequence"
    make_plot(collected, "output", ("sageattn3", "sageattn4"), f"{title_prefix} output error distributions", output_path)
    make_plot(collected, "k", ("direct", "residual_reconstruct"), f"{title_prefix} K reconstruction error distributions", k_path)
    make_hist_plot(collected, "output", ("sageattn3", "sageattn4"), f"{title_prefix} output error histograms", output_hist_path, args.hist_bins, args.hist_cap_stds)
    make_hist_plot(collected, "k", ("direct", "residual_reconstruct"), f"{title_prefix} K reconstruction error histograms", k_hist_path, args.hist_bins, args.hist_cap_stds)
    make_log_hist_plot(collected, "output", ("sageattn3", "sageattn4"), f"{title_prefix} output log-error histograms", output_log_hist_path, args.hist_bins, args.log_hist_cap_quantile)
    make_log_hist_plot(collected, "k", ("direct", "residual_reconstruct"), f"{title_prefix} K reconstruction log-error histograms", k_log_hist_path, args.hist_bins, args.log_hist_cap_quantile)

    print(output_path)
    print(k_path)
    print(output_hist_path)
    print(k_hist_path)
    print(output_log_hist_path)
    print(k_log_hist_path)


if __name__ == "__main__":
    main()
