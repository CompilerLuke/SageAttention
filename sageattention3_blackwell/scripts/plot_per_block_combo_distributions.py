#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from scripts.compare_decode_constraint_variants import eval_token_slice
from scripts.compare_sageattn_blackwell_fixtures import output_distribution_metrics
from scripts.plot_per_block_combo_barchart import (
    VARIANT_LABELS,
    sageattn3_full_mean_unrounded,
    sageattn4_full_mean_unrounded,
    sageattn4_full_mean_unrounded_resid,
    sageattn4_full_mean_unrounded_resid_perm,
    sageattn4_full_mean_unrounded_split,
    sageattn4_full_mean_unrounded_split_perm,
)


METRICS = ("rmse", "one_minus_cos")
PLOT_METRICS = ("rmse", "log_rmse", "one_minus_cos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", nargs="*", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-heads", type=int, default=0)
    parser.add_argument("--variants", nargs="+", choices=VARIANT_LABELS.keys())
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--merge-json", nargs="*", type=Path, default=[])
    parser.add_argument("--plot-prefix", type=Path)
    parser.add_argument("--bins", type=int, default=28)
    parser.add_argument("--cap-quantile", type=float, default=0.98)
    return parser.parse_args()


def call_silently(fn: Callable, *args, **kwargs) -> torch.Tensor:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out


def fixture_label(path: Path, meta: dict) -> str:
    layer = meta.get("layer")
    if layer is not None:
        return f"L{layer}"
    return path.stem


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "q50": float(np.quantile(arr, 0.50)),
        "q80": float(np.quantile(arr, 0.80)),
        "q95": float(np.quantile(arr, 0.95)),
        "q98": float(np.quantile(arr, 0.98)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


@torch.no_grad()
def collect_fixture(path: Path, device: str, max_heads: int, variants: list[str]) -> dict:
    import sageattn3
    import sageattn4

    payload = torch.load(path, map_location="cpu")
    q_cpu = payload["q"].contiguous()
    k_cpu = payload["k"].contiguous()
    v_cpu = payload["v"].contiguous()
    meta = payload.get("meta", {})
    token_slice = eval_token_slice(meta, q_cpu.shape[2], True)
    kv_end = token_slice.start
    k_context_cpu = k_cpu[:, :, :kv_end, :].contiguous()
    v_context_cpu = v_cpu[:, :, :kv_end, :].contiguous()

    n_q_heads = q_cpu.shape[1]
    n_kv_heads = k_context_cpu.shape[1]
    group_size = n_q_heads // n_kv_heads
    if max_heads > 0:
        n_q_heads = min(n_q_heads, max_heads)

    chunks = {name: {metric: [] for metric in METRICS} for name in variants}
    for q_head in range(n_q_heads):
        kv_head = q_head // group_size
        q = q_cpu[:, q_head : q_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        k = k_context_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        v = v_context_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=False)

        outputs = {}
        if "s3_per_block_false" in variants:
            outputs["s3_per_block_false"] = call_silently(
                sageattn3_full_mean_unrounded, q.clone(), k.clone(), v.clone(), is_causal=False
            )
        if "s3_per_block_true" in variants:
            outputs["s3_per_block_true"] = call_silently(
                sageattn3.sageattn3_blackwell, q.clone(), k.clone(), v.clone(), is_causal=False, per_block_mean=True
            )
        if "s4_per_block_false" in variants:
            outputs["s4_per_block_false"] = call_silently(
                sageattn4_full_mean_unrounded, q.clone(), k.clone(), v.clone(), is_causal=False
            )
        if "s4_per_block_false_split" in variants:
            outputs["s4_per_block_false_split"] = call_silently(
                sageattn4_full_mean_unrounded_split, q.clone(), k.clone(), v.clone(), is_causal=False
            )
        if "s4_per_block_false_split_perm" in variants:
            outputs["s4_per_block_false_split_perm"] = call_silently(
                sageattn4_full_mean_unrounded_split_perm, q.clone(), k.clone(), v.clone(), is_causal=False
            )
        if "s4_per_block_false_resid" in variants:
            outputs["s4_per_block_false_resid"] = call_silently(
                sageattn4_full_mean_unrounded_resid, q.clone(), k.clone(), v.clone(), is_causal=False
            )
        if "s4_per_block_false_resid_perm" in variants:
            outputs["s4_per_block_false_resid_perm"] = call_silently(
                sageattn4_full_mean_unrounded_resid_perm, q.clone(), k.clone(), v.clone(), is_causal=False
            )
        if "s4_per_block_true" in variants:
            outputs["s4_per_block_true"] = call_silently(
                sageattn4.sageattn4_blackwell, q.clone(), k.clone(), v.clone(), is_causal=False, per_block_mean=True
            )

        for name, out in outputs.items():
            metric_values = output_distribution_metrics(out, ref, token_slice)
            for metric in METRICS:
                chunks[name][metric].append(metric_values[metric].cpu())

        del q, k, v, ref, outputs

    return {
        "label": fixture_label(path, meta),
        "source": Path(meta["text_file"]).name if meta.get("text_file") else path.name,
        "q_seq": int(q_cpu.shape[2]),
        "kv_seq": int(k_context_cpu.shape[2]),
        "eval_range": [int(token_slice.start), int(token_slice.stop)],
        "values": {
            name: {
                metric: torch.cat(metric_chunks).float().tolist()
                for metric, metric_chunks in chunks_by_metric.items()
            }
            for name, chunks_by_metric in chunks.items()
        },
    }


def aggregate(items: list[dict]) -> dict:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for item in items:
        grouped.setdefault(item["label"], []).append(item)

    out = {}
    for label, rows in grouped.items():
        variants = sorted(
            {variant for row in rows for variant in row["values"]},
            key=list(VARIANT_LABELS).index,
        )
        out[label] = {
            "sources": [row["source"] for row in rows],
            "q_seqs": [row["q_seq"] for row in rows],
            "kv_seqs": [row["kv_seq"] for row in rows],
            "eval_ranges": [row["eval_range"] for row in rows],
            "values": {},
            "summary": {},
        }
        for variant in variants:
            out[label]["values"][variant] = {}
            out[label]["summary"][variant] = {}
            for metric in METRICS:
                values = [
                    value
                    for row in rows
                    if variant in row["values"]
                    for value in row["values"][variant][metric]
                ]
                out[label]["values"][variant][metric] = values
                out[label]["summary"][variant][metric] = summarize(values)
    return out


def deep_merge(results: list[dict]) -> dict:
    merged: dict = {}
    for result in results:
        for layer, layer_payload in result.items():
            dst = merged.setdefault(
                layer,
                {
                    "sources": layer_payload.get("sources", []),
                    "q_seqs": layer_payload.get("q_seqs", []),
                    "kv_seqs": layer_payload.get("kv_seqs", []),
                    "eval_ranges": layer_payload.get("eval_ranges", []),
                    "values": {},
                    "summary": {},
                },
            )
            dst["values"].update(layer_payload["values"])
            dst["summary"].update(layer_payload.get("summary", {}))
    for layer_payload in merged.values():
        for variant, metrics in layer_payload["values"].items():
            layer_payload.setdefault("summary", {}).setdefault(variant, {})
            for metric, values in metrics.items():
                layer_payload["summary"][variant][metric] = summarize(values)
    return merged


def values_for_plot(summary: dict, layer: str, variant: str, metric: str) -> np.ndarray:
    if metric == "log_rmse":
        values = np.asarray(summary[layer]["values"][variant]["rmse"], dtype=np.float64)
        return np.log10(np.maximum(values, np.finfo(np.float64).tiny))
    return np.asarray(summary[layer]["values"][variant][metric], dtype=np.float64)


def plot_metric(summary: dict, metric: str, out_path: Path, bins: int, cap_quantile: float) -> None:
    layers = sorted(summary.keys(), key=lambda label: int(label[1:]) if label.startswith("L") and label[1:].isdigit() else label)
    colors = {
        "s3_per_block_false": "#4c78a8",
        "s3_per_block_true": "#72b7b2",
        "s4_per_block_false": "#f58518",
        "s4_per_block_true": "#e45756",
    }
    label = {
        "rmse": "Answer-token RMSE",
        "log_rmse": "log10(answer-token RMSE)",
        "one_minus_cos": "Answer-token 1-cos",
    }[metric]
    fig, axes = plt.subplots(1, len(layers), figsize=(6.4 * len(layers), 4.2), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        all_values = np.concatenate(
            [
                values_for_plot(summary, layer, variant, metric)
                for variant in VARIANT_LABELS
                if variant in summary[layer]["values"]
            ]
        )
        low = float(all_values.min()) if metric == "log_rmse" else 0.0
        cap = float(np.quantile(all_values, cap_quantile))
        edges = np.linspace(low, cap, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for variant in VARIANT_LABELS:
            if variant not in summary[layer]["values"]:
                continue
            values = values_for_plot(summary, layer, variant, metric)
            visible = values[values <= cap]
            hist, _ = np.histogram(visible, bins=edges, density=True)
            ax.plot(
                centers,
                hist,
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=VARIANT_LABELS[variant],
                color=colors[variant],
            )
            ax.axvline(np.median(values), color=colors[variant], alpha=0.25, linewidth=1.0)
        ax.set_title(f"{layer} (x cap p{cap_quantile * 100:.0f})")
        ax.set_xlabel(label)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{label} distribution, post-RoPE Q/K")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = []
    if args.fixtures:
        variants = args.variants or list(VARIANT_LABELS)
        collected = [collect_fixture(path, args.device, args.max_heads, variants) for path in args.fixtures]
        summary = aggregate(collected)
        results.append(summary)
        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
            print(args.out_json)
    for path in args.merge_json:
        results.append(json.loads(path.read_text()))
    if args.plot_prefix is not None:
        merged = deep_merge(results)
        for metric in PLOT_METRICS:
            out_path = args.plot_prefix.with_name(f"{args.plot_prefix.name}_{metric}.png")
            plot_metric(merged, metric, out_path, args.bins, args.cap_quantile)
            print(out_path)
        print(json.dumps({layer: payload["summary"] for layer, payload in merged.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
