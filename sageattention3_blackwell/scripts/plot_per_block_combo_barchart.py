#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import importlib
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


VARIANT_LABELS = OrderedDict(
    [
        ("s3_per_block_false", "s3 pB=False"),
        ("s3_per_block_true", "s3 pB=True"),
        ("s4_per_block_false", "s4 pB=False"),
        ("s4_per_block_false_split", "s4 pB=False split"),
        ("s4_per_block_false_split_perm", "s4 pB=False split+perm"),
        ("s4_per_block_false_resid", "s4 pB=False residual"),
        ("s4_per_block_false_resid_perm", "s4 pB=False residual+perm"),
        ("s4_per_block_true", "s4 pB=True"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", nargs="*", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-heads", type=int, default=0)
    parser.add_argument("--variants", nargs="+", choices=VARIANT_LABELS.keys())
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--merge-json", nargs="*", type=Path, default=[])
    parser.add_argument("--plot-out", type=Path)
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


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean().item()),
        "q95": float(torch.quantile(values, 0.95).item()),
    }


def summarize_cosim(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "q50": float(torch.quantile(values, 0.50).item()),
        "q95": float(torch.quantile(values, 0.95).item()),
    }


def pad_to_block(x: torch.Tensor, block: int) -> torch.Tensor:
    pad_len = (block - x.size(2) % block) % block
    if pad_len == 0:
        return x.contiguous()
    return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()


def query_cosim_to_means(q: torch.Tensor, token_slice: slice, max_heads: int) -> dict[str, dict[str, float]]:
    if max_heads > 0:
        q = q[:, :max_heads]
    q = q.float()
    q_eval = q[:, :, token_slice, :]

    full_mean = q.mean(dim=-2, keepdim=True)
    full_cosim = F.cosine_similarity(q_eval, full_mean.expand(-1, -1, q.size(2), -1)[:, :, token_slice, :], dim=-1)

    q_padded = pad_to_block(q, 128)
    bsz, n_heads, _, head_dim = q_padded.shape
    q_blocks = q_padded.reshape(bsz, n_heads, q_padded.size(2) // 128, 128, head_dim)
    block_mean = q_blocks.mean(dim=-2, keepdim=True).expand_as(q_blocks).reshape_as(q_padded)
    block_cosim = F.cosine_similarity(q_eval, block_mean[:, :, token_slice, :], dim=-1)

    return {
        "full_mean": summarize_cosim(full_cosim.reshape(-1).cpu()),
        "block128_mean": summarize_cosim(block_cosim.reshape(-1).cpu()),
    }


def sageattn3_full_mean_unrounded(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    s3_api = importlib.import_module("sageattn3.api")
    q_len = q.size(2)
    k_len = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16

    q_mean = q.mean(dim=-2, keepdim=True)
    q = q - q_mean
    k = k - k.mean(dim=-2, keepdim=True)
    q, k, v = [pad_to_block(x, 128) for x in (q, k, v)]
    delta_s = torch.matmul(q_mean, k.transpose(-2, -1)).to(torch.float32).contiguous()

    qlist = s3_api.scale_and_quant_fp4(q)
    klist = s3_api.scale_and_quant_fp4_permute(k)
    vlist = s3_api.scale_and_quant_fp4_transpose(v)
    return s3_api.blockscaled_fp4_attn(
        qlist,
        klist,
        vlist,
        delta_s,
        k_len,
        is_causal,
        False,
        is_bf16,
    )[0][:, :, :q_len, :].contiguous()


def sageattn4_full_mean_unrounded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    quant_block: int = 8,
    split_selected: bool = False,
    residual_split: bool = False,
    permute_first: bool = False,
) -> torch.Tensor:
    s4_api = importlib.import_module("sageattn4.api")
    assert quant_block == 8

    q_len = q.size(2)
    k_len = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    bsz, n_heads, _, head_dim = q.shape
    device = q.device
    pack_group = quant_block - 1

    def quant_pack_k(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = pad_to_block(x, pack_group)
        seq = x.size(-2)
        x_block = x.reshape(bsz, n_heads, seq // pack_group, pack_group, head_dim)
        x_mean = x_block.mean(dim=-2)
        x_mean = s4_api.round_to_blockscaled_fp4(x_mean)
        x_mean_f = x_mean.float()
        x_norm_sq = (x_mean_f * x_mean_f).sum(dim=-1, keepdim=True)
        x_mean_dir = torch.where(
            x_norm_sq > 0,
            x_mean_f / x_norm_sq,
            torch.zeros_like(x_mean_f),
        )
        lamb = torch.einsum("b h n m d, b h n d -> b h n m", x_block.float(), x_mean_dir)
        x_res = (x_block.float() - lamb.unsqueeze(-1) * x_mean_f.unsqueeze(-2)).to(x.dtype)
        lamb = torch.cat(
            [
                torch.zeros((bsz, n_heads, seq // pack_group, 1), device=device, dtype=torch.float32),
                lamb.to(torch.float32),
            ],
            dim=-1,
        )
        x_block = torch.cat([x_mean.unsqueeze(-2), x_res], dim=-2)
        return (
            x_block.reshape(bsz, n_heads, seq // pack_group * quant_block, head_dim),
            lamb.reshape(bsz, n_heads, seq // pack_group * quant_block),
        )

    def quant_pack_v(x: torch.Tensor) -> torch.Tensor:
        x = pad_to_block(x, pack_group)
        seq = x.size(-2)
        x_block = x.reshape(bsz, n_heads, seq // pack_group, pack_group, head_dim)
        if residual_split:
            x_zero = torch.zeros((bsz, n_heads, seq // pack_group, head_dim), device=device, dtype=x.dtype)
            x_base = torch.cat([x_zero.unsqueeze(-2), x_block], dim=-2)
            x_base = x_base.reshape(bsz, n_heads, seq // pack_group * quant_block, head_dim)
            x_base_rounded = s4_api.round_to_token_blockscaled_fp4(pad_to_block(x_base, 128))[:, :, : x_base.size(2), :]
            x_base_rounded = x_base_rounded.reshape(bsz, n_heads, seq // pack_group, quant_block, head_dim)
            x_hi = x_base_rounded[:, :, :, 1, :].contiguous()
            x_slot = (x_block[:, :, :, 0, :].float() - x_hi.float()).to(x.dtype)
            x_split = x_block.clone()
            x_split[:, :, :, 0, :] = x_hi.to(x.dtype)
        elif not split_selected:
            x_slot = torch.zeros((bsz, n_heads, seq // pack_group, head_dim), device=device, dtype=x.dtype)
            x_split = x_block
        else:
            x_slot = (x_block[:, :, :, 0, :].float() * 0.5).to(x.dtype)
            x_split = x_block.clone()
            x_split[:, :, :, 0, :] = x_slot
        x_block = torch.cat([x_slot.unsqueeze(-2), x_split], dim=-2)
        return x_block.reshape(bsz, n_heads, seq // pack_group * quant_block, head_dim)

    def permute_important_first(k_src: torch.Tensor, v_src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k_padded = pad_to_block(k_src, pack_group)
        v_padded = pad_to_block(v_src, pack_group)
        seq = k_padded.size(-2)
        groups = seq // pack_group
        k_block = k_padded.reshape(bsz, n_heads, groups, pack_group, head_dim)
        v_block = v_padded.reshape(bsz, n_heads, groups, pack_group, head_dim)
        score = torch.einsum("b h d, b h n m d -> b h n m", q_mean.squeeze(-2).float(), k_block.float())
        valid = torch.arange(seq, device=device).reshape(1, 1, groups, pack_group) < k_len
        selected = score.masked_fill(~valid, -torch.inf).argmax(dim=-1, keepdim=True)
        lane = torch.arange(pack_group, device=device).reshape(1, 1, 1, pack_group)
        rank = torch.where(lane == selected, torch.full_like(lane, -1), lane)
        perm = rank.argsort(dim=-1).expand(bsz, n_heads, groups, pack_group)
        gather_idx = perm[..., None].expand(-1, -1, -1, -1, head_dim)
        k_perm = torch.gather(k_block, dim=3, index=gather_idx).reshape(bsz, n_heads, seq, head_dim)
        v_perm = torch.gather(v_block, dim=3, index=gather_idx).reshape(bsz, n_heads, seq, head_dim)
        return k_perm[:, :, :k_src.size(2), :].contiguous(), v_perm[:, :, :v_src.size(2), :].contiguous()

    k = k - k.mean(dim=-2, keepdim=True)
    q_mean = q.mean(dim=-2, keepdim=True)
    if permute_first:
        k, v = permute_important_first(k, v)
    q = q - q_mean
    q = pad_to_block(q, 128)
    k, lambda_k = quant_pack_k(k)
    v = quant_pack_v(v)
    k = pad_to_block(k, 128)
    lambda_k = pad_to_block(lambda_k.unsqueeze(-1), 128).squeeze(-1)
    delta_s = torch.matmul(q_mean, k.transpose(-2, -1)).to(torch.float32).contiguous()
    v = pad_to_block(v, 128)

    qlist = s4_api.scale_and_quant_fp4(q)
    klist = s4_api.scale_and_quant_fp4_permute(k)
    vlist = s4_api.scale_and_quant_fp4_transpose(v)
    packed_k_len = k_len // pack_group * quant_block if k_len % pack_group == 0 else k_len // pack_group * quant_block + 1 + (k_len % pack_group)
    return s4_api.blockscaled_fp4_attn(
        qlist,
        klist,
        vlist,
        delta_s,
        lambda_k,
        q_len,
        packed_k_len,
        is_causal,
        False,
        is_bf16,
    )[0][:, :, :q_len, :].contiguous()


def sageattn4_full_mean_unrounded_split(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    return sageattn4_full_mean_unrounded(q, k, v, is_causal, split_selected=True)


def sageattn4_full_mean_unrounded_split_perm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    return sageattn4_full_mean_unrounded(q, k, v, is_causal, split_selected=True, permute_first=True)


def sageattn4_full_mean_unrounded_resid(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    return sageattn4_full_mean_unrounded(q, k, v, is_causal, residual_split=True)


def sageattn4_full_mean_unrounded_resid_perm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    return sageattn4_full_mean_unrounded(q, k, v, is_causal, residual_split=True, permute_first=True)


@torch.no_grad()
def collect_metrics(path: Path, device: str, max_heads: int, variants: list[str]) -> dict:
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

    chunks = {name: {"rmse": [], "one_minus_cos": []} for name in variants}
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
        if "s4_per_block_true" in variants:
            outputs["s4_per_block_true"] = call_silently(
                sageattn4.sageattn4_blackwell, q.clone(), k.clone(), v.clone(), is_causal=False, per_block_mean=True
            )

        for name, out in outputs.items():
            metric_values = output_distribution_metrics(out, ref, token_slice)
            chunks[name]["rmse"].append(metric_values["rmse"].cpu())
            chunks[name]["one_minus_cos"].append(metric_values["one_minus_cos"].cpu())

        del q, k, v, ref, outputs

    return {
        "label": fixture_label(path, meta),
        "source": Path(meta["text_file"]).name if meta.get("text_file") else path.name,
        "q_seq": int(q_cpu.shape[2]),
        "kv_seq": int(k_context_cpu.shape[2]),
        "eval_range": [int(token_slice.start), int(token_slice.stop)],
        "query_cosim_to_mean": query_cosim_to_means(q_cpu, token_slice, n_q_heads),
        "metrics": {
            name: {
                metric: summarize(torch.cat(values))
                for metric, values in metric_chunks.items()
            }
            for name, metric_chunks in chunks.items()
        },
    }


def aggregate(items: list[dict]) -> dict:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for item in items:
        grouped.setdefault(item["label"], []).append(item)

    out = {}
    for label, rows in grouped.items():
        variants = sorted({variant for row in rows for variant in row["metrics"]}, key=list(VARIANT_LABELS).index)
        out[label] = {
            "sources": [row["source"] for row in rows],
            "q_seqs": [row["q_seq"] for row in rows],
            "kv_seqs": [row["kv_seq"] for row in rows],
            "eval_ranges": [row["eval_range"] for row in rows],
            "query_cosim_to_mean": {},
            "metrics": {},
        }
        for mean_name in ("full_mean", "block128_mean"):
            out[label]["query_cosim_to_mean"][mean_name] = {
                metric: float(np.mean([row["query_cosim_to_mean"][mean_name][metric] for row in rows]))
                for metric in ("mean", "std", "q50", "q95")
            }
        for variant in variants:
            out[label]["metrics"][variant] = {}
            for metric in ("rmse", "one_minus_cos"):
                means = [row["metrics"][variant][metric]["mean"] for row in rows if variant in row["metrics"]]
                q95s = [row["metrics"][variant][metric]["q95"] for row in rows if variant in row["metrics"]]
                out[label]["metrics"][variant][metric] = {
                    "mean": float(np.mean(means)),
                    "q95": float(np.mean(q95s)),
                }
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
                    "query_cosim_to_mean": layer_payload.get("query_cosim_to_mean", {}),
                    "metrics": {},
                },
            )
            if not dst.get("query_cosim_to_mean") and layer_payload.get("query_cosim_to_mean"):
                dst["query_cosim_to_mean"] = layer_payload["query_cosim_to_mean"]
            dst["metrics"].update(layer_payload["metrics"])
    return merged


def plot_bars(summary: dict, out_path: Path) -> None:
    layers = sorted(summary.keys(), key=lambda label: int(label[1:]) if label.startswith("L") and label[1:].isdigit() else label)
    variants = list(VARIANT_LABELS)
    colors = {
        "s3_per_block_false": "#4c78a8",
        "s3_per_block_true": "#72b7b2",
        "s4_per_block_false": "#f58518",
        "s4_per_block_true": "#e45756",
    }
    x = np.arange(len(layers))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(variants))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for offset, variant in zip(offsets, variants):
        values = [
            summary[layer]["metrics"].get(variant, {}).get("rmse", {}).get("mean", np.nan)
            for layer in layers
        ]
        ax.bar(x + offset, values, width=width, label=VARIANT_LABELS[variant], color=colors[variant])

    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel("Answer-token RMSE mean")
    ax.set_title("Full-Q, context-only K/V attention error\npost-RoPE Q/K; pB=False uses unrounded full-mean dS")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = []
    if args.fixtures:
        variants = args.variants or list(VARIANT_LABELS)
        collected = [collect_metrics(path, args.device, args.max_heads, variants) for path in args.fixtures]
        summary = aggregate(collected)
        results.append(summary)
        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
            print(args.out_json)
    for path in args.merge_json:
        results.append(json.loads(path.read_text()))
    if args.plot_out is not None:
        merged = deep_merge(results)
        plot_bars(merged, args.plot_out)
        print(args.plot_out)
        print(json.dumps(merged, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
