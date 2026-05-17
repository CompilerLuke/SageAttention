#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from scripts.compare_sageattn_blackwell_fixtures import output_distribution_metrics, question_token_slice


FixturePath = Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-heads", type=int, default=0)
    parser.add_argument("--question-section", action="store_true")
    parser.add_argument("--eval-section", action="store_true")
    return parser.parse_args()


def call_silently(fn: Callable, *args, **kwargs) -> torch.Tensor:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out


def sageattn3_unrounded_ds_global_q(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    import sageattn3.api as sageattn3_api

    def pad_128(x: torch.Tensor) -> torch.Tensor:
        pad_len = (128 - x.size(-2) % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()

    q_len = q.size(-2)
    k_len = k.size(-2)
    is_bf16 = q.dtype == torch.bfloat16

    k = k - k.mean(dim=-2, keepdim=True)
    q, k, v = map(pad_128, (q, k, v))
    qm = q.mean(dim=-2, keepdim=True)
    q = q - qm
    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()

    qlist = sageattn3_api.scale_and_quant_fp4(q)
    klist = sageattn3_api.scale_and_quant_fp4_permute(k)
    vlist = sageattn3_api.scale_and_quant_fp4_transpose(v)
    return sageattn3_api.blockscaled_fp4_attn(
        qlist,
        klist,
        vlist,
        delta_s,
        k_len,
        False,
        False,
        is_bf16,
    )[0][:, :, :q_len, :].contiguous()


def fixture_label(path: Path, meta: dict) -> str:
    layer = meta.get("layer")
    if layer is not None:
        return f"L{layer}"
    return path.stem


def eval_token_slice(meta: dict, seq_len: int, enabled: bool) -> slice | None:
    if not enabled:
        return None
    start = int(meta["eval_token_start"])
    end = int(meta["eval_token_end"])
    start = max(0, min(start, seq_len))
    end = max(start, min(end, seq_len))
    if end <= start:
        raise ValueError(f"empty eval token span after truncation: start={start}, end={end}, seq_len={seq_len}")
    return slice(start, end)


def selected_token_slice(meta: dict, seq_len: int, question_section: bool, eval_section: bool) -> slice | None:
    if question_section and eval_section:
        raise ValueError("use only one of --question-section or --eval-section")
    if eval_section:
        return eval_token_slice(meta, seq_len, True)
    return question_token_slice(meta, seq_len, question_section)


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean().item()),
        "q50": float(torch.quantile(values, 0.50).item()),
        "q80": float(torch.quantile(values, 0.80).item()),
        "q95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


@torch.no_grad()
def collect_fixture(path: Path, device: str, max_heads: int, question_section: bool, eval_section: bool) -> dict:
    import sageattn3
    import sageattn4

    payload = torch.load(path, map_location="cpu")
    q_cpu = payload["q"].contiguous()
    k_cpu = payload["k"].contiguous()
    v_cpu = payload["v"].contiguous()
    meta = payload.get("meta", {})
    token_slice = selected_token_slice(meta, q_cpu.shape[2], question_section, eval_section)
    kv_end = token_slice.start if eval_section and token_slice is not None else k_cpu.shape[2]
    k_context_cpu = k_cpu[:, :, :kv_end, :].contiguous()
    v_context_cpu = v_cpu[:, :, :kv_end, :].contiguous()
    n_q_heads = q_cpu.shape[1]
    n_kv_heads = k_context_cpu.shape[1]
    group_size = n_q_heads // n_kv_heads
    if max_heads > 0:
        n_q_heads = min(n_q_heads, max_heads)

    variants: OrderedDict[str, list[dict[str, torch.Tensor]]] = OrderedDict(
        [
            ("s3_unrounded_ds_per_block_false", []),
            ("s3_rounded_ds_per_block_true", []),
            ("s4_rounded_ds_per_block_true", []),
        ]
    )

    for q_head in range(n_q_heads):
        kv_head = q_head // group_size
        q = q_cpu[:, q_head : q_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        k = k_context_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        v = v_context_cpu[:, kv_head : kv_head + 1].to(device=device, dtype=torch.bfloat16).contiguous()
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=False)

        outputs = {
            "s3_unrounded_ds_per_block_false": call_silently(sageattn3_unrounded_ds_global_q, q.clone(), k.clone(), v.clone()),
            "s3_rounded_ds_per_block_true": call_silently(
                sageattn3.sageattn3_blackwell,
                q.clone(),
                k.clone(),
                v.clone(),
                is_causal=False,
                per_block_mean=True,
            ),
            "s4_rounded_ds_per_block_true": call_silently(
                sageattn4.sageattn4_blackwell,
                q.clone(),
                k.clone(),
                v.clone(),
                is_causal=False,
                per_block_mean=True,
            ),
        }
        for name, out in outputs.items():
            variants[name].append(output_distribution_metrics(out, ref, token_slice))

        del q, k, v, ref, outputs

    return {
        "label": fixture_label(path, meta),
        "source": Path(meta["text_file"]).name if meta.get("text_file") else path.name,
        "seq": int(q_cpu.shape[2]),
        "kv_seq": int(k_context_cpu.shape[2]),
        "token_range": None if token_slice is None else (int(token_slice.start), int(token_slice.stop)),
        "tokens_compared": None if token_slice is None else int(token_slice.stop - token_slice.start),
        "variants": {
            name: {
                metric: torch.cat([chunk[metric].cpu() for chunk in chunks])
                for metric in ("rmse", "one_minus_cos")
            }
            for name, chunks in variants.items()
        },
    }


def main() -> None:
    args = parse_args()
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for path in args.fixtures:
        item = collect_fixture(path, args.device, args.max_heads, args.question_section, args.eval_section)
        grouped.setdefault(item["label"], []).append(item)

    for label, items in grouped.items():
        sources = ", ".join(item["source"] for item in items)
        seqs = ", ".join(str(item["seq"]) for item in items)
        kv_seqs = ", ".join(str(item["kv_seq"]) for item in items)
        token_ranges = ", ".join(str(item["token_range"]) for item in items)
        tokens_compared = ", ".join(str(item["tokens_compared"]) for item in items)
        print(f"\n{label}: sources={sources}")
        print(f"q_seqs={seqs} kv_seqs={kv_seqs} token_ranges={token_ranges} tokens_compared={tokens_compared}")
        print("variant                              RMSE_mean   RMSE_q50   RMSE_q80   RMSE_q95    1-cos_mean     1-cos_q95")
        stats = {}
        variant_names = items[0]["variants"].keys()
        for name in variant_names:
            rmse = torch.cat([item["variants"][name]["rmse"] for item in items])
            one_minus_cos = torch.cat([item["variants"][name]["one_minus_cos"] for item in items])
            rm = summarize(rmse)
            oc = summarize(one_minus_cos)
            stats[name] = {"rmse": rm, "one_minus_cos": oc}
            print(
                f"{name:<36} {rm['mean']:10.8f} {rm['q50']:10.8f} {rm['q80']:10.8f} "
                f"{rm['q95']:10.8f} {oc['mean']:14.10f} {oc['q95']:14.10f}"
            )

        base = stats["s3_unrounded_ds_per_block_false"]
        smooth = stats["s4_rounded_ds_per_block_true"]
        print(
            "s4 rounded per_block true vs s3 unrounded per_block false: "
            f"RMSE_mean_reduction={100.0 * (1.0 - smooth['rmse']['mean'] / base['rmse']['mean']):.2f}% "
            f"RMSE_q95_reduction={100.0 * (1.0 - smooth['rmse']['q95'] / base['rmse']['q95']):.2f}% "
            f"1-cos_mean_reduction={100.0 * (1.0 - smooth['one_minus_cos']['mean'] / base['one_minus_cos']['mean']):.2f}% "
            f"1-cos_q95_reduction={100.0 * (1.0 - smooth['one_minus_cos']['q95'] / base['one_minus_cos']['q95']):.2f}%"
        )


if __name__ == "__main__":
    main()
