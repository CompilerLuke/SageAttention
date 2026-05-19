from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sageattn3.api as s3
import sageattn4.api as s4


KERNEL_LABELS = {
    "flash": "Flash SDPA",
    "flash_fp16": "Flash SDPA FP16",
    "sageattn3": "SageAttention3",
    "sageattn4": "SageAttention4",
}

KERNEL_COLORS = {
    "flash": "#4c78a8",
    "flash_fp16": "#4c78a8",
    "sageattn3": "#f58518",
    "sageattn4": "#54a24b",
}


def packed_sageattn4_key_length(seq_len: int, quant_block_size: int) -> int:
    local_tokens = quant_block_size - 1
    full_groups, rem = divmod(seq_len, local_tokens)
    if rem == 0:
        return full_groups * quant_block_size
    return full_groups * quant_block_size + 1 + rem


def bench_sageattention4_kernel_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    quant_block_size: int = 16,
    per_block_mean: bool = True,
    v_trick: bool = False,
    is_causal: bool = True,
    is_bf16: bool = True,
):
    q_len = q.size(2)
    k_len = k.size(2)
    q, k, v, delta_s, lambda_k = s4.preprocess_qkv(
        q.clone(),
        k.clone(),
        v.clone(),
        per_block_mean=per_block_mean,
        quant_block_size=quant_block_size,
        v_trick=v_trick,
    )
    qlist_from_cuda = s4.scale_and_quant_fp4(q)
    klist_from_cuda = s4.scale_and_quant_fp4_permute(k)
    vlist_from_cuda = s4.scale_and_quant_fp4_transpose(v)
    logical_k_len = packed_sageattn4_key_length(k_len, quant_block_size)

    def kernel():
        return s4.blockscaled_fp4_attn(
            qlist_from_cuda,
            klist_from_cuda,
            vlist_from_cuda,
            delta_s,
            lambda_k,
            q_len,
            logical_k_len,
            is_causal,
            per_block_mean,
            is_bf16,
        )[0][:, :, :q_len, :].contiguous()

    return kernel


def bench_sageattention3_kernel_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    per_block_mean: bool = True,
    is_causal: bool = True,
    is_bf16: bool = True,
):
    q_len = q.size(2)
    k_len = k.size(2)
    q, k, v, delta_s = s3.preprocess_qkv(q.clone(), k.clone(), v.clone(), per_block_mean=per_block_mean)
    qlist_from_cuda = s3.scale_and_quant_fp4(q)
    klist_from_cuda = s3.scale_and_quant_fp4_permute(k)
    vlist_from_cuda = s3.scale_and_quant_fp4_transpose(v)

    def kernel():
        return s3.blockscaled_fp4_attn(
            qlist_from_cuda,
            klist_from_cuda,
            vlist_from_cuda,
            delta_s,
            k_len,
            is_causal,
            per_block_mean,
            is_bf16,
        )[0][:, :, :q_len, :].contiguous()

    return kernel


def bench_cuda(func, *, num_elem: int, io: int, flops: int, warmup: int, repeats: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()

        times_ms = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(repeats):
            start.record()
            func()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

    mean_time = float(np.mean(times_ms))
    median_time = float(np.median(times_ms))
    throughput = num_elem / median_time * 1e3
    tflops = flops / 1e9 / median_time
    bandwidth = io / median_time * 1e3

    return {
        "mean_ms": mean_time,
        "median_ms": median_time,
        "tokens_per_s": float(throughput),
        "tflops": float(tflops),
        "io_gb_s": float(bandwidth / 1e9),
    }


def attention_flops(batch: int, heads: int, seq_len: int, head_dim: int, is_causal: bool) -> int:
    dense_flops = 4 * batch * heads * seq_len * seq_len * head_dim
    return dense_flops // 2 if is_causal else dense_flops


def shape_for_sequence(args: argparse.Namespace, seq_len: int) -> tuple[int, int]:
    if not args.scale_batch:
        return args.batch, args.heads

    base_seq_len = min(args.seq_lengths)
    scale = seq_len / base_seq_len
    batch = max(1, round(args.batch / (scale * scale)))
    return batch, args.heads


def make_kernel_fn(
    kernel: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    per_block_mean: bool,
    v_trick: bool,
    sageattn4_quant_block_size: int,
):
    if kernel == "flash":
        return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    if kernel == "flash_fp16":
        q_fp16 = q.to(torch.float16)
        k_fp16 = k.to(torch.float16)
        v_fp16 = v.to(torch.float16)
        return lambda: F.scaled_dot_product_attention(q_fp16, k_fp16, v_fp16, is_causal=is_causal)
    if kernel == "sageattn3":
        return bench_sageattention3_kernel_only(q, k, v, per_block_mean=per_block_mean, is_causal=is_causal)
    if kernel == "sageattn4":
        return bench_sageattention4_kernel_only(
            q,
            k,
            v,
            quant_block_size=sageattn4_quant_block_size,
            per_block_mean=per_block_mean,
            v_trick=v_trick,
            is_causal=is_causal,
        )
    raise ValueError(f"Unknown kernel: {kernel}")


def run_benchmarks(args: argparse.Namespace) -> list[dict]:
    results = []
    device = "cuda"
    kernels = args.kernels

    for seq_len in args.seq_lengths:
        batch, heads = shape_for_sequence(args, seq_len)
        torch.manual_seed(args.seed + seq_len)
        q = torch.randn((batch, heads, seq_len, args.head_dim), dtype=torch.bfloat16, device=device)
        k = torch.randn((batch, heads, seq_len, args.head_dim), dtype=torch.bfloat16, device=device)
        v = torch.randn((batch, heads, seq_len, args.head_dim), dtype=torch.bfloat16, device=device)

        flops = attention_flops(batch, heads, seq_len, args.head_dim, args.causal)
        io = 3 * batch * heads * seq_len * args.head_dim * q.element_size()
        num_elem = batch * heads * seq_len

        for kernel in kernels:
            print(f"benchmarking seq={seq_len} batch={batch} heads={heads} kernel={kernel}", flush=True)
            bench_fn = make_kernel_fn(
                kernel,
                q,
                k,
                v,
                is_causal=args.causal,
                per_block_mean=args.per_block_mean,
                v_trick=args.v_trick,
                sageattn4_quant_block_size=args.sageattn4_quant_block_size,
            )
            stats = bench_cuda(
                bench_fn,
                flops=flops,
                io=io,
                num_elem=num_elem,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            result = {
                "seq_len": seq_len,
                "batch": batch,
                "heads": heads,
                "head_dim": args.head_dim,
                "kernel": kernel,
                "label": KERNEL_LABELS[kernel],
                **stats,
            }
            results.append(result)
            print(
                f"  median={stats['median_ms']:.3f} ms "
                f"tflops={stats['tflops']:.2f} tokens/s={stats['tokens_per_s']:.0f}",
                flush=True,
            )
            del bench_fn
            torch.cuda.empty_cache()

        del q, k, v
        torch.cuda.empty_cache()

    add_flash_relative_throughput(results)
    return results


def add_flash_relative_throughput(results: list[dict]) -> None:
    baseline_kernel = "flash" if any(row["kernel"] == "flash" for row in results) else "flash_fp16"
    flash_by_shape = {
        (row["seq_len"], row["batch"], row["heads"]): row["tflops"]
        for row in results
        if row["kernel"] == baseline_kernel
    }
    for row in results:
        flash_tflops = flash_by_shape.get((row["seq_len"], row["batch"], row["heads"]))
        row["relative_to_flash"] = row["tflops"] / flash_tflops if flash_tflops else None


def seq_label(seq_len: int) -> str:
    return f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)


def plot_results(results: list[dict], out: Path, title: str) -> None:
    seq_lengths = sorted({row["seq_len"] for row in results})
    kernels = [kernel for kernel in KERNEL_LABELS if any(row["kernel"] == kernel for row in results)]
    by_key = {(row["seq_len"], row["kernel"]): row for row in results}
    shape_by_seq = {
        seq_len: next(row for row in results if row["seq_len"] == seq_len)
        for seq_len in seq_lengths
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=180)
    centers = np.arange(len(seq_lengths), dtype=float)
    width = min(0.24, 0.78 / max(1, len(kernels)))
    offsets = [-(len(kernels) - 1) * width / 2 + idx * width for idx in range(len(kernels))]

    max_y = 0.0
    for kernel, offset in zip(kernels, offsets):
        xs = centers + offset
        ys = [by_key[(seq_len, kernel)]["tflops"] for seq_len in seq_lengths]
        max_y = max(max_y, max(ys))
        bars = ax.bar(
            xs,
            ys,
            width=width,
            label=KERNEL_LABELS[kernel],
            color=KERNEL_COLORS[kernel],
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, seq_len in zip(bars, seq_lengths):
            ratio = by_key[(seq_len, kernel)].get("relative_to_flash")
            label = f"{ratio:.2f}x" if ratio is not None else "n/a"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Effective TFLOP/s")
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [
            f"{seq_label(seq_len)}\nB{shape_by_seq[seq_len]['batch']} H{shape_by_seq[seq_len]['heads']}"
            for seq_len in seq_lengths
        ]
    )
    ax.set_ylim(0, max_y * 1.18 if max_y > 0 else 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_outputs(results: list[dict], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shapes = [
        {"seq_len": seq_len, "batch": shape_for_sequence(args, seq_len)[0], "heads": shape_for_sequence(args, seq_len)[1]}
        for seq_len in args.seq_lengths
    ]
    metadata = {
        "base_batch": args.batch,
        "base_heads": args.heads,
        "head_dim": args.head_dim,
        "seq_lengths": args.seq_lengths,
        "scale_batch": args.scale_batch,
        "sageattn4_quant_block_size": args.sageattn4_quant_block_size,
        "shapes": shapes,
        "kernels": args.kernels,
        "causal": args.causal,
        "per_block_mean": args.per_block_mean,
        "v_trick": args.v_trick,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "flops_convention": "causal uses half of dense attention FLOPs" if args.causal else "dense attention FLOPs",
    }
    payload = {"metadata": metadata, "results": results}
    args.json_out.write_text(json.dumps(payload, indent=2))

    shape_desc = f"scaled B, fixed H={args.heads}" if args.scale_batch else f"B={args.batch}, H={args.heads}"
    title = f"SageAttention kernel throughput, {shape_desc}, D={args.head_dim}, {'causal' if args.causal else 'non-causal'}"
    plot_results(results, args.plot_out, title)


def print_table(results: list[dict]) -> None:
    header = f"{'seq':>6}  {'B':>3}  {'H':>3}  {'kernel':<15}  {'median ms':>10}  {'TFLOP/s':>10}  {'vs flash':>9}"
    print(header)
    print("-" * len(header))
    for row in results:
        ratio = row.get("relative_to_flash")
        ratio_s = f"{ratio:.2f}x" if ratio is not None else "n/a"
        print(
            f"{seq_label(row['seq_len']):>6}  {row['batch']:>3}  {row['heads']:>3}  {row['label']:<15}  "
            f"{row['median_ms']:>10.3f}  {row['tflops']:>10.2f}  {ratio_s:>9}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Flash SDPA, SageAttention3, and SageAttention4 kernels.")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[16 * 1024, 32 * 1024, 64 * 1024])
    parser.add_argument("--kernels", nargs="+", choices=list(KERNEL_LABELS), default=list(KERNEL_LABELS))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--sageattn4-quant-block-size", type=int, choices=[16], default=16)
    parser.add_argument("--scale-batch", dest="scale_batch", action="store_true", default=True)
    parser.add_argument("--no-scale-batch", dest="scale_batch", action="store_false")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--plot-out", type=Path, default=Path("reports/bench_tflops_by_sequence.png"))
    parser.add_argument("--json-out", type=Path, default=Path("reports/bench_tflops_by_sequence.json"))
    parser.add_argument("--causal", dest="causal", action="store_true", default=True)
    parser.add_argument("--non-causal", dest="causal", action="store_false")
    parser.add_argument("--per-block-mean", dest="per_block_mean", action="store_true", default=True)
    parser.add_argument("--no-per-block-mean", dest="per_block_mean", action="store_false")
    parser.add_argument("--v-trick", dest="v_trick", action="store_true", default=False)
    parser.add_argument("--no-v-trick", dest="v_trick", action="store_false")
    return parser.parse_args()


def bench_main() -> None:
    args = parse_args()
    results = run_benchmarks(args)
    write_outputs(results, args)
    print_table(results)
    print(args.plot_out)
    print(args.json_out)


if __name__ == "__main__":
    bench_main()
