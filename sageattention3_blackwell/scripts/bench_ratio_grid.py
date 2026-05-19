from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEQ_LENGTHS = [8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024]
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64]

# These are excluded from the default run because the benchmark does not
# complete on the current machine. The failures happen before a ratio can be
# measured, in SageAttention3 preprocessing.
KNOWN_FAILED_CELLS = {
    (32 * 1024, 64): "S3 preprocess CUBLAS_STATUS_ALLOC_FAILED",
    (64 * 1024, 32): "S3 preprocess CUBLAS_STATUS_ALLOC_FAILED",
    (64 * 1024, 64): "S3 preprocess CUDA OOM",
}


def seq_label(seq_len: int) -> str:
    return f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)


def cell_stem(seq_len: int, batch: int) -> str:
    return f"bench_ratio_cell_s{seq_len // 1024}k_b{batch}"


def cell_json_path(out_dir: Path, seq_len: int, batch: int) -> Path:
    return out_dir / f"{cell_stem(seq_len, batch)}.json"


def cell_plot_path(out_dir: Path, seq_len: int, batch: int) -> Path:
    return out_dir / f"{cell_stem(seq_len, batch)}.png"


def group_rows_by_shape(rows: list[dict]) -> dict[tuple[int, int, int], dict[str, dict]]:
    grouped: dict[tuple[int, int, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[(row["seq_len"], row["batch"], row["heads"])][row["kernel"]] = row
    return grouped


def read_ratios_from_files(
    paths: list[Path],
    seq_lengths: set[int],
    batches: set[int],
) -> tuple[dict[tuple[int, int], list[float]], list[dict]]:
    ratios: dict[tuple[int, int], list[float]] = defaultdict(list)
    raw: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for (seq_len, batch, heads), rows in group_rows_by_shape(payload.get("results", [])).items():
            if seq_len not in seq_lengths or batch not in batches:
                continue
            if "sageattn3" not in rows or "sageattn4" not in rows:
                continue
            s3_tflops = rows["sageattn3"]["tflops"]
            s4_tflops = rows["sageattn4"]["tflops"]
            ratio = s4_tflops / s3_tflops
            ratios[(seq_len, batch)].append(ratio)
            raw.append(
                {
                    "source": path.name,
                    "seq_len": seq_len,
                    "batch": batch,
                    "heads": heads,
                    "sageattn3_tflops": s3_tflops,
                    "sageattn4_tflops": s4_tflops,
                    "s4_s3_ratio": ratio,
                }
            )
    return ratios, raw


def collect_measurements(
    out_dir: Path,
    seq_lengths: list[int],
    batches: list[int],
    source_jsons: list[Path],
) -> tuple[dict[tuple[int, int], list[float]], list[dict]]:
    seq_set = set(seq_lengths)
    batch_set = set(batches)

    cell_paths = [cell_json_path(out_dir, seq_len, batch) for seq_len in seq_lengths for batch in batches]
    cell_ratios, cell_raw = read_ratios_from_files(cell_paths, seq_set, batch_set)

    fallback_paths = []
    fallback_paths.extend(sorted(out_dir.glob("bench_scaled_b*_8k_64k_tflops_by_sequence.json")))
    fallback_paths.extend(source_jsons)
    fallback_ratios, fallback_raw = read_ratios_from_files(fallback_paths, seq_set, batch_set)

    ratios: dict[tuple[int, int], list[float]] = {}
    raw: list[dict] = []
    for seq_len in seq_lengths:
        for batch in batches:
            key = (seq_len, batch)
            if key in cell_ratios:
                ratios[key] = cell_ratios[key]
                raw.extend(item for item in cell_raw if item["seq_len"] == seq_len and item["batch"] == batch)
            elif key in fallback_ratios:
                ratios[key] = fallback_ratios[key]
                raw.extend(item for item in fallback_raw if item["seq_len"] == seq_len and item["batch"] == batch)
    return ratios, raw


def run_cell(args: argparse.Namespace, seq_len: int, batch: int) -> dict:
    json_out = cell_json_path(args.out_dir, seq_len, batch)
    plot_out = cell_plot_path(args.out_dir, seq_len, batch)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "bench.py"),
        "--seq-lengths",
        str(seq_len),
        "--batch",
        str(batch),
        "--heads",
        str(args.heads),
        "--head-dim",
        str(args.head_dim),
        "--no-scale-batch",
        "--kernels",
        "sageattn3",
        "sageattn4",
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--out-dir",
        str(args.out_dir),
        "--plot-out",
        str(plot_out),
        "--json-out",
        str(json_out),
    ]
    if args.non_causal:
        cmd.append("--non-causal")
    if not args.per_block_mean:
        cmd.append("--no-per-block-mean")

    if args.dry_run:
        print(" ".join(cmd), flush=True)
        return {"seq_len": seq_len, "batch": batch, "status": "dry_run"}

    print(f"running {seq_label(seq_len)} B{batch}", flush=True)
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        return {"seq_len": seq_len, "batch": batch, "status": "failed", "reason": "timeout"}
    except subprocess.CalledProcessError as exc:
        return {"seq_len": seq_len, "batch": batch, "status": "failed", "reason": f"exit {exc.returncode}"}
    return {"seq_len": seq_len, "batch": batch, "status": "ok"}


def make_summary(
    ratios: dict[tuple[int, int], list[float]],
    seq_lengths: list[int],
    batches: list[int],
    failures: dict[tuple[int, int], str],
) -> list[dict]:
    summary = []
    for seq_len in seq_lengths:
        for batch in batches:
            key = (seq_len, batch)
            values = ratios.get(key, [])
            entry = {"seq_len": seq_len, "batch": batch}
            if values:
                entry.update(
                    {
                        "status": "ok",
                        "ratio_mean": float(np.mean(values)),
                        "ratio_min": float(np.min(values)),
                        "ratio_max": float(np.max(values)),
                        "num_measurements": len(values),
                    }
                )
            elif key in failures:
                entry.update({"status": "failed", "reason": failures[key]})
            else:
                entry.update({"status": "missing"})
            summary.append(entry)
    return summary


def plot_heatmap(
    ratios: dict[tuple[int, int], list[float]],
    seq_lengths: list[int],
    batches: list[int],
    failures: dict[tuple[int, int], str],
    out: Path,
) -> None:
    matrix = np.full((len(batches), len(seq_lengths)), np.nan)
    counts = np.zeros((len(batches), len(seq_lengths)), dtype=int)
    for (seq_len, batch), values in ratios.items():
        if seq_len not in seq_lengths or batch not in batches:
            continue
        row = batches.index(batch)
        col = seq_lengths.index(seq_len)
        matrix[row, col] = float(np.mean(values))
        counts[row, col] = len(values)

    fig, ax = plt.subplots(figsize=(9.2, 6.1), dpi=180)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="#f2f2f2")
    image = ax.imshow(np.ma.masked_invalid(matrix * 100.0), aspect="auto", origin="lower", cmap=cmap, vmin=85, vmax=100)

    ax.set_title("SageAttention4 / SageAttention3 Throughput Ratio")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Batch size")
    ax.set_xticks(range(len(seq_lengths)))
    ax.set_xticklabels([seq_label(seq_len) for seq_len in seq_lengths])
    ax.set_yticks(range(len(batches)))
    ax.set_yticklabels([str(batch) for batch in batches])

    for row, batch in enumerate(batches):
        for col, seq_len in enumerate(seq_lengths):
            key = (seq_len, batch)
            if not np.isnan(matrix[row, col]):
                value = matrix[row, col] * 100.0
                text = f"{value:.1f}%"
                if counts[row, col] > 1:
                    text += f"\n(n={counts[row, col]})"
                color = "white" if value < 93 else "black"
                ax.text(col, row, text, ha="center", va="center", color=color, fontsize=8)
            elif key in failures:
                ax.text(col, row, "fail", ha="center", va="center", color="#555555", fontsize=8, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, len(seq_lengths), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(batches), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("S4 / S3 TFLOP/s (%)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_notes(path: Path, failures: dict[tuple[int, int], str], seq_lengths: list[int], batches: list[int]) -> None:
    lines = [
        "# SageAttention4/SageAttention3 Ratio Grid Notes",
        "",
        "Default grid:",
        f"- sequence lengths: {', '.join(seq_label(seq_len) for seq_len in seq_lengths)}",
        f"- batch sizes: {', '.join(str(batch) for batch in batches)}",
        "- heads: 16",
        "- kernels: sageattn3 and sageattn4 only",
        "",
        "Known skipped configurations:",
    ]
    for (seq_len, batch), reason in sorted(failures.items()):
        lines.append(f"- {seq_label(seq_len)}, B={batch}: {reason}")
    lines.extend(
        [
            "",
            "128k is not part of the default grid because previous runs failed during TMA descriptor initialization before S3/S4 timing completed.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def print_table(summary: list[dict], seq_lengths: list[int], batches: list[int]) -> None:
    by_key = {(row["seq_len"], row["batch"]): row for row in summary}
    print("seq\t" + "\t".join(f"B{batch}" for batch in batches))
    for seq_len in seq_lengths:
        row = [seq_label(seq_len)]
        for batch in batches:
            entry = by_key[(seq_len, batch)]
            if entry["status"] == "ok":
                row.append(f"{entry['ratio_mean'] * 100:.1f}%")
            elif entry["status"] == "failed":
                row.append("fail")
            else:
                row.append("")
        print("\t".join(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and plot an S4/S3 throughput-ratio grid.")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=DEFAULT_SEQ_LENGTHS)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--plot-out", type=Path, default=Path("reports/bench_s4_s3_ratio_heatmap_by_batch_seq.png"))
    parser.add_argument("--json-out", type=Path, default=Path("reports/bench_s4_s3_ratio_heatmap_by_batch_seq.json"))
    parser.add_argument("--notes-out", type=Path, default=Path("reports/bench_s4_s3_ratio_heatmap_by_batch_seq_notes.md"))
    parser.add_argument("--source-json", type=Path, action="append", default=[])
    parser.add_argument("--plot-only", action="store_true", help="Do not launch benchmarks; only plot existing JSON results.")
    parser.add_argument("--rerun", action="store_true", help="Rerun cells even if an existing measurement is available.")
    parser.add_argument("--include-known-failures", action="store_true", help="Try known failing/OOM cells instead of marking them failed.")
    parser.add_argument("--dry-run", action="store_true", help="Print benchmark commands without running them.")
    parser.add_argument("--non-causal", action="store_true")
    parser.add_argument("--no-per-block-mean", dest="per_block_mean", action="store_false", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seq_lengths = sorted(args.seq_lengths)
    batches = sorted(args.batches)
    known_failures = {
        key: reason
        for key, reason in KNOWN_FAILED_CELLS.items()
        if key[0] in seq_lengths and key[1] in batches and not args.include_known_failures
    }

    measurements, _ = collect_measurements(args.out_dir, seq_lengths, batches, args.source_json)
    run_results = []
    if not args.plot_only:
        for seq_len in seq_lengths:
            for batch in batches:
                key = (seq_len, batch)
                if key in known_failures:
                    print(f"skipping known failure {seq_label(seq_len)} B{batch}: {known_failures[key]}", flush=True)
                    continue
                if not args.rerun and key in measurements:
                    print(f"skipping existing {seq_label(seq_len)} B{batch}", flush=True)
                    continue
                run_results.append(run_cell(args, seq_len, batch))
        measurements, raw = collect_measurements(args.out_dir, seq_lengths, batches, args.source_json)
    else:
        raw = collect_measurements(args.out_dir, seq_lengths, batches, args.source_json)[1]

    unexpected_failures = {
        (item["seq_len"], item["batch"]): item["reason"]
        for item in run_results
        if item.get("status") == "failed"
    }
    failures = {**known_failures, **unexpected_failures}
    summary = make_summary(measurements, seq_lengths, batches, failures)

    plot_heatmap(measurements, seq_lengths, batches, failures, args.plot_out)
    args.json_out.write_text(
        json.dumps(
            {
                "seq_lengths": seq_lengths,
                "batches": batches,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "known_failures": [
                    {"seq_len": seq_len, "batch": batch, "reason": reason}
                    for (seq_len, batch), reason in sorted(known_failures.items())
                ],
                "run_results": run_results,
                "summary": summary,
                "raw": raw,
            },
            indent=2,
        )
    )
    write_notes(args.notes_out, known_failures, seq_lengths, batches)
    print_table(summary, seq_lengths, batches)
    print(args.plot_out)
    print(args.json_out)
    print(args.notes_out)


if __name__ == "__main__":
    main()
