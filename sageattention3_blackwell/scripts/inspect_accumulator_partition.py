#!/usr/bin/env python3
"""Inspect the CuTe accumulator partition used by the Blackwell QK MMA.

This compiles a tiny host-side CuTe probe for the same tiled MMA shape used by
the block-lambda kernel, then annotates each accumulator element with the
current mean/residual layout fields:

  Q block-local, b=15: row = q_group * 16 + q_slot
  K slot-major,  b=15: col = k_slot * 8 + k_group

The important output is not performance. It is the ownership relation between a
real residual/residual accumulator element and its reconstruction sources:

  q_mean_k_res:  (q_group * 16, col)
  q_res_k_mean:  (row, k_group)
  q_mean_k_mean: (q_group * 16, k_group)

Example:
  python scripts/inspect_accumulator_partition.py --warp 0
  python scripts/inspect_accumulator_partition.py --warp 0 --sample-limit 0
  python scripts/inspect_accumulator_partition.py --dump-csv /tmp/acc.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


PROBE_SOURCE = r"""
#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "sageattn3/blackwell/cute_extension.h"

using namespace cute;

int main() {
    static constexpr int kHeadDim = @@HEAD_DIM@@;
    static constexpr int kBlockM = @@BLOCK_M@@;
    static constexpr int kBlockN = @@BLOCK_N@@;

    using Element = cutlass::float_e2m1_t;
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using PermTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));
    using PermTileN = _32;
    using PermTileK = Int<kHeadDim>;
    using AtomLayoutMNK = std::conditional_t<
        kBlockM == 128,
        Layout<Shape<_8, _1, _1>>,
        Layout<Shape<_4, _1, _1>>
    >;
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
        AtomLayoutMNK{},
        Tile<PermTileM, PermTileN, PermTileK>{}
    ));

    TiledMmaQK tiled_mma_qk;
    Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));

    std::cout << "# block_m," << kBlockM << "\n";
    std::cout << "# block_n," << kBlockN << "\n";
    std::cout << "# head_dim," << kHeadDim << "\n";
    std::cout << "# tiled_mma_threads," << int(size(tiled_mma_qk)) << "\n";
    std::cout << "tid,warp,lane,elem,row,col\n";

    for (int tid = 0; tid < int(size(tiled_mma_qk)); ++tid) {
        auto thread_mma_qk = tiled_mma_qk.get_thread_slice(tid);
        Tensor tScS = thread_mma_qk.partition_C(cS);
        for (int i = 0; i < int(size(tScS)); ++i) {
            auto coord = tScS(i);
            int row = int(get<0>(coord));
            int col = int(get<1>(coord));
            std::cout
                << tid << ","
                << (tid / 32) << ","
                << (tid % 32) << ","
                << i << ","
                << row << ","
                << col << "\n";
        }
    }
    return 0;
}
"""


@dataclass(frozen=True)
class Owner:
    tid: int
    warp: int
    lane: int
    elem: int


@dataclass
class AnnotatedEntry:
    tid: int
    warp: int
    lane: int
    elem: int
    row: int
    col: int
    q_group: int
    q_slot: int
    k_group: int
    k_slot: int
    is_real_score: bool
    q_mean_row: int
    k_mean_col: int
    q_mean_tid: int | None
    q_mean_lane: int | None
    q_mean_elem: int | None
    k_mean_tid: int | None
    k_mean_lane: int | None
    k_mean_elem: int | None
    qk_mean_tid: int | None
    qk_mean_lane: int | None
    qk_mean_elem: int | None
    q_mean_same_warp: bool | None
    q_mean_expected_lane: bool | None
    k_mean_same_thread: bool | None
    qk_mean_same_warp: bool | None
    qk_mean_expected_lane: bool | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def build_probe(args: argparse.Namespace, root: Path) -> Path:
    nvcc = shutil.which(args.nvcc)
    if nvcc is None:
        raise SystemExit(f"could not find nvcc executable: {args.nvcc}")

    source = (
        PROBE_SOURCE
        .replace("@@BLOCK_M@@", str(args.block_m))
        .replace("@@BLOCK_N@@", str(args.block_n))
        .replace("@@HEAD_DIM@@", str(args.head_dim))
    )
    key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    build_dir = Path(args.build_dir or (Path(tempfile.gettempdir()) / "sage_acc_partition_probe"))
    build_dir.mkdir(parents=True, exist_ok=True)
    src = build_dir / f"probe_{key}.cu"
    exe = build_dir / f"probe_{key}"
    src.write_text(source)

    if exe.exists() and not args.force_rebuild:
        return exe

    cmd = [
        nvcc,
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-I",
        str(root),
        "-I",
        str(root / "csrc" / "cutlass" / "include"),
        "-I",
        str(root / "csrc" / "cutlass" / "tools" / "util" / "include"),
        "-gencode",
        f"arch=compute_{args.cuda_arch},code=sm_{args.cuda_arch}",
        str(src),
        "-o",
        str(exe),
    ]
    if args.verbose:
        print("+", " ".join(cmd), file=sys.stderr)
    run(cmd, cwd=root)
    return exe


def read_probe(exe: Path) -> tuple[dict[str, str], list[dict[str, int]]]:
    proc = subprocess.run([str(exe)], check=True, text=True, stdout=subprocess.PIPE)
    metadata: dict[str, str] = {}
    csv_lines: list[str] = []
    for line in proc.stdout.splitlines():
        if line.startswith("# "):
            key, value = line[2:].split(",", 1)
            metadata[key] = value
        elif line:
            csv_lines.append(line)

    reader = csv.DictReader(csv_lines)
    rows: list[dict[str, int]] = []
    for row in reader:
        rows.append({key: int(value) for key, value in row.items()})
    return metadata, rows


def owner_tuple(owner: Owner | None) -> tuple[int | None, int | None, int | None]:
    if owner is None:
        return None, None, None
    return owner.tid, owner.lane, owner.elem


def annotate(rows: list[dict[str, int]], residual_block: int) -> list[AnnotatedEntry]:
    group_width = residual_block + 1
    if 128 % group_width != 0:
        raise SystemExit(f"128 is not divisible by residual_block + 1 ({group_width})")
    groups_per_tile = 128 // group_width

    owners: dict[tuple[int, int], Owner] = {}
    for row in rows:
        owners[(row["row"], row["col"])] = Owner(
            tid=row["tid"],
            warp=row["warp"],
            lane=row["lane"],
            elem=row["elem"],
        )

    annotated: list[AnnotatedEntry] = []
    for row in rows:
        q_group = row["row"] // group_width
        q_slot = row["row"] % group_width
        k_slot = row["col"] // groups_per_tile
        k_group = row["col"] % groups_per_tile
        is_real = q_slot != 0 and k_slot != 0

        q_mean_row = q_group * group_width
        k_mean_col = k_group

        q_mean = owners.get((q_mean_row, row["col"]))
        k_mean = owners.get((row["row"], k_mean_col))
        qk_mean = owners.get((q_mean_row, k_mean_col))
        q_mean_tid, q_mean_lane, q_mean_elem = owner_tuple(q_mean)
        k_mean_tid, k_mean_lane, k_mean_elem = owner_tuple(k_mean)
        qk_mean_tid, qk_mean_lane, qk_mean_elem = owner_tuple(qk_mean)

        expected_lane = row["lane"] & 3
        annotated.append(
            AnnotatedEntry(
                tid=row["tid"],
                warp=row["warp"],
                lane=row["lane"],
                elem=row["elem"],
                row=row["row"],
                col=row["col"],
                q_group=q_group,
                q_slot=q_slot,
                k_group=k_group,
                k_slot=k_slot,
                is_real_score=is_real,
                q_mean_row=q_mean_row,
                k_mean_col=k_mean_col,
                q_mean_tid=q_mean_tid,
                q_mean_lane=q_mean_lane,
                q_mean_elem=q_mean_elem,
                k_mean_tid=k_mean_tid,
                k_mean_lane=k_mean_lane,
                k_mean_elem=k_mean_elem,
                qk_mean_tid=qk_mean_tid,
                qk_mean_lane=qk_mean_lane,
                qk_mean_elem=qk_mean_elem,
                q_mean_same_warp=None if q_mean is None else q_mean.warp == row["warp"],
                q_mean_expected_lane=None if q_mean is None else q_mean.warp == row["warp"] and q_mean.lane == expected_lane,
                k_mean_same_thread=None if k_mean is None else k_mean.tid == row["tid"],
                qk_mean_same_warp=None if qk_mean is None else qk_mean.warp == row["warp"],
                qk_mean_expected_lane=None if qk_mean is None else qk_mean.warp == row["warp"] and qk_mean.lane == expected_lane,
            )
        )
    return annotated


def write_csv(path: Path, entries: list[AnnotatedEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(entries[0]).keys()))
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def write_json(path: Path, metadata: dict[str, str], entries: list[AnnotatedEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "entries": [asdict(entry) for entry in entries],
    }
    path.write_text(json.dumps(payload, indent=2))


def summarize(metadata: dict[str, str], entries: list[AnnotatedEntry], args: argparse.Namespace) -> None:
    real = [entry for entry in entries if entry.is_real_score]
    q_struct = [entry for entry in entries if entry.q_slot == 0]
    k_struct = [entry for entry in entries if entry.k_slot == 0]
    by_tid: dict[int, list[AnnotatedEntry]] = {}
    for entry in entries:
        by_tid.setdefault(entry.tid, []).append(entry)
    entries_per_thread = max(len(thread_entries) for thread_entries in by_tid.values())
    paired_q_mean_shuffles = (entries_per_thread + 1) // 2
    useful_q_mean_pairs = max(
        len({
            entry.elem & ~8
            for entry in thread_entries
            if entry.q_slot != 0 and entry.k_slot != 0
        })
        for thread_entries in by_tid.values()
    )
    qk_mean_shuffles = 2

    def count_real(attr: str) -> int:
        return sum(1 for entry in real if getattr(entry, attr) is True)

    print("Accumulator Partition Probe")
    print(f"  block_m/block_n/head_dim: {metadata.get('block_m')}/{metadata.get('block_n')}/{metadata.get('head_dim')}")
    print(f"  mma threads: {metadata.get('tiled_mma_threads')}")
    print(f"  accumulator entries: {len(entries)}")
    print(f"  real residual/residual entries: {len(real)}")
    print(f"  Q structural entries: {len(q_struct)}")
    print(f"  K structural entries: {len(k_struct)}")
    print()
    print("Reconstruction Source Locality For Real Entries")
    print(f"  q_mean_k_res same warp:       {count_real('q_mean_same_warp')}/{len(real)}")
    print(f"  q_mean_k_res lane == lane&3:  {count_real('q_mean_expected_lane')}/{len(real)}")
    print(f"  q_res_k_mean same thread:     {count_real('k_mean_same_thread')}/{len(real)}")
    print(f"  q_mean_k_mean same warp:      {count_real('qk_mean_same_warp')}/{len(real)}")
    print(f"  q_mean_k_mean lane == lane&3: {count_real('qk_mean_expected_lane')}/{len(real)}")
    print()
    print("No-Mask Reconstruction Shuffle Model")
    print("  per number below is warp-wide SHFL instructions per reconstruction call after unrolling")
    print(f"  original per-entry q_mean + qk_mean: {entries_per_thread} + {entries_per_thread} = {2 * entries_per_thread}")
    print(f"  qk_mean hoisted only:                {entries_per_thread} + {qk_mean_shuffles} = {entries_per_thread + qk_mean_shuffles}")
    print(f"  paired q_mean + qk hoist:            {paired_q_mean_shuffles} + {qk_mean_shuffles} = {paired_q_mean_shuffles + qk_mean_shuffles}")
    print(f"  paired lower bound if skipping K0:   {useful_q_mean_pairs} + {qk_mean_shuffles} = {useful_q_mean_pairs + qk_mean_shuffles}")
    print("  current no-mask path uses the paired lower bound and masks K0 by direct accumulator index")

    by_warp: dict[int, list[AnnotatedEntry]] = {}
    for entry in entries:
        by_warp.setdefault(entry.warp, []).append(entry)
    print()
    print("Warp Coverage")
    for warp in sorted(by_warp):
        subset = by_warp[warp]
        rows = sorted({entry.row for entry in subset})
        cols = sorted({entry.col for entry in subset})
        print(
            f"  warp {warp:2d}: rows {rows[0]:3d}..{rows[-1]:3d} "
            f"({len(rows):2d} rows), cols {cols[0]:3d}..{cols[-1]:3d} ({len(cols):3d} cols)"
        )

    if args.warp is not None:
        sample = [entry for entry in entries if entry.warp == args.warp]
    elif args.tid is not None:
        sample = [entry for entry in entries if entry.tid == args.tid]
    else:
        sample = [entry for entry in entries if entry.warp == 0]

    if args.sample_limit > 0:
        sample = sample[: args.sample_limit]

    print()
    label = f"warp {args.warp}" if args.warp is not None else f"tid {args.tid}" if args.tid is not None else "warp 0"
    suffix = "" if args.sample_limit == 0 else f" first {len(sample)} entries"
    print(f"Sample: {label}")
    if suffix:
        print(f"  showing{suffix}; use --sample-limit 0 or --dump-csv for the full layout")
    print(
        "  tid lane elem | row col | q(g,s) k(g,s) | "
        "qmean(t,l,e) kmean(t,l,e) qkmean(t,l,e)"
    )
    for entry in sample:
        print(
            f"  {entry.tid:3d} {entry.lane:4d} {entry.elem:4d} | "
            f"{entry.row:3d} {entry.col:3d} | "
            f"({entry.q_group:1d},{entry.q_slot:2d}) ({entry.k_group:1d},{entry.k_slot:2d}) | "
            f"({entry.q_mean_tid},{entry.q_mean_lane},{entry.q_mean_elem}) "
            f"({entry.k_mean_tid},{entry.k_mean_lane},{entry.k_mean_elem}) "
            f"({entry.qk_mean_tid},{entry.qk_mean_lane},{entry.qk_mean_elem})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--residual-block", type=int, default=15, help="Mean/residual block size b; group width is b + 1.")
    parser.add_argument("--cuda-arch", default="120a", help="CUDA arch suffix, e.g. 120a.")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--dump-csv", type=Path, help="Write annotated entries as CSV.")
    parser.add_argument("--dump-json", type=Path, help="Write annotated entries as JSON.")
    parser.add_argument("--warp", type=int, help="Sample entries for one warp.")
    parser.add_argument("--tid", type=int, help="Sample entries for one thread.")
    parser.add_argument("--sample-limit", type=int, default=128, help="Rows to print from the selected sample; 0 prints all.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_m not in (64, 128):
        raise SystemExit("--block-m must be 64 or 128 to match kernel_traits.h")
    root = repo_root()
    exe = build_probe(args, root)
    metadata, rows = read_probe(exe)
    entries = annotate(rows, args.residual_block)
    if args.dump_csv:
        write_csv(args.dump_csv, entries)
        print(f"Wrote {args.dump_csv}")
    if args.dump_json:
        write_json(args.dump_json, metadata, entries)
        print(f"Wrote {args.dump_json}")
    summarize(metadata, entries, args)


if __name__ == "__main__":
    main()
