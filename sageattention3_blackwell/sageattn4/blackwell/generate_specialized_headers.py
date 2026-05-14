#!/usr/bin/env python3
"""Generate concrete SageAttention4 Blackwell type headers."""

from __future__ import annotations

import argparse
import difflib
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from string import Template

sys.dont_write_bytecode = True

from specialized_codegen import (
    CUTEDSL_STATUS,
    CppType,
    append_stage_layout,
    blockscaled_sf_atom,
    blockscaled_smem_layout_atom,
    call_expr,
    copy_atom_ldsm_u32x4_n,
    copy_atom_universal,
    cpp_bool,
    cute_blocked_product,
    cute_tile_to_shape,
    cute_int,
    cute_layout,
    cute_shape,
    cutedsl_context,
    cutlass_dtype,
    decltype,
    dynamic_int,
    gmma_k_smem_selector,
    instance,
    lambda_smem_layout_atom,
    multiline_call,
    named,
    render_namespace_type_aliases,
    sm120_rr_smem_selector,
    template_type,
    tiled_mma,
)


TEMPLATE_PATH = Path(__file__).with_name("generated_templates") / "sageattn4_fwd_specialization.h.in"


@dataclass(frozen=True)
class FwdSpec:
    head_dim: int = 128
    block_m: int = 128
    block_n: int = 128
    stages: int = 3
    cluster_m: int = 1
    block_mean: bool = True
    is_causal: bool = False

    @property
    def namespace(self) -> str:
        return (
            f"sageattn4_fwd_hdim{self.head_dim}_bm{self.block_m}_"
            f"bn{self.block_n}_s{self.stages}_blockmean{int(self.block_mean)}"
        )

    @property
    def header_name(self) -> str:
        return f"{self.namespace}.h"

    @property
    def n_warps(self) -> int:
        if self.block_m == 128:
            return 12
        if self.block_m == 64:
            return 8
        raise ValueError(f"unsupported block_m={self.block_m}")

    @property
    def atom_layout_m(self) -> int:
        if self.block_m == 128:
            return 8
        if self.block_m == 64:
            return 4
        raise ValueError(f"unsupported block_m={self.block_m}")

    @property
    def block_mean_cpp(self) -> str:
        return cpp_bool(self.block_mean)

    @property
    def is_causal_cpp(self) -> str:
        return cpp_bool(self.is_causal)


def build_type_table(spec: FwdSpec) -> dict[str, CppType]:
    types: dict[str, CppType] = {}

    types["M"] = cute_int(spec.block_m)
    types["N"] = cute_int(spec.block_n)
    types["K"] = cute_int(spec.head_dim)
    types["Stage"] = cute_int(spec.stages)

    m = named("M")
    n = named("N")
    k = named("K")
    tile_shape = cute_shape((m, n, k))

    element_dtype = cutlass_dtype("Float4E2M1FN")
    element_sf_dtype = cutlass_dtype("Float8E4M3FN")
    element_accum_dtype = cutlass_dtype("Float32")
    output_dtype = cutlass_dtype("BFloat16")
    index_dtype = cutlass_dtype("Int64")

    atom_layout_shape = (spec.atom_layout_m, 1, 1)
    atom_layout_mnk = cute_layout(atom_layout_shape)
    sfq_atom = blockscaled_smem_layout_atom(spec.block_m, spec.head_dim)
    sfk_atom = blockscaled_smem_layout_atom(spec.block_n, spec.head_dim)
    sfv_atom = blockscaled_smem_layout_atom(spec.block_n, spec.head_dim)
    sfvt_atom = blockscaled_smem_layout_atom(spec.head_dim, spec.block_n)
    lambda_k_atom = lambda_smem_layout_atom(spec.block_n)
    smem_layout_atom_ds = cute_layout((spec.block_m, spec.block_n), stride=(0, 1))
    layout_sfp = cute_layout(((16, 4), 1, spec.block_n // 64), stride=((0, 1), 0, 4))
    layout_p = cute_layout(((8, 2, 2), 1, spec.block_n // 64), stride=((1, 8, 16), 0, 32))
    smem_layout_lambda_k = append_stage_layout(lambda_k_atom, spec.stages)
    smem_layout_sfk = append_stage_layout(sfk_atom, spec.stages)
    smem_layout_sfv = append_stage_layout(sfv_atom, spec.stages)
    smem_layout_sfvt = append_stage_layout(sfvt_atom, spec.stages)
    smem_atom_k = sm120_rr_smem_selector(element_dtype, spec.head_dim)
    smem_atom_n = sm120_rr_smem_selector(element_dtype, spec.block_n)
    smem_layout_q = cute_tile_to_shape(smem_atom_k, (spec.block_m, spec.head_dim), (0, 1))
    smem_layout_k = cute_tile_to_shape(smem_atom_k, (spec.block_n, spec.head_dim, spec.stages), (0, 1, 2))
    smem_layout_v = cute_tile_to_shape(smem_atom_k, (spec.block_n, spec.head_dim, spec.stages), (0, 1, 2))
    smem_layout_vt = cute_tile_to_shape(smem_atom_n, (spec.head_dim, spec.block_n, spec.stages), (0, 1, 2))
    smem_layout_ds = cute_tile_to_shape(smem_layout_atom_ds, (spec.block_m, spec.block_n, spec.stages), (0, 1, 2))
    smem_atom_o = gmma_k_smem_selector(output_dtype, spec.head_dim)
    smem_layout_o = cute_tile_to_shape(smem_atom_o, (spec.block_m, spec.head_dim), (0, 1))

    smem_copy_atom_q = copy_atom_ldsm_u32x4_n(element_dtype)
    smem_copy_atom_kv = copy_atom_ldsm_u32x4_n(element_dtype)
    smem_copy_atom_sf = copy_atom_universal(element_sf_dtype)
    smem_copy_atom_lamb = copy_atom_universal(element_sf_dtype)
    smem_copy_atom_ds = copy_atom_universal(element_accum_dtype)

    tiled_mma_qk = tiled_mma(atom_layout_shape, (spec.block_m, 32, spec.head_dim))
    tiled_mma_pv = tiled_mma(atom_layout_shape, (spec.block_m, 32, spec.head_dim))
    dyn = dynamic_int()

    types["ElementPairType"] = template_type("cutlass::nv_float4_t", element_dtype)
    types["OutputType"] = output_dtype
    types["ElementSF"] = element_sf_dtype
    types["Element"] = element_dtype
    types["ElementAccum"] = element_accum_dtype
    types["ElementOut"] = output_dtype
    types["index_t"] = index_dtype
    types["TileShape_MNK"] = tile_shape
    types["ClusterShape_MNK"] = cute_shape((1, 1, 1))
    types["PermTileM"] = m
    types["PermTileN"] = cute_int(32)
    types["PermTileK"] = k
    types["AtomLayoutMNK"] = atom_layout_mnk

    kernel_input_to_mma = template_type(
        "cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element",
        element_dtype,
    )
    types["ElementQMma"] = decltype(call_expr(kernel_input_to_mma))
    types["ElementKMma"] = decltype(call_expr(kernel_input_to_mma))
    types["TiledMmaQK"] = tiled_mma_qk
    types["TiledMmaPV"] = tiled_mma_pv

    types["GmemTiledCopy"] = named("cute::SM90_TMA_LOAD")
    types["GmemTiledCopySF"] = named("cute::SM90_TMA_LOAD")
    types["GmemTiledCopyLambda"] = named("cute::SM90_TMA_LOAD")

    types["SmemLayoutAtomQ"] = smem_atom_k
    types["SmemLayoutAtomK"] = smem_atom_k
    types["SmemLayoutAtomV"] = smem_atom_k
    types["SmemLayoutAtomVt"] = smem_atom_n
    types["SmemLayoutQ"] = smem_layout_q
    types["SmemLayoutK"] = smem_layout_k
    types["SmemLayoutV"] = smem_layout_v
    types["SmemLayoutVt"] = smem_layout_vt
    types["SmemLayoutAtomDS"] = smem_layout_atom_ds
    types["SmemLayoutDS"] = smem_layout_ds

    types["SmemCopyAtomQ"] = smem_copy_atom_q
    types["SmemCopyAtomKV"] = smem_copy_atom_kv
    types["SmemCopyAtomSF"] = smem_copy_atom_sf
    types["SmemCopyAtomLamb"] = smem_copy_atom_lamb
    types["SmemCopyAtomDS"] = smem_copy_atom_ds

    types["BlkScaledConfig"] = template_type("flash::BlockScaledConfig", "kSFVectorSize")
    types["LayoutSF"] = named("BlkScaledConfig::LayoutSF")
    types["SfAtom"] = blockscaled_sf_atom(16)
    types["SmemLayoutAtomSFQ"] = sfq_atom
    types["SmemLayoutAtomSFK"] = sfk_atom
    types["SmemLayoutAtomLambK"] = lambda_k_atom
    types["SmemLayoutAtomSFV"] = sfv_atom
    types["SmemLayoutAtomSFVt"] = sfvt_atom
    types["LayoutSFP"] = layout_sfp
    types["LayoutP"] = layout_p
    types["SmemLayoutSFQ"] = sfq_atom
    types["SmemLayoutLambdaK"] = smem_layout_lambda_k
    types["SmemLayoutSFK"] = smem_layout_sfk
    types["SmemLayoutSFV"] = smem_layout_sfv
    types["SmemLayoutSFVt"] = smem_layout_sfvt
    types["LayoutDS"] = cute_blocked_product(
        smem_layout_atom_ds,
        (dyn, dyn, dyn, dyn),
        (dyn, 1, dyn, dyn),
    )
    types["LayoutLambdaKV"] = cute_blocked_product(
        lambda_k_atom,
        (dyn, dyn, dyn),
        (dyn, dyn, dyn),
    )

    types["SmemLayoutAtomO"] = smem_atom_o
    types["SmemLayoutO"] = smem_layout_o
    types["SharedStorage"] = template_type(
        "::SharedStorageQKVOwithSF",
        "kStages",
        "kEpiStages",
        element_dtype,
        element_sf_dtype,
        output_dtype,
        "SmemLayoutQ",
        "SmemLayoutK",
        "SmemLayoutV",
        "SmemLayoutDS",
        "SmemLayoutO",
        "SmemLayoutSFQ",
        "SmemLayoutSFK",
        "SmemLayoutSFVt",
        "SmemLayoutLambdaK",
    )
    types["MainloopPipeline"] = template_type("cutlass::PipelineTmaAsync", "kStages")
    types["PipelineState"] = template_type("cutlass::PipelineState", "kStages")
    types["MainloopPipelineQ"] = template_type("cutlass::PipelineTmaAsync", "1")
    types["PipelineParamsQ"] = named("MainloopPipelineQ::Params")
    types["PipelineStateQ"] = template_type("cutlass::PipelineState", "1")
    types["EpilogueBarrier"] = template_type("flash::OrderedSequenceBarrierVarGroupSize", "kEpiStages", "2")

    return types


def render_header(spec: FwdSpec) -> str:
    types = build_type_table(spec)
    substitutions = {
        "namespace": spec.namespace,
        "head_dim": spec.head_dim,
        "block_m": spec.block_m,
        "block_n": spec.block_n,
        "stages": spec.stages,
        "cluster_m": spec.cluster_m,
        "block_mean_cpp": spec.block_mean_cpp,
        "is_causal_cpp": spec.is_causal_cpp,
        "n_warps": spec.n_warps,
        "cutedsl_status": CUTEDSL_STATUS,
        "type_aliases": render_namespace_type_aliases(types),
    }
    return Template(TEMPLATE_PATH.read_text()).substitute(substitutions)


def write_or_check(path: Path, content: str, check: bool) -> bool:
    old = path.read_text() if path.exists() else ""
    if old == content:
        return True
    if check:
        diff = difflib.unified_diff(
            old.splitlines(),
            content.splitlines(),
            fromfile=str(path),
            tofile=f"{path} (generated)",
            lineterm="",
        )
        print("\n".join(diff))
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("generated"),
        help="directory for generated headers",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in headers are up to date without writing",
    )
    args = parser.parse_args()

    ok = True
    for spec in [FwdSpec()]:
        ok = write_or_check(args.output_dir / spec.header_name, render_header(spec), args.check) and ok
    return 0 if ok else 1


def main_and_force_exit() -> None:
    with cutedsl_context():
        try:
            rc = main()
        except BaseException:
            traceback.print_exc()
            rc = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)


if __name__ == "__main__":
    main_and_force_exit()
