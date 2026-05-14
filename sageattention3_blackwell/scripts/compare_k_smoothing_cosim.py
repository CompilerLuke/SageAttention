#!/usr/bin/env python3
"""
Compare attention-output cosine similarity for old K centering vs new K smoothing.

The default input path uses TransformerLens to extract real Q/K/V activations from
a sample text. The FP4 old/new kernels are run in separate subprocesses because
both packages export extension modules with the same names.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_TEXT = """
SageAttention trades small numerical approximations for large bandwidth wins in
attention kernels. A useful quality test should use activations from real model
layers, because token statistics, head specialization, and projection geometry
are not well represented by independent Gaussian tensors.
"""


CHILD_RUNNER = r"""
import argparse
import sys
from pathlib import Path

import torch


def _dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported kernel dtype: {name}")


parser = argparse.ArgumentParser()
parser.add_argument("--package-path", required=True)
parser.add_argument("--variant", choices=["old", "new"], required=True)
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--dtype", choices=["bf16", "fp16"], required=True)
parser.add_argument("--causal", action="store_true")
parser.add_argument("--no-per-block-mean", action="store_true")
args = parser.parse_args()

package_path = Path(args.package_path).resolve()
sys.path.insert(0, str(package_path))

from sageattn4 import api  # noqa: E402

device = torch.device("cuda")
dtype = _dtype(args.dtype)
payload = torch.load(args.input, map_location="cpu")
q = payload["q"].to(device=device, dtype=dtype).contiguous()
k = payload["k"].to(device=device, dtype=dtype).contiguous()
v = payload["v"].to(device=device, dtype=dtype).contiguous()
qlen = q.shape[-2]
klen = k.shape[-2]
per_block_mean = not args.no_per_block_mean
is_bf16 = dtype is torch.bfloat16

if args.variant == "old":
    q_p, k_p, v_p, delta_s = api.preprocess_qkv(
        q.clone(), k.clone(), v.clone(), per_block_mean
    )
    qlist = api.scale_and_quant_fp4(q_p)
    klist = api.scale_and_quant_fp4_permute(k_p)
    vlist = api.scale_and_quant_fp4_transpose(v_p)
    out = api.blockscaled_fp4_attn(
        qlist, klist, vlist, delta_s, klen, args.causal, per_block_mean, is_bf16
    )[0][..., :qlen, :].contiguous()
else:
    q_p, k_p, v_p, delta_s, lambda_q, lambda_k, q_orig, k_orig, _ = api.preprocess_qkv(
        q.clone(), k.clone(), v.clone(), per_block_mean
    )
    qlist = api.scale_and_quant_fp4(q_p)
    klist = api.scale_and_quant_fp4_permute(k_p)
    vlist = api.scale_and_quant_fp4_transpose(v_p)
    out = api.blockscaled_fp4_attn(
        qlist,
        klist,
        vlist,
        lambda_q,
        lambda_k,
        delta_s,
        q_orig,
        k_orig,
        args.causal,
        per_block_mean,
        is_bf16,
    )[0][..., :qlen, :].contiguous()

torch.cuda.synchronize()
torch.save(
    {
        "out": out.detach().cpu(),
        "package_path": str(package_path),
        "api_path": str(Path(api.__file__).resolve()),
        "q_shape": tuple(q.shape),
    },
    args.output,
)
"""


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compare cosine similarity of attention outputs from the old "
            "global-K-centering kernel and the new block-lambda K smoothing kernel."
        )
    )
    parser.add_argument(
        "--model",
        default="EleutherAI/pythia-6.9b",
        help="TransformerLens model name; default is ungated and has 128-wide attention heads",
    )
    parser.add_argument("--layer", type=int, default=0, help="Attention layer to sample")
    parser.add_argument("--seq-len", type=int, default=512, help="Token sequence length")
    parser.add_argument("--heads", type=int, default=0, help="Limit heads; 0 keeps all heads")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Sample text for TransformerLens")
    parser.add_argument("--text-file", type=Path, help="Read sample text from this file")
    parser.add_argument("--device", default="cuda", help="Device for TransformerLens and SDPA")
    parser.add_argument("--model-dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--kernel-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--causal", action="store_true", help="Use causal attention")
    parser.add_argument(
        "--no-per-block-mean",
        action="store_true",
        help="Use a single Q mean instead of per-128-token Q means",
    )
    parser.add_argument(
        "--old-package-path",
        type=Path,
        default=Path("/tmp/SageAttention/sageattention4_blackwell"),
        help="Checkout/build containing the old K-centering sageattn4 package",
    )
    parser.add_argument(
        "--new-package-path",
        type=Path,
        default=repo_root,
        help="Checkout/build containing the new K-smoothing sageattn4 package",
    )
    parser.add_argument("--skip-fp4", action="store_true", help="Only run dense SDPA sanity checks")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic Q/K/V instead of TransformerLens; useful for smoke tests",
    )
    parser.add_argument("--synthetic-heads", type=int, default=4)
    parser.add_argument("--synthetic-head-dim", type=int, default=128)
    parser.add_argument("--json-out", type=Path, help="Optional JSON metrics output path")
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def _read_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        return args.text_file.read_text()
    return args.text


@torch.no_grad()
def load_transformerlens_qkv(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise SystemExit(
            "TransformerLens is required for real activations. Install it with "
            "`pip install transformer-lens`, or pass `--synthetic` for a smoke test."
        ) from exc

    dtype = _torch_dtype(args.model_dtype)
    model = HookedTransformer.from_pretrained_no_processing(
        args.model,
        device=args.device,
        dtype=dtype,
    )
    model.eval()

    base_text = _read_text(args).strip()
    text = base_text
    tokens = model.to_tokens(text, prepend_bos=True).to(args.device)
    while tokens.shape[-1] < args.seq_len:
        text = text + "\n\n" + base_text
        tokens = model.to_tokens(text, prepend_bos=True).to(args.device)
    tokens = tokens[:, : args.seq_len]

    hook_names = {
        "q": f"blocks.{args.layer}.attn.hook_q",
        "k": f"blocks.{args.layer}.attn.hook_k",
        "v": f"blocks.{args.layer}.attn.hook_v",
    }
    _, cache = model.run_with_cache(
        tokens,
        return_type=None,
        stop_at_layer=args.layer + 1,
        names_filter=lambda name: name in set(hook_names.values()),
    )
    q = cache[hook_names["q"]].permute(0, 2, 1, 3).contiguous()
    k = cache[hook_names["k"]].permute(0, 2, 1, 3).contiguous()
    v = cache[hook_names["v"]].permute(0, 2, 1, 3).contiguous()
    if args.heads > 0:
        q = q[:, : args.heads]
        k = k[:, : args.heads]
        v = v[:, : args.heads]
    meta = {
        "source": "transformer_lens",
        "model": args.model,
        "layer": args.layer,
        "tokens": int(tokens.shape[-1]),
        "model_dtype": args.model_dtype,
    }
    return q, k, v, meta


def load_synthetic_qkv(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device=args.device).manual_seed(123)
    shape = (1, args.synthetic_heads, args.seq_len, args.synthetic_head_dim)
    dtype = _torch_dtype(args.model_dtype)
    q = torch.randn(shape, generator=generator, device=args.device, dtype=dtype)
    k = torch.randn(shape, generator=generator, device=args.device, dtype=dtype)
    v = torch.randn(shape, generator=generator, device=args.device, dtype=dtype)
    return q, k, v, {"source": "synthetic", "model_dtype": args.model_dtype}


def old_centered_k_dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    qf = q.float()
    k_centered = k.float() - k.float().mean(dim=-2, keepdim=True)
    vf = v.float()
    scores = torch.matmul(qf, k_centered.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if causal:
        q_len, k_len = q.shape[-2], k.shape[-2]
        mask = torch.ones(q_len, k_len, device=q.device, dtype=torch.bool).tril(
            diagonal=k_len - q_len
        )
        scores = scores.masked_fill(~mask, -torch.inf)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs.to(vf.dtype), vf)


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    diff = af - bf
    per_head_a = a.float().flatten(2)
    per_head_b = b.float().flatten(2)
    per_head_cos = F.cosine_similarity(per_head_a, per_head_b, dim=-1)
    return {
        "cos": float(F.cosine_similarity(af, bf, dim=0).item()),
        "per_head_cos_mean": float(per_head_cos.mean().item()),
        "per_head_cos_min": float(per_head_cos.min().item()),
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "rel_l2": float((diff.norm() / bf.norm().clamp_min(1e-12)).item()),
    }


def run_kernel_variant(
    package_path: Path,
    variant: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace,
    tmpdir: Path,
) -> dict[str, Any]:
    if not package_path.exists():
        raise FileNotFoundError(f"{variant} package path does not exist: {package_path}")
    input_path = tmpdir / "qkv.pt"
    output_path = tmpdir / f"{variant}_out.pt"
    if not input_path.exists():
        torch.save({"q": q.detach().cpu(), "k": k.detach().cpu(), "v": v.detach().cpu()}, input_path)
    cmd = [
        sys.executable,
        "-c",
        CHILD_RUNNER,
        "--package-path",
        str(package_path),
        "--variant",
        variant,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--dtype",
        args.kernel_dtype,
    ]
    if args.causal:
        cmd.append("--causal")
    if args.no_per_block_mean:
        cmd.append("--no-per-block-mean")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_path)
    subprocess.run(cmd, check=True, cwd=str(package_path), env=env)
    return torch.load(output_path, map_location="cpu")


def print_metrics(rows: dict[str, dict[str, float]]) -> None:
    headers = ("comparison", "cos", "head_mean", "head_min", "mae", "max_abs", "rel_l2")
    print(
        f"{headers[0]:<28} {headers[1]:>12} {headers[2]:>12} {headers[3]:>12} "
        f"{headers[4]:>12} {headers[5]:>12} {headers[6]:>12}"
    )
    for name, row in rows.items():
        print(
            f"{name:<28} {row['cos']:12.8f} {row['per_head_cos_mean']:12.8f} "
            f"{row['per_head_cos_min']:12.8f} {row['mae']:12.6g} "
            f"{row['max_abs']:12.6g} {row['rel_l2']:12.6g}"
        )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required unless you choose a CPU device and --skip-fp4")

    if args.synthetic:
        q, k, v, meta = load_synthetic_qkv(args)
    else:
        q, k, v, meta = load_transformerlens_qkv(args)
    if q.shape[-1] != 128 and not args.skip_fp4:
        raise SystemExit(
            f"Loaded model produced head_dim={q.shape[-1]}; FP4 comparison expects real 128-wide heads. "
            "Use a 128-head-dim model such as a Llama 8B family checkpoint."
        )

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=args.causal)
    old_dense = old_centered_k_dense(q, k, v, args.causal)
    rows: dict[str, dict[str, float]] = {
        "old_dense_vs_sdpa": metrics(old_dense.cpu(), ref.cpu()),
    }
    outputs: dict[str, Any] = {"meta": meta, "shape": tuple(q.shape), "metrics": rows}

    if not args.skip_fp4:
        with tempfile.TemporaryDirectory(prefix="k_smoothing_cosim_") as tmp:
            tmpdir = Path(tmp)
            old_payload = run_kernel_variant(args.old_package_path, "old", q, k, v, args, tmpdir)
            new_payload = run_kernel_variant(args.new_package_path, "new", q, k, v, args, tmpdir)
        old_out = old_payload["out"]
        new_out = new_payload["out"]
        ref_cpu = ref.detach().cpu()
        rows["old_fp4_vs_sdpa"] = metrics(old_out, ref_cpu)
        rows["new_fp4_vs_sdpa"] = metrics(new_out, ref_cpu)
        rows["new_fp4_vs_old_fp4"] = metrics(new_out, old_out)
        outputs["old_api_path"] = old_payload["api_path"]
        outputs["new_api_path"] = new_payload["api_path"]

    print("QKV source:", json.dumps(meta, sort_keys=True))
    print("QKV shape:", tuple(q.shape), "causal:", args.causal)
    if not args.skip_fp4:
        print("old api:", outputs["old_api_path"])
        print("new api:", outputs["new_api_path"])
    print_metrics(rows)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
