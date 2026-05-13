"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa

try:
    import fp4attn_cuda
except ImportError:  # pragma: no cover - depends on local extension build
    fp4attn_cuda = None

try:
    import fp4quant_cuda
except ImportError:  # pragma: no cover - depends on local extension build
    fp4quant_cuda = None


MEAN_RESIDUAL_BLOCK_SIZE = 15
MEAN_RESIDUAL_EXPANDED_TILE = 128
MEAN_RESIDUAL_K_SMOOTHING_STRENGTH = 0.1
MEAN_RESIDUAL_K_LAYOUT = "block_local"


@triton.jit
def group_mean_kernel(
    q_ptr,          
    q_out_ptr,      
    qm_out_ptr,     
    B, H, L, D: tl.constexpr,    
    stride_qb, stride_qh, stride_ql, stride_qd,  
    stride_qmb, stride_qmh, stride_qml, stride_qmd,  
    GROUP_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)
    
    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    
    q_offsets = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + tl.arange(0, D)[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets)
    
    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE
    
    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group)

    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D) * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor):
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE
    
    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype) 
    
    grid = (B, H, num_groups)
    
    group_mean_kernel[grid](
        q, q_out, qm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE
    )
    return q_out, qm


def _mean_residual_tile_params(block_size: int = MEAN_RESIDUAL_BLOCK_SIZE):
    group_width = block_size + 1
    if MEAN_RESIDUAL_EXPANDED_TILE % group_width != 0:
        raise ValueError("block_size + 1 must divide the 128-token expanded tile")
    groups_per_tile = MEAN_RESIDUAL_EXPANDED_TILE // group_width
    token_tile = groups_per_tile * block_size
    return group_width, groups_per_tile, token_tile


def _pad_tokens(x: torch.Tensor, multiple: int):
    pad_len = (multiple - x.size(-2) % multiple) % multiple
    if pad_len == 0:
        return x.contiguous()
    return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()


def _pack_group_slots(
    expanded: torch.Tensor,
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
    layout: str = "block_local",
    value_ndim: int = 0,
):
    group_width, groups_per_tile, _ = _mean_residual_tile_params(block_size)
    slot_dim = expanded.ndim - value_ndim - 1
    group_dim = slot_dim - 1
    tile_dim = group_dim - 1
    if expanded.shape[group_dim] != groups_per_tile or expanded.shape[slot_dim] != group_width:
        raise ValueError("expanded tensor has incompatible mean/residual group shape")
    if layout == "block_local":
        ordered = expanded
    elif layout == "slot_major":
        ordered = expanded.movedim(slot_dim, group_dim)
    else:
        raise ValueError(f"unsupported mean/residual layout: {layout}")
    prefix = ordered.shape[:tile_dim]
    tiles = ordered.shape[tile_dim]
    tail = ordered.shape[-value_ndim:] if value_ndim else ()
    return ordered.reshape(*prefix, tiles * groups_per_tile * group_width, *tail)


def _unpack_group_slots(
    packed: torch.Tensor,
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
    layout: str = "block_local",
    value_ndim: int = 0,
):
    group_width, groups_per_tile, _ = _mean_residual_tile_params(block_size)
    seq_dim = packed.ndim - value_ndim - 1
    expanded_len = packed.shape[seq_dim]
    if expanded_len % MEAN_RESIDUAL_EXPANDED_TILE != 0:
        raise ValueError("packed mean/residual length must be a multiple of 128")
    tiles = expanded_len // MEAN_RESIDUAL_EXPANDED_TILE
    prefix = packed.shape[:seq_dim]
    tail = packed.shape[-value_ndim:] if value_ndim else ()
    grouped = packed.reshape(*prefix, tiles, group_width, groups_per_tile, *tail)
    if layout == "slot_major":
        return grouped.movedim(seq_dim + 1, seq_dim + 2)
    if layout == "block_local":
        return grouped.reshape(*prefix, tiles, groups_per_tile, group_width, *tail)
    raise ValueError(f"unsupported mean/residual layout: {layout}")


def decompose_mean_residual_blocks(
    x: torch.Tensor,
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
    layout: str = "block_local",
    eps: float = 1e-6,
    smoothing_strength: float = 1.0,
):
    if x.ndim != 4:
        raise ValueError("expected x with shape [B, H, L, D]")
    original_len = x.size(-2)
    group_width, groups_per_tile, token_tile = _mean_residual_tile_params(block_size)
    padded = _pad_tokens(x, token_tile)
    bsz, heads, _, dim = padded.shape
    tiles = padded.size(-2) // token_tile
    blocks = padded.reshape(bsz, heads, tiles, groups_per_tile, block_size, dim)
    mean = blocks.mean(dim=-2)
    mean_f = mean.float()
    denom = mean_f.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    lambda_vals = (
        (blocks.float() * mean_f.unsqueeze(-2)).sum(dim=-1, keepdim=True) / denom.unsqueeze(-2)
    ).squeeze(-1).to(x.dtype)
    lambda_vals = (lambda_vals.float() * smoothing_strength).to(x.dtype)
    residual = blocks - lambda_vals.unsqueeze(-1) * mean.unsqueeze(-2)
    expanded = torch.cat([mean.unsqueeze(-2), residual], dim=-2)
    lambda_expanded = torch.cat(
        [torch.zeros_like(lambda_vals[..., :1]), lambda_vals],
        dim=-1,
    )
    packed = _pack_group_slots(expanded, block_size, layout, value_ndim=1).contiguous()
    lambda_packed = _pack_group_slots(lambda_expanded, block_size, layout).contiguous()
    return packed, lambda_packed, original_len


def reconstruct_mean_residual_scores(
    expanded_scores: torch.Tensor,
    lambda_q: torch.Tensor,
    lambda_k: torch.Tensor,
    q_original_len: int,
    k_original_len: int,
    q_layout: str = "block_local",
    k_layout: str = "slot_major",
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
):
    q_grouped = _unpack_group_slots(lambda_q, block_size, q_layout)
    k_grouped = _unpack_group_slots(lambda_k, block_size, k_layout)
    bsz, heads, q_tiles, q_groups, group_width = q_grouped.shape
    k_tiles, k_groups = k_grouped.shape[-3:-1]
    scores = expanded_scores.reshape(
        bsz,
        heads,
        q_tiles,
        MEAN_RESIDUAL_EXPANDED_TILE,
        k_tiles,
        MEAN_RESIDUAL_EXPANDED_TILE,
    )
    if q_layout == "block_local":
        scores = scores.reshape(bsz, heads, q_tiles, q_groups, group_width, k_tiles, MEAN_RESIDUAL_EXPANDED_TILE)
    elif q_layout == "slot_major":
        scores = scores.reshape(bsz, heads, q_tiles, group_width, q_groups, k_tiles, MEAN_RESIDUAL_EXPANDED_TILE)
        scores = scores.movedim(3, 4)
    else:
        raise ValueError(f"unsupported mean/residual layout: {q_layout}")
    if k_layout == "block_local":
        scores = scores.reshape(bsz, heads, q_tiles, q_groups, group_width, k_tiles, k_groups, group_width)
    elif k_layout == "slot_major":
        scores = scores.reshape(bsz, heads, q_tiles, q_groups, group_width, k_tiles, group_width, k_groups)
        scores = scores.movedim(-2, -1)
    else:
        raise ValueError(f"unsupported mean/residual layout: {k_layout}")
    a00 = scores[:, :, :, :, 0, :, :, 0]
    q_mean_k_res = scores[:, :, :, :, 0, :, :, 1:]
    q_res_k_mean = scores[:, :, :, :, 1:, :, :, 0]
    q_res_k_res = scores[:, :, :, :, 1:, :, :, 1:]
    lq = q_grouped[..., 1:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    lk = k_grouped[..., 1:].unsqueeze(2).unsqueeze(3).unsqueeze(4)
    reconstructed = (
        q_res_k_res
        + lq * q_mean_k_res.unsqueeze(4)
        + lk * q_res_k_mean.unsqueeze(-1)
        + lq * lk * a00.unsqueeze(4).unsqueeze(-1)
    )
    reconstructed = reconstructed.reshape(
        bsz,
        heads,
        q_tiles * q_groups * block_size,
        k_tiles * k_groups * block_size,
    )
    return reconstructed[..., :q_original_len, :k_original_len]


def expand_v_for_mean_residual_blocks(
    v: torch.Tensor,
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
    layout: str = "slot_major",
):
    if v.ndim != 4:
        raise ValueError("expected v with shape [B, H, L, D]")
    original_len = v.size(-2)
    group_width, groups_per_tile, token_tile = _mean_residual_tile_params(block_size)
    padded = _pad_tokens(v, token_tile)
    bsz, heads, _, dim = padded.shape
    tiles = padded.size(-2) // token_tile
    blocks = padded.reshape(bsz, heads, tiles, groups_per_tile, block_size, dim)
    expanded = torch.zeros(
        bsz,
        heads,
        tiles,
        groups_per_tile,
        group_width,
        dim,
        device=v.device,
        dtype=v.dtype,
    )
    expanded[..., 1:, :] = blocks
    return _pack_group_slots(expanded, block_size, layout, value_ndim=1).contiguous(), original_len


def _pack_k_scores_for_layout(
    scores: torch.Tensor,
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
    layout: str = "slot_major",
):
    group_width, groups_per_tile, token_tile = _mean_residual_tile_params(block_size)
    if scores.size(-1) % token_tile != 0:
        raise ValueError("score columns must be padded to the K token tile")
    tiles = scores.size(-1) // token_tile
    blocks = scores.reshape(*scores.shape[:-1], tiles, groups_per_tile, block_size)
    expanded = torch.zeros(
        *scores.shape[:-1],
        tiles,
        groups_per_tile,
        group_width,
        device=scores.device,
        dtype=scores.dtype,
    )
    expanded[..., 1:] = blocks
    return _pack_group_slots(expanded, block_size, layout).contiguous()


def _pack_k_scores_for_slot_major(
    scores: torch.Tensor,
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
):
    return _pack_k_scores_for_layout(scores, block_size, "slot_major")


def pack_probs_q_mean_residual_blocks(
    probs: torch.Tensor,
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
):
    original_len = probs.size(-2)
    group_width, groups_per_tile, token_tile = _mean_residual_tile_params(block_size)
    pad_len = (token_tile - original_len % token_tile) % token_tile
    if pad_len:
        probs = F.pad(probs, (0, 0, 0, pad_len), value=0)
    tiles = probs.size(-2) // token_tile
    blocks = probs.reshape(*probs.shape[:-2], tiles, groups_per_tile, block_size, probs.size(-1))
    mean = blocks.mean(dim=-2, keepdim=True)
    expanded = torch.cat([mean, blocks - mean], dim=-2)
    return _pack_group_slots(expanded, block_size, "block_local", value_ndim=1).contiguous(), original_len


def unpack_q_mean_residual_output(
    out: torch.Tensor,
    original_len: int,
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
):
    grouped = _unpack_group_slots(out, block_size, "block_local", value_ndim=1)
    restored = grouped[..., 1:, :] + grouped[..., :1, :]
    restored = restored.reshape(*out.shape[:-2], -1, out.size(-1))
    return restored[..., :original_len, :]


def unpack_q_block_local_output(
    out: torch.Tensor,
    original_len: int,
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
):
    grouped = _unpack_group_slots(out, block_size, "block_local", value_ndim=1)
    restored = grouped[..., 1:, :].reshape(*out.shape[:-2], -1, out.size(-1))
    return restored[..., :original_len, :]


def unpack_q_slot_major_output(
    out: torch.Tensor,
    original_len: int,
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
):
    grouped = _unpack_group_slots(out, block_size, "slot_major", value_ndim=1)
    restored = grouped[..., 1:, :].reshape(*out.shape[:-2], -1, out.size(-1))
    return restored[..., :original_len, :]


def mean_residual_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    block_size: int = MEAN_RESIDUAL_BLOCK_SIZE,
):
    q_packed, lambda_q, q_len = decompose_mean_residual_blocks(q, block_size, "slot_major")
    k_packed, lambda_k, k_len = decompose_mean_residual_blocks(k, block_size, "slot_major")
    expanded_scores = torch.matmul(q_packed, k_packed.transpose(-2, -1))
    scores = reconstruct_mean_residual_scores(
        expanded_scores,
        lambda_q,
        lambda_k,
        q_len,
        k_len,
        "slot_major",
        "slot_major",
        block_size,
    )
    if is_causal:
        causal = torch.ones(q_len, k_len, device=q.device, dtype=torch.bool).tril(diagonal=k_len - q_len)
        scores = scores.masked_fill(~causal, -torch.inf)
    probs = torch.softmax(scores * (q.size(-1) ** -0.5), dim=-1)
    return torch.matmul(probs.to(v.dtype), v)


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):
    q_original_len = q.size(-2)
    _, _, k_token_tile = _mean_residual_tile_params()
    q_packed = _pad_tokens(q, MEAN_RESIDUAL_EXPANDED_TILE)
    k = k - k.mean(dim=-2, keepdim=True)
    k_padded = _pad_tokens(k, k_token_tile)
    if per_block_mean:
        q_packed, q_mean = triton_group_mean(q_packed)
    else:
        q_mean = q_packed.mean(dim=-2, keepdim=True)
        q_packed = q_packed - q_mean
    k_packed, lambda_k, k_original_len = decompose_mean_residual_blocks(
        k,
        layout=MEAN_RESIDUAL_K_LAYOUT,
        smoothing_strength=MEAN_RESIDUAL_K_SMOOTHING_STRENGTH,
    )
    v_packed, v_original_len = expand_v_for_mean_residual_blocks(v, layout=MEAN_RESIDUAL_K_LAYOUT)
    delta_s_compact = torch.matmul(q_mean, k_padded.transpose(-2, -1)).to(torch.float32)
    delta_s = _pack_k_scores_for_layout(delta_s_compact, layout=MEAN_RESIDUAL_K_LAYOUT)
    lambda_q = torch.empty(
        q_packed.size(0),
        q_packed.size(1),
        q_packed.size(2),
        device=q.device,
        dtype=torch.bfloat16,
    )
    lambda_k = lambda_k.to(torch.bfloat16).contiguous()
    return q_packed, k_packed, v_packed, delta_s, lambda_q, lambda_k, q_original_len, k_original_len, v_original_len

def scale_and_quant_fp4(x: torch.Tensor, lambda_: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    if fp4quant_cuda is None:
        raise RuntimeError("fp4quant_cuda is not installed")
    B, H, N, D = x.shape
    if lambda_ is not None and D != 128:
        raise ValueError("lambda sign-bit packing requires head_dim == 128")
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    if lambda_ is None:
        fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    else:
        fp4quant_cuda.scaled_fp4_quant_with_lambda(x, packed_fp4, fp8_scale, lambda_.float().contiguous(), 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_permute(x: torch.Tensor, lambda_: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    if fp4quant_cuda is None:
        raise RuntimeError("fp4quant_cuda is not installed")
    B, H, N, D = x.shape
    if lambda_ is not None and D != 128:
        raise ValueError("lambda sign-bit packing requires head_dim == 128")
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    if lambda_ is None:
        fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    else:
        fp4quant_cuda.scaled_fp4_quant_permute_with_lambda(x, packed_fp4, fp8_scale, lambda_.float().contiguous(), 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    if fp4quant_cuda is None:
        raise RuntimeError("fp4quant_cuda is not installed")
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def blockscaled_fp4_attn(qlist: Tuple, 
                         klist: Tuple,
                         vlist: Tuple,
                         lambda_q: torch.Tensor,
                         lambda_k: torch.Tensor,
                         delta_s: torch.Tensor,
                         QL: int,
                         KL: int,
                         is_causal: bool = False, 
                         per_block_mean: bool = True,
                         is_bf16: bool = True
                        ):
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return fp4attn_cuda.fwd(
        qlist[0], klist[0], vlist[0],
        qlist[1], klist[1], vlist[1],
        lambda_q, lambda_k,
        delta_s, QL, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16,
    )


def quantize_mean_residual_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    q_packed, k_packed, v_packed, _, lambda_q, lambda_k, q_len, k_len, v_len = preprocess_qkv(q, k, v)
    return {
        "q": scale_and_quant_fp4(q_packed),
        "k": scale_and_quant_fp4_permute(k_packed),
        "v": scale_and_quant_fp4_transpose(v_packed),
        "lambda_q": lambda_q,
        "lambda_k": lambda_k,
        "q_original_len": q_len,
        "k_original_len": k_len,
        "v_original_len": v_len,
    }


def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, **kwargs):
    if fp4attn_cuda is None or q.size(-1) != 128:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v, delta_s, lambda_q, lambda_k, q_original_len, k_original_len, _ = preprocess_qkv(q, k, v, per_block_mean)
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)
    o_expanded = blockscaled_fp4_attn(
        qlist_from_cuda,
        klist_from_cuda,
        vlist_from_cuda,
        lambda_q,
        lambda_k,
        delta_s,
        q_original_len,
        KL,
        is_causal,
        per_block_mean,
        is_bf16,
    )[0]
    return o_expanded[..., :QL, :].contiguous()
