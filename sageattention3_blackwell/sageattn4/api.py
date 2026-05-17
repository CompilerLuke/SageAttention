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
import einops as E
import sageattn4_fwd_cuda as fp4attn4_cuda
import sageattn4_quant_cuda as fp4quant4_cuda

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

def ceil_div(a,b):
    return (a+b-1) // b

def round_to_blockscaled_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round as the CUDA block-scaled FP4 quantizer, returning dequantized values."""
    assert x.size(-1) % 16 == 0
    orig_dtype = x.dtype
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], xf.shape[-1] // 16, 16)

    scale = (blocks.abs().amax(dim=-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn).float()
    scaled = torch.where(scale == 0, torch.zeros_like(blocks), blocks / scale)
    scaled_abs = scaled.abs().clamp(max=6.0)

    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x.device,
        dtype=torch.float32,
    )
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=x.device,
        dtype=torch.float32,
    )
    rounded_abs = levels[torch.bucketize(scaled_abs, boundaries)]
    rounded = rounded_abs.copysign(scaled)
    return (rounded * scale).reshape_as(xf).to(orig_dtype)

def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool, quant_block_size: int):
    B,H,_,D = q.shape
    device = q.device

    def pad_to_block(x, N):
        L = x.size(2)
        pad_len = (N - L % N) % N
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()
    def quant_pack_k(x):
        Q = quant_block_size-1
        x = pad_to_block(x, Q)
        T = x.size(-2)
        x_block = E.rearrange(x, 'b h (n m) d -> b h n m d', n=T//Q, m=Q)
        x_mean = x_block.mean(dim=-2)
        x_mean = round_to_blockscaled_fp4(x_mean)
        x_mean_f = x_mean.float()
        x_norm_sq = (x_mean_f * x_mean_f).sum(dim=-1, keepdim=True)
        x_mean_dir = torch.where(
            x_norm_sq > 0,
            x_mean_f / x_norm_sq,
            torch.zeros_like(x_mean_f),
        )
        lamb = torch.einsum('b h n m d, b h n d -> b h n m', x_block.float(), x_mean_dir)

        x_res = (x_block.float() - lamb.unsqueeze(-1) * x_mean_f.unsqueeze(-2)).to(x.dtype)
        lamb = torch.cat([
            torch.zeros((B,H,T//Q,1), device=device, dtype=torch.float32),
            lamb.to(torch.float32)
        ], dim=-1)
        x_block = torch.cat([
            x_mean.unsqueeze(-2),
            x_res
        ], dim=-2)
        x_block = x_block.reshape(B,H,T//Q*(Q+1),D)
        lamb = lamb.reshape(B,H,T//Q*(Q+1))
        return x_block, lamb
    def quant_pack_v(x):
        Q = quant_block_size-1
        x = pad_to_block(x, Q)
        T = x.size(-2)
        x_block = E.rearrange(x, 'b h (n m) d -> b h n m d', n=T//Q, m=Q)

        x_mean = torch.zeros((B,H,T//Q,D), device=device, dtype=x.dtype)
        x_res = x_block

        #TODO: SMOOTH V-VALUES AS WELL to avoid padding waste
        #x_mean = x_block.mean(dim=-2)
        #x_res = x_block - x_mean.unsqueeze(-2)

        x_block = torch.cat([
            x_mean.unsqueeze(-2),
            x_res
        ], dim=-2)
        x_block = x_block.reshape(B,H,T//Q*(Q+1),D)
        return x_block

    BLOCK_SIZE = 128

    # Match sageattn3's softmax-invariant global K centering before padding/packing.
    k = k - k.mean(dim=-2, keepdim=True)

    q = pad_to_block(q, N=BLOCK_SIZE)
    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm

    k, k_lambda = quant_pack_k(k)
    v = quant_pack_v(v)

    k = pad_to_block(k, N=BLOCK_SIZE)
    k_lambda = pad_to_block(k_lambda.unsqueeze(-1), N=BLOCK_SIZE).squeeze(-1)

    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()

    v = pad_to_block(v, N=BLOCK_SIZE)

    print("k=", k.shape, "v=", v.shape)

    return q, k, v, delta_s, k_lambda

def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant4_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant4_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant4_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def blockscaled_fp4_attn(qlist: Tuple, 
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         lambda_k: torch.Tensor,
                         KL: int,
                         is_causal: bool = False, 
                         per_block_mean: bool = True,
                         is_bf16: bool = True
                        ):
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return fp4attn4_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, lambda_k, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16)


def last_fwd_used_specialized() -> bool:
    return fp4attn4_cuda.last_fwd_used_specialized()


def sageattn4_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, quant_block = 8, **kwargs):
    assert quant_block in [8]

    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)

    assert q.dtype == torch.bfloat16, "TODO: support inputs other than bfloat16"

    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v, delta_s, lambdaK = preprocess_qkv(q, k, v, per_block_mean, quant_block)
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)

    # CORRECT KEY LENGTH TO ACCOUNT FOR K_PACKING
    # K_PACKING PACKS [K_MEAN K_RES0 K_RES1 ... K_RES_{Q-1}], which increases the sequence length
    Q = quant_block - 1

    print("KL BEFORE", KL)

    if KL % Q == 0:
        KL = KL // Q * (Q+1)
    else:
        KL = KL // Q * (Q+1) + 1 + (KL%Q)

    print("KL AFTER", KL)

    o_fp4 = blockscaled_fp4_attn(
    qlist_from_cuda,
    klist_from_cuda, 
    vlist_from_cuda,
    delta_s,
    lambdaK,
    KL,
    is_causal,
    per_block_mean,
    is_bf16
    )[0][:, :, :QL, :].contiguous()
    print("output fp4")
    return o_fp4
