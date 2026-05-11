import pytest
import torch
import torch.nn.functional as F

try:
    from sageattention3_blackwell.sageattn3 import api as mr
except ModuleNotFoundError:
    from sageattn3 import api as mr


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    ("q_layout", "k_layout"),
    [
        ("block_local", "block_local"),
        ("block_local", "slot_major"),
        ("slot_major", "block_local"),
        ("slot_major", "slot_major"),
    ],
)
def test_mean_residual_qk_reconstructs_dense_scores(q_layout, k_layout):
    torch.manual_seed(0)
    device = _device()
    q = torch.randn(1, 2, 123, 32, device=device)
    k = torch.randn(1, 2, 119, 32, device=device)

    q_packed, lambda_q, q_len = mr.decompose_mean_residual_blocks(q, layout=q_layout)
    k_packed, lambda_k, k_len = mr.decompose_mean_residual_blocks(k, layout=k_layout)

    expanded_scores = torch.matmul(q_packed, k_packed.transpose(-2, -1))
    scores = mr.reconstruct_mean_residual_scores(
        expanded_scores,
        lambda_q,
        lambda_k,
        q_len,
        k_len,
        q_layout=q_layout,
        k_layout=k_layout,
    )

    torch.testing.assert_close(scores, torch.matmul(q, k.transpose(-2, -1)), atol=2e-5, rtol=2e-5)


def test_zero_mean_v_slots_with_masked_structural_columns_matches_dense_pv():
    torch.manual_seed(1)
    device = _device()
    scores = torch.randn(1, 2, 123, 119, device=device)
    probs = torch.softmax(scores, dim=-1)
    v = torch.randn(1, 2, 119, 32, device=device)

    v_packed, v_len = mr.expand_v_for_mean_residual_blocks(v)
    assert v_len == v.shape[-2]

    block_size = mr.MEAN_RESIDUAL_BLOCK_SIZE
    group_width, groups_per_tile, token_tile = mr._mean_residual_tile_params(block_size)
    probs = F.pad(probs, (0, (token_tile - probs.shape[-1] % token_tile) % token_tile), value=0)
    tiles = probs.shape[-1] // token_tile
    p_blocks = probs.reshape(*probs.shape[:-1], tiles, groups_per_tile, block_size)
    expanded = torch.zeros(
        *probs.shape[:-1],
        tiles,
        groups_per_tile,
        group_width,
        device=probs.device,
        dtype=probs.dtype,
    )
    expanded[..., 1:] = p_blocks
    probs_packed = mr._pack_group_slots(expanded, block_size, "slot_major")

    torch.testing.assert_close(torch.matmul(probs_packed, v_packed), torch.matmul(probs[..., :119], v), atol=3e-6, rtol=3e-6)


def test_q_row_mean_probability_packing_matches_dense_pv():
    torch.manual_seed(2)
    device = _device()
    probs = torch.softmax(torch.randn(1, 2, 123, 119, device=device), dim=-1)
    v = torch.randn(1, 2, 119, 32, device=device)

    probs_packed, q_len = mr.pack_probs_q_mean_residual_blocks(probs)
    out_packed = torch.matmul(probs_packed, v)
    out = mr.unpack_q_mean_residual_output(out_packed, q_len)

    torch.testing.assert_close(out, torch.matmul(probs, v), atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("is_causal", [False, True])
def test_mean_residual_attention_reference_matches_sdpa(is_causal):
    torch.manual_seed(3)
    device = _device()
    q_len = 123
    k_len = 123 if is_causal else 119
    q = torch.randn(1, 2, q_len, 32, device=device)
    k = torch.randn(1, 2, k_len, 32, device=device)
    v = torch.randn(1, 2, k_len, 32, device=device)

    out = mr.mean_residual_attention_reference(q, k, v, is_causal=is_causal)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    torch.testing.assert_close(out, expected, atol=3e-6, rtol=3e-6)


def _skip_unless_specialized_blackwell_kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the specialized mean/residual kernel")
    if mr.fp4attn_cuda is None or not hasattr(mr.fp4attn_cuda, "fwd"):
        pytest.skip("fp4attn_cuda.fwd is not installed")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in {(12, 0), (12, 1)}:
        pytest.skip("the specialized FP4 attention kernel is built for Blackwell")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the specialized mean/residual kernel")
def test_sageattn3_blackwell_specialized_constant_value_tile():
    _skip_unless_specialized_blackwell_kernel()

    q = torch.zeros(1, 1, 112, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.ones_like(q)

    out = mr.sageattn3_blackwell(q, k, v, is_causal=True)
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    assert out.min().item() >= 0
    assert out.max().item() <= 1.05


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the mean/residual kernel")
def test_sageattn3_blackwell_specialized_random_smoke_matches_sdpa_loose(is_causal):
    _skip_unless_specialized_blackwell_kernel()

    torch.manual_seed(6)
    q = torch.randn(1, 1, 112, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = mr.sageattn3_blackwell(q, k, v, is_causal=is_causal)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    diff = (out.float() - expected.float()).abs()

    assert torch.isfinite(out).all()
    assert diff.mean().item() < 0.25
    assert diff.max().item() < 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the specialized mean/residual kernel")
def test_sageattn3_blackwell_specialized_pads_partial_kv_tiles():
    _skip_unless_specialized_blackwell_kernel()

    q = torch.randn(1, 1, 112, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, 111, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    out = mr.sageattn3_blackwell(q, k, v, is_causal=True)
    torch.cuda.synchronize()

    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for NVFP4 quantization")
def test_quantize_mean_residual_qkv_shapes_and_dtypes():
    if mr.fp4quant_cuda is None:
        pytest.skip("fp4quant_cuda is not installed")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in {(12, 0), (12, 1)}:
        pytest.skip("the packaged FP4 quantizer is built for Blackwell")

    torch.manual_seed(4)
    q = torch.randn(1, 2, 100, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 90, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2, 90, 128, device="cuda", dtype=torch.bfloat16)

    packed = mr.quantize_mean_residual_qkv(q, k, v)
    torch.cuda.synchronize()

    q_fp4, q_scale = packed["q"]
    k_fp4, k_scale = packed["k"]
    v_fp4, v_scale = packed["v"]

    assert q_fp4.shape == (1, 2, 128, 64)
    assert k_fp4.shape == (1, 2, 128, 64)
    assert v_fp4.shape == (1, 2, 128, 64)
    assert q_scale.shape == (1, 2, 128, 8)
    assert k_scale.shape == (1, 2, 128, 8)
    assert v_scale.shape == (1, 2, 128, 8)
    assert q_fp4.dtype == torch.uint8
    assert k_fp4.dtype == torch.uint8
    assert v_fp4.dtype == torch.uint8
    assert q_scale.dtype == torch.float8_e4m3fn
    assert k_scale.dtype == torch.float8_e4m3fn
    assert v_scale.dtype == torch.float8_e4m3fn
    assert packed["lambda_q"].shape == (1, 2, 128)
    assert packed["lambda_k"].shape == (1, 2, 128)
    assert packed["q_original_len"] == 100
    assert packed["k_original_len"] == 90
    assert packed["v_original_len"] == 90
