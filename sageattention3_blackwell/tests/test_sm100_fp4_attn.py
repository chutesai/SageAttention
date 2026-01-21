import os
import sys

import pytest
import torch


TEST_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sageattn3.api import (
    preprocess_qkv,
    sageattn3_blackwell,
    scale_and_quant_fp4_sm100,
)
import fp4quant_cuda


def _is_sm100():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major == 10 and minor == 0


def _dequant_fp4_sm100(packed, scales):
    return fp4quant_cuda.scaled_fp4_dequant_sm100(packed, scales)


def _reference_attn(q, k, v, delta_s, softmax_scale, is_causal, per_block_mean):
    qk = torch.matmul(q, k.transpose(-2, -1))
    if per_block_mean:
        group_ids = torch.arange(q.size(2), device=q.device) // 128
        delta_s_q = delta_s.index_select(2, group_ids)
    else:
        delta_s_q = delta_s[:, :, :1, :].expand_as(qk)
    qk = qk + delta_s_q

    if is_causal:
        q_idx = torch.arange(q.size(2), device=q.device)
        k_idx = torch.arange(k.size(2), device=q.device)
        valid = k_idx[None, :] < (q_idx[:, None] + 1 + (k.size(2) - q.size(2)))
        mask = torch.where(valid, 0.0, float("-inf"))
        qk = qk + mask

    attn = torch.softmax(qk * softmax_scale, dim=-1)
    return torch.matmul(attn, v)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm100_fp4_attention_matches_reference(head_dim, dtype, is_causal):
    if not _is_sm100():
        pytest.skip("SM100 GPU required")

    torch.manual_seed(0)
    device = "cuda"
    batch, heads, qlen, klen = 1, 2, 128, 128
    q = torch.randn(batch, heads, qlen, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, klen, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, klen, head_dim, device=device, dtype=dtype)

    out = sageattn3_blackwell(q, k, v, is_causal=is_causal, per_block_mean=True)

    q_p, k_p, v_p, delta_s = preprocess_qkv(q, k, v, per_block_mean=True)
    q_packed, q_sf = scale_and_quant_fp4_sm100(q_p)
    k_packed, k_sf = scale_and_quant_fp4_sm100(k_p)
    v_packed, v_sf = scale_and_quant_fp4_sm100(v_p)

    q_deq = _dequant_fp4_sm100(q_packed, q_sf)
    k_deq = _dequant_fp4_sm100(k_packed, k_sf)
    v_deq = _dequant_fp4_sm100(v_packed, v_sf)

    softmax_scale = head_dim ** -0.5
    ref = _reference_attn(q_deq, k_deq, v_deq, delta_s, softmax_scale, is_causal, True)

    torch.testing.assert_close(out.float(), ref.float(), rtol=5e-2, atol=5e-2)


def test_sm100_fp4_attention_per_block_mean_false():
    if not _is_sm100():
        pytest.skip("SM100 GPU required")

    torch.manual_seed(1)
    device = "cuda"
    batch, heads, qlen, klen, head_dim = 1, 2, 128, 128, 64
    q = torch.randn(batch, heads, qlen, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, klen, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, klen, head_dim, device=device, dtype=torch.float16)

    out = sageattn3_blackwell(q, k, v, is_causal=False, per_block_mean=False)

    q_p, k_p, v_p, delta_s = preprocess_qkv(q, k, v, per_block_mean=False)
    q_packed, q_sf = scale_and_quant_fp4_sm100(q_p)
    k_packed, k_sf = scale_and_quant_fp4_sm100(k_p)
    v_packed, v_sf = scale_and_quant_fp4_sm100(v_p)

    q_deq = _dequant_fp4_sm100(q_packed, q_sf)
    k_deq = _dequant_fp4_sm100(k_packed, k_sf)
    v_deq = _dequant_fp4_sm100(v_packed, v_sf)

    softmax_scale = head_dim ** -0.5
    ref = _reference_attn(q_deq, k_deq, v_deq, delta_s, softmax_scale, False, False)

    torch.testing.assert_close(out.float(), ref.float(), rtol=5e-2, atol=5e-2)
