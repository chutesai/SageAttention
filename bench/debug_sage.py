"""Debug SageAttention step by step."""
import torch
import torch.nn.functional as F

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Import SageAttention components
from sageattn3.api import (
    preprocess_qkv, scale_and_quant_fp4, scale_and_quant_fp4_permute,
    scale_and_quant_fp4_transpose, _matmul_no_cublas, triton_group_mean
)

B, H, S, D = 1, 32, 512, 128
dtype = torch.bfloat16

print(f"Test config: B={B}, H={H}, S={S}, D={D}, dtype={dtype}")
print()

# Create inputs
q = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
k = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
v = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
print("Created inputs")

# Step 1: k mean subtraction
print("\nStep 1: k.mean() subtraction")
try:
    k_mean = k.mean(dim=-2, keepdim=True)
    k_centered = k - k_mean
    torch.cuda.synchronize()
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Step 2: Padding
print("\nStep 2: Padding")
try:
    def pad_128(x):
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()

    q_pad = pad_128(q)
    k_pad = pad_128(k_centered)
    v_pad = pad_128(v)
    torch.cuda.synchronize()
    print(f"  PASS - shapes: q={q_pad.shape}, k={k_pad.shape}, v={v_pad.shape}")
except Exception as e:
    print(f"  FAIL: {e}")

# Step 3: Group mean
print("\nStep 3: triton_group_mean")
try:
    q_out, qm = triton_group_mean(q_pad)
    torch.cuda.synchronize()
    print(f"  PASS - q_out={q_out.shape}, qm={qm.shape}")
except Exception as e:
    print(f"  FAIL: {e}")

# Step 4: Triton matmul for delta_s
print("\nStep 4: _matmul_no_cublas (delta_s = qm @ k^T)")
try:
    delta_s = _matmul_no_cublas(qm, k_pad)
    torch.cuda.synchronize()
    print(f"  PASS - delta_s={delta_s.shape}, dtype={delta_s.dtype}")
except Exception as e:
    print(f"  FAIL: {e}")

# Step 5: FP4 quantization
print("\nStep 5: scale_and_quant_fp4 (Q)")
try:
    q_fp4, q_scale = scale_and_quant_fp4(q_out)
    torch.cuda.synchronize()
    print(f"  PASS - q_fp4={q_fp4.shape}, q_scale={q_scale.shape}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\nStep 6: scale_and_quant_fp4_permute (K)")
try:
    k_fp4, k_scale = scale_and_quant_fp4_permute(k_pad)
    torch.cuda.synchronize()
    print(f"  PASS - k_fp4={k_fp4.shape}, k_scale={k_scale.shape}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\nStep 7: scale_and_quant_fp4_transpose (V)")
try:
    v_fp4, v_scale = scale_and_quant_fp4_transpose(v_pad)
    torch.cuda.synchronize()
    print(f"  PASS - v_fp4={v_fp4.shape}, v_scale={v_scale.shape}")
except Exception as e:
    print(f"  FAIL: {e}")

# Step 8: Full preprocess
print("\nStep 8: Full preprocess_qkv")
try:
    q2 = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    k2 = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    v2 = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    q_p, k_p, v_p, delta_s_p = preprocess_qkv(q2, k2, v2, per_block_mean=True)
    torch.cuda.synchronize()
    print(f"  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Step 9: Full sageattn3_blackwell
print("\nStep 9: Full sageattn3_blackwell")
try:
    from sageattn3 import sageattn3_blackwell
    q3 = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    k3 = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    v3 = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    out = sageattn3_blackwell(q3, k3, v3, is_causal=False, per_block_mean=True)
    torch.cuda.synchronize()
    print(f"  PASS - output shape: {out.shape}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\nDone!")
