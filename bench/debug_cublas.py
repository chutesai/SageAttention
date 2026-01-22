"""Debug cuBLAS issues on B200."""
import torch
import torch.nn.functional as F

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
print()

# Clear any CUDA errors
torch.cuda.synchronize()

# Test 1: Simple matmul
print("Test 1: Simple 2D matmul")
try:
    a = torch.randn(128, 128, device='cuda', dtype=torch.float32)
    b = torch.randn(128, 128, device='cuda', dtype=torch.float32)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: Batched matmul with batch=1
print("Test 2: Batched matmul (batch=1)")
try:
    a = torch.randn(1, 128, 128, device='cuda', dtype=torch.float32)
    b = torch.randn(1, 128, 128, device='cuda', dtype=torch.float32)
    c = torch.bmm(a, b)
    torch.cuda.synchronize()
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 3: 4D matmul (like attention)
print("Test 3: 4D matmul (B=1, H=8, S=256, D=128)")
try:
    q = torch.randn(1, 8, 256, 128, device='cuda', dtype=torch.float32)
    k = torch.randn(1, 8, 256, 128, device='cuda', dtype=torch.float32)
    # This triggers cublasSgemmStridedBatched
    attn = torch.matmul(q, k.transpose(-2, -1))
    torch.cuda.synchronize()
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: einsum version
print("Test 4: einsum (B=1, H=8, S=256, D=128)")
try:
    q = torch.randn(1, 8, 256, 128, device='cuda', dtype=torch.float32)
    k = torch.randn(1, 8, 256, 128, device='cuda', dtype=torch.float32)
    attn = torch.einsum('bhqd,bhkd->bhqk', q, k)
    torch.cuda.synchronize()
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 5: Loop-based reference (no batched GEMM)
print("Test 5: Loop-based matmul (no batched GEMM)")
try:
    q = torch.randn(1, 8, 256, 128, device='cuda', dtype=torch.float32)
    k = torch.randn(1, 8, 256, 128, device='cuda', dtype=torch.float32)
    results = []
    for b in range(q.size(0)):
        for h in range(q.size(1)):
            qbh = q[b, h]  # (256, 128)
            kbh = k[b, h]  # (256, 128)
            attn_bh = torch.mm(qbh, kbh.t())  # (256, 256)
            results.append(attn_bh)
    torch.cuda.synchronize()
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 6: SDPA
print("Test 6: scaled_dot_product_attention")
try:
    q = torch.randn(1, 8, 256, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, 8, 256, 128, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(1, 8, 256, 128, device='cuda', dtype=torch.bfloat16)
    out = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 7: Check if error is persistent
print("\nTest 7: Check CUDA error state")
try:
    torch.cuda.synchronize()
    print(f"  Last error: {torch.cuda.get_last_error()}")
except Exception as e:
    print(f"  Error checking state: {e}")
