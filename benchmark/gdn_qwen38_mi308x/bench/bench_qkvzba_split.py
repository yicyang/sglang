"""Framework-side cost of the GDN input split for Qwen3.8 TP2 (HV/HK = 3).

With SGLANG_USE_AITER=1, qwen3_5.py only routes ratios (1, 2, 4, 8) to the fused
Triton split; ratio 3 takes the unfused path (two .contiguous() + torch.cat).
This compares both paths on real-shaped projections and checks they agree.
"""
import sys

import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
)

HK, HV, K, V = 8, 24, 128, 128


def unfused(qkvz, ba):
    # Mirrors Qwen3_5GatedDeltaNet.fix_query_key_value_ordering + the fallback in forward().
    T = qkvz.shape[0]
    q, k, v, z = torch.split(qkvz, [HK * K, HK * K, HV * V, HV * V], dim=-1)
    b, a = torch.split(ba, [HV, HV], dim=-1)
    z = z.reshape(T, HV, V)
    b = b.contiguous()
    a = a.contiguous()
    mixed_qkv = torch.cat((q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)), dim=-1)
    return mixed_qkv, z, b, a


def fused(qkvz, ba):
    return fused_qkvzba_split_reshape_cat_contiguous(qkvz, ba, HK, HV, K, V)


def t_us(fn, *a, n=50):
    for _ in range(5):
        fn(*a)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn(*a)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / n


for T in [1, 8, 32, 2048, 12000, 16384]:
    qkvz = torch.randn(T, 2 * HK * K + 2 * HV * V, device="cuda", dtype=torch.bfloat16)
    ba = torch.randn(T, 2 * HV, device="cuda", dtype=torch.bfloat16)
    ref = unfused(qkvz, ba)
    got = fused(qkvz, ba)
    same = all(torch.equal(x.reshape(-1), y.reshape(-1)) for x, y in zip(ref, got))
    print(f"T={T:6d}: unfused {t_us(unfused, qkvz, ba):8.1f} us | fused {t_us(fused, qkvz, ba):8.1f} us | bit-identical={same}")
    sys.stdout.flush()
