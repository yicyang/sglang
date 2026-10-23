"""Decode GDN micro-benchmark for Qwen3.8-Flash-Next TP2 (gfx942).

Inputs mirror sglang's packed decode: mixed_qkv [B, 2*Hg*K + H*V] bf16 (after
conv1d), a/b [B, H] bf16, A_log [H] fp32, dt_bias [H] bf16, state pool
[slots, H, V, K] fp32, int32 cache_indices. Each candidate goes from these
inputs to (o [B, H, V], updated pool); any split/copy/transposition it needs
is part of its time. Timing replays a captured HIP graph (decode runs under
CUDA graphs in sglang), plus the profiler's device-kernel sum.
"""

import argparse
import json
import os
import sys
import traceback

import torch

H, HG, K, V = 24, 8, 128, 128
QKV_DIM = 2 * HG * K + H * V
DEV = "cuda"
SCALE = K ** -0.5


def make_inputs(B, n_slots=80, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return dict(
        B=B,
        mixed_qkv=torch.randn(B, QKV_DIM, device=DEV, dtype=torch.bfloat16, generator=g),
        a=torch.randn(B, H, device=DEV, dtype=torch.bfloat16, generator=g),
        b=torch.randn(B, H, device=DEV, dtype=torch.bfloat16, generator=g),
        A_log=torch.randn(H, device=DEV, dtype=torch.float32, generator=g) * 0.5,
        dt_bias=torch.randn(H, device=DEV, dtype=torch.bfloat16, generator=g) * 0.5,
        pool=torch.randn(n_slots, H, V, K, device=DEV, dtype=torch.float32, generator=g) * 0.05,
        cu=torch.arange(B + 1, device=DEV, dtype=torch.int32),
        idx=(torch.randperm(n_slots - 1, device=DEV, generator=g)[:B] + 1).to(torch.int32),
    )


def split_qkv(d):
    B = d["B"]
    q, k, v = torch.split(d["mixed_qkv"], [HG * K, HG * K, H * V], dim=-1)
    return q.view(B, HG, K), k.view(B, HG, K), v.view(B, H, V)


def reference(d, pool):
    q, k, v = (x.float() for x in split_qkv(d))
    q = torch.nn.functional.normalize(q, dim=-1).repeat_interleave(H // HG, 1)
    k = torch.nn.functional.normalize(k, dim=-1).repeat_interleave(H // HG, 1)
    g = -torch.exp(d["A_log"]) * torch.nn.functional.softplus(d["a"].float() + d["dt_bias"].float())
    beta = torch.sigmoid(d["b"].float())
    idx = d["idx"].long()
    S = pool[idx] * torch.exp(g)[:, :, None, None]
    kv = torch.einsum("bhvk,bhk->bhv", S, k)
    S = S + ((v - kv) * beta[..., None])[..., None] * k[:, :, None, :]
    o = torch.einsum("bhvk,bhk->bhv", S, q) * SCALE
    pool[idx] = S
    return o


# ----------------------------------------------------------------------------- candidates


def cand_sglang_packed():
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    def run(d, pool, out):
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=d["mixed_qkv"], a=d["a"], b=d["b"], A_log=d["A_log"], dt_bias=d["dt_bias"],
            scale=SCALE, initial_state=pool, out=out, ssm_state_indices=d["idx"],
            use_qk_l2norm_in_kernel=True,
        )
        return out.view(d["B"], H, V)
    return run


def cand_sglang_split():
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    def run(d, pool, out):
        B = d["B"]
        q, k, v = split_qkv(d)
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=d["A_log"], dt_bias=d["dt_bias"], q=q.view(1, B, HG, K), k=k.view(1, B, HG, K),
            v=v.view(1, B, H, V), a=d["a"], b=d["b"], initial_state_source=pool,
            initial_state_indices=d["idx"], scale=SCALE, use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0, softplus_threshold=20.0,
            cu_seqlens=d["cu"],
        )
        return o.view(B, H, V)
    return run


def cand_aiter_flydsl():
    from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode

    def run(d, pool, out):
        B = d["B"]
        q, k, v = split_qkv(d)
        o = out.view(B, 1, H, V)
        # need_shuffle_state=False: the kernel reads the [N, H, V, K] pool as is
        flydsl_gdr_decode(
            query=q.view(B, 1, HG, K), key=k.view(B, 1, HG, K), value=v.view(B, 1, H, V),
            a=d["a"].view(B, 1, H), b=d["b"].view(B, 1, H), dt_bias=d["dt_bias"], A_log=d["A_log"],
            indices=d["idx"],
            state=pool, out=o, use_qk_l2norm=True, need_shuffle_state=False,
        )
        return out.view(B, H, V)
    return run


def cand_aiter_rearrange():
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule,
    )

    def run(d, pool, out):
        o, _ = fused_rearrange_sigmoid_gated_delta_rule(
            A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"], qkv=d["mixed_qkv"],
            key_dim=HG * K, value_dim=H * V, head_k_dim=K, head_v_dim=V, scale=SCALE,
            initial_state=pool, inplace_final_state=True, ssm_state_indices=d["idx"],
            use_qk_l2norm_in_kernel=True, core_attn_out=out, cu_seqlens=d["cu"],
        )
        return out.view(d["B"], H, V)
    return run


def cand_aiter_sigmoid():
    from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    def run(d, pool, out):
        B = d["B"]
        q, k, v = split_qkv(d)
        # this kernel uses a [N, H, K, V] state: transpose touched slots in / out
        idx = d["idx"].long()
        kv = pool.index_select(0, idx).transpose(-1, -2).contiguous()
        local = torch.arange(B, device=DEV, dtype=d["idx"].dtype)
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=d["A_log"], a=d["a"], dt_bias=d["dt_bias"], softplus_beta=1.0, softplus_threshold=20.0,
            q=q.view(1, B, HG, K), k=k.view(1, B, HG, K), v=v.view(1, B, H, V), b=d["b"],
            initial_state_source=kv, initial_state_indices=local, scale=SCALE,
            use_qk_l2norm_in_kernel=True, cu_seqlens=d["cu"],
        )
        pool.index_copy_(0, idx, kv.transpose(-1, -2))
        return o.view(B, H, V)
    return run


def cand_atom():
    sys.path.insert(0, os.environ.get("ATOM_ROOT", "/root/workspace/gdn_optimization/ATOM"))
    from atom.model_ops.fla_ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

    def run(d, pool, out):
        B = d["B"]
        q, k, v = split_qkv(d)
        o, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dt_bias"],
            q=q.view(1, B, HG, K), k=k.view(1, B, HG, K), v=v.view(1, B, H, V), o=out.view(1, B, H, V),
            scale=SCALE, initial_state=pool, inplace_final_state=True,
            cu_seqlens=d["cu"], ssm_state_indices=d["idx"], use_qk_l2norm_in_kernel=True,
        )
        return out.view(B, H, V)
    return run


CANDS = {
    "sglang_packed_decode(triton)": cand_sglang_packed,
    "sglang_fused_sigmoid_gating(triton)": cand_sglang_split,
    "aiter_flydsl_gdr_decode": cand_aiter_flydsl,
    "aiter_fused_rearrange_sigmoid_gdr(triton)": cand_aiter_rearrange,
    "aiter_fused_sigmoid_gating_kv(triton,+transpose)": cand_aiter_sigmoid,
    "atom_fused_sigmoid_gating_vk(triton)": cand_atom,
}


def check(run):
    d = make_inputs(13, seed=3)
    ref_pool = d["pool"].clone()
    o_ref = reference(d, ref_pool)
    pool = d["pool"].clone()
    out = torch.empty(d["B"], 1, H, V, device=DEV, dtype=torch.bfloat16)
    o = run(d, pool, out).float()
    torch.cuda.synchronize()
    idx = d["idx"].long()
    eo = ((o - o_ref).abs().max() / o_ref.abs().max()).item()
    es = ((pool[idx] - ref_pool[idx]).abs().max() / ref_pool[idx].abs().max()).item()
    mask = torch.ones(pool.shape[0], dtype=torch.bool, device=DEV)
    mask[idx] = False
    clob = (pool[mask] != d["pool"][mask]).any().item()
    return eo, es, clob


def bench(run, B, iters=20, layers=36):
    """One graph = one decode step over `layers` GDN layers, each with its own
    state pool (as in the model), so state reads are cache-cold like in serving."""
    d = make_inputs(B)
    pools = [d["pool"].clone() for _ in range(layers)]
    out = torch.empty(B, 1, H, V, device=DEV, dtype=torch.bfloat16)
    for p in pools[:3]:
        run(d, p, out)
    torch.cuda.synchronize()
    graph_us = None
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run(d, pools[0], out)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for p in pools:
                run(d, p, out)
        gr.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            e0.record()
            gr.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000 / layers)
        ts.sort()
        graph_us = ts[len(ts) // 2]
    except Exception as ex:
        print(f"    graph capture failed: {type(ex).__name__}: {str(ex)[:100]}", flush=True)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for p in pools:
            run(d, p, out)
        torch.cuda.synchronize()
    kern = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            kern[ev.name] = kern.get(ev.name, 0) + ev.device_time / layers
    return dict(graph_us=graph_us, gpu_us=sum(kern.values()), kernels=kern)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--bs", default="1,2,4,8,16,32,64")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import aiter
    res = []
    for name, fac in CANDS.items():
        if args.only and not any(s in name for s in args.only.split(",")):
            continue
        row = dict(impl=name + args.tag)
        try:
            run = fac()
            eo, es, clob = check(run)
            ok = eo < 2e-2 and es < 2e-2 and not clob
            row.update(err_o=eo, err_state=es, clobbers_other_slots=clob, correct=ok)
            print(f"[{name}] err_o={eo:.2e} err_state={es:.2e} clobber={clob} -> {'OK' if ok else 'FAIL'}", flush=True)
            for B in map(int, args.bs.split(",")):
                r = bench(run, B)
                row[f"bs{B}"] = r
                g = f"{r['graph_us']:.1f}" if r["graph_us"] is not None else "n/a"
                print(f"  bs={B:3d}: graph {g} us | gpu {r['gpu_us']:.1f} us | kernels={len(r['kernels'])}", flush=True)
        except Exception as ex:
            row["error"] = f"{type(ex).__name__}: {ex}"[:400]
            print(f"[{name}] ERROR {row['error']}", flush=True)
            traceback.print_exc()
        res.append(row)
    json.dump(dict(meta=dict(aiter=aiter.__file__), results=res), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
