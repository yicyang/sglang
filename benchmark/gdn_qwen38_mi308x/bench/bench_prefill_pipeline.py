"""Prefill GDN pipeline micro-benchmark for Qwen3.8-Flash-Next TP2 (gfx942).

Inputs mirror what sglang's GDNAttnBackend.forward_extend hands to the kernel:
  q, k: [1, T, Hg, K] bf16 ; v: [1, T, H, V] bf16 ; g, beta: [1, T, H] fp32
  state pool [slots, H, V, K] fp32 (VK) + int32/int64 cache_indices, in-place.
Every candidate is timed from those exact inputs to (o, updated pool), so any
layout conversion / gather / scatter a backend needs is included in its time.

The aiter variant under test is whatever `import aiter` resolves to; run this
script once per variant with a different PYTHONPATH and --aiter-label.
"""

import argparse
import inspect
import json
import os
import sys
import traceback

import torch

H, HG, K, V = 24, 8, 128, 128  # TP2 per-rank: 48/2 value heads, 16/2 key heads
DEV = "cuda"


def make_inputs(seqlens, n_slots=64, seed=0, realistic_gate=False):
    g_ = torch.Generator(device=DEV).manual_seed(seed)
    T = sum(seqlens)
    q = torch.randn(1, T, HG, K, device=DEV, dtype=torch.bfloat16, generator=g_)
    k = torch.randn(1, T, HG, K, device=DEV, dtype=torch.bfloat16, generator=g_)
    v = torch.randn(1, T, H, V, device=DEV, dtype=torch.bfloat16, generator=g_)
    # realistic gating: g = -softplus(a+dt)*exp(A_log) <= 0 ; beta = sigmoid(b)
    a = torch.randn(1, T, H, device=DEV, generator=g_)
    if realistic_gate:
        # Real Qwen3.8 layers reach |g| ~ 1e2 per token (exp(A_log) up to ~150); this
        # range broke the pre-fix FlyDSL prepare kernel while mild gates did not.
        A_log = torch.empty(H, device=DEV).uniform_(-1.0, 5.0, generator=g_)
        g = -torch.exp(A_log) * torch.nn.functional.softplus(a * 2.0)
    else:
        g = -torch.nn.functional.softplus(a) * 0.5
    beta = torch.sigmoid(torch.randn(1, T, H, device=DEV, generator=g_))
    pool = torch.randn(n_slots, H, V, K, device=DEV, dtype=torch.float32, generator=g_) * 0.05
    N = len(seqlens)
    perm = torch.randperm(n_slots - 1, device=DEV, generator=g_)[:N] + 1
    idx = perm.to(torch.int32)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0).tolist()), device=DEV, dtype=torch.int64)
    return dict(q=q, k=k, v=v, g=g, beta=beta, pool=pool, idx=idx, cu=cu, seqlens=seqlens)


# ----------------------------------------------------------------------------- candidates


def cand_sglang():
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

    def run(d, pool):
        out = chunk_gated_delta_rule(
            q=d["q"], k=d["k"], v=d["v"], g=d["g"], beta=d["beta"],
            initial_state=pool, initial_state_indices=d["idx"],
            cu_seqlens=d["cu"], head_first=False, use_qk_l2norm_in_kernel=True,
        )
        return out[0]
    return run


def _aiter_opt_vk(**flags):
    from aiter.ops.triton.gated_delta_net.gated_delta_rule import chunk_gated_delta_rule_opt_vk
    params = inspect.signature(chunk_gated_delta_rule_opt_vk).parameters
    for f, val in flags.items():
        if val and f not in params:
            raise RuntimeError(f"{f} unsupported by this aiter")
    build_meta = None
    if "prefill_metadata" in params:
        try:
            from aiter.ops.prefill_batch_metadata import build_gated_delta_rule_prefill_metadata as build_meta
        except ImportError:
            build_meta = None

    mode = {"gather": "initial_state_indices" not in params}

    def run(d, pool):
        kw = dict(
            q=d["q"], k=d["k"], v=d["v"], g=d["g"], beta=d["beta"],
            output_final_state=True, use_qk_l2norm_in_kernel=True,
            cu_seqlens=d["cu"], **flags,
        )
        if mode["gather"]:
            # backend cannot index the pool: gather / scatter around the call
            idx = d["idx"].long()
            kw["initial_state"] = pool.index_select(0, idx)
        else:
            kw.update(initial_state=pool, initial_state_indices=d["idx"])
            if "inplace_final_state" in params:
                kw["inplace_final_state"] = True
        if "state_dtype" in params:
            kw["state_dtype"] = torch.float32
        if build_meta is not None:
            # metadata is built once per batch in a real integration (shared by 36 layers)
            if "_meta" not in d:
                d["_meta"] = build_meta(d["seqlens"], cu_seqlens=d["cu"])
            kw["prefill_metadata"] = d["_meta"]
        kw = {a: b for a, b in kw.items() if a in params}
        try:
            o, ht = chunk_gated_delta_rule_opt_vk(**kw)
        except ValueError as ex:
            if mode["gather"] or "Indexed state pools" not in str(ex):
                raise
            mode["gather"] = True
            print("    (indexed pool unsupported -> gather/scatter fallback, cost included)", flush=True)
            return run(d, pool)
        if mode["gather"]:
            pool.index_copy_(0, d["idx"].long(), ht.to(pool.dtype))
        return o
    return run


def cand_atom_vk():
    sys.path.insert(0, os.environ.get("ATOM_ROOT", "/root/workspace/gdn_optimization/ATOM"))
    from atom.model_ops.fla_ops.chunk_vk import chunk_gated_delta_rule_vk

    def run(d, pool):
        idx = d["idx"].long()
        h0 = pool.index_select(0, idx)  # ATOM has no index arg: gather / scatter around call
        o, ht = chunk_gated_delta_rule_vk(
            q=d["q"], k=d["k"], v=d["v"], g=d["g"], beta=d["beta"],
            initial_state=h0, output_final_state=True, cu_seqlens=d["cu"],
            use_qk_l2norm_in_kernel=True,
        )
        pool.index_copy_(0, idx, ht.to(pool.dtype))
        return o
    return run


def cand_atom_kv():
    """ATOM native (fused cumsum-kkt / merge-recompute) kernels use a KV state; a
    sglang integration must transpose the VK pool slots in and out."""
    sys.path.insert(0, os.environ.get("ATOM_ROOT", "/root/workspace/gdn_optimization/ATOM"))
    from atom.model_ops.fla_ops.chunk import chunk_gated_delta_rule

    def run(d, pool):
        idx = d["idx"].long()
        h0 = pool.index_select(0, idx).transpose(-1, -2).contiguous()
        o, ht = chunk_gated_delta_rule(
            q=d["q"], k=d["k"], v=d["v"], g=d["g"], beta=d["beta"],
            initial_state=h0, output_final_state=True, cu_seqlens=d["cu"],
            use_qk_l2norm_in_kernel=True,
        )
        pool.index_copy_(0, idx, ht.transpose(-1, -2).to(pool.dtype))
        return o
    return run


def get_candidates(label):
    c = {}
    if label == "sglang":
        c["sglang_fla_triton"] = cand_sglang
        c["atom_chunk_vk"] = cand_atom_vk
        c["atom_chunk_kv(+transpose)"] = cand_atom_kv
    else:
        c[f"{label}:opt_vk_triton"] = lambda: _aiter_opt_vk()
        c[f"{label}:opt_vk_hipK5"] = lambda: _aiter_opt_vk(use_chunk_hip=True)
        c[f"{label}:opt_vk_flydslK5"] = lambda: _aiter_opt_vk(use_chunk_flydsl=True)
        c[f"{label}:opt_vk_flydslPrep+hipK5"] = lambda: _aiter_opt_vk(use_chunk_hip=True, use_prepare_flydsl=True)
        c[f"{label}:opt_vk_flydslPrep+flydslK5"] = lambda: _aiter_opt_vk(use_chunk_flydsl=True, use_prepare_flydsl=True)
    return c


# ----------------------------------------------------------------------------- reference


def torch_reference(d, pool):
    """fp32 naive recurrence (sequential over tokens) used for correctness."""
    q, k, v, g, beta = d["q"][0].float(), d["k"][0].float(), d["v"][0].float(), d["g"][0], d["beta"][0]
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    rep = H // HG
    q = q.repeat_interleave(rep, dim=1)
    k = k.repeat_interleave(rep, dim=1)
    o = torch.empty(q.shape[0], H, V, device=DEV)
    scale = K ** -0.5
    cu = d["cu"].tolist()
    for n in range(len(cu) - 1):
        slot = int(d["idx"][n])
        S = pool[slot].clone()  # [H, V, K]
        for t in range(cu[n], cu[n + 1]):
            S = S * torch.exp(g[t])[:, None, None]
            kv = torch.einsum("hvk,hk->hv", S, k[t])
            delta = (v[t] - kv) * beta[t][:, None]
            S = S + delta[:, :, None] * k[t][:, None, :]
            o[t] = torch.einsum("hvk,hk->hv", S, q[t]) * scale
        pool[slot] = S
    return o[None]


REAL_INPUTS = os.path.join(os.path.dirname(__file__), "..", "results", "kernel", "real_inputs")


def real_cases():
    """Inputs captured from the running server (one sequence each, zero initial state)."""
    import glob
    for p in sorted(glob.glob(os.path.join(REAL_INPUTS, "*.pt"))):
        x = torch.load(p, map_location=DEV)
        x = x.get("inputs", x)
        T = x["q"].shape[1]
        yield os.path.basename(p), dict(
            q=x["q"], k=x["k"], v=x["v"], g=x["g"], beta=x["beta"],
            pool=torch.zeros(4, H, V, K, device=DEV), idx=torch.tensor([2], device=DEV, dtype=torch.int32),
            cu=torch.tensor([0, T], device=DEV, dtype=torch.int64), seqlens=[T])


def correlated_key_case(corr, T=512, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    base = torch.randn(1, 1, HG, K, device=DEV, generator=gen)
    k = (corr * base + (1 - corr) * torch.randn(1, T, HG, K, device=DEV, generator=gen)).bfloat16()
    return dict(q=torch.randn(1, T, HG, K, device=DEV, generator=gen).bfloat16(), k=k,
                v=torch.randn(1, T, H, V, device=DEV, generator=gen).bfloat16(),
                g=-torch.rand(1, T, H, device=DEV, generator=gen) * 0.1,
                beta=torch.full((1, T, H), 0.99, device=DEV),
                pool=torch.zeros(4, H, V, K, device=DEV), idx=torch.tensor([2], device=DEV, dtype=torch.int32),
                cu=torch.tensor([0, T], device=DEV, dtype=torch.int64), seqlens=[T])


def check_against(run, d, ref_run):
    pool_ref, pool = d["pool"].clone(), d["pool"].clone()
    o_ref = ref_run(dict(d), pool_ref).float()
    o = run(dict(d), pool).float()
    idx = d["idx"].long()
    eo = ((o - o_ref).abs().max() / o_ref.abs().max()).item()
    es = ((pool[idx] - pool_ref[idx]).abs().max() / pool_ref[idx].abs().max().clamp_min(1e-9)).item()
    return eo, es


def check(run, seqlens, realistic_gate=False):
    d = make_inputs(seqlens, seed=1, realistic_gate=realistic_gate)
    ref_pool = d["pool"].clone()
    o_ref = torch_reference(d, ref_pool)
    pool = d["pool"].clone()
    o = run(d, pool)
    torch.cuda.synchronize()
    idx = d["idx"].long()
    err_o = ((o.float() - o_ref).abs().max() / o_ref.abs().max()).item()
    err_s = ((pool[idx] - ref_pool[idx]).abs().max() / ref_pool[idx].abs().max()).item()
    untouched = torch.ones(pool.shape[0], dtype=torch.bool, device=DEV)
    untouched[idx] = False
    clobber = (pool[untouched] != d["pool"][untouched]).any().item()
    return err_o, err_s, clobber


# ----------------------------------------------------------------------------- timing


def bench(run, seqlens, iters, warmup=5):
    d = make_inputs(seqlens)
    pools = [d["pool"].clone() for _ in range(2)]
    for i in range(warmup):
        run(d, pools[i % 2])
    torch.cuda.synchronize()
    times = []
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=DEV)
    for i in range(iters):
        pools[0].copy_(d["pool"])
        flush.zero_()  # evict L2 / MALL between iterations
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        run(d, pools[0])
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    # GPU busy time: sum of device kernel durations over back-to-back calls (no
    # host gaps, no reset kernels inside the window). Primary kernel metric.
    from torch.profiler import ProfilerActivity, profile
    n = 10
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            run(d, pools[1])
        torch.cuda.synchronize()
    per_kernel = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            per_kernel[ev.name] = per_kernel.get(ev.name, 0.0) + ev.device_time / n
    gpu_us = sum(per_kernel.values())
    return dict(gpu_us=gpu_us, median_us=times[len(times) // 2], min_us=times[0],
                p90_us=times[int(len(times) * 0.9)], kernels=per_kernel)


SHAPES = {
    "1x1024": [1024],
    "1x2048": [2048],
    "1x4096": [4096],
    "1x8192": [8192],
    "1x12000": [12000],
    "1x16384": [16384],
    "4x4096": [4096] * 4,
    "16x1024": [1024] * 16,
    "mix5": [3000, 517, 8000, 129, 4738],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aiter-label", default="sglang", help="'sglang' runs sglang+ATOM; otherwise label for aiter variant")
    ap.add_argument("--only", default=None, help="comma-separated candidate substrings")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tag", default="", help="suffix appended to candidate names")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import aiter  # noqa: F401  (record which aiter is loaded)
    meta = dict(aiter_file=aiter.__file__, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(), env={k: v for k, v in os.environ.items() if k.startswith(("SGLANG_GDN", "FLYDSL", "GATED_DELTA", "AITER"))})
    results = []
    for name, factory in get_candidates(args.aiter_label).items():
        if args.only and not any(s in name for s in args.only.split(",")):
            continue
        row = dict(impl=name + args.tag)
        try:
            run = factory()
            eo, es, clob = check(run, [300, 1, 700, 64, 129])
            row.update(err_o=eo, err_state=es, clobbers_other_slots=clob)
            ok = eo < 2e-2 and es < 2e-2 and not clob
            # fp32 reference with realistic (large) gates
            eo2, es2, clob2 = check(run, [300, 1, 700, 64, 129], realistic_gate=True)
            row.update(err_o_realgate=eo2, err_state_realgate=es2)
            ok = ok and eo2 < 2e-2 and es2 < 2e-2 and not clob2
            # correlated keys (as in real layers) make the WY triangular solve ill-conditioned
            for corr in (0.5, 0.9):
                eo4, es4 = check_against(run, correlated_key_case(corr), cand_sglang())
                row[f"err_o_corrkey{corr}"] = eo4
                ok = ok and eo4 < 5e-2 and es4 < 5e-2
                print(f"    correlated keys {corr}: err_o={eo4:.2e} err_state={es4:.2e}", flush=True)
            # captured server inputs, against the sglang FLA triton kernel
            ref_run = cand_sglang()
            for cname, d in real_cases():
                eo3, es3 = check_against(run, d, ref_run)
                row[f"err_o_{cname}"] = eo3
                ok = ok and eo3 < 2e-2 and es3 < 2e-2
                print(f"    real input {cname}: err_o={eo3:.2e} err_state={es3:.2e}", flush=True)
            print(f"    realistic gates: err_o={eo2:.2e} err_state={es2:.2e}", flush=True)
            row["correct"] = ok
            print(f"[{name}] correctness err_o={eo:.2e} err_state={es:.2e} clobber={clob} -> {'OK' if ok else 'FAIL'}", flush=True)
            for sname in args.shapes.split(","):
                r = bench(run, SHAPES[sname], args.iters)
                row[sname] = r
                print(f"  {sname:>8}: gpu {r['gpu_us']:9.1f} us | wall median {r['median_us']:9.1f} us (min {r['min_us']:.1f}, p90 {r['p90_us']:.1f})", flush=True)
        except Exception as ex:  # record failures, they are results too
            row["error"] = f"{type(ex).__name__}: {ex}"[:500]
            print(f"[{name}] ERROR {row['error']}", flush=True)
            traceback.print_exc()
        results.append(row)
    with open(args.out, "w") as f:
        json.dump(dict(meta=meta, results=results), f, indent=1)


if __name__ == "__main__":
    main()
