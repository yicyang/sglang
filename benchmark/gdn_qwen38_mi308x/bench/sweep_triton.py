"""Triton config sweep for GDN prefill kernels on gfx942 (Qwen3.8 TP2 shapes).

Each candidate config is applied to ONE kernel, the full pipeline is run on real
shaped inputs, and only that kernel's device time is read back from the torch
profiler. Search space follows the ROCm MI300X Triton tuning guide: tile sizes,
num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim, kpack.

  --target sglang_k5      sglang FLA chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (Autotuner)
  --target sglang_o       sglang FLA chunk_fwd_kernel_o (plain jit, hardcoded BK/BV)
  --target sglang_wu      sglang FLA recompute_w_u_fwd_kernel (plain jit)
  --target aiter_o_vk     aiter chunk_fwd_kernel_o_opt_vk (Autotuner)
"""

import argparse
import itertools
import json
import os
import sys

os.environ.setdefault("FLA_CACHE_RESULTS", "0")
import torch  # noqa: E402
import triton  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from bench_prefill_pipeline import SHAPES, _aiter_opt_vk, cand_sglang, make_inputs  # noqa: E402


class Proxy:
    """Stands in for a plain @triton.jit kernel and overrides launch meta-params."""

    def __init__(self, kern, overrides):
        self.kern, self.overrides = kern, overrides

    def __getitem__(self, grid):
        launcher = self.kern[grid]

        def call(*a, **kw):
            kw.update(self.overrides)
            return launcher(*a, **kw)
        return call


def target_spec(name):
    if name == "sglang_k5":
        import sglang.kernels.ops.attention.fla.chunk_delta_h as m
        return dict(kind="autotune", obj=m.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
                    kname="chunk_gated_delta_rule_fwd_kernel_h_blockdim64", run=cand_sglang(),
                    tiles=[{"BV": bv} for bv in (16, 32, 64)],
                    default=dict(BV=32, num_warps=4, num_stages=2))
    if name == "sglang_o":
        import sglang.kernels.ops.attention.fla.chunk_o as m
        return dict(kind="proxy", mod=m, attr="chunk_fwd_kernel_o", kname="chunk_fwd_kernel_o",
                    run=cand_sglang(),
                    tiles=[{"BK": bk, "BV": bv} for bk in (32, 64, 128) for bv in (32, 64, 128)],
                    default=dict(BK=128, BV=64, num_warps=4, num_stages=2))
    if name == "sglang_wu":
        import sglang.kernels.ops.attention.fla.wy_fast as m
        return dict(kind="proxy", mod=m, attr="recompute_w_u_fwd_kernel", kname="recompute_w_u_fwd_kernel",
                    run=cand_sglang(),
                    tiles=[{"BK": bk, "BV": bv} for bk in (32, 64, 128) for bv in (32, 64, 128)],
                    default=dict(BK=64, BV=64, num_warps=4, num_stages=3))
    if name == "aiter_o_vk":
        import aiter.ops.triton._triton_kernels.gated_delta_net.prefill.chunk_o as m
        kern = m.chunk_fwd_kernel_o_opt_vk
        at = kern.fn if not hasattr(kern, "configs") else kern  # Heuristics -> Autotuner
        return dict(kind="autotune", obj=at, kname="chunk_fwd_kernel_o_opt_vk",
                    run=_aiter_opt_vk(use_chunk_flydsl=True, use_prepare_flydsl=True),
                    tiles=[{"BK": bk, "BV": bv} for bk in (32, 64, 128) for bv in (32, 64, 128)],
                    default=dict(at.configs[0].kwargs, num_warps=at.configs[0].num_warps,
                                 num_stages=at.configs[0].num_stages))
    raise ValueError(name)


def apply(spec, cfg):
    meta = {k: v for k, v in cfg.items() if k not in ("num_warps", "num_stages")}
    if spec["kind"] == "autotune":
        at = spec["obj"]
        at.configs = [triton.Config(meta, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])]
        at.cache.clear()
    else:
        if "orig" not in spec:
            spec["orig"] = getattr(spec["mod"], spec["attr"])
        setattr(spec["mod"], spec["attr"], Proxy(spec["orig"], dict(cfg)))


def measure(spec, shapes, n=5):
    from torch.profiler import ProfilerActivity, profile
    out = {}
    for s in shapes:
        d = make_inputs(SHAPES[s])
        pool = d["pool"].clone()
        for _ in range(2):
            spec["run"](d, pool)  # compile + warmup
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                spec["run"](d, pool)
            torch.cuda.synchronize()
        t = sum(ev.device_time for ev in prof.events()
                if ev.device_type == torch.autograd.DeviceType.CUDA and spec["kname"] in ev.name) / n
        out[s] = t
    return out


def geomean(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p ** (1 / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--shapes", default="1x2048,1x12000,4x4096")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shapes = args.shapes.split(",")
    spec = target_spec(args.target)
    results = []

    def trial(cfg):
        apply(spec, cfg)
        try:
            t = measure(spec, shapes)
        except Exception as ex:  # invalid configs (LDS overflow etc.)
            print(f"  skip {cfg}: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
            return None
        r = dict(cfg=cfg, us=t, gmean=geomean(list(t.values())))
        results.append(r)
        print(f"  {cfg} -> " + ", ".join(f"{k}={v:.1f}" for k, v in t.items()) + f" | gmean {r['gmean']:.1f}", flush=True)
        return r

    base = trial(spec["default"]) if spec["default"] else None
    # stage 1: tiles x warps x stages
    for tile, w, st in itertools.product(spec["tiles"], (2, 4, 8), (1, 2, 3)):
        trial(dict(tile, num_warps=w, num_stages=st))
    ok = sorted([r for r in results if r], key=lambda r: r["gmean"])
    # stage 2: AMD-specific knobs on the top-5
    for r in ok[:5]:
        for wpe, nk, kp in itertools.product((0, 1, 2, 4), (16, 32), (1, 2)):
            if (wpe, nk, kp) == (0, 16, 1):
                continue
            trial(dict(r["cfg"], waves_per_eu=wpe, matrix_instr_nonkdim=nk, kpack=kp))
    ok = sorted([r for r in results if r], key=lambda r: r["gmean"])
    best = ok[0]
    print("BEST", best)
    if base:
        print("DEFAULT", base, f"speedup {base['gmean'] / best['gmean']:.3f}x")
    json.dump(dict(target=args.target, shapes=shapes, default=base, best=best, all=ok), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
