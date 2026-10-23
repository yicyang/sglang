"""Per-batch-size launch config sweep for GDN decode kernels (Qwen3.8 TP2, gfx942).

  --target sglang_packed  sglang fused_recurrent_gated_delta_rule_packed_decode
                          (BV x num_warps x num_stages [x waves_per_eu])
  --target aiter_flydsl   aiter flydsl_gdr_decode
                          (NUM_BLOCKS_PER_V_DIM x NUM_WARPS x WARP_THREADS_K);
                          writes rows in gdr_decode_tuned.csv format
Timing: HIP-graph replay, same harness as bench_decode.py.
"""

import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import bench_decode as bd  # noqa: E402


def sweep_sglang(bs_list):
    import sglang.kernels.ops.attention.fla.fused_recurrent as fr

    run = bd.cand_sglang_packed()
    out = {}
    space = [dict(BV=bv, num_warps=w, num_stages=s)
             for bv, w, s in itertools.product((16, 32, 64, 128), (1, 2, 4, 8), (1, 2, 3))]
    for B in bs_list:
        res = []
        for cfg in space:
            fr.PACKED_DECODE_GFX942_CONFIGS = [(1 << 30, cfg)]
            try:
                r = bd.bench(run, B)
            except Exception as ex:
                print(f"  B={B} skip {cfg}: {type(ex).__name__}", flush=True)
                continue
            res.append((r["graph_us"], cfg))
        res.sort(key=lambda x: x[0])
        best_t, best = res[0]
        for wpe in (1, 2, 4):  # AMD occupancy hint on the winner
            cfg = dict(best, waves_per_eu=wpe)
            fr.PACKED_DECODE_GFX942_CONFIGS = [(1 << 30, cfg)]
            r = bd.bench(run, B)
            res.append((r["graph_us"], cfg))
        res.sort(key=lambda x: x[0])
        fr.PACKED_DECODE_GFX942_CONFIGS = []
        default = bd.bench(run, B)["graph_us"]
        out[B] = dict(default_us=default, best_us=res[0][0], best=res[0][1], top5=res[:5])
        print(f"B={B}: default {default:.1f} us -> best {res[0][0]:.1f} us {res[0][1]}", flush=True)
    return out


def sweep_flydsl(bs_list):
    import aiter.ops.flydsl.linear_attention_kernels as lak

    run = bd.cand_aiter_flydsl()
    out = {}
    lak.get_default_kwargs("torch.bfloat16", "torch.float32", 1, 1, bd.HG, bd.H, bd.K, bd.V)  # load csv
    for B in bs_list:
        key = ("torch.bfloat16", "torch.float32", lak.GDR_GPU_ARCH, B, 1, bd.HG, bd.H, bd.K, bd.V)
        lak.GDR_GLOBAL_CONFIG_MAP.pop(key, None)
        default = bd.bench(run, B)["graph_us"]
        res = []
        for nb, nw, wk in itertools.product((1, 2, 4, 8, 16), (1, 2, 4), (4, 8, 16)):
            cfg = dict(NUM_BLOCKS_PER_V_DIM=nb, NUM_WARPS=nw, WARP_THREADS_K=wk)
            lak.GDR_GLOBAL_CONFIG_MAP[key] = cfg
            lak.GDR_GLOBAL_CONFIG_MAP[key[:3] + (13,) + key[4:]] = cfg  # bd.check uses B=13
            try:
                eo, es, clob = bd.check(run)
                if eo > 2e-2 or es > 2e-2 or clob:
                    print(f"  B={B} {cfg} WRONG RESULT", flush=True)
                    continue
                r = bd.bench(run, B)
            except Exception as ex:
                print(f"  B={B} skip {cfg}: {type(ex).__name__}: {str(ex)[:80]}", flush=True)
                continue
            res.append((r["graph_us"], cfg))
        res.sort(key=lambda x: x[0])
        lak.GDR_GLOBAL_CONFIG_MAP.pop(key, None)
        out[B] = dict(default_us=default, best_us=res[0][0], best=res[0][1], top5=res[:5])
        print(f"B={B}: default {default:.1f} us -> best {res[0][0]:.1f} us {res[0][1]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["sglang_packed", "aiter_flydsl"])
    ap.add_argument("--bs", default="1,2,4,8,16,24,32,48,64")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bs_list = [int(x) for x in args.bs.split(",")]
    res = sweep_sglang(bs_list) if args.target == "sglang_packed" else sweep_flydsl(bs_list)
    json.dump(res, open(args.out, "w"), indent=1)
    if args.target == "aiter_flydsl":
        import aiter.ops.flydsl.linear_attention_kernels as lak
        with open(args.out.replace(".json", ".csv"), "w") as f:
            for B, r in res.items():
                c = r["best"]
                f.write(f"{lak.GDR_GPU_ARCH},torch.bfloat16,torch.float32,{B},1,{bd.HG},{bd.H},{bd.K},{bd.V},"
                        f"{c['NUM_BLOCKS_PER_V_DIM']},{c['NUM_WARPS']},{c['WARP_THREADS_K']},{r['best_us']}\n")


if __name__ == "__main__":
    main()
