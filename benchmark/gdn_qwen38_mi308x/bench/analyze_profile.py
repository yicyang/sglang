"""Summarize sglang torch-profiler traces: GDN kernel time and layout/copy kernels.

usage: analyze_profile.py <trace dir> [--rank 0]
Reports, for the TP rank's trace: total GPU kernel time, time in GDN-related
kernels (by kernel), and every copy/transpose/cat/fill-like kernel with its
call count, so integrations can be checked for extra layout conversions.
"""
import argparse
import glob
import gzip
import json
import re
from collections import defaultdict

GDN = re.compile(
    r"gated_delta|fwd_h|chunk_fwd|chunk_gdn|gdn|gdr|recompute_w_u|l2norm|chunk_local_cumsum|"
    r"kkt|solve_tril|merge_16x16|causal_conv1d|packed_decode|sigmoid_gating|fused_recurrent|"
    r"qkvzba|qkv_split|gdn_prepare|chunk_scaled_dot",
    re.I,
)
COPY = re.compile(
    r"copy|contiguous|CatArray|cat_|transpose|permute|index_select|indexSelect|index_copy|"
    r"FillFunctor|(?<!pre)fill_|zero_|elementwise_kernel",
    re.I,
)


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--rank", default="0")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.trace_dir}/**/*TP-{args.rank}*.json*", recursive=True)) or sorted(
        glob.glob(f"{args.trace_dir}/**/*.json*", recursive=True))
    if not files:
        raise SystemExit("no trace found")
    tr = load(files[0])
    kern = defaultdict(lambda: [0.0, 0])
    for e in tr.get("traceEvents", []):
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            k = kern[e["name"]]
            k[0] += e.get("dur", 0)
            k[1] += 1
    total = sum(v[0] for v in kern.values())
    gdn = {n: v for n, v in kern.items() if GDN.search(n)}
    cp = {n: v for n, v in kern.items() if COPY.search(n) and n not in gdn and not n.startswith("moe")}
    print(f"trace: {files[0]}")
    print(f"total GPU kernel time: {total/1e3:.1f} ms over {sum(v[1] for v in kern.values())} kernels")
    g = sum(v[0] for v in gdn.values())
    print(f"GDN-related kernels: {g/1e3:.1f} ms ({100*g/total:.1f}%)")
    for n, (t, c) in sorted(gdn.items(), key=lambda x: -x[1][0])[: args.top]:
        print(f"  {t/1e3:9.2f} ms  x{c:6d}  {n[:120]}")
    c_ = sum(v[0] for v in cp.values())
    print(f"copy/layout-like kernels (non-GDN-named): {c_/1e3:.1f} ms ({100*c_/total:.1f}%)")
    for n, (t, c) in sorted(cp.items(), key=lambda x: -x[1][0])[: args.top]:
        print(f"  {t/1e3:9.2f} ms  x{c:6d}  {n[:120]}")


if __name__ == "__main__":
    main()
