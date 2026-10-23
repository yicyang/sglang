"""Aggregate e2e bench_serving results: per variant x workload, median over rounds
(min..max shown), plus delta vs base. Usage: summarize_e2e.py <results/e2e dir> [variants...]"""
import glob
import json
import os
import re
import statistics
import sys

root = sys.argv[1]
variants = sys.argv[2:] or ["base", "sgl_tuned", "aiter_pf", "aiter_all"]
WL = {"A": "12000/350 c1", "B": "12000/1 c1", "C": "4000/350 c16", "D": "2000/500 c32"}
METRICS = [("median_ttft_ms", "TTFT p50 ms", -1), ("mean_ttft_ms", "TTFT mean ms", -1),
           ("median_tpot_ms", "TPOT p50 ms", -1), ("median_itl_ms", "ITL p50 ms", -1),
           ("output_throughput", "out tok/s", 1), ("total_throughput", "total tok/s", 1)]


def load(v, w):
    vals = {}
    for p in sorted(glob.glob(f"{root}/{v}/round[0-9]/bench_{w}.jsonl")):
        if not re.search(r"round[1-9]/", p):
            continue
        d = json.loads(open(p).readline())
        for k, *_ in METRICS:
            vals.setdefault(k, []).append(d[k])
    return vals


for w, desc in WL.items():
    print(f"\n### Workload {w}: in/out/conc = {desc}\n")
    print("| metric | " + " | ".join(variants) + " |")
    print("|---|" + "---|" * len(variants))
    base = load("base", w)
    data = {v: load(v, w) for v in variants}
    for k, name, sign in METRICS:
        if w == "B" and ("tpot" in k or "itl" in k or "output" in k):
            continue
        cells = []
        for v in variants:
            xs = data[v].get(k)
            if not xs:
                cells.append("-")
                continue
            m = statistics.median(xs)
            cell = f"{m:.1f} ({min(xs):.0f}-{max(xs):.0f}, n={len(xs)})" if m > 100 else f"{m:.2f} ({min(xs):.2f}-{max(xs):.2f}, n={len(xs)})"
            if v != "base" and base.get(k):
                b = statistics.median(base[k])
                gain = (m / b - 1) * 100 * sign
                cell += f" **{gain:+.1f}%**"
            cells.append(cell)
        print(f"| {name} | " + " | ".join(cells) + " |")

print("\n### GSM8K (round 1, 1319 questions, 5-shot)\n")
for v in variants:
    p = f"{root}/{v}/round1/gsm8k.log"
    if os.path.exists(p):
        acc = re.findall(r"Accuracy: ([0-9.]+)", open(p).read())
        inv = re.findall(r"Invalid: ([0-9.]+)", open(p).read())
        print(f"- {v}: accuracy {acc[-1] if acc else '?'}, invalid {inv[-1] if inv else '?'}")
