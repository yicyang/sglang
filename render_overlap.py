"""Render one Qwen3.5 decoder layer of a 32k-token TP2 prefill as an SVG timeline:
RCCL all-reduce (before) vs pipelined SDMA all-reduce (this PR), from torch-profiler traces.

usage: render_overlap.py <base.trace.json.gz> <pr.trace.json.gz> <out.svg> [tokens=32768]
"""
import gzip
import html
import json
import re
import sys

H, BW_GBPS, CHUNKS = 2048, 46.0, 4  # SDMA push-AR measured 46 GB/s (128 MB in ~2.9 ms)
C = {  # reference palette slots 1-4 in documented order, neutral for the rest
    "gemm": "#2a78d6", "moe": "#eb6834", "sdma": "#1baf7a", "ar": "#eda100", "other": "#c3c2b7",
}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def kernels(path):
    d = json.load(gzip.open(path))
    ev = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_op", "Kernel")]
    return sorted(ev, key=lambda e: e["ts"])


def cat(name):
    if re.search(r"nccl|cross_device_reduce|all_?reduce", name, re.I):
        return "ar"
    if re.search(r"moe_gemm|moe_sorting|topk|shared_experts", name, re.I):
        return "moe"
    if re.search(r"Flatmm|gemm_xdl|Cijk|gemm_a8w8", name):
        return "gemm"
    return "other"


def layer_window(ev):
    """Kernels from the end of the 3rd full-attention FMHA of the longest forward
    to the start of the next layer's GDN core (gdn_prefill_front)."""
    emb = [i for i, e in enumerate(ev) if "_vocab_parallel_embedding" in e["name"]] + [len(ev)]
    fwds = [ev[a:b] for a, b in zip(emb, emb[1:]) if sum("Fmha" in e["name"] for e in ev[a:b]) >= 10]
    fwd = max(fwds, key=lambda f: sum(e["dur"] for e in f))
    fmha = [i for i, e in enumerate(fwd) if "Fmha" in e["name"]]
    a = fmha[2]
    b = next(i for i in range(a, len(fwd)) if "gdn_prefill_front" in fwd[i]["name"])
    t0 = fwd[a]["ts"] + fwd[a]["dur"]
    return [e for e in fwd[a + 1 : b]], t0, fwd[b]["ts"]


def sdma_copies(win, tokens):
    """Estimated copy-engine intervals: one per push, queued, bytes / measured bandwidth."""
    dur = tokens // CHUNKS * H * 2 / (BW_GBPS * 1e3)  # us
    out, free = [], 0.0
    for e in win:
        if e["name"].startswith("push_"):
            s = max(e["ts"] + e["dur"], free)
            free = s + dur
            out.append((s, free))
    return out


def main(base_path, pr_path, out_path, tokens=32768):
    rows = []
    for label, path in (("Before: RCCL all-reduce", base_path), ("This PR: SDMA all-reduce, 4 row chunks", pr_path)):
        win, t0, t1 = layer_window(kernels(path))
        rows.append((label, win, t0, t1))
    span = max(t1 - t0 for _, _, t0, t1 in rows) / 1000  # ms
    W, L, R = 1100, 190, 30
    x = lambda ms: L + ms / span * (W - L - R)
    lane_h, gap = 26, 10
    y, parts = 96, []
    for label, win, t0, t1 in rows:
        is_pr = "SDMA" in label
        parts.append(f'<text x="16" y="{y - 8}" font-size="13" font-weight="600" fill="{INK}">{label}</text>')
        ar_ms = sum(e["dur"] for e in win if cat(e["name"]) == "ar") / 1000
        wait_ms = sum(e["dur"] for e in win if e["name"].startswith("wait_")) / 1000
        total = (t1 - t0) / 1000
        lanes = [("compute stream", y)] + ([("SDMA engine (est.)", y + lane_h + gap)] if is_pr else [])
        for name, ly in lanes:
            parts.append(f'<text x="{L - 8}" y="{ly + 17}" font-size="11" fill="{INK2}" text-anchor="end">{name}</text>')
            parts.append(f'<rect x="{L}" y="{ly}" width="{x(total) - L:.1f}" height="{lane_h}" fill="#f0efec"/>')
        for e in win:
            c = cat(e["name"])
            s, w = x((e["ts"] - t0) / 1000), max(e["dur"] / 1000 / span * (W - L - R), 0.6)
            if w < 0.8 and c == "other":
                continue
            parts.append(f'<rect x="{s:.2f}" y="{y}" width="{w:.2f}" height="{lane_h}" fill="{C[c]}" stroke="{SURF}" stroke-width="1"><title>{html.escape(e["name"][:80])} {e["dur"]/1000:.3f} ms</title></rect>')
        if is_pr:
            sy = y + lane_h + gap
            for s, e in sdma_copies(win, tokens):
                parts.append(f'<rect x="{x((s - t0) / 1000):.2f}" y="{sy}" width="{(e - s) / 1000 / span * (W - L - R):.2f}" height="{lane_h}" rx="3" fill="{C["sdma"]}" stroke="{SURF}" stroke-width="2"/>')
        note = (f"layer {total:.2f} ms; exposed all-reduce wait {wait_ms:.2f} ms" if is_pr
                else f"layer {total:.2f} ms; all-reduce on the critical path {ar_ms:.2f} ms")
        ly = y + (2 * lane_h + gap if is_pr else lane_h) + 16
        parts.append(f'<line x1="{x(total):.1f}" y1="{y - 4}" x2="{x(total):.1f}" y2="{ly - 12}" stroke="{INK}" stroke-width="1.5"/>')
        parts.append(f'<text x="{x(total) - 4:.1f}" y="{ly}" font-size="12" fill="{INK}" text-anchor="end">{note}</text>')
        y = ly + 40
    axis_y = y - 18
    for ms in range(0, int(span) + 1):
        parts.append(f'<line x1="{x(ms):.1f}" y1="78" x2="{x(ms):.1f}" y2="{axis_y}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{x(ms):.1f}" y="{axis_y + 14}" font-size="11" fill="{MUTED}" text-anchor="middle">{ms} ms</text>')
    lg, lx = axis_y + 40, 16
    for key, text in (("gemm", "projection GEMM (o_proj, in_proj)"), ("moe", "MoE"), ("ar", "RCCL all-reduce"),
                      ("sdma", "SDMA copy (estimated: chunk bytes / 46 GB/s)"), ("other", "norm / quant / add / sort")):
        parts.append(f'<rect x="{lx}" y="{lg - 10}" width="12" height="12" rx="2" fill="{C[key]}"/>'
                     f'<text x="{lx + 18}" y="{lg}" font-size="12" fill="{INK2}">{text}</text>')
        lx += 18 + 7.2 * len(text) + 22
    hgt = lg + 24
    title = "Qwen3.5-35B-A3B TP2 prefill (32k tokens, MI308X, rank 0): one full-attention layer, o_proj to the next GDN core"
    sub = "Measured kernels from torch-profiler traces; the GEMM / MoE run per row chunk while the copy engine moves the previous chunk"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{hgt}" viewBox="0 0 {W} {hgt}" font-family="Helvetica, Arial, sans-serif">'
           f'<rect width="{W}" height="{hgt}" fill="{SURF}"/>'
           f'<text x="16" y="28" font-size="15" font-weight="700" fill="{INK}">{title}</text>'
           f'<text x="16" y="48" font-size="12" fill="{INK2}">{sub}</text>' + "".join(parts) + "</svg>")
    open(out_path, "w").write(svg)
    for label, win, t0, t1 in rows:
        print(f"{label}: window {(t1 - t0) / 1000:.2f} ms, kernels {len(win)}")


if __name__ == "__main__":
    main(*sys.argv[1:4], *(int(a) for a in sys.argv[4:5]))
