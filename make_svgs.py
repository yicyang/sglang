"""Static SVG figures for the GDN FlyDSL PRs (light surface, reference palette slots 1-2 + neutral)."""
import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/workspace/bench_qwen38_gdn_flydsl/pr_assets"
os.makedirs(OUT, exist_ok=True)

SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BASE = "#b9b8b2"  # neutral baseline
S1 = "#2a78d6"  # FlyDSL decode
S2 = "#eb6834"  # FlyDSL decode + prefill
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"


def svg(w, h, body, title, desc):
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
        f"role='img' aria-labelledby='t d' {FONT}>\n<title id='t'>{title}</title><desc id='d'>{desc}</desc>\n"
        f"<rect width='{w}' height='{h}' fill='{SURF}'/>\n{body}</svg>\n"
    )


def text(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' "
            f"font-weight='{weight}'>{s}</text>\n")


def bar(x, y, w, h, fill, r=4):
    # rounded data end on the right, square at the baseline (left)
    r = min(r, w / 2, h / 2)
    return (f"<path d='M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h - r} "
            f"Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z' fill='{fill}'/>\n")


# ---------------------------------------------------------------- 1. decode fusion
def fig_decode():
    W, H = 860, 440
    b = text(24, 32, "GDN decode, one layer (TP2, batch 8, MI308X): 7 kernels to 2, Triton interface kept", 16, weight="600")
    b += text(24, 52, "Kernel time per layer from a torch profile of the serving decode step", 12, INK2)
    x0, y0, scale = 170, 78, 10.5  # px per us
    before = [("b/a copy x2", 10.8), ("cat(q, k, v)", 6.2), ("causal_conv1d_update", 5.8),
              ("packed recurrence", 26.2), ("z copy", 5.4), ("gated RMSNorm", 5.5)]
    b += text(24, y0 + 16, "Triton chain", 13, weight="600")
    x = x0
    for name, us in before:
        w = us * scale
        b += f"<rect x='{x:.1f}' y='{y0}' width='{w - 2:.1f}' height='26' rx='3' fill='{BASE}'/>\n"
        if w > len(name) * 6.6 + 10:
            b += text(x + w / 2 - 1, y0 + 17, name, 11, INK, "middle")
        x += w
    tot_b = sum(u for _, u in before)
    b += text(x + 6, y0 + 17, f"{tot_b:.0f} us", 12, INK, weight="600")
    # small labels below for narrow segments
    xs = x0
    for i, (name, us) in enumerate(before):
        w = us * scale
        if w <= len(name) * 6.6 + 10:
            b += f"<line x1='{xs + w / 2:.1f}' y1='{y0 + 27}' x2='{xs + w / 2:.1f}' y2='{y0 + 38}' stroke='{MUTED}'/>\n"
            b += text(xs + w / 2, y0 + 50 + (i % 2) * 13, name, 10, INK2, "middle")
        xs += w
    y1 = y0 + 92
    b += text(24, y1 + 17, "FlyDSL", 13, weight="600")
    w1, w2 = 15.9 * scale, 5.5 * scale
    b += f"<rect x='{x0}' y='{y1}' width='{w1 - 2:.1f}' height='26' rx='3' fill='{S1}'/>\n"
    b += text(x0 + w1 / 2, y1 + 17, "fused decode", 11, SURF, "middle")
    b += f"<rect x='{x0 + w1:.1f}' y='{y1}' width='{w2 - 2:.1f}' height='26' rx='3' fill='{BASE}'/>\n"
    b += f"<line x1='{x0 + w1 + w2 / 2:.1f}' y1='{y1 + 27}' x2='{x0 + w1 + w2 / 2:.1f}' y2='{y1 + 38}' stroke='{MUTED}'/>\n"
    b += text(x0 + w1 + w2 / 2, y1 + 50, "gated RMSNorm (unchanged)", 10, INK2, "middle")
    b += text(x0 + w1 + w2 + 6, y1 + 17, "21.4 us", 12, INK, weight="600")
    b += text(x0 + w1 + w2 + 66, y1 + 17, "(36 layers: 2.17 to 0.77 ms per decode step, step -10%)", 12, INK2)
    # how it works
    y2 = y1 + 80
    b += text(24, y2, "Inside the fused kernel (one workgroup = one token x one v-head, 4 waves)", 13, weight="600")
    items = [
        ("1", "Scalar s_load of slot index / gates; all 16 dwordx4 fp32 state loads per lane issued first"),
        ("2", "Unpack + conv update from the in_proj output; q/k conv state shared by 3 v-heads is shifted once by the"),
        ("", "last reader (self-resetting atomic arrival counter, CUDA-graph safe)"),
        ("3", "K-dim reductions with DPP (quad_perm, row_half_mirror, row_mirror): no LDS round trips"),
        ("4", "State load + store with the nt cache policy: the in-place write-back of just-read lines"),
        ("", "is the bound on MI308X; nt cuts it ~25% (see the second figure)"),
        ("5", "Writes o [B, 1, HV, V] like the Triton kernel plus a contiguous z; the model's own gated RMSNorm"),
        ("", "runs as before (an in-kernel norm variant, 17.5 us, is available via norm_weight=)"),
    ]
    yy = y2 + 24
    for k, s in items:
        if k:
            b += f"<circle cx='34' cy='{yy - 4}' r='8' fill='{S1}'/>" + text(34, yy, k, 10, SURF, "middle", "600")
        b += text(50, yy, s, 12, INK)
        yy += 20
    return svg(W, H, b, "GDN decode fusion", "Seven decode kernels per GDN layer replaced by one FlyDSL kernel")


# ---------------------------------------------------------------- 2. in-place RMW + steps
def fig_rmw():
    W, H = 860, 290
    b = text(24, 32, "Why the decode recurrence is bound by the state write-back (MI308X)", 16, weight="600")
    b += text(24, 52, "Left: 25 MB read+write microbenchmark. Right: recurrence kernel, TP2 batch 8 (us, lower is better)", 12, INK2)
    # left: bandwidth bars
    rows = [("copy to another buffer", 2.79, BASE), ("in place, default cache policy", 1.27, BASE),
            ("in place, nt load + nt store", 1.50, S1)]
    x0, y = 230, 84
    sc = 60  # px per TB/s
    for name, v, c in rows:
        b += text(x0 - 8, y + 15, name, 12, INK, "end")
        b += bar(x0, y, v * sc, 22, c)
        b += text(x0 + v * sc + 6, y + 15, f"{v:.2f} TB/s", 12, INK)
        y += 36
    b += text(24, y + 14, "A store to a line the kernel has just read is slow; writes to lines read", 11, INK2)
    b += text(24, y + 29, "&gt;6 MB earlier, or not yet read, run at 2.0 to 3.0 TB/s.", 11, INK2)
    # right: step chart
    steps = [("sglang Triton", 26.2, BASE), ("FlyDSL v0", 19.9, S1), ("+ s_load, DPP, straight-line", 19.6, S1),
             ("+ nt load/store", 14.5, S1)]
    x1, y = 630, 84
    sc = 6.2
    for name, v, c in steps:
        b += text(x1 - 8, y + 15, name, 12, INK, "end")
        b += bar(x1, y, v * sc, 22, c)
        b += text(x1 + v * sc + 6, y + 15, f"{v:.1f}", 12, INK)
        y += 36
    b += text(452, y + 14, "With all compute removed (the loaded state stored back), the kernel", 11, INK2)
    b += text(452, y + 29, "still takes 19.7 us: compute is hidden, the write path sets the floor.", 11, INK2)
    return svg(W, H, b, "In-place state write-back", "Bandwidth of in-place read-modify-write and decode kernel latency steps")


# ---------------------------------------------------------------- 3. prefill
def fig_prefill():
    W, H = 860, 300
    b = text(24, 32, "GDN prefill, one layer (TP2, 12k tokens): kernel time, us (lower is better)", 16, weight="600")
    b += text(24, 52, "Same outputs: conv cache bit-exact, chunk_o and the norm bit-identical, q/k/v within 1 bf16 ulp", 12, INK2)
    rows = [("front end: cat + conv + split + 2x l2norm", 980, 325, "1 FlyDSL kernel"),
            ("chunk_fwd_o (Triton)", 642, 421, "BK=64, BV=128 on gfx942"),
            ("z copy + gated RMSNorm", 301, 194, "norm reads strided z")]
    x0, y = 330, 80
    sc = 0.42
    for name, a, f, how in rows:
        b += text(x0 - 8, y + 15, name, 12, INK, "end")
        b += bar(x0, y, a * sc, 20, BASE)
        b += text(x0 + a * sc + 6, y + 15, f"{a}", 12, INK)
        b += bar(x0, y + 24, f * sc, 20, S2)
        b += text(x0 + f * sc + 6, y + 39, f"{f}  ({how})", 12, INK)
        y += 62
    b += f"<rect x='{x0}' y='{y + 4}' width='12' height='12' rx='2' fill='{BASE}'/>" + text(x0 + 18, y + 14, "before", 12, INK2)
    b += f"<rect x='{x0 + 80}' y='{y + 4}' width='12' height='12' rx='2' fill='{S2}'/>" + text(x0 + 98, y + 14, "after", 12, INK2)
    return svg(W, H, b, "GDN prefill kernels", "Prefill GDN kernel time before and after per layer")


# ---------------------------------------------------------------- 4. QPS
def fig_qps(data):
    """data: {"TP2": {"c8": [baseline, pr], ...}, "TP4": {...}} in req/s."""
    W, H = 860, 300
    b = text(24, 32, "End-to-end QPS, Qwen3.8-Flash-Next FP8 on MI308X (12000 in / 350 out)", 16, weight="600")
    b += text(24, 52, "Same-session interleaved runs, mean of 2 (TP2) / 3 (TP4) rounds, higher is better", 12, INK2)
    groups = [(tp, c) for tp in ("TP2", "TP4") for c in ("c8", "c12")]
    x, y0, hmax = 110, 250, 150
    vmax = max(v for tp in data.values() for vs in tp.values() for v in vs) * 1.15
    b += f"<line x1='80' y1='{y0}' x2='{W - 40}' y2='{y0}' stroke='{GRID}'/>\n"
    cols = [BASE, S1]
    for tp, c in groups:
        vals = data[tp][c]
        for i, v in enumerate(vals):
            h = v / vmax * hmax
            bx = x + i * 54
            b += (f"<path d='M{bx},{y0} V{y0 - h + 4} Q{bx},{y0 - h} {bx + 4},{y0 - h} H{bx + 44} "
                  f"Q{bx + 48},{y0 - h} {bx + 48},{y0 - h + 4} V{y0} Z' fill='{cols[i]}'/>\n")
            lab = f"{v:.3f}" if i == 0 else f"{v:.3f} ({(v / vals[0] - 1) * 100:+.1f}%)"
            b += text(bx + 24 + (14 if i else 0), y0 - h - 6, lab, 11, INK, "middle", "600" if i else "normal")
        b += text(x + 51, y0 + 18, f"{tp} {c}", 12, INK2, "middle")
        x += 180
    lx = 600
    for i, name in enumerate(["Triton baseline", "this PR (FlyDSL GDN)"]):
        b += f"<rect x='{lx}' y='{64 + i * 18}' width='12' height='12' rx='2' fill='{cols[i]}'/>"
        b += text(lx + 18, 74 + i * 18, name, 12, INK2)
    return svg(W, H, b, "End-to-end QPS", "QPS for the Triton baseline and this PR at TP2 and TP4")


# ---------------------------------------------------------------- fusion figures (concise)
def _chain(b, x0, y, segs, scale, total_label, fill):
    x = x0
    for i, (name, v) in enumerate(segs):
        w = v * scale
        b += f"<rect x='{x:.1f}' y='{y}' width='{max(w - 2, 1):.1f}' height='26' rx='3' fill='{fill}'/>\n"
        if w > len(name) * 6.6 + 10:
            b += text(x + w / 2, y + 17, name, 11, INK if fill == BASE else SURF, "middle")
        else:
            b += f"<line x1='{x + w / 2:.1f}' y1='{y + 27}' x2='{x + w / 2:.1f}' y2='{y + 37}' stroke='{MUTED}'/>\n"
            b += text(x + w / 2, y + 49 + (i % 2) * 13, name, 10, INK2, "middle")
        x += w
    b += text(x + 6, y + 17, total_label, 12, INK, weight="600")
    return b


def fig_fusion(title, sub, before, after, after_extra, scale, unit, notes, H, n_before):
    W = 860
    b = text(24, 32, title, 16, weight="600") + text(24, 52, sub, 12, INK2)
    x0, y0 = 150, 76
    tb = sum(v for _, v in before)
    b += text(24, y0 + 17, "before", 13, weight="600")
    b = _chain(b, x0, y0, before, scale, f"{tb:.0f} {unit}, {n_before} kernels", BASE)
    y1 = y0 + 84
    b += text(24, y1 + 17, "after", 13, weight="600")
    ta = sum(v for _, v in after)
    b = _chain(b, x0, y1, after, scale, "", S1)
    x = x0 + sum(v for _, v in after) * scale
    for name, v in after_extra:
        w = v * scale
        b += f"<rect x='{x:.1f}' y='{y1}' width='{w - 2:.1f}' height='26' rx='3' fill='{BASE}'/>\n"
        b += f"<line x1='{x + w / 2:.1f}' y1='{y1 + 27}' x2='{x + w / 2:.1f}' y2='{y1 + 37}' stroke='{MUTED}'/>\n"
        b += text(x + w / 2, y1 + 49, name, 10, INK2, "middle")
        x += w
        ta += v
    b += text(x + 6, y1 + 17, f"{ta:.1f} {unit} ({(ta / tb - 1) * 100:+.0f}%)" if ta < 100 else f"{ta:.0f} {unit} ({(ta / tb - 1) * 100:+.0f}%)", 12, INK, weight="600")
    yy = y1 + 82
    for s in notes:
        b += f"<circle cx='30' cy='{yy - 4}' r='3' fill='{S1}'/>" + text(42, yy, s, 12, INK)
        yy += 20
    return svg(W, H, b, title, sub)


def fig_decode_fusion():
    return fig_fusion(
        "GDN decode: 6 kernels fused into 1 (TP2, batch 8, per layer)",
        "Serving torch profile on MI308X",
        [("b/a copy x2", 10.8), ("cat(q,k,v)", 6.2), ("conv1d_update", 5.8), ("Triton packed recurrence", 26.2),
         ("z copy", 5.4), ("gated RMSNorm", 5.5)],
        [("FlyDSL fused decode", 15.9)], [("gated RMSNorm (unchanged)", 5.5)],
        9.0, "us",
        ["fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl: unpack + conv + gating + l2norm + recurrence",
         "Bound by the in-place fp32 state write-back; nt loads/stores of the state cut it ~25%",
         "q/k conv state shared by 3 v-heads is shifted once, by the last workgroup that read it"],
        292, 7)


def fig_prefill_fusion():
    us = lambda ms: ms * 1000 / 36  # noqa: E731  per-layer from per-step totals
    return fig_fusion(
        "GDN prefill front end: 5 kernels fused into 1 (TP2, per layer)",
        "Serving torch profile on MI308X, prefill step of 8 x 12000-token requests",
        [("cat(q,k,v)", us(12.4)), ("Triton causal_conv1d_fwd", us(22.8)), ("qkv split", us(5.0)),
         ("l2norm x2", us(3.8))],
        [("FlyDSL prefill front", us(13.6))], [],
        0.45, "us",
        ["flydsl_gdn_prefill_front: varlen causal conv + SiLU + q/k/v split + q/k l2norm, one read of in_proj",
         "Was VALU-bound: exact bf16 product rounding via Veltkamp splitting packs into v_pk_* (2.5x fewer VALU ops)",
         "Conv cache bit-exact vs Triton; q/k/v within 1 bf16 ulp"],
        292, 5)


if __name__ == "__main__":
    open(f"{OUT}/gdn_decode_fusion.svg", "w").write(fig_decode_fusion())
    open(f"{OUT}/gdn_prefill_fusion.svg", "w").write(fig_prefill_fusion())
    open(f"{OUT}/gdn_state_writeback.svg", "w").write(fig_rmw())
    open(f"{OUT}/gdn_prefill.svg", "w").write(fig_prefill())
    qf = os.path.join(OUT, "qps.json")
    if os.path.exists(qf):
        open(f"{OUT}/gdn_qps.svg", "w").write(fig_qps(json.load(open(qf))))
    print("written to", OUT)
