"""Generate the two PR figures (static SVG, light/dark via prefers-color-scheme)."""
import os
OUT = os.path.dirname(os.path.abspath(__file__))
FONT = 'system-ui,-apple-system,"Segoe UI",sans-serif'
STYLE = """<style>
.bg{fill:#fcfcfb}.ink{fill:#0b0b0b}.ink2{fill:#52514e}.mut{fill:#898781}
.grid{stroke:#e1e0d9}.axis{stroke:#c3c2b7}
.box{fill:#fcfcfb;stroke:#c3c2b7}.gone{fill:none;stroke:#898781;stroke-dasharray:4 3}
.fused{fill:#2a78d6}.moved{fill:#fcfcfb;stroke:#eb6834;stroke-width:2}
.s1{fill:#2a78d6}.s2{fill:#eb6834}.arrow{stroke:#898781;fill:none}.ag{fill:#e1e0d9}
.strike{stroke:#898781}
@media (prefers-color-scheme: dark){
.bg{fill:#1a1a19}.ink{fill:#f0efec}.ink2{fill:#c3c2b7}.grid{stroke:#2c2c2a}.axis{stroke:#383835}
.box{fill:#1a1a19;stroke:#383835}.fused{fill:#3987e5}.moved{fill:#1a1a19;stroke:#d95926}
.s1{fill:#3987e5}.s2{fill:#d95926}.ag{fill:#2c2c2a}}
text{font-family:%s}
</style>""" % FONT


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, cls="ink", size=13, anchor="middle", weight=400):
    return f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>'


# ---------------- figure 1: which kernels are fused ----------------
def fusion():
    W = 1120
    BW, BH, G = 92, 44, 8          # box width/height, gap
    AGW = 40                        # AG marker slot
    x0 = 120
    # column x positions: AG slot, 3 routing, AG slot, 5 MoE stages
    cols = [x0]
    x = x0 + AGW + G
    for _ in range(3):
        cols.append(x); x += BW + G
    cols.append(x); x += AGW + G
    for _ in range(6):
        cols.append(x); x += BW + G
    # cols: 0=AG, 1..3 routing, 4=AG, 5 sort, 6 quant x, 7 gate-up, 8 quant h, 9 down, 10 sum
    out = [STYLE, f'<rect class="bg" width="{W}" height="470" rx="8"/>']
    out.append(text(20, 30, "Qwen4 MoE kernels: baseline vs this PR + the aiter PR (gfx942, PTPC FP8)", "ink", 16, "start", 600))

    def box(col, y, lines, kind="box", span=1):
        x = cols[col]
        w = BW * span + (G * (span - 1)) if col not in (0, 4) else AGW
        if span > 1:
            w = cols[col + span - 1] + BW - x
        cls = {"box": "box", "gone": "gone", "fused": "fused", "moved": "moved"}[kind]
        r = [f'<rect x="{x}" y="{y}" width="{w}" height="{BH}" rx="4" class="{cls}"/>']
        tcls = {"box": "ink", "gone": "mut", "fused": "", "moved": "ink"}[kind]
        for i, ln in enumerate(lines):
            ty = y + BH / 2 + (i - (len(lines) - 1) / 2) * 15 + 4
            if kind == "fused":
                r.append(f'<text x="{x + w / 2}" y="{ty}" fill="#ffffff" font-size="12.5" text-anchor="middle" font-weight="{600 if i == 0 else 400}">{esc(ln)}</text>')
            else:
                r.append(text(x + w / 2, ty, ln, tcls, 12.5))
        if kind == "gone":
            r.append(f'<line x1="{x + 6}" y1="{y + BH / 2}" x2="{x + w - 6}" y2="{y + BH / 2}" class="strike" stroke-width="1.5"/>')
        return "".join(r)

    def ag(col, y, label):
        x = cols[col]
        return (f'<rect x="{x}" y="{y}" width="{AGW}" height="{BH}" rx="4" class="ag"/>'
                + text(x + AGW / 2, y + BH / 2 + 4, label, "ink2", 12, weight=600))

    def rowlabel(y, a, b):
        return text(20, y + BH / 2 - 2, a, "ink", 13, "start", 600) + text(20, y + BH / 2 + 14, b, "ink2", 12, "start")

    # prefill
    out.append(text(20, 62, "Prefill (SP + SDMA), per MoE layer", "ink2", 13, "start", 600))
    y = 78
    out.append(rowlabel(y, "baseline", "aiter fused_moe"))
    out.append(ag(0, y, "AG"))
    for c, l in ((1, ["router GEMM", "all T rows"]), (2, ["topk", "all T rows"]), (3, ["+shared exp.", "all T rows"]),
                 (5, ["moe sorting"]), (6, ["quant x"]), (7, ["gate-up", "+SiLU"]), (8, ["quant h", "T*topk rows"]),
                 (9, ["down", "8x1 + tail"]), (10, ["top-k sum"])):
        out.append(box(c, y, l))
    y = 140
    out.append(rowlabel(y, "this PR", "local routing"))
    for c, l in ((1, ["router GEMM", "local T/TP rows"]), (2, ["topk", "local rows"]), (3, ["+shared exp.", "local rows"])):
        out.append(box(c, y, l, "moved"))
    out.append(ag(4, y, "AG"))
    out.append(box(5, y, ["moe sorting"]))
    out.append(box(6, y, ["quant x"]))
    out.append(box(7, y, ["gate-up + SiLU", "+ fp8 quant (fused)"], "fused", span=2))
    out.append(box(9, y, ["down", "8x1 + tail"]))
    out.append(box(10, y, ["top-k sum"]))
    out.append(text(cols[4] + AGW / 2, y + BH + 16, "bf16 rows + 96 B", "ink2", 11))
    out.append(text(cols[4] + AGW / 2, y + BH + 29, "routing row", "ink2", 11))

    # decode
    out.append(text(20, 250, "Decode (CUDA graph), per MoE layer", "ink2", 13, "start", 600))
    y = 266
    out.append(rowlabel(y, "baseline", "aiter tuned"))
    for c, l in ((1, ["topkGating", "softmax (aiter)"]), (2, ["+shared exp."]), (3, ["zero-fill out"]),
                 (5, ["gate-up", "batch1"]), (6, ["down", "batch1"])):
        out.append(box(c, y, l))
    y = 328
    out.append(rowlabel(y, "this PR", "aiter batch1"))
    out.append(box(1, y, ["Triton router", "(gfx942)"], "moved"))
    out.append(box(2, y, ["+shared exp."]))
    out.append(box(3, y, ["zero-fill out"], "gone"))
    out.append(box(5, y, ["gate-up b1", "+ out clear"], "fused"))
    out.append(box(6, y, ["down", "batch1"]))

    # legend
    ly = 420
    lx = 20
    items = [("fused", "fused kernel (one launch)"), ("moved", "moved to local rows / faster kernel"),
             ("gone", "launch removed"), ("box", "unchanged")]
    for cls, lab in items:
        out.append(f'<rect x="{lx}" y="{ly}" width="26" height="16" rx="3" class="{cls}"/>')
        out.append(text(lx + 34, ly + 12.5, lab, "ink2", 12.5, "start"))
        lx += 34 + 8 * len(lab) + 30
    out.append(text(20, 455, "AG / RS: SDMA all-gather / reduce-scatter (bf16), unchanged; RS not drawn.", "mut", 12, "start"))
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 470" width="{W}" height="470">' + "".join(out) + "</svg>"


# ---------------- figure 2: kernel -> step -> E2E ----------------
def compare():
    rows = [
        ("Prefill MoE layer kernel", "time, M=12000, single GPU", 10.7, 13.8),
        ("MoE + router kernels", "per prefill step (profile)", 15.0, 25.5),
        ("Prefill step", "wall time (profile)", 2.2, 5.6),
        ("Decode step", "wall time, bs 8 (profile)", 3.7, 9.5),
        ("E2E QPS, c8", "12000 in / 350 out", 3.1, 3.7),
        ("E2E QPS, c12", "12000 in / 350 out", 2.9, 5.2),
    ]
    W, H = 900, 500
    L, R, T = 250, 60, 92
    BT, GAP, BAND = 14, 2, 58
    xmax = 30.0
    pw = W - L - R
    sx = lambda v: L + v / xmax * pw
    out = [STYLE, f'<rect class="bg" width="{W}" height="{H}" rx="8"/>']
    out.append(text(20, 30, "Fused MoE gain vs baseline: kernel -> step -> end to end", "ink", 16, "start", 600))
    out.append(text(20, 50, "Improvement in % (time reduction, or QPS gain). Higher is better.", "ink2", 12.5, "start"))
    # legend
    lx = L
    for cls, lab in (("s1", "TP2 (2 GPUs)"), ("s2", "TP4 (4 GPUs)")):
        out.append(f'<rect x="{lx}" y="64" width="12" height="12" rx="3" class="{cls}"/>')
        out.append(text(lx + 18, 74.5, lab, "ink2", 12.5, "start"))
        lx += 130
    # grid
    yb = T + BAND * len(rows)
    for v in range(0, 31, 5):
        x = sx(v)
        out.append(f'<line x1="{x}" y1="{T - 6}" x2="{x}" y2="{yb}" class="{"axis" if v == 0 else "grid"}" stroke-width="1"/>')
        out.append(text(x, yb + 18, f"{v}%", "mut", 12))
    for i, (name, sub, a, b) in enumerate(rows):
        y = T + i * BAND + 10
        if i == 4:
            out.append(f'<line x1="20" y1="{y - 12}" x2="{W - 20}" y2="{y - 12}" class="grid" stroke-width="1"/>')
        out.append(text(L - 14, y + 12, name, "ink", 13, "end", 600))
        out.append(text(L - 14, y + 28, sub, "ink2", 11.5, "end"))
        for j, (v, cls) in enumerate(((a, "s1"), (b, "s2"))):
            by = y + j * (BT + GAP)
            x1 = sx(v)
            r = 4
            # square at the baseline, 4px rounded data end
            d = (f"M{L},{by} H{x1 - r} Q{x1},{by} {x1},{by + r} V{by + BT - r} "
                 f"Q{x1},{by + BT} {x1 - r},{by + BT} H{L} Z")
            out.append(f'<path d="{d}" class="{cls}"><title>{("TP2", "TP4")[j]} {name}: {v}%</title></path>')
            out.append(text(x1 + 6, by + 11, f"{v:.1f}%", "ink2", 12, "start", 600 if i >= 4 else 400))
    out.append(text(20, H - 16, "MI308X, SP + SDMA overlap on in both arms, bf16 communication. "
                    "E2E: TP2 mean of 2 rounds, TP4 median of 4 rounds.", "mut", 11.5, "start"))
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">' + "".join(out) + "</svg>"


# ---------------- figure 3 (aiter PR): MoE-internal fusion only ----------------
def fusion_aiter():
    W, Hh = 980, 430
    BW, BH, G = 112, 44, 10
    x0 = 150
    cols = [x0 + i * (BW + G) for i in range(6)]
    out = [STYLE, f'<rect class="bg" width="{W}" height="{Hh}" rx="8"/>']
    out.append(text(20, 30, "aiter fused_moe, gfx942 FlyDSL path (PTPC FP8): kernels fused by this PR", "ink", 16, "start", 600))

    def box(col, y, lines, kind="box", span=1):
        x = cols[col]
        w = BW * span + G * (span - 1)
        cls = {"box": "box", "gone": "gone", "fused": "fused"}[kind]
        r = [f'<rect x="{x}" y="{y}" width="{w}" height="{BH}" rx="4" class="{cls}"/>']
        for i, ln in enumerate(lines):
            ty = y + BH / 2 + (i - (len(lines) - 1) / 2) * 15 + 4
            if kind == "fused":
                r.append(f'<text x="{x + w / 2}" y="{ty}" fill="#ffffff" font-size="12.5" text-anchor="middle" font-weight="{600 if i == 0 else 400}">{esc(ln)}</text>')
            else:
                r.append(text(x + w / 2, ty, ln, "mut" if kind == "gone" else "ink", 12.5))
        if kind == "gone":
            r.append(f'<line x1="{x + 6}" y1="{y + BH / 2}" x2="{x + w - 6}" y2="{y + BH / 2}" class="strike" stroke-width="1.5"/>')
        return "".join(r)

    def rowlabel(y, a, b):
        return text(20, y + BH / 2 - 2, a, "ink", 13, "start", 600) + text(20, y + BH / 2 + 14, b, "ink2", 12, "start")

    out.append(text(20, 62, "Prefill MoE (per layer)", "ink2", 13, "start", 600))
    y = 78
    out.append(rowlabel(y, "before", "_run_prefill"))
    for c, l in ((0, ["moe sorting"]), (1, ["quant x"]), (2, ["gate-up + SiLU", "bf16 out"]), (3, ["quant h", "re-read bf16"]),
                 (4, ["down", "8x1 + tail"]), (5, ["top-k sum"])):
        out.append(box(c, y, l))
    y = 140
    out.append(rowlabel(y, "this PR", "_run_prefill"))
    out.append(box(0, y, ["moe sorting"]))
    out.append(box(1, y, ["quant x"]))
    out.append(box(2, y, ["gate-up + SiLU + fp8 quant", "full-row tile, one launch"], "fused", span=2))
    out.append(box(4, y, ["down", "8x1 + tail"]))
    out.append(box(5, y, ["top-k sum"]))
    cx = cols[2] + BW + G / 2
    out.append(text(cx, y + BH + 17, "M=12000: TP2 1.63 -> 1.35 ms (-17%), TP4 1.05 -> 0.75 ms (-28%)", "ink2", 12))
    out.append(text(cx, y + BH + 32, "bf16 [M*topk, inter] write + re-read removed; output bit-exact", "ink2", 12))

    out.append(text(20, 262, "Decode MoE (per layer, fp8 batch1 path: M <= 8 for Qwen3.8)", "ink2", 13, "start", 600))
    y = 278
    out.append(rowlabel(y, "before", "_run_batch1"))
    out.append(box(0, y, ["zero-fill out"]))
    out.append(box(1, y, ["gate-up", "batch1"]))
    out.append(box(2, y, ["down", "batch1"]))
    y = 340
    out.append(rowlabel(y, "this PR", "_run_batch1"))
    out.append(box(0, y, ["zero-fill out"], "gone"))
    out.append(box(1, y, ["gate-up batch1", "+ out clear"], "fused"))
    out.append(box(2, y, ["down", "batch1"]))
    out.append(text(cols[3], y + 18, "one launch less per layer;", "ink2", 12, "start"))
    out.append(text(cols[3], y + 34, "CUDA-graph safe", "ink2", 12, "start"))

    lx, ly = 20, 404
    for cls, lab in (("fused", "fused kernel (one launch)"), ("gone", "launch removed"), ("box", "unchanged")):
        out.append(f'<rect x="{lx}" y="{ly}" width="26" height="16" rx="3" class="{cls}"/>')
        out.append(text(lx + 34, ly + 12.5, lab, "ink2", 12.5, "start"))
        lx += 34 + 8 * len(lab) + 30
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}">' + "".join(out) + "</svg>"


open(os.path.join(OUT, "qwen4_fused_moe_fusion.svg"), "w").write(fusion())
open(os.path.join(OUT, "qwen4_fused_moe_gain.svg"), "w").write(compare())
print("ok")
open(os.path.join(OUT, "aiter_fused_moe_gfx942_fusion.svg"), "w").write(fusion_aiter())
