"""Schematic overlap figure (kernel order, widths not to scale) and QPS-gain bars for the PR.

usage: gen_figs.py <out_dir>
"""
import html
import sys

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
GRAY, ORANGE, BLUE, GREEN = "#dcdad5", "#eb6834", "#2a78d6", "#1baf7a"
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def text(self, x, y, s, size=18, fill=INK, weight="normal", anchor="start", italic=False):
        style = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}"{style}>{html.escape(s)}</text>'
        )

    def box(self, x, y, w, h, fill, label="", sub="", fg=None):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}"/>')
        fg = fg or (INK if fill == GRAY else "#ffffff")
        if label and sub:
            self.text(x + w / 2, y + h / 2 - 3, label, 15, fg, anchor="middle")
            self.text(x + w / 2, y + h / 2 + 15, sub, 15, fg, anchor="middle")
        elif label:
            self.text(x + w / 2, y + h / 2 + 6, label, 17, fg, anchor="middle")

    def line(self, x1, y1, x2, y2, color=INK2, width=1.6, arrow=False):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{marker}/>'
        )

    def svg(self):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>'
            f'<rect width="{self.w}" height="{self.h}" fill="{SURF}"/>' + "".join(self.parts) + "</svg>"
        )


def lane(svg, y, items, x0=200, gap=10, h=58):
    """items: (width, fill, label, sub, tag); returns {tag: (x, width)} for the tagged boxes."""
    x, at = x0, {}
    for w, fill, label, sub, tag in items:
        svg.box(x, y, w, h, fill, label, sub)
        if tag:
            at[tag] = (x, w)
        x += w + gap
    return at


def push_copies(svg, at, y_compute, y_sdma, copies, h=58):
    """copies: (push tag, copy x, copy width, label); arrow from the push to the copy start."""
    for tag, cx, cw, label in copies:
        px, pw = at[tag]
        svg.text(px + pw / 2, y_compute - 10, f"push {tag.split(':')[1]}", 14, INK2, anchor="middle")
        svg.line(px + pw / 2, y_compute + h + 4, cx + 6, y_sdma - 6, arrow=True)
        svg.box(cx, y_sdma, cw, h, BLUE, label)


def bracket(svg, x1, x2, y, note):
    svg.line(x1, y - 10, x1, y)
    svg.line(x1, y, x2, y)
    svg.line(x2, y - 10, x2, y)
    svg.text((x1 + x2) / 2, y + 24, note, 17, INK2, anchor="middle", italic=True)


def overlap_figure():
    s = Svg(2000, 1480)
    s.text(36, 58, "Qwen3.5-35B-A3B TP2 prefill over SDMA: where each all-reduce overlaps", 30, weight="bold")
    s.text(36, 98, "one full-attention layer, one rank, kernel order and times from a TP2 32k-token prefill "
                   "trace (TP0); widths not to scale", 19, INK2)

    s.text(36, 172, "Baseline (#117)", 24, weight="bold")
    s.text(36, 210, "two all-reduces per layer, each an RCCL kernel (ncclDevKernel) on the compute stream, "
                    "2.9 ms each at 32k tokens", 19, INK2)
    s.text(150, 290, "compute", 18, INK2, anchor="end")
    lane(s, 260, [
        (150, GRAY, "attn / GDN", "", None), (120, GRAY, "o_proj", "", None),
        (170, GREEN, "AR (rccl)", "2.95 ms", None), (150, GRAY, "add + norm", "", None),
        (150, GRAY, "MoE", "", None), (170, GREEN, "AR (rccl)", "2.91 ms", None),
        (150, GRAY, "add + norm", "", None), (170, GRAY, "in_proj (next)", "", None),
    ], x0=170)

    # After attention: o_proj per chunk, each partial pushed; norm + MoE consume chunk by chunk.
    s.text(36, 420, "This PR - after attention: o_proj -> all-reduce -> add + norm -> MoE, chunk by chunk", 24, weight="bold")
    s.text(36, 458, "each o_proj chunk's partial sum goes to the peer right after its GEMM; "
                    "each landed chunk is summed, normed and run through the MoE", 19, INK2)
    yc, ys = 540, 660
    s.text(150, yc + 34, "compute", 18, INK2, anchor="end")
    s.text(150, ys + 34, "SDMA", 18, INK2, anchor="end")
    at = lane(s, yc, [
        (130, GRAY, "attn / GDN", "", None),
        (110, GRAY, "o_proj c0", "", None), (22, ORANGE, "", "", "push:c0"),
        (110, GRAY, "o_proj c1", "", None), (22, ORANGE, "", "", "push:c1"),
        (110, GRAY, "o_proj c2", "", None), (22, ORANGE, "", "", "push:c2"),
        (110, GRAY, "o_proj c3", "", None), (22, ORANGE, "", "", "push:c3"),
        (86, ORANGE, "wait c0", "22us", "wait0"),
        (170, GRAY, "norm + MoE c0", "", "moe0"), (22, ORANGE, "", "", "wait:c1"),
        (170, GRAY, "norm + MoE c1", "", "moe1"), (22, ORANGE, "", "", "wait:c2"),
        (170, GRAY, "norm + MoE c2", "", "moe2"), (22, ORANGE, "", "", "wait:c3"),
        (170, GRAY, "norm + MoE c3", "", None),
    ], x0=170)
    for i in (1, 2, 3):
        wx, ww = at[f"wait:c{i}"]
        s.text(wx + ww / 2, yc - 10, f"wait c{i}", 14, INK2, anchor="middle")
    # One copy engine: c0 runs under o_proj c1..c3 and lands as o_proj ends (22 us wait);
    # c1..c3 queue behind it and each runs under the previous chunk's MoE (4-5 us waits).
    c0x = at["push:c0"][0] + at["push:c0"][1] + 6
    copies = [("push:c0", c0x, at["wait0"][0] - c0x - 4, "copy c0 -> peer")]
    copies += [(f"push:c{i}", at[f"moe{i - 1}"][0], at[f"moe{i - 1}"][1], f"copy c{i} -> peer") for i in (1, 2, 3)]
    push_copies(s, at, yc, ys, copies)
    bracket(s, c0x, at["moe2"][0] + at["moe2"][1], ys + 92,
            "c0 runs under the remaining o_proj chunks, c1..c3 under the previous chunk's MoE; "
            "thin orange = push / wait (4-5 us once landed)")

    # After the MoE: each chunk's output pushed; the next layer's add + norm consume them.
    s.text(36, 860, "This PR - after the MoE: the same pipeline into the next layer", 24, weight="bold")
    s.text(36, 898, "each MoE chunk's partial sum goes out right after it; the next layer's add + norm run per "
                    "chunk, then in_proj reads all rows", 19, INK2)
    yc, ys = 980, 1100
    s.text(150, yc + 34, "compute", 18, INK2, anchor="end")
    s.text(150, ys + 34, "SDMA", 18, INK2, anchor="end")
    at = lane(s, yc, [
        (170, GRAY, "norm + MoE c0", "", None), (22, ORANGE, "", "", "push:m0"),
        (170, GRAY, "norm + MoE c1", "", "moe1"), (22, ORANGE, "", "", "push:m1"),
        (170, GRAY, "norm + MoE c2", "", "moe2"), (22, ORANGE, "", "", "push:m2"),
        (170, GRAY, "norm + MoE c3", "", "moe3"), (22, ORANGE, "", "", "push:m3"),
        (100, ORANGE, "wait m3", "381us", "waitm3"),
        (300, GRAY, "add + norm c0..c3 (next)", "", None),
        (190, GRAY, "in_proj (next)", "", None),
    ], x0=170)
    under = ["moe1", "moe2", "moe3", "waitm3"]
    copies = [(f"push:m{i}", at[u][0], at[u][1], f"copy m{i}" + (" -> peer" if i < 3 else "")) for i, u in enumerate(under)]
    push_copies(s, at, yc, ys, copies)
    bracket(s, at["moe1"][0], at["moe3"][0] + at["moe3"][1], ys + 92,
            "m0..m2 run under the next MoE chunks; m3 has no compute after it and is the one exposed wait")

    s.text(36, 1290, "Not overlapped: attention / GDN and the next layer's in_proj. Left exposed per layer: the first "
                     "chunk's wait (~22 us) and the last MoE chunk's copy (~380 us),", 19)
    s.text(36, 1322, "vs 5.9 ms of RCCL per layer on #117: this layer goes from 16.1 ms to 11.6 ms. "
                     "Chunks are 1/4 of the batch rows; rank 0 rows + rank 1 rows keep the replicas bit-identical.", 19)
    lx = 36
    for fill, label in ((GRAY, "compute kernel"), (ORANGE, "mori push / wait kernel (compute stream)"),
                        (BLUE, "SDMA copy (copy engine, under compute)"), (GREEN, "RCCL all-reduce (exposed)")):
        s.parts.append(f'<rect x="{lx}" y="1398" width="34" height="24" rx="3" fill="{fill}"/>')
        s.text(lx + 46, 1417, label, 19, INK2)
        lx += 46 + 11 * len(label) + 40
    return s.svg()


def qps_figure():
    rows = [("12k c8", 5.8, 15.5), ("12k c16", 2.6, 17.2), ("12k c32", 3.1, 19.1), ("8k c8 (CI)", 3.7, 12.4)]
    W, H, L, B, T = 1120, 520, 90, 450, 70
    s = Svg(W, H)
    s.text(L - 4, 40, "QPS gain over #117, Qwen3.5-35B-A3B TP2, 500 output tokens, MI308X", 21)
    ymax = 20
    y = lambda v: B - v / ymax * (B - T)
    for v in range(0, ymax + 1, 4):
        s.line(L, y(v), W - 20, y(v), GRID, 1)
        s.text(L - 12, y(v) + 6, str(v), 17, INK2, anchor="end")
    s.parts.append(f'<text x="30" y="{(B + T) / 2}" font-size="18" fill="{INK2}" text-anchor="middle" '
                   f'transform="rotate(-90 30 {(B + T) / 2})">QPS gain over #117 (%)</text>')
    slot = (W - 20 - L) / len(rows)
    for i, (name, gdn, pr) in enumerate(rows):
        cx = L + slot * (i + 0.5)
        for j, (v, fill) in enumerate(((gdn, BLUE), (pr, ORANGE))):
            bx = cx - 74 + j * 78
            s.parts.append(f'<rect x="{bx}" y="{y(v)}" width="70" height="{B - y(v)}" fill="{fill}"/>')
            s.text(bx + 35, y(v) - 8, f"+{v:.1f}%", 17, INK, anchor="middle")
        s.text(cx, B + 30, name, 19, INK2, anchor="middle")
    for k, (fill, label) in enumerate(((BLUE, "+ FlyDSL GDN decode"), (ORANGE, "+ SDMA all-reduce (this PR)"))):
        s.parts.append(f'<rect x="{L + 20}" y="{T + 6 + 30 * k}" width="30" height="18" fill="{fill}"/>')
        s.text(L + 60, T + 21 + 30 * k, label, 18)
    return s.svg()


if __name__ == "__main__":
    out = sys.argv[1]
    open(f"{out}/qwen35_tp2_sdma_overlap_stages.svg", "w").write(overlap_figure())
    open(f"{out}/qwen35_tp2_qps_gain.svg", "w").write(qps_figure())
