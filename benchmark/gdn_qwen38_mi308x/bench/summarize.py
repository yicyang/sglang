"""Print a table (gpu_us / wall_us) from one or more bench json files."""
import json, sys
rows = []
shapes = None
for p in sys.argv[1:]:
    d = json.load(open(p))
    for r in d["results"]:
        rows.append(r)
        s = [k for k, v in r.items() if isinstance(v, dict) and "gpu_us" in v]
        shapes = shapes or s
print("| impl | correct | " + " | ".join(shapes) + " |")
print("|---|---|" + "---|" * len(shapes))
for r in rows:
    if "error" in r:
        print(f"| {r['impl']} | ERROR: {r['error'][:120]} |" + " |" * len(shapes)); continue
    cells = [f"{r[s]['gpu_us']:.0f} / {r[s]['median_us']:.0f}" if s in r else "-" for s in shapes]
    print(f"| {r['impl']} | {r.get('correct')} (o {r.get('err_o',0):.1e}, s {r.get('err_state',0):.1e}) | " + " | ".join(cells) + " |")
