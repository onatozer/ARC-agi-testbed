"""Highlight REPORTS_SGD.md's main results table in place: per metric column, the best value bold, the other values in
the top 10% underlined. Cells may hold several values ("1 / 2 / 7", ".753 / .797"); each is ranked on its own. Re-run
after adding rows (existing highlighting is stripped first).

    python scripts/miniarc5/highlight_results.py [REPORTS_SGD.md]
"""
import math
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "REPORTS_SGD.md"
METRICS = slice(2, 7)  # pass@1, pass@2, pass@5, pass@100, token acc
STRIP = re.compile(r"\*\*|</?u>")


def values(cell):
    out = []
    for part in cell.split("/"):
        try:
            out.append(float(part.strip()))
        except ValueError:
            pass
    return out


def render(cell, best, cut):
    parts = []
    for part in cell.split(" / "):
        p = part.strip()
        try:
            v = float(p)
        except ValueError:
            parts.append(p); continue
        parts.append(f"**{p}**" if v == best else f"<u>{p}</u>" if v >= cut else p)
    return " / ".join(parts)


lines = open(PATH).read().split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith("## Results"))
rows = [i for i in range(start, len(lines)) if lines[i].startswith("| ") and not lines[i].startswith("| checkpoint")]
end = next(i for i in range(rows[0], len(lines)) if not lines[i].startswith("|"))
rows = [i for i in rows if i < end]
cells = {i: [STRIP.sub("", c).strip() for c in lines[i].strip().strip("|").split("|")] for i in rows}
ncol = len(next(iter(cells.values())))
for col in range(ncol)[METRICS]:
    vals = sorted((v for c in cells.values() for v in values(c[col])), reverse=True)
    if not vals:
        continue
    best, cut = vals[0], vals[max(math.ceil(0.1 * len(vals)) - 1, 0)]
    for i, c in cells.items():
        c[col] = render(c[col], best, cut)
for i, c in cells.items():
    lines[i] = "| " + " | ".join(c) + " |"
note = "Bold: the best value in its column; underlined: the rest of the column's top 10% (`scripts/miniarc5/highlight_results.py`)."
if note not in lines[start + 1 : start + 4]:
    lines.insert(start + 1, ""); lines.insert(start + 2, note)
open(PATH, "w").write("\n".join(lines))
print(f"highlighted {len(rows)} rows")
