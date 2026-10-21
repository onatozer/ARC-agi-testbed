"""Picks the inner step size from step-size probes (ttt.sh runs with maml_steps=3 on 2 tasks): the largest alpha whose
support loss fell on every probed task (before step 1 -> before step 3) by at least 5%; none qualifying: the
smallest. usage: pick_alpha.py <run dir> <suffix prefix> <alpha> [<alpha> ...]  (probe logs: <run dir>/ttt_<prefix>_a<alpha with . -> p>.log)"""
import re
import sys
from pathlib import Path

run_dir, prefix, alphas = Path(sys.argv[1]), sys.argv[2], [float(a) for a in sys.argv[3:]]
ok = []
for a in alphas:
    log = run_dir / f"ttt_{prefix}_a{str(a).replace('.', 'p')}.log"
    pairs = re.findall(r"support loss ([0-9.eE+-]+) -> ([0-9.eE+-]+)", log.read_text(errors="replace")) if log.exists() else []
    if pairs and all(float(after) <= 0.95 * float(before) for before, after in pairs):
        ok.append(a)
print(max(ok) if ok else min(alphas))
