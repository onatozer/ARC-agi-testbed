"""Round 18's alpha init: the round-17 VARC init with the highest pass@1 at 1000 stretched steps, ties broken by token
accuracy there. Prints "weights rows" for the chosen init.

    python scripts/miniarc5/pick_alpha_init.py [runs_dir]
"""
import glob, json, os, sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/ARC-agi-testbed/autoresearch_runs/miniarc5")
CANDIDATES = [("msgd17_v_a1e-3", "ms13a3", "1e-3 3e-2"), ("msgd17_v_a0.1", "ms13a01", "0.1 0.1"), ("msgd17_v_a1", "ms13a1", "1 1")]
scores = []
for run, pfx, init in CANDIDATES:
    paths = glob.glob(os.path.join(RUNS, run, "outputs", "*", "*", "*", "eval_preds", f"step_*_{pfx}_st1000", "ttt_metrics.json"))
    if not paths:
        sys.exit(f"missing round-17 result {run}/{pfx}_st1000")
    m = json.load(open(sorted(paths)[-1]))["metrics"]
    scores.append(((m["ARC/pass@1"], m["all/accuracy"]), init, run))
    print(f"{run}: pass@1 {m['ARC/pass@1']:.3f} tok {m['all/accuracy']:.3f}", file=sys.stderr)
best = max(scores)
print(f"chosen: {best[2]}", file=sys.stderr)
print(best[1])
