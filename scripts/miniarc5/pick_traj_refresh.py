"""Round 27's target refresh: round 26's better of built-once (traj_refresh 1) and rebuilt-every-25% (0.25), by pass@1 at
1000 steps, ties by token accuracy. Prints the traj_refresh value.

    python scripts/miniarc5/pick_traj_refresh.py [runs_dir]
"""
import glob, json, os, sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/ARC-agi-testbed/autoresearch_runs/miniarc5")
scores = []
for run, sfx, refresh in (("msgd26_traj_once", "ms20o_sg1000", "1"), ("msgd26_traj_q25", "ms20q_sg1000", "0.25")):
    paths = glob.glob(os.path.join(RUNS, run, "outputs", "*", "*", "*", "eval_preds", f"step_*_{sfx}", "ttt_metrics.json"))
    if not paths:
        sys.exit(f"missing round-26 result {run}/{sfx}")
    m = json.load(open(sorted(paths)[-1]))["metrics"]
    scores.append(((m["ARC/pass@1"], m["all/accuracy"]), refresh, run))
    print(f"{run}: pass@1 {m['ARC/pass@1']:.3f} tok {m['all/accuracy']:.3f}", file=sys.stderr)
best = max(scores)
print(f"chosen: {best[2]}", file=sys.stderr)
print(best[1])
