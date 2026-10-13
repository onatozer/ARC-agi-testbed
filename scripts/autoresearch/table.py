"""One line per run under a runs directory: status, final train exact, and the eval metric at every eval step.
usage: table.py <runs dir> [metric, default all/exact_accuracy]"""
import glob, os, re, sys

from wandb_offline import history

root = sys.argv[1]
metric = sys.argv[2] if len(sys.argv) > 2 else "all/exact_accuracy"
for run in sorted(glob.glob(root + "/*/")):
    name = os.path.basename(run.rstrip("/"))
    log = open(run + "train.log").read() if os.path.exists(run + "train.log") else ""
    m = re.search(r"exit=(\d+) seconds=(\d+)", log)
    status = f"exit={m.group(1)} {int(m.group(2)) // 60}min" if m else "running " + (re.findall(r"(\d+)/\d+ \[", log) or ["?"])[-1]
    evals, train = {}, None
    for row in history(run):
        if metric in row:
            evals[int(row["step"])] = row[metric]
        elif "train/exact_accuracy" in row:
            train = row["train/exact_accuracy"]
    curve = " ".join(f"{s}:{v:.3f}" for s, v in sorted(evals.items()))
    print(f"{name:32s} {status:18s} train={train if train is None else round(train, 3)}  {curve}")
