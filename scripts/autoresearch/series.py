"""Time series of train metrics from an offline W&B run, every `every` steps. usage: series.py <run dir> <every> <key...>"""
import sys

from wandb_offline import history

run, every, keys = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
for row in history(run):
    step = int(row.get("step", -1))
    if any(k in row for k in keys) and (step % every == 0 or step == 1):
        print(step, " ".join(f"{k.split('/')[-1]}={row[k]:.4g}" for k in keys if k in row and isinstance(row[k], (int, float))))
