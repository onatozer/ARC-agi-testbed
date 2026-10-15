"""Prints the eval rows (and the last train row) of an offline W&B run. usage: run_summary.py <run dir>"""
import json, sys

from wandb_offline import history

KEEP = ("NQueens", "all/", "eval")
last_train = None
for row in history(sys.argv[1]):
    if any(k.startswith(KEEP) for k in row):
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in sorted(row.items()) if not k.startswith("_") and not isinstance(v, dict)}))
    elif "train/lm_loss" in row:
        last_train = row
if last_train:
    print("last train:", json.dumps({k: round(v, 4) for k, v in sorted(last_train.items()) if k.startswith("train/") and isinstance(v, float)}))
