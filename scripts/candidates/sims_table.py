"""Prints one row per eval-sims setting of a sims_sweep.sh run. usage: sims_table.py <run dir with eval_preds/> [...]"""
import json
import re
import sys
from pathlib import Path

KEYS = ("all/exact_accuracy", "NQueens/accuracy", "all/search/argmax_agree", "all/search/post_entropy", "eval_time_sec", "eval/gpu_alloc_mb")


def main() -> None:
    for run in sys.argv[1:]:
        rows = []
        for metrics_file in Path(run).glob("eval_preds/*sims_s*_arch_eval_mcts_simulations-*/eval_metrics_*.json"):
            seed = int(re.search(r"sims_s(\d+)", metrics_file.parent.name).group(1))
            sims = int(re.search(r"simulations-(\d+)", metrics_file.parent.name).group(1))
            data = json.loads(metrics_file.read_text())
            flat = {"eval_time_sec": data.get("eval_time_sec"), **data.get("metrics", {})}
            rows.append((seed, sims, flat))
        print(run)
        print("seed  sims  " + "  ".join(k.split("/")[-1] for k in KEYS))
        for seed, sims, flat in sorted(rows):
            print(f"{seed:4d}  {sims:4d}  " + "  ".join(f"{flat[k]:.4g}" if isinstance(flat.get(k), (int, float)) else "-" for k in KEYS))


if __name__ == "__main__":
    main()
