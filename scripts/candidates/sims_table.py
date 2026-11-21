"""Prints one row per (eval seed, pick, sims) of a sims_sweep.sh run. usage: sims_table.py <run dir with eval_preds/> [...]"""
import json
import re
import sys
from pathlib import Path

KEYS = ("all/exact_accuracy", "NQueens/accuracy", "ARC/pass@2", "all/search/argmax_agree", "all/search/post_entropy",
        "all/search/pick_ce_regret", "all/search/pick_is_best", "all/search/cand_ce_gap", "all/search/value_rank_corr", "eval_time_sec")


def main() -> None:
    for run in sys.argv[1:]:
        rows = []
        for metrics_file in Path(run).glob("eval_preds/*sims_s*_arch_eval_mcts_simulations-*/eval_metrics_*.json"):
            name = metrics_file.parent.name
            seed = int(re.search(r"sims_s(\d+)", name).group(1))
            sims = int(re.search(r"simulations-(\d+)", name).group(1))
            pick = (re.search(r"eval_pick-(\w+)", name) or [None, "sample"])[1]
            data = json.loads(metrics_file.read_text())
            rows.append((seed, pick, sims, {"eval_time_sec": data.get("eval_time_sec"), **data.get("metrics", {})}))
        print(run)
        print("seed  pick    sims  " + "  ".join(k.split("/")[-1] for k in KEYS))
        for seed, pick, sims, flat in sorted(rows, key=lambda row: row[:3]):
            print(f"{seed:4d}  {pick:6s}  {sims:4d}  " + "  ".join(f"{flat[k]:.4g}" if isinstance(flat.get(k), (int, float)) else "-" for k in KEYS))


if __name__ == "__main__":
    main()
