"""Prints one row per eval of a checkpoint's eval_preds/ (gram_width.yaml and treegram_from_gram.yaml runs): the
test-time scaling ladder of REPORTS_GRAM_TTS.md. usage: ladder_table.py <run dir with eval_preds/> [...]"""
import json
import sys
from pathlib import Path

KEYS = ("all/exact_accuracy", "NQueens/accuracy", "different_init/majority_vote_acc", "different_init/lprm_best_acc",
        "different_init/any_correct", "all/search/pick_is_best", "all/search/value_rank_corr", "all/search/cand_ce_gap", "all/nfe", "eval_time_sec")


def main() -> None:
    for run in sys.argv[1:]:
        print(run)
        print("eval  " + "  ".join(k.split("/")[-1] for k in KEYS))
        for metrics_file in sorted(Path(run).glob("eval_preds/*/eval_metrics_*.json")):
            data = json.loads(metrics_file.read_text())
            flat = {"eval_time_sec": data.get("eval_time_sec"), **data.get("metrics", {})}
            print(f"{metrics_file.parent.name}  " + "  ".join(f"{flat[k]:.4g}" if isinstance(flat.get(k), (int, float)) else "-" for k in KEYS))


if __name__ == "__main__":
    main()
