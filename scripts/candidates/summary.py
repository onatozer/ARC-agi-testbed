"""One line per eval of a run (exact accuracy, search diagnostics) plus the last train row, wall-clock and peak memory,
read from the run's local .wandb file and train.log. usage: summary.py <run dir> [<run dir> ...]"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcts_bench"))
from wandb_offline import history  # noqa: E402

EVAL_KEYS = ("NQueens/accuracy", "all/exact_accuracy", "all/accuracy", "all/lm_loss", "all/search/cand_div", "all/search/post_entropy", "all/search/argmax_agree", "all/search/kl_nats", "eval/gpu_alloc_mb")
TRAIN_KEYS = ("train/lm_loss", "train/kl_loss", "train/search/cand_loss", "train/search/cand_ce_gap", "train/search/cand_div", "train/search/post_entropy", "train/search/argmax_agree", "train/search/cand_best_agree", "train/search/kl_nats", "train/search/recover_acc", "train/search/pick_entropy", "train/search/balance", "train/search/cand_eps")


def summarize(run_dir: str) -> None:
    print(f"## {Path(run_dir).name}")
    log = Path(run_dir) / "train.log"
    if log.exists():
        text = log.read_text(errors="replace")
        elapsed = re.findall(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (?:(\d+):)?(\d+):([\d.]+)", text)
        rss = re.findall(r"Maximum resident set size \(kbytes\): (\d+)", text)
        params = re.findall(r"Model parameters: (\d+)", text)
        exit_code = re.findall(r"^exit=(\d+)", text, re.M)
        secs = int(elapsed[-1][0] or 0) * 3600 + int(elapsed[-1][1]) * 60 + float(elapsed[-1][2]) if elapsed else None
        print(f"params {params[-1] if params else '?'} | wall {f'{secs:.0f} s' if secs is not None else '?'} | peak RSS {int(rss[-1]) // 1024 if rss else '?'} MB | exit {exit_code[-1] if exit_code else '?'}")
    train = []
    for row in history(run_dir):
        if "all/exact_accuracy" in row:
            print("eval", json.dumps({k: round(row[k], 4) for k in EVAL_KEYS if k in row} | {"step": row.get("_step")}))
        elif "train/lm_loss" in row:
            train.append(row)
    # The search metrics are averaged over the rows that halted this step, so they are zero on most steps and a row
    # does not always carry every key: report each key's mean over its nonzero values in the last 200 train rows
    if train:
        tail = train[-200:]
        means = {}
        for k in TRAIN_KEYS:
            vals = [r[k] for r in tail if r.get(k, 0) != 0]
            if vals:
                means[k] = round(sum(vals) / len(vals), 4)
        print("train (mean of last 200 rows)", json.dumps(means | {"last_step": tail[-1].get("_step")}))


for run in sys.argv[1:]:
    summarize(run)
