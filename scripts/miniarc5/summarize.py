"""Summarizes MiniARC 5x5 runs for REPORTS_ARC.md: the eval rows (ARC pass@k, exact accuracy, memory) of each run's W&B
history (online runs keep a local .wandb copy too), the last train row, wall-clock and peak RSS from train.log, and any
TTT results under eval_preds/. usage: summarize.py <run dir> [<run dir> ...]"""
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcts_bench"))
from wandb_offline import history  # noqa: E402

KEYS = ("ARC/pass@1", "ARC/pass@2", "ARC/pass@5", "all/exact_accuracy", "all/accuracy", "eval/gpu_alloc_mb", "eval/gpu_reserved_mb", "eval/host_rss_mb")


def summarize(run_dir: str) -> None:
    print(f"## {Path(run_dir).name}")
    log = Path(run_dir) / "train.log"
    if log.exists():
        text = log.read_text(errors="replace")
        wall = re.findall(r"exit=(\d+) seconds=(\d+)", text)
        rss = re.findall(r"Maximum resident set size \(kbytes\): (\d+)", text)
        params = re.findall(r"Model parameters: (\d+)", text)
        print(f"params {params[-1] if params else '?'} | wall {wall[-1][1] + ' s (exit ' + wall[-1][0] + ')' if wall else '?'} | peak RSS {int(rss[-1]) // 1024 if rss else '?'} MB")
    last_train = None
    for row in history(run_dir):
        if any(k in row for k in KEYS[:4]):
            print("eval step", row.get("step"), {k: round(row[k], 4) for k in KEYS if k in row})
        elif "train/lm_loss" in row:
            last_train = row
    if last_train:
        print("last train:", {k: round(v, 4) for k, v in sorted(last_train.items()) if k.startswith("train/") and isinstance(v, float) and ("loss" in k or "accuracy" in k or "mb" in k or k.endswith("/lr") or "search/" in k)})
    for path in sorted(glob.glob(f"{run_dir}/**/ttt_metrics.json", recursive=True)):
        m = json.load(open(path))
        times = [json.load(open(p)).get("time_sec", 0) for p in glob.glob(str(Path(path).parent / "tasks" / "*.json"))]
        print(f"ttt {Path(path).parent.name}: {m['num_tasks']} tasks, {{k: round(v, 4) for k, v in m['metrics'].items() if 'pass@' in k or k.endswith('exact_accuracy')}}, {sum(times) / max(len(times), 1):.1f} s/task")


if __name__ == "__main__":
    for run_dir in sys.argv[1:]:
        summarize(run_dir)
