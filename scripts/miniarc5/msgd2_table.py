"""Markdown table of the VARC Meta-SGD re-run (scripts/miniarc5/msgd2_varc.sh) from the ttt_metrics.json files on disk.

    python scripts/miniarc5/msgd2_table.py [runs_dir]
"""
import glob
import json
import os
import sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/ARC-agi-testbed/autoresearch_runs/miniarc5")
ROWS = [  # (run, eval suffix, label)
    ("varc_ho2", "ms4v_adam", "plain VARC, ttt_mini (ordinary fine-tune)"),
    ("varc_ho2", "ms4ve_s5", "plain VARC, fixed alpha 1 on the rows, 5 steps"),
    ("varc_ho2", "ms4ve_s10", "plain VARC, fixed alpha 1 on the rows, 10 steps"),
    ("varc_ho2", "ms4vl_s5", "plain VARC, fixed alpha 0.01 on the layers, 5 steps"),
    ("varc_ho2", "ms4vl_s10", "plain VARC, fixed alpha 0.01 on the layers, 10 steps"),
    ("msgd2_varc_emb", "ms4ve_s1", "Meta-SGD rows (alpha0 1), learned loop 1 step"),
    ("msgd2_varc_emb", "ms4ve_s5", "Meta-SGD rows, learned loop 5 steps"),
    ("msgd2_varc_emb", "ms4ve_s10", "Meta-SGD rows, learned loop 10 steps"),
    ("msgd2_varc_emb", "ms4ve_adam", "Meta-SGD rows, then ttt_mini"),
    ("msgd2_varc_lh", "ms4vl_s1", "Meta-SGD layers (alpha0 0.01), learned loop 1 step"),
    ("msgd2_varc_lh", "ms4vl_s5", "Meta-SGD layers, learned loop 5 steps"),
    ("msgd2_varc_lh", "ms4vl_s10", "Meta-SGD layers, learned loop 10 steps"),
    ("msgd2_varc_lh", "ms4vl_adam", "Meta-SGD layers, then ttt_mini"),
    ("msgd2_varc_all", "ms4va_s1", "Meta-SGD everything (alpha0 0.01), learned loop 1 step"),
    ("msgd2_varc_all", "ms4va_s5", "Meta-SGD everything, learned loop 5 steps"),
    ("msgd2_varc_all", "ms4va_s10", "Meta-SGD everything, learned loop 10 steps"),
    ("msgd2_varc_all", "ms4va_adam", "Meta-SGD everything, then ttt_mini"),
    ("varc_ho2_cont", "ms4vc_adam", "plain VARC continued 2000 steps (batch 16), then ttt_mini"),
    ("msgd2_varc_all_ft", "ms4vaf_adam", "Meta-SGD everything from the plain checkpoint, 2000 meta-steps, then ttt_mini"),
    ("msgd2_varc_all_ft", "ms4vaf_s10", "Meta-SGD everything from the plain checkpoint, learned loop 10 steps"),
    ("msgd2_varc_all_full", "ms4vafull_adam", "Meta-SGD everything from scratch, 4000 x batch 128 (baseline budget), then ttt_mini"),
    ("msgd2_varc_all_full", "ms4vafull_s10", "Meta-SGD everything from scratch at the baseline budget, learned loop 10 steps"),
    ("varc_ho2_cont2", "ms5vc_adam", "plain VARC continued 1000 steps (batch 256, lr 1e-4), then ttt_mini"),
    ("msgd3_varc_ft", "ms5va_s1", "round 3: meta-fine-tuned, 16-copy episodes, fresh rows, alpha0 rows 1 / weights 0.01; learned loop 1 step"),
    ("msgd3_varc_ft", "ms5va_s5", "round 3, learned loop 5 steps"),
    ("msgd3_varc_ft", "ms5va_s10", "round 3, learned loop 10 steps"),
    ("msgd3_varc_ft", "ms5va_adam", "round 3, then ttt_mini"),
    ("varc_ho2", "ms6v_none", "plain VARC, no adaptation (0 epochs)"),
    ("varc_ho2", "ms6v_seed1", "plain VARC, ttt_mini, eval seed 1"),
    ("varc_ho2", "ms6v_seed2", "plain VARC, ttt_mini, eval seed 2"),
    ("varc_ho2", "ms6v_views4", "plain VARC, ttt_mini, 4 canvas views pooled"),
    ("varc_ho2", "ms6v_runs2", "plain VARC, ttt_mini, 2 runs pooled"),
    ("varc_ho2", "ms6v_e40", "plain VARC, ttt_mini, 40 epochs (4 warmup)"),
    ("varc_ho2", "ms6v_aux32", "plain VARC, ttt_mini, 32 aux tasks"),
]


def metrics(run: str, suffix: str):
    paths = glob.glob(os.path.join(RUNS, run, "outputs", "*", "*", "*", "eval_preds", f"step_*_{suffix}", "ttt_metrics.json"))
    if not paths:
        return None
    d = json.load(open(sorted(paths)[-1]))
    m, n = d["metrics"], d["num_tasks"]
    return n, [round(m[f"ARC/pass@{k}"] * n) for k in (1, 2, 5, 100)], m["all/accuracy"]


def alpha(run: str):
    """(geometric mean, max) of the learned step sizes at the end of training, from the run's train.log."""
    try:
        lines = [l for l in open(os.path.join(RUNS, run, "train.log"), errors="replace") if "meta/alpha_mean" in l]
    except FileNotFoundError:
        return None
    return lines[-1].strip() if lines else None


print("| checkpoint | test-time procedure | pass@1 | pass@2 | pass@5 | pass@100 | token acc |")
print("|---|---|---|---|---|---|---|")
for run, suffix, label in ROWS:
    r = metrics(run, suffix)
    cells = ["pending"] * 5 if r is None else [*map(str, r[1]), f"{r[2]:.3f}"]
    print(f"| {run} | {label} | " + " | ".join(cells) + " |")
