# GRAM re-tune and test-time search over the prior

Branch `treerm-scaling` (continues PR #66). The pivot (user, 2026-10-18): training-time search (TreeRM, TreeGRAM's
training search, REPORTS_SCALING.md) did not work. The idea kept: at test time, K epsilons drawn from GRAM's prior
Gaussian, equal mass, refined by an MCTS with the LPRM value head, as a plug-in replacement for the GRAM paper's
parallel depth / width scaling. No head outputs a distribution over next states. Two steps: (1) re-tune the GRAM
recipe itself, which has underperformed the paper here; (2) from one GRAM checkpoint, measure the search against
naive depth and width scaling at matched compute. One GPU job at a time (4 GB laptop).

## 0. Scoreboard

| stage | what | result |
|---|---|---|
| 1 | plain GRAM baseline, N-Queens 8x8 fast tier (128 x 3000, lr 3e-4), current gram.yaml, mean guidance at eval | 12.5% test at 3000 steps, train lm loss 0.003: the memorisation gap (TRM-192 20-21%, max-steps TRM control 40.8%) |

## 1. Baseline (seed 0)

`scripts/candidates/queue.sh nqueens8 queues/gram_base.txt`: `arch=gram +arch.eval_prior_mean=true` (the key is
read by `gram.py` but defined in no yaml). 1.02 M params, 514 s. Test accuracy by 500 steps: 0 / 0 / 0.1 / 1.9 /
7.4 / 12.5; train lm loss 0.003 and KL 0.010 at the end. Research notes for the recipe gap and the test-time ladder
are under `research_notes/GRAM retune and test time search/` (untracked).
