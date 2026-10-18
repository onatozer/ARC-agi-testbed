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
| 1b | ladder rung 0 on that checkpoint | TreeGRAM K=1 / 1 sim = 12.5% (the harness reproduces the mean inference); width 1 with sampled guidance (the paper's N=1) = 0.1%: the prior's draws are unusable |

## 1. Baseline (seed 0)

`scripts/candidates/queue.sh nqueens8 queues/gram_base.txt`: `arch=gram +arch.eval_prior_mean=true` (the key is
read by `gram.py` but defined in no yaml). 1.02 M params, 514 s. Test accuracy by 500 steps: 0 / 0 / 0.1 / 1.9 /
7.4 / 12.5; train lm loss 0.003 and KL 0.010 at the end. Research notes for the recipe gap and the test-time ladder
are under `research_notes/GRAM retune and test time search/` (untracked).

## 2. What the research found (notes: `gram_recipe_gap.md`, `test_time_ladder.md`)

- The repo's GRAM is not missing the paper's recipe so much as it has the earlier `gram-posterior-fixes` branch's
  fixes merged (#59: token-mean KL, no posterior on the no-grad cycles, the N-Queens evaluator) but its two measured
  levers switched off in `config/arch/gram.yaml`: `prior_sample_frac` 0 (on held-out 4x4 Sudoku, 0.25 took GRAM
  from 11.7% to 76.4% over 3 seeds, the prior's variance widening to 2.0) and beta 0.05 tuned with it (the paper's
  App. B.2 value for N-Queens is 0.07; 0.1 was seed-unstable, >= 0.2 collapsed).
- The paper (arXiv 2605.19376): N-Queens 8x8 99.5% at N=1, 100 at N=20 (its benchmark has 1-3 givens and held-out
  inputs; the repo's has 0-7 givens with 95% of test inputs in training). Test-time scaling: depth = more
  supervision steps, width = N prior-sampled trajectories combined by majority vote (Algorithm 2); LPRM best-of-N
  (App. A.4) is offered but never used for a reported number. GRAM 16 iters x N=20 costs 8.48 TFLOPs for 96.1% on
  Sudoku-Extreme vs TRM 320 iters 8.19 TFLOPs for 92.2%.
- Harness: `different_init: N` runs N copies per puzzle (random latent inits plus, with `eval_prior_mean: false`,
  independent prior draws) and votes (`different_init/majority_vote_acc`); LPRM best-of-N added here
  (`different_init/lprm_best_acc`, `evaluators/eval_fn.py`). TreeGRAM runs on a GRAM checkpoint through
  `config/eval/treegram_from_gram.yaml` (init_override of the arch name and the TreeRM keys; `loss_head_name` sets the
  TreeRM eval head since `arch.loss.name` is resolved before init_override). `all/nfe` counts no search passes; by
  the fused search's structure the search's forward passes per step are P x (1 + S), independent of K, so S
  simulations match width N = S + 1. `scripts/candidates/ladder.sh <ckpt> <rung>` runs rung 0 (checks), 1 (depth
  {8,16} x width {1,4,8,16}), 2 (TreeGRAM K {4,8} x sims {K,2K,4K} x depth {8,16}); `ladder_table.py` prints them.
- Stop criterion for "GRAM works": test exact at or above the 40.8% same-backbone max-steps control with a
  train-test gap of TRM's order; sampled eval close to mean eval (rung 0 of the baseline: 0.1 vs 12.5); KL in the low
  single-digit nats per puzzle; majority@N above N=1.

## 3. Re-tune sweep (N-Queens 8x8 fast tier, seed 0; `queues/gram_retune.txt`)

All `arch=gram +arch.eval_prior_mean=true` plus the override; 3000 steps, 555-654 s each; train metrics are the
mean of the last 200 steps from the local W&B history (`train_exact.py` in the job tmp dir; `summary.py` does not
print them).

| run | test exact (mean guidance), by 500 steps | train exact | train lm | train KL (x 0.05) |
|---|---|---|---|---|
| frac 0 (baseline) | 0 / 0 / 0.1 / 1.9 / 7.4 / **12.5** | 99.97 | 0.003 | 0.010 |
| frac 0.1 | 0 / 0 / 3.5 / 7.2 / 10.4 / **14.0** | 99.1 | 0.009 | 0.017 |
| frac 0.25 | 0 / 0 / 1.6 / 2.1 / 4.5 / **5.5** | 98.8 | 0.020 | 0.024 |
| frac 0.25, per-puzzle draw | 0 / 0 / 1.8 / 0.4 / 1.1 / **9.9** | 76.7 | 0.075 | 0.041 |
| frac 0.25, beta 0.07 | 0 / 0 / 0.8 / 3.2 / 5.4 / **5.9** | - | 0.024 | 0.032 |
| frac 0.5 | 0 / 0 / 2.5 / 4.8 / 7.1 / **11.0** | 93.9 | 0.050 | 0.036 |
| control: same backbone, max-steps head, no guidance (TreeRM K=1, no candidate loss) | **40.8** | 45.1 | 0.171 | - |

Reading: the prior-sample fraction does not close the gap on this tier at the mean-guidance eval. Every GRAM run
solves 94-100% of its training puzzles (the control 45%) and 5-14% at test: with the posterior on the grad cycle
the backbone outsources the answer to the label-carrying guidance and never learns to reason, and prior-guided
puzzles (up to half of them at frac 0.5) are still solved in training because the inputs are memorised (95% of
test inputs are in the training set, so this is not a transfer gap but a guidance gap). The runs with frac > 0
trained on prior *samples* while this eval used the prior *mean*, which the widened variance puts off-distribution:
the sampled evals (rung s: width 1, and width 8 with vote and LPRM best-of-N) are the decisive check, below.

Rung s (sampled guidance, depth 8, eval seed 0, 1000 puzzles; `ladder.sh <ckpt> s`):

| checkpoint | width 1 sampled | width 8: per-copy | vote | LPRM best-of-8 | any of 8 |
|---|---|---|---|---|---|
| frac 0 (baseline) | 0.1 | - | - | - | - |
| frac 0.1 | 0.7 | 0.5 | 0.6 | 0.7 | 3.2 |
| frac 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 |
| frac 0.5 | 1.0 | 0.6 | 1.0 | 1.2 | 4.4 |

Reading: the prior's draws are unusable on this tier whatever the prior-sample fraction, so there is nothing for
width, vote, LPRM or a search to work with; the mean is the only guidance that scores at all (5-14%), and that is
below the no-guidance control (40.8). "GRAM works as it should" is not reachable by these levers here. The one
setting where the earlier branch showed GRAM working (held-out 4x4 Sudoku, frac 0.25, beta 0.05, augmentation:
84.1% vs TRM 30.7 at N=1, 98.9% majority@20) is the anchor to replicate in this harness before the ladder.

## 4. Anchor: the setting where GRAM is known to work (held-out 4x4 Sudoku)

The `gram-posterior-fixes` report (worktree `autoreasch`, REPORT.md Part 2) showed GRAM beating TRM only on a
held-out, multi-solution testbed: 4x4 Sudoku, 4 clues, 0% of test inputs in training, 95% of test puzzles with
several solutions; hidden 128, 2 heads, H 3 / L 3, 4 supervision steps, batch 128, lr 3e-4. Unaugmented, 5000
steps: TRM 15.4% vs GRAM (frac 0.25, beta 0.05) 52.9% single-sample prior valid (seed 1: 13.1 vs 54.3); augmented,
10k steps: 30.8 vs 79.8 over two seeds, 98.4% majority@20. Its reasons the N-Queens tiers cannot test GRAM: 6x6 has
four solutions that share no queens; 8x8 here has its test inputs in training.

Rebuilt here as `data/sudoku-4ho` (`build_sudoku_dataset grid_size=4 source_repo=null num_train=20000 num_test=1000
num_aug=0 clue_ratio=0.3 seed=0`; the held-out split is in the merged builder) with a `sudoku4` tier in
`scripts/candidates/{run,queue}.sh`. Queue `queues/gram_anchor.txt`: `arch=trm` and `arch=gram
+arch.eval_prior_mean=false arch.prior_sample_frac=0.25` (sampled guidance, the paper's N=1).
