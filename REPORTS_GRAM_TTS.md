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
| 2 | GRAM re-tune sweep on N-Queens (prior_sample_frac 0.1-0.5, per-puzzle draw, beta 0.07) | 5.5-14.0% mean guidance, 0-1% sampled, train exact 94-100%: the tier cannot test GRAM |
| 3 | anchor on held-out 4x4 Sudoku (the earlier branch's testbed), 5000 steps | multi-solution: GRAM 25.8 vs TRM 10.3 exact; unique-solution (clue 0.4): GRAM 89.2 vs TRM 99.5 |
| 4 | the ladder from the unique-solution GRAM checkpoint | vote +3 at N=16 over N=1 (93.8-94.5 vs 91); MCTS over draws 91 -> 89 with budget, 4-5 pts under the vote at matched passes; LPRM best-of-N 81-84: the value head cannot rank draws |

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

Result (seed 0, 5000 steps, held-out 1000, exact match; `queues/gram_anchor.txt`):

| run | exact by 1000 steps | params | wall |
|---|---|---|---|
| TRM | 0 / 0.5 / 9.3 / 9.1 / **10.3** | 526k | 397 s |
| GRAM, frac 0.25, sampled guidance | 0 / 0 / 1.0 / 12.9 / **25.8** | 1.02 M | 443 s |

GRAM works here as the earlier branch found (its validity metric gave 15.4 vs 52.9; exact match undercounts valid
answers on multi-solution puzzles, and the harness's vote / best-of-N also score against the label). For the ladder
a held-out *unique-solution* build (`data/sudoku-4hu`: clue_ratio 0.4, `unique_solution=true`) makes exact match the
validity for every rung.

## 5. The testbed for the ladder: held-out unique-solution 4x4 Sudoku (`data/sudoku-4hu`)

`build_sudoku_dataset grid_size=4 source_repo=null num_train=20000 num_test=1000 num_aug=0 clue_ratio=0.4
unique_solution=true seed=0` (the generated splits had ignored `unique_solution`: fixed in 7594eb5). Tier
`sudoku4u`, the section 4 recipe, 5000 steps, seed 0, exact match = validity:

| run | exact by 1000 steps | wall |
|---|---|---|
| TRM | 2.1 / 61.5 / 94.5 / 97.1 / **99.5** | 377 s |
| GRAM, frac 0.25, sampled guidance | 0.6 / 27.5 / 65.5 / 82.7 / **89.2** | 413 s |

With 6-7 clues and one solution the task is near TRM's ceiling, so GRAM is the weaker model here (the earlier
branch's GRAM advantage was on 4-clue multi-solution puzzles); it still leaves 9-11 pts of headroom for test-time
scaling, which is what the ladder measures.

## 6. The ladder, from the GRAM checkpoint (no retraining; `ladder.sh <ckpt> s|1|2`, eval seed 0)

Compute: `all/nfe` counts the plain passes only (96 at depth 8, 192 at depth 16); the search's extra passes per
step are P x (1 + S), independent of K, so S simulations match width N = S + 1 (section 2). Width N runs N copies
per puzzle with independent prior draws and random latent inits (the harness cannot share the init), scored per
copy, by majority vote and by LPRM best-of-N; the search picks the most-visited of K draws (slot 0 = the mean) with
LPRM-valued rollouts.

| method | matched width | depth 8 | depth 16 |
|---|---|---|---|
| width 1, sampled guidance (the paper's N=1) | 1 | 90.8 | 91.1 |
| width 4: per copy / vote / LPRM best-of-4 / any | 4 | 82.4 / 89.9 / 81.3 / 96.0 | 82.8 / 90.2 / 83.1 / 95.7 |
| width 8: per copy / vote / LPRM best-of-8 / any | 8 | 83.3 / 93.3 / 83.9 / 98.2 | 83.8 / 92.8 / 83.3 / 98.5 |
| width 16: per copy / vote / LPRM best-of-16 / any | 16 | 82.5 / 93.8 / 81.4 / 99.4 | 83.8 / 94.5 / 82.4 / 99.0 |
| MCTS K=4, S=4 / 8 / 16 | 5 / 9 / 17 | 91.1 / 89.5 / 89.2 | 91.1 / 89.9 / 89.1 |
| MCTS K=8, S=8 / 16 / 32 | 9 / 17 / 33 | 91.2 / 89.2 / 89.1 | 91.2 / 88.9 / 89.1 |

Search diagnostics: pick_is_best 0.64 at S=K (K=4; the pick is then the first child, the mean) falling to 0.36-0.39
at 2K-4K; K=8 0.51 -> 0.22; value_rank_corr 0.05 (K=4) and -0.01 to 0.00 (K=8); cand_ce_gap 0.008-0.015 nats.
Eval wall 7-33 s per setting after the first compile (60-88 s).

Reading:
1. **Majority vote is the method that scales**: +2.5 at N=8 and +3.0 at N=16 over a single sample (depth 16:
   +3.4), with 98-99% of puzzles solved by at least one copy. Per-copy accuracy is 7-8 pts below the single run
   because the copies also get random latent inits; the vote more than recovers it.
2. **The search does not beat width at matched compute and gets worse with budget**: 91.1-91.2 at S=K, where the
   argmax pick is the prior mean, down to 89 at 2K-4K, i.e. 4-5 pts below the vote at the same forward passes and
   below the paper's N=1. **LPRM best-of-N fails the same way** (81-84, below a single sample): GRAM's LPRM value
   head, trained as the paper trains it, cannot rank the prior's draws (rank correlation at most 0.05), so any
   selection that follows it moves away from the mean toward draws that score well on a blind value. The draws
   themselves are not the problem: the vote over the same draws works.
3. **Depth is flat** (depth 16 = depth 8 everywhere): the model halts at the 4 supervision steps it was trained for.

## 7. Where this leaves the pivot

- GRAM re-tuned: on the N-Queens 8x8 tier nothing works (the posterior memorises, the prior's draws score 0-1%;
  sections 3); on held-out 4x4 Sudoku the earlier branch's recipe (frac 0.25, beta 0.05, no posterior on the
  no-grad cycles) reproduces: 25.8 vs TRM 10.3 exact on multi-solution puzzles, 89.2 vs 99.5 on unique ones.
- The test-time search over prior draws, as a plug-in for the paper's width scaling, loses to the paper's own
  majority vote by 4-5 pts at matched compute and does not improve with budget, for the same reason LPRM best-of-N
  loses: the value head is blind to which draw is better. Until a value head ranks draws (the same gap REPORTS_SCALING
  section 12 found on MiniARC), width + vote is the test-time scaling to use, and the search is a worse way to
  spend the same passes.
- Single seed, one checkpoint, two tiers; the width baseline carries a random-init confound that works against it,
  which only strengthens the conclusion. Not run: augmentation and 10k steps (the earlier branch's 80% regime), the
  multi-solution ladder (needs validity-scored vote / best-of-N), more seeds.
