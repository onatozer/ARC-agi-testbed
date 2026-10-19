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
| 4 | the ladder from the unique-solution GRAM checkpoint | draws-only width: vote +3.3 at N=16 over N=1 (94.2 vs 91); MCTS over draws 91 -> 89 with budget, 3-5 pts under the vote at matched passes; LPRM best-of-N level with one sample: the value head cannot rank draws |
| 5 | value bootstrapping of the LPRM head on prior draws (`lprm_prior_rollouts: 4`), same ladder | rank correlation 0.05 -> 0.12; LPRM best-of-N still level with one sample (86-88), MCTS still 89 -> 86 with budget, vote +3.3 (91.5-91.8 at N=16); single sample 88 vs 89-91 |

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
per puzzle that differ only by their prior draws (`different_init_reset_std: null`, added in c5cf386: the harness
had always given every copy a random latent init as well, which is not the paper's width and not what the search's
K draws do; the first pass of this rung with that init had per-copy 82-84, vote 90-94.5, LPRM best-of-N 81-84),
scored per copy, by majority vote and by LPRM best-of-N; the search picks the most-visited of K draws (slot 0 = the mean) with
LPRM-valued rollouts.

| method | matched width | depth 8 | depth 16 |
|---|---|---|---|
| width 1, sampled guidance (the paper's N=1) | 1 | 90.8 | 91.1 |
| width 4: per copy / vote / LPRM best-of-4 / any | 4 | 90.5 / 93.3 / 92.1 / 96.9 | 90.6 / 92.2 / 90.4 / 96.8 |
| width 8: per copy / vote / LPRM best-of-8 / any | 8 | 90.0 / 92.6 / 91.1 / 97.7 | 90.5 / 93.7 / 90.8 / 97.9 |
| width 16: per copy / vote / LPRM best-of-16 / any | 16 | 90.4 / 94.2 / 90.2 / 98.7 | 90.9 / 94.2 / 91.4 / 98.7 |
| MCTS K=4, S=4 / 8 / 16 | 5 / 9 / 17 | 91.1 / 89.5 / 89.2 | 91.1 / 89.9 / 89.1 |
| MCTS K=8, S=8 / 16 / 32 | 9 / 17 / 33 | 91.2 / 89.2 / 89.1 | 91.2 / 88.9 / 89.1 |

Search diagnostics: pick_is_best 0.64 at S=K (K=4; the pick is then the first child, the mean) falling to 0.36-0.39
at 2K-4K; K=8 0.51 -> 0.22; value_rank_corr 0.05 (K=4) and -0.01 to 0.00 (K=8); cand_ce_gap 0.008-0.015 nats.
Eval wall 7-33 s per setting after the first compile (60-88 s).

Reading:
1. **Majority vote is the method that scales**: +2.5 at N=4, +1.8/+2.6 at N=8 and +3.3 at N=16 over a single
   sample, with 97-99% of puzzles solved by at least one copy; per-copy accuracy stays at the single sample's.
2. **The search does not beat width at matched compute and gets worse with budget**: 91.1-91.2 at S=K, where the
   argmax pick is the prior mean, down to 89 at 2K-4K, i.e. 3-5 pts below the vote at the same forward passes and
   below the paper's N=1. **LPRM best-of-N is level with a single sample** (90.2-92.1): GRAM's LPRM value head,
   trained as the paper trains it, carries no information about which draw is better (rank correlation at most
   0.05), so selecting by it is a random pick among the copies, and a search that follows it step by step drifts
   from the mean. The draws themselves are not the problem: the vote over the same draws works.
3. **Depth is flat** (depth 16 = depth 8 everywhere): the model halts at the 4 supervision steps it was trained for.

## 7. Where this leaves the pivot

- GRAM re-tuned: on the N-Queens 8x8 tier nothing works (the posterior memorises, the prior's draws score 0-1%;
  sections 3); on held-out 4x4 Sudoku the earlier branch's recipe (frac 0.25, beta 0.05, no posterior on the
  no-grad cycles) reproduces: 25.8 vs TRM 10.3 exact on multi-solution puzzles, 89.2 vs 99.5 on unique ones.
- The test-time search over prior draws, as a plug-in for the paper's width scaling, loses to the paper's own
  majority vote by 3-5 pts at matched compute and does not improve with budget, for the same reason LPRM best-of-N
  gains nothing: the value head is blind to which draw is better. Until a value head ranks draws (the same gap REPORTS_SCALING
  section 12 found on MiniARC), width + vote is the test-time scaling to use, and the search is a worse way to
  spend the same passes.
- Single seed, one checkpoint, two tiers. Not run: augmentation and 10k steps (the earlier branch's 80% regime), the
  multi-solution ladder (needs validity-scored vote / best-of-N), more seeds.

## 8. Value bootstrapping of the LPRM head on prior draws (user, 2026-10-18 evening)

In TreeRM the bootstrap trained the value head on the training search's rollout leaves (REPORTS.md); it never
touched the backbone (the leaves are detached), and REPORTS_SCALING.md's K=1 control had it on with nothing to pick
between, so its old "+8/+14" is confounded. GRAM has no training search, and its LPRM is trained by the loss head
on the grad cycle's posterior-guided state, which is label-informed: the head never sees the prior's draws it is
asked to rank at test (except on the prior-sample puzzles). The analogue, `arch.lprm_prior_rollouts: K` (commit
0207646, `GRAM_Inner._lprm_rollout_loss`): at each grad cycle K draws of eps from the prior at u_t, each decoded
without gradient, and the head trained by `value_bootstrap_weight` x MSE to predict each draw's token accuracy
from its first token. Run `gram_u_vb` (section 5 recipe plus K=4, 427 s) and the ladder from its checkpoint:

| method | matched width | depth 8 | depth 16 |
|---|---|---|---|
| width 1, sampled guidance | 1 | 87.9 | 88.5 |
| width 4: per copy / vote / LPRM best-of-4 / any | 4 | 86.7 / 89.3 / 86.7 / 95.3 | 86.9 / 89.3 / 87.6 / 95.2 |
| width 8: per copy / vote / LPRM best-of-8 / any | 8 | 87.4 / 91.3 / 88.4 / 97.7 | 86.8 / 89.7 / 86.3 / 97.6 |
| width 16: per copy / vote / LPRM best-of-16 / any | 16 | 86.8 / 91.5 / 85.8 / 98.7 | 87.2 / 91.8 / 87.4 / 98.4 |
| MCTS K=4, S=4 / 8 / 16 | 5 / 9 / 17 | 89.1 / 85.8 / 86.7 | 89.1 / 85.8 / 85.6 |
| MCTS K=8, S=8 / 16 / 32 | 9 / 17 / 33 | 89.0 / 86.1 / 86.0 | 89.1 / 86.1 / 86.1 |

value_rank_corr 0.11-0.13 at K=4 and 0.08-0.10 at K=8 (0.05 / 0.00 without the bootstrap); pick_is_best 0.70 at
S=K (K=4) falling to 0.37-0.39; cand_ce_gap 0.009-0.017.

Reading: the bootstrap gives the head a real but weak ranking signal (the correlation doubles to ~0.12), not enough
to make any selection pay: LPRM best-of-N stays level with a single sample and the search still loses ~3 pts as its
budget grows, while the vote gains its usual +3.3 at N=16. The single-sample accuracy is ~2 pts lower than the run
without the bootstrap (87.9-88.5 vs 89.2-91.1), inside one seed's noise but to be checked with a second seed before
reading it as a cost. A head that ranks draws needs more than this target: the draw's first token after one
rms-norm is a thin view of a 16-token state, and token accuracy of the draw itself is a noisy label for a 4-step
trajectory's outcome; the paper's target (final correctness from every intermediate state) or a head over the
whole state are the next things to try, neither built.

## 9. Three changes to the LPRM head (user, 2026-10-18 night): pooled read-out, draws on every cycle, final target

Built (commit 308eba7), each behind a key on `arch` (gram.yaml): `lprm_pool: mean` (the head reads the mean over
tokens of the rms-normed state instead of its first token, through one `_lprm_token` / `_lprm_value` that the loss
head, the bootstrap, TreeRM's search leaves and the pick diagnostics all use; at a step's last cycle the search
reads the head on the draw itself, whose noise on the other tokens the first token cannot see);
`lprm_rollout_cycles: all` (the section 8 bootstrap draws K prior draws on every cycle, each rolled out with plain
updates to the end of the step, the search's rollout policy, instead of on the grad cycle only); `lprm_target:
final` (the loss head regresses the value onto the trajectory's final correctness, the paper's App. A.4 target:
each row's remaining supervision steps rolled out without gradient as inference runs them, `GRAM._final_correct`,
instead of this step's exact match). All on top of section 8's run (frac 0.25, bootstrap K=4); unique 4x4 build,
5000 steps, seed 0; the ladder's width rung with draws-only copies.

| run | wall | single sample | vote N=4 / 8 / 16 | LPRM best-of-4 / 8 / 16 | any of 16 |
|---|---|---|---|---|---|
| section 8 (bootstrap, first token) | 427 s | 87.9 / 88.5 | 89.3 / 91.3 / 91.5 | 86.7 / 88.4 / 85.8 | 98.7 |
| + `lprm_pool: mean` | 1107 s | 92.3 / 92.5 | 94.9 / 95.8 / 96.0 | 96.6 / 97.4 / 97.5 | 99.0 |
| + pool + `lprm_target: final` | 646 s | **94.0 / 92.9** | 96.0 / 95.9 / 96.6 | 97.4 / **98.8** / 98.3 | 99.4 |
| + pool + `lprm_rollout_cycles: all` | 1325 s | 88.9 / 89.6 | 92.0 / 93.4 / 93.3 | 87.8 / 87.5 / 85.2 | 98.5 |
| all three | 979 s | 86.9 / 87.5 | 89.5 / 92.0 / 92.0 | 94.4 / 96.7 / 97.7 | 99.4 |

(depth 8 / depth 16 for the single sample; the N columns are depth 8, depth 16 within 0.7.)

Reading:
1. **The pooled read-out is the fix.** With the head reading every token, LPRM best-of-N goes from below a single
   sample to 97.5% at N=16, 1.5-2 pts above the vote and within 1.5 of the any-correct ceiling. The single-sample
   model also gains 4-5 pts (92.3 vs 87.9): the pooled value loss is a better auxiliary signal for the backbone.
2. **The final-correctness target adds on top**: single sample 94.0, best-of-8 98.8, 0.6 under its ceiling, the
   best numbers of the loop. The paper defined this target and never used it.
3. **Draws on every cycle hurt** (single sample -3, best-of-N back below the vote): rolled-out depth-1 and depth-2
   states mixed with depth-0 draws under one target confuse the head and the backbone, at 2.5x the wall; dropped.
4. The search rung failed on every checkpoint (exit 66, below) and is rerun in section 10.
