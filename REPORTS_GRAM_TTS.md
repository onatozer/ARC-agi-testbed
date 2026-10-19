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
| 6 | pooled LPRM read-out (`lprm_pool: mean`) and the paper's final-correctness target (`lprm_target: final`) | the head ranks draws (corr 0.34 / 0.67); LPRM best-of-N 97.5 / 98.8 at N=16 / 8, above the vote (96) and within 1 of the any-correct ceiling; single sample 92-94; draws on every cycle hurt |
| 7 | the search on those checkpoints | still falls with budget (95.2 -> 93.4 at K=4, 94.9 -> 92.0 at K=8) despite the head: 4-5 pts under best-of-N at equal passes; the step-wise criterion is myopic |
| 8 | the search with the full horizon (`mcts_horizon: full`: a simulation rolls through the remaining supervision steps) | improves with budget for the first time: 95.2 / 97.9 / 97.2 at S = 4 / 8 / 16 (step horizon 95.2 / 93.7 / 92.3), above the vote (96.6 at N=16), just under best-of-N (98.8), at ~5x the passes per simulation |
| 9 | MiniARC 5x5, GRAM pool+final, the ladder | GRAM is at the floor there (single sample 0.04% exact, pass@2 1.3 vs TRM 4.7); every rung 0.04-0.23%; not a usable tier for GRAM at this recipe |
| 10 | held-out 8x8 N-Queens, 1-3 givens, 3k and 10k steps | TRM 23.7 -> 68.9 (= the paper's 66.8); GRAM pool+final 0 at both (75% train exact with the posterior; mean guidance 3.2; width 8 any-valid 0.4%): the shortcut; ladder skipped |
| 11 | the discrepancy with the public and official GRAM code (sections 15-16) | ad3002: label-blind posterior (TRM + noise); pixeli99: same mechanism as ours at 33x the budget, its objective / precision / carry change nothing here; official: the KL is a token **sum** (~110x ours per token) |
| 12 | GRAM with the official KL strength, held-out N-Queens, 10k steps | 0 -> **66.2%** (TRM 68.9, the paper's TRM 66.8), train exact 97 -> 32; vote +4 at N=8 / 16 (69.7 / 70.4), any-valid 75-77% |

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

## 7. Where this leaves the pivot (final; sections 8-10 came after the first version of this section)

- GRAM re-tuned: the N-Queens 8x8 tier cannot test it (sections 3); on held-out 4x4 Sudoku the
  `gram-posterior-fixes` recipe reproduces (section 4-5), and three changes to the LPRM head (section 9) lift the
  single-sample model from 89 to 94 and make its value head rank prior draws (correlation 0.05 -> 0.67).
- Test-time scaling on that model, from one checkpoint at matched forward passes (sections 6, 9, 10): LPRM
  best-of-N 98.8 at N=8 (ceiling 99.4) > majority vote 96 > single sample 94 > the MCTS over prior draws, which
  falls with budget (95 -> 92-93) even once its value head works, because its per-step choice is myopic. The paper's
  never-used selector, with a read-out over the whole state and its own final-correctness target, is the test-time
  scaling to use; the search as built is dominated at every budget.
- Section 11 (later): with the value read at the trajectory's end (`mcts_horizon: full`) the search improves with
  budget and beats the vote (97.9 at S=8), still just under best-of-N at several times the passes per simulation.
- Single seed, one checkpoint per cell, one testbed near TRM's ceiling. Not run: more seeds, the multi-solution
  build (needs a validity-scored vote / best-of-N), augmentation and 10k steps, a search whose value is the
  trajectory's final correctness rather than the step's.

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

## 10. The search on the section 9 checkpoints (rung 2, pooled head, depth 8, eval seed 0)

The chain's first attempt had a wrong lock path (exit 66); rerun with `eval=treegram_from_gram lprm_pool=mean`.

| checkpoint | K | S = K / 2K / 4K | value_rank_corr | pick_is_best (chance 1/K) | best-of-(S+1) from section 9 |
|---|---|---|---|---|---|
| + pool | 4 | 95.8 / 93.2 / 93.8 | 0.31-0.34 | 0.54 / 0.60 / 0.60 | 97.4 (N=8) |
| + pool + final | 4 | 95.2 / 94.8 / 93.4 | 0.65-0.67 | 0.51 / 0.78 / 0.77 | 98.8 (N=8) |
| + pool + final | 8 | 94.9 / 93.6 / 92.0 | 0.59-0.60 | 0.36 / 0.71 / 0.72 | 98.3 (N=16) |
| + pool + all | 4 | 91.4 / 88.4 / 88.5 | 0.55-0.58 | 0.32 / 0.70 / 0.69 | 87.5 |
| all three | 4 | 90.3 / 88.2 / 87.3 | 0.30-0.33 | 0.40 / 0.51 / 0.51 | 96.7 |

Reading: the value head now ranks the draws (0.67 against 0.05 in section 6) and the pick is the CE-best candidate
78% of the time at K=4, yet accuracy still falls as the budget grows, and at equal forward passes the search trails
best-of-N by 4-5 pts (94.8 vs 98.8 at S=8 / N=9). With the head no longer the limit, the search's own criterion is:
it chooses, at each of the 12 z_H updates of a trajectory, the draw whose rollout decodes best at the end of that
step, and those greedy choices compound into a worse final board than either taking the mean (S=K, the best
search setting) or judging whole trajectories once by their final value (best-of-N). A search would need the
trajectory's final correctness as its leaf value (the full horizon, at 4x the passes per simulation; stripped in
#64 as `mcts_horizon=full`) to be asking the same question best-of-N asks. Not built.

## 11. The full-horizon search (user, 2026-10-18 night)

The step horizon was a training-time constraint (the KL needs the search inside the supervision step) that the eval
search had inherited. `arch.mcts_horizon: full` (commit efe80cc): at eval the outer model passes the steps left, a
simulation's allowed depth grows by H_cycles per step left, the state is rms-normed at the step boundaries as
inference does, and the LPRM scores the trajectory's end. TreeGRAM's rollouts (and GRAM's bootstrap rollouts)
now continue as mean inference does, u + mu per update, instead of plain updates. The compiled search round is
specialised per rollout length, which the full horizon turns into a 10+ min compile storm (up to 11 updates, about
a dozen shapes), so `treegram_from_gram.yaml` now runs the search eagerly.

Sudoku pool+final checkpoint (section 9), K=4, depth 8, eval seed 0, eager search:

| horizon | S=4 (= K: the mean is picked) | S=8 | S=16 | eval s at S=4 / 8 / 16 |
|---|---|---|---|---|
| step (mean-guided rollouts) | 95.2 | 93.7 | 92.3 | 60 / 9 / 9 |
| full | 95.2 | **97.9** | 97.2 | 1925 / 126 / 235 |
| for reference: vote N=8 / 16 | 95.9 / 96.6 | | | |
| for reference: LPRM best-of-8 / 16 | 98.8 / 98.3 | | | |

Reading: with the value read at the trajectory's end, the search improves with budget for the first time in this
project (+2.7 from S=4 to S=8) and beats the vote at N=8 and N=16. It stays just under LPRM best-of-N, and each
full-horizon simulation costs about five times the forward passes of a step-horizon one (a rollout of up to 11
updates instead of at most 2), so at matched passes best-of-N remains the more efficient selector; the
`pick_is_best` / `value_rank_corr` columns measure the step-level criterion and no longer describe this search. The
S=4 wall (1925 s) is an outlier of unknown cause (a shared machine; the S=8 / S=16 rows took 126 / 235 s).

## 12. MiniARC 5x5 (standard build, 4000 steps, the recipes.sh recipe plus puzzle_emb_lr 3e-2, seed 0)

`scripts/miniarc5/queue_gram.sh`: GRAM with the pooled head, the final target, frac 0.25 and the draw bootstrap,
sampled guidance (a first run died in the ARC evaluator, which requires `q_halt_logits`; GRAM now emits the LPRM
value's logit for it, as TreeRM does; 8a36eb2). 1570 s. Ladder from its checkpoint (4768 test rows, batch 149):

| | single sample | width 4 / 8 / 16 vote | LPRM best-of-4 / 8 / 16 | step search S=4 / 8 / 16 | full-horizon search S=4 / 8 / 16 |
|---|---|---|---|---|---|
| exact | 0.04 | 0.06 / 0.06 / 0.06 | 0.08 / 0.08 / 0.08 | 0.15 / 0.08 / 0.08 | 0.15 / 0.23 / 0.17 |
| pass@2 (in-training eval) | 1.3 | | | | |

Eval wall: width 29-145 s, step search 120-200 s, full-horizon search 1968-3928 s per setting.

Reading: GRAM at this recipe does not work on MiniARC (TRM and TreeRM reach pass@2 4.7 here; GRAM 1.3, token
accuracy 56%), so every rung is at the floor and the value head's rank correlation is negative; no test-time
procedure can be measured on it. GRAM needs its own tuning for ARC-style tasks (the paper trains it at ~10 M
params for 50% on ARC-1), which this loop did not attempt. The full-horizon search is also impractical at this
size without the compiled round (up to 65 min per setting).

## 13. 8x8 N-Queens the paper's way (held-out inputs, 1-3 givens; `data/nqueens-8ho`, tier `nqueens8ho`)

Rebuilt with the fixed builder (`build_nqueens_dataset board_size=8 num_train=8000 num_test=1000 num_aug=0
min_given_queens=1 max_given_queens=3 seed=0`; the builder's held-out set had been taken after the inputs were
re-encoded and so had never excluded anything, dab6e9c): 0 test inputs in training; the test set holds only 2-3
givens because all 64 one-queen inputs occur in the 8000 training puzzles. Scored by `NQueens/accuracy` (a valid
board that keeps the givens), the N-copy rungs by the new validity metrics. The nqueens8 recipe (128 x 3000, lr
3e-4), seed 0:

| run | accuracy by 500 steps | wall |
|---|---|---|
| TRM | 0 / 0 / 0 / 0.3 / 6.1 / **23.7** (still rising) | 482 s |
| GRAM, pooled head + final target, frac 0.25, bootstrap, sampled guidance | 0 / 0 / 0 / 0 / 0 / **0** | 1600 s |

GRAM has not taken off by 3000 steps on this tier (the earlier branch saw late, seed-sensitive take-offs on
held-out Sudoku, and TRM itself is far from converged here), so its ladder (rung 1 and the step search were run
before this was seen) measures nothing. Next: the same pair at 10k steps.

At 10k steps (tier `nqueens8ho10k`: eval every 2000; checkpoints kept):

| run | accuracy by 2000 steps | train exact (last 200 steps) | wall |
|---|---|---|---|
| TRM | 0.1 / 52.4 / 61.4 / 65.8 / **68.9** | 24.5 | 1606 s |
| GRAM pool+final, frac 0.25, bootstrap | 0 / 0 / 0 / 0.3 / **0** (mean guidance 3.2; width 8 sampled: any valid 0.4%) | 74.7 (posterior) | 5006 s |

Reading: TRM lands on the paper's TRM number (68.9 vs 66.8 +- 5.6) on this rebuilt benchmark, so the testbed is
the paper's. GRAM does not: with the posterior it solves 75% of training puzzles at 98% token accuracy, with the
prior it produces no valid board (token accuracy 88%, every board a near miss), at 3x TRM's wall. This is the
section 3 shortcut on a held-out task: a multi-solution problem lets the label-carrying guidance pick the solution
in training, and the prior sample fraction (0.25) that cures it on 4x4 Sudoku does not here. The earlier branch
had the same experience on 8x8 N-Queens ("none of the priors had started solving") before moving to 4x4 Sudoku;
the paper's GRAM reaches 99.5% at ~10 M params, batch 768, ~28k steps. The ladder was skipped (nothing to scale).

## 14. Where the project stands (final for this session)

- **Test-time scaling on a GRAM that works** (held-out unique 4x4 Sudoku, one checkpoint, seed 0): LPRM best-of-N
  with the pooled head and the paper's final-correctness target is the efficient selector (98.8 at N=8, ceiling
  99.4); majority vote 96; the search over prior draws is the only search in this project that improves with
  budget once its value is read at the trajectory's end (`mcts_horizon: full`: 95.2 -> 97.9 at S=8, above the vote)
  but each simulation costs ~5x a step-horizon one, so best-of-N wins at matched passes. The step-horizon search is
  dominated everywhere.
- **GRAM's reach**: it works on held-out 4x4 Sudoku (multi- and unique-solution) and not, at this scale and
  recipe, on held-out 8x8 N-Queens (0%; the posterior shortcut) or MiniARC 5x5 (pass@2 1.3 vs TRM 4.7). Both
  would need GRAM-side work (the prior-sample mechanism, scale, steps) before any test-time ladder means anything.
- **Harness kept on the branch**: `ladder.sh` / `ladder_table.py`, `gram_width.yaml` (draws-only width, mean or
  sampled, vote, LPRM best-of-N, validity-scored variants), `treegram_from_gram.yaml` (step or full horizon,
  eager), tiers `sudoku4`, `sudoku4u`, `nqueens8ho`, `nqueens8ho10k`, MiniARC queues; builders fixed (unique
  Sudoku, held-out N-Queens); GRAM emits `q_halt_logits`.
- Single seed throughout. Not run: a second seed of the Sudoku ladder, augmentation, the paper's scale, a
  compiled full-horizon round (per-length shapes), GRAM tuning for N-Queens / ARC.

## 15. Why the public re-implementations learn N-Queens and ours does not (user, 2026-10-19)

Two researchers diffed https://github.com/ad3002/gram and https://github.com/pixeli99/GRAM (commit 7639e5a)
against our GRAM, file by file (notes `diff_ad3002.md`, `diff_pixeli99.md`). The KL (balanced, alpha 0.8, summed
over the hidden dim, averaged over tokens), the reparameterisation and variance floor, guidance on every
transition, the posterior on the grad transition only and prior draws on the no-grad ones, and the accuracy
definition are the same on all three sides. The differences:

**ad3002/gram (89.5% held-out N-Queens at 1500 steps, 11 M params)**: its posterior cannot see the label. Its
default `posterior_y_encoding: decoder_embed_mean` averages the label's embedding over the 64 positions, and every
valid N-Queens board has exactly 8 queens and 56 empties, so that vector is the same for every puzzle (their own
docstring: completions differ by 3e-7 on a norm of 10.7; KL 0.007 nats). Their GRAM is TRM plus learned,
label-blind noise, and its prior trivially matches its posterior. Our per-position label is the arm they also
built (`per_position`): 29.5% test at 40x our budget, and 0.0% held-out when the no-grad transitions sampled the
posterior; they call both readings broken. Second-order: they train one canonical (lex-min) completion per unique
input; our build draws a random solution per row (568 of 2575 unique training inputs carry two or more targets).

**pixeli99/GRAM (98.1% held-out, 10.5 M params, ~27k optimizer steps x 768 x 16 supervision steps, ~33x our
puzzle-steps)**: the same per-position mechanism as ours, with: CE + beta KL only (no LPRM loss, no prior-sample
fraction, no bootstrap); fp32 / TF32, with a code comment that bf16 autocast "stalls the recursion because gradients
to the zero-init Gaussian heads round to zero" (ours runs those heads in bf16); the raw state carried between
supervision steps (rms-normed only in the decode; ours carries the normed state); lr 1e-4 constant, EMA 0.9999,
clip 1.0, 16 supervision steps, separate H and L blocks; data enumerating all 92 solutions x all 1/2/3-queen subsets
with every consistent solution as a row and 15% of unique inputs held out (their deterministic baseline 64.1 vs our
TRM 68.9). No learning curve is published, so their take-off step is unknown.

**A trap in ours**: `+arch.eval_prior_mean=true` also removed the draws from training's no-grad cycles (every
section 3 run); fixed (the key now applies at eval only). The held-out runs of section 13 used `false`.

Options added to test the candidates (commits 4570d25, 6866ecb): `posterior_y_pool: position | mean` (ad3002's
label-blind posterior), `lprm_weight` (0 = pixeli99's objective), `carry_norm: true | false` (the raw carry),
`forward_dtype=float32` (existing). Runs on the held-out tier at 3000 steps, sampled guidance (TRM 23.7, GRAM
pool+final 0 at this budget):

| run | what | accuracy at 3000 | train exact |
|---|---|---|---|
| gram_ypool | label-blind posterior, otherwise plain | 0 / 0 / 0 / 0 / 0.5 / **4.4** (624 s) | 10.2 (KL 0.003 nats: posterior = prior) |
| gram_ypool_pf | label-blind posterior + pooled head, final target, frac 0.25, bootstrap | 0 at every eval (3008 s) | 0.0 |
| gram_plain | pixeli99 objective (no LPRM loss), bf16 | 0 at every eval (1138 s) | 96.7 (KL 0.058) |
| gram_fp32 | the same in fp32 | 0 at every eval (1347 s) | 96.4 (KL 0.056) |
| gram_raw | the same with the raw carry | 0 at every eval (1189 s) | 99.3 (KL 0.031) |

Reading: the label-blind posterior removes the shortcut (train exact 10%, KL ~0) and the prior path starts to
learn (4.4% at 3000 steps, behind TRM's 23.7), which is the mechanism behind ad3002's numbers: their GRAM is TRM
plus learned noise. None of pixeli99's recipe differences matters at our budget: without the LPRM loss and the
prior-sample fraction, in bf16 or fp32, with the normed or the raw carry, the faithful posterior memorises 96-99% of
the training puzzles and the prior produces no valid board. What separates pixeli99's 98% from our 0% is therefore
not in the objective, precision or carry but in scale and budget (10.5 M params, ~33x the puzzle-steps) or in the
data (all consistent solutions as rows), neither affordable here; the official release is diffed next.

## 16. The official release (https://github.com/ahn-ml/gram-release; notes `diff_official.md`)

The official GRAM is this codebase's own shape (an HRM/TRM fork: one shared 2-layer block, H 3 / L 4, AdamATan2,
bf16 backbone, stablemax CE, ACT halting, 16-token puzzle prefix, EMA 0.999) with a per-token label-conditioned
posterior like ours (one-hot y concatenated with u_t; `gram.py:435, 467`). The load-bearing difference is the
**KL reduction**: `compute_kl` (`models/losses.py:125, 146-151, 184`) is a plain `torch.sum` over batch x tokens x
the 256 latent dims, set against a per-row token-mean CE, with beta 0.07 (`scripts/nqueens_8.sh:14`); per content
token that weights the KL 0.07 x 64 = 4.5 nats for nats. Ours averages the KL over the 80 tokens and multiplies by
0.05 (`gram.py:131-139`): 0.04 per content token, about 110x weaker. The posterior KL our runs settle at (0.2-0.8
nats per token, 16-66 per puzzle) would cost 1-4 nats of loss per row under the official weighting against a CE
saving of at most ~0.17, so the label shortcut is a net loss in their objective and nearly free in ours. Both
public re-implementations (and the earlier `gram-posterior-fixes` branch, whose "fix 1" moved the KL from a token
sum to a token mean as "64x too strong") made our choice, not the official one.

Also settled: the Gaussian heads run in fp32 inside the bf16 forward and are default-initialised (not zeroed);
logvar parameterisation, no floor, no temperature; the posterior is sampled on every transition in training,
including the no-grad cycles (`gram.py:420-473`), the prior on all at eval; no free bits, warm-up, annealing,
dropout or stop-gradient; a 256-dim latent projected and added to u_t; no value head on N-Queens (`use_value_head:
False`; the official N-Queens objective is CE + 0.07 KL + 0.5 halt BCE); raw carry; 45,000 optimizer steps x 768 at
hidden 512 (27x our supervision-step samples); accuracy = any valid board (ours also requires the givens kept);
data = the paper's enumeration (all 92 solutions x 1-3-queen subsets, every consistent solution as a row, 15% of
the 5,148 unique inputs held out). No curves or take-off step are published.

Test (plain objective: `lprm_weight=0`, no prior-sample fraction, no bootstrap, sampled guidance; held-out tier,
3000 steps, TRM 23.7):

| run | kl_strength | posterior on no-grad cycles | accuracy by 500 steps | train exact |
|---|---|---|---|---|
| gram_plain (section 15) | 0.05 | no | 0 at every eval | 96.7 |
| gram_kl56 | 5.6 (0.07 x 80 tokens) | no | 0 / 0 / 0 / 0 / 0.3 / **2.6** (807 s) | 12.9 (KL 0.015) |
| gram_kl45 | 4.48 (0.07 x 64 content tokens) | no | 0 / 0 / 0 / 0 / 0.1 / **3.0** (1182 s) | 12.7 (KL 0.030) |
| gram_kl56_pnc | 5.6 | yes (the official training) | 0 / 0 / 0 / 0 / 0 / **0.6** (1479 s) | 11.8 (KL 0.014) |

Reading: at the official KL strength the shortcut is gone (train exact 97 -> 12-13%, TRM's range; KL 0.015-0.03)
and the faithful posterior's prior path produces valid boards for the first time (2.6-3.0% at 3000 steps, taking
off between 2500 and 3000; TRM 23.7). Sampling the posterior on the no-grad cycles too, as the official training
does, is slightly behind at this budget (0.6). The 110x KL weighting is the difference between our GRAM and the
paper's; whether the prior path then catches TRM and passes it (the paper: 99.5 vs 66.8) is a question of budget,
tested next at 10k steps (TRM 68.9).

At 10k steps (tier `nqueens8ho10k`, the plain objective, `kl_strength=5.6`):

| run | accuracy by 2000 steps | train exact | wall |
|---|---|---|---|
| TRM (section 13) | 0.1 / 52.4 / 61.4 / 65.8 / **68.9** | 24.5 | 1606 s |
| GRAM, our KL (section 13, with the pooled head etc.) | 0 / 0 / 0 / 0.3 / **0** | 74.7 | 5006 s |
| GRAM, the official KL strength | 0 / 22.7 / 49.5 / 61.2 / **66.2** (still rising) | 32.3 (KL 0.014) | 3776 s |

Reading: one config value takes the faithful GRAM from 0 to 66.2% on held-out N-Queens, level with TRM (68.9) and
with the paper's TRM (66.8 +- 5.6), the posterior no longer memorising (train exact 32 vs 97 / 75). The paper's
GRAM is 99.5 there at ~27x the samples and hidden 512; whether the crossover appears at our scale with more steps
is open (the curve has not flattened). The reduction is now a config option, `kl_reduction: mean | sum`
(commit below), so the paper's recipe is `kl_reduction: sum, kl_strength: 0.07` rather than a token-count-scaled
beta; the default stays `mean` so every earlier number keeps its meaning.

## 17. Where the discrepancy investigation leaves things

- **Why our GRAM scored 0 on N-Queens**: the KL was ~110x too weak per token (a token mean x 0.05 against the
  official token sum x 0.07), so the label-conditioned posterior could carry the answer at almost no cost and the
  prior never had to learn. The `gram-posterior-fixes` branch's "fix 1" introduced the mean as a correction of a
  "64x too strong" KL; it was the official strength. Both public re-implementations use the mean; ad3002's works
  on N-Queens only because its posterior is label-blind there.
- **What the fix restores**: held-out N-Queens 0 -> 66.2 at 10k steps (TRM 68.9), with the faithful per-position
  posterior; the posterior-on-no-grad-cycles variant and the content-token beta (4.48) behave the same at 3k.
- **What it means for the rest of this file**: the Sudoku ladder (sections 6-11) was run on a GRAM trained with
  the weak KL; its single-sample / vote / best-of-N / search numbers stand as measured, but the model under them
  was partly shortcutting too (train exact 100% there), so the ladder should be repeated on a `kl_reduction: sum`
  checkpoint before any number is quoted as GRAM's. Not done in this session.
- Still open: the paper's 99.5 needs its scale or budget; the vote rung on the 66.2 checkpoint (next); a value
  head trained alongside (`lprm_weight`) at the official KL, since best-of-N and the search need one.

The paper's width scaling on the 66.2 checkpoint (prior draws, draws-only copies, validity-scored; the LPRM head
was not trained in this run, `lprm_weight=0`):

| width | per copy valid | majority vote valid | LPRM best-of-N valid (untrained head) | any copy valid |
|---|---|---|---|---|
| 1 | 66.1 | | | |
| 8 | 66.4 | **69.7** | 66.9 | 75.4 |
| 16 | 66.4 | **70.4** | 65.0 | 76.6 |

Reading: on the restored GRAM the paper's width scaling works on N-Queens as it did on Sudoku: +3.6 / +4.2 valid
boards by voting 8 / 16 prior draws, with 75-77% of puzzles solved by at least one draw (the paper: 99.5 -> 100 at
N=20 with coverage 89.7). The 9-10 pt gap between the vote and any-valid is what a trained selector (LPRM best-of-N
with the pooled head, or the full-horizon search) would have to close; this run trained no value head.

## 18. Replication attempt (user, 2026-10-19): the paper's N-Queens construction and recipe

Data: `build_nqueens_dataset ... enumerate_all=true heldout_frac=0.15 min_given_queens=1 max_given_queens=3`
(`data/nqueens-8off`): every one of the 92 solutions x every 1-3-queen subset is an input, every consistent solution
a row, 15% of the unique inputs held out: 7,164 train rows over 4,376 inputs, 1,300 test rows over 772 inputs
(8,464 pairs, 5,148 inputs, the paper's counts), 1-3 givens in the test split, no overlap.

Recipe: the official one (hidden 512, 8 heads, H 3 / L 6, 16 supervision steps, batch 768, lr 1e-4, 2000 warm-up,
45k steps; the user's 2026-10-10 W&B run `GRAM-Nqueens-8x8` matched it on another machine) is **out of memory on
the 4 GB laptop GPU** at the first step (13.1 M params; `pretrain.py` has no gradient accumulation), and at its
cost (~100x a hidden-128 / batch-128 step) 45k steps would take days here. The attempt is therefore at the
laptop's scale: our backbone (hidden 128, 1.0 M params), the official objective (`kl_reduction: sum`, beta 0.07,
posterior on every transition, no value head), on the paper's build:

| run | steps | accuracy (valid board keeping the givens) | train exact |
|---|---|---|---|
| TRM | 10k | cancelled by the user (the chain was stopped 15 min in) | |
| GRAM, official objective | 10k | cancelled | |
| GRAM, official objective | 30k | cancelled (the crossover with budget alone was the question: at 10k on the random build GRAM was still rising, TRM flat) | |
| + vote over 8 / 20 draws | | cancelled | |

The paper: TRM 66.8 +- 5.6, GRAM 99.5 +- 0.3 (N=1), 100 (N=20).
