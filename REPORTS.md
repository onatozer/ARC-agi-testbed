# TreeRM autoresearch report

**Question.** Can TreeRM (MCTS over K candidate z_H updates, trained with GRAM's ELBO) beat TRM and GRAM at equal
parameter count and equal training budget on a small N-Queens setting? If so, which hyper-params / training tricks
make it work?

This document is the running log: setup decisions, every experiment, and the reasoning between them. Newest
entries are at the bottom of "Iteration log". The "Results" table is kept current.

---

## 1. Setup decisions

### Benchmark: 8x8 N-Queens, not 6x6
The request said 6x6. I built it first and checked it: 6x6 has only **4 solutions**, so the builder yields just 249
distinct puzzles, and with the default `num_given_queens: null` (0..N-1 random givens) **every test puzzle also
appears in train**. Accuracy there measures memorisation of 4 boards; any arch would saturate it and the comparison
would be uninformative. 8x8 has 92 solutions and is the GRAM paper's own N-Queens setting, so I used that.

Caveats that still apply to 8x8:
- The builder seeds train and test from one rng, so a single build makes test == the first 1000 train puzzles. I
  built twice (seeds 0 and 1) and took train from one, test from the other (`data/nqueens-8` in the main checkout).
- Even so 94.8% of test inputs occur in train: the whole 8x8 puzzle space is only ~92 solutions x 2^8 subsets of
  givens. So this benchmark measures *solving an ambiguous puzzle* (many givens-subsets admit several valid
  completions), not generalisation to unseen boards. That's fine for an arch comparison but should be kept in mind.
- Train = 10k puzzles x (1 + 8 dihedral augs) = 90k rows. Test = 1000 puzzles, givens uniform over 0..7.

6x6 Sudoku would also have been an option; I didn't explore it because N-Queens was already working and the
evaluator (`NQueens/accuracy`) gives a clean "any valid completion keeping the givens" metric, which is the right
thing for an ambiguous-solution task. Exact-match against the stored label is also reported.

### Parity
- **Parameter parity** at ~0.87–0.89M:
  | arch | hidden | params |
  |---|---|---|
  | TRM | 192 | 886,274 |
  | GRAM | 112 | 865,313 |
  | TreeRM (K=4) | 128 | 886,402 |

  TreeRM's extra parameters are the candidate generator (`2H -> H -> SwiGLU -> K*H`) and the cross-attention
  scorer; at hidden 128 they cost about what TRM gets from going 128 -> 192 wide. That's a real handicap for
  TreeRM's backbone (128 vs 192), which is the honest comparison the user asked for.
- **Budget parity**: 3000 optimizer steps, global batch 128 (384k samples, ~4.3 epochs over the 90k rows), warmup
  100, EMA 0.999, AdamATan2, weight decay 1.0. Eval on the full 1000-puzzle test set at steps 1000/2000/3000, EMA
  weights, every model runs all `halt_max_steps=8` supervision steps at eval.
- Shared backbone settings: L_layers=2, H_cycles=3, L_cycles=4, halt_max_steps=8, num_heads=4, expansion=4,
  RoPE, bf16, no puzzle embedding (`puzzle_emb_ndim=0`, as the N-Queens config sets).
- TreeRM does more compute per step (the search). Wall-clock is reported next to accuracy so the cost is visible;
  "same budget" here means same optimizer steps / samples seen, as requested.
- Hardware: one RTX 3050 Ti laptop GPU (4 GB). This caps what's feasible: ~15 min per TRM run, ~40 min per TreeRM
  run at K=4, 8 simulations.

### Harness
Runs are launched from a scratch directory with `+wandb_mode=offline`; eval rows are parsed back out of the offline
`.wandb` file (`summary.py`). Nothing is uploaded. Each run's config and log are under
`scratchpad/runs/<name>/`.

---

## 2. Engineering changes to TreeRM made during the loop

These are changes to the model code that were needed to run the experiments at all, not hyper-parameter choices.

1. **Lockstep batched MCTS** (commit `5e710b3`). The proof-of-concept ran one Python MCTS per batch row,
   sequentially, each rollout a separate forward pass. At batch 128 that was **40 s per training step** (vs 0.15 s
   for TRM), i.e. 33 hours for 3000 steps. The fix keeps one tree per puzzle but advances all trees in lockstep:
   each simulation, every tree selects its leaf, then *one* batched call expands all leaves and *one* batched call
   runs all rollouts (padding shorter rollouts by evaluating them at their own depth inside the loop). Semantics
   are unchanged (same tests pass); **0.8 s per step**, ~50x faster. Search cost is now
   `mcts_simulations x (rollout depth) x L-cycle passes` per z_H update, batched over the puzzles.

2. **`candidate_residual` option.** With full-vector candidates the generator has to reproduce the whole z_H from
   scratch at init, so early in training every candidate is far from the plain TRM update `u_t` and the search is
   choosing among four bad states. `candidate_residual: true` makes candidates `u_t + delta_k`, so at init the model
   is roughly a TRM with small perturbations and the search can only help. Off by default (the user's outline was
   full vectors); tested as a variant.

---

## 3. Results

`acc` = `NQueens/accuracy` on the 1000-puzzle test set (valid 8-queens board keeping every given), at 1k / 2k / 3k
steps. `exact` = exact match with the stored label at 3k. Wall-clock is the full run including evals.

| run | arch | params | lr | acc @1k | acc @2k | acc @3k | exact @3k | wall | notes |
|---|---|---|---|---|---|---|---|---|---|
| trm192_lr3e-4 | TRM | 886k | 3e-4 | 0.008 | 0.132 | 0.210 | 0.210 | 12.3 min | baseline |
| trm192_lr1e-3 | TRM | 886k | 1e-3 | | | | | | baseline |
| gram112_lr3e-4 | GRAM | 865k | 3e-4 | | | | | | baseline |
| gram112_lr1e-3 | GRAM | 865k | 1e-3 | | | | | | baseline |
| treerm128_base | TreeRM | 886k | 3e-4 | | | | | | K=4, 8 sims, horizon=step |
| treerm128_lr1e-3 | TreeRM | 886k | 1e-3 | | | | | | |
| treerm128_resid | TreeRM | 886k | 3e-4 | | | | | | candidate_residual |
| treerm128_full | TreeRM | 886k | 3e-4 | | | | | | horizon=full |

---

## 4. Iteration log

### 0. Timing and sanity (before the loop)
- TRM hidden 128, 200 steps: 6.7 it/s, 0.875 token accuracy, 0 boards solved. Fine as a smoke test.
- TreeRM hidden 128, K=4, 8 sims, batch 128, original per-row search: 40–50 s/it. Not runnable at budget.
  -> engineering change 1.
- After batching: 1.2 it/s. Loss finite, KL ~0.004 after 20 steps (prior already close to the visit distribution;
  with 8 simulations over 4 children the visit distribution is coarse — see question (b) below).

### 1. Baselines (queue q1, running)
TRM and GRAM at parity, lr in {3e-4, 1e-3}. The lr sweep is there because TRM's default 1e-4 is for 50k-step
runs; at 3k steps a higher lr is probably needed, and I don't want to handicap the baselines.

TRM 3e-4: 0.8% / 13.2% / 21.0% at 1k / 2k / 3k, 12.3 min. Still climbing steeply at 3k, so the budget is in the
regime where the model is just starting to solve boards — differences between archs should show up here rather
than being saturated away. Observation: `NQueens/accuracy` == `exact_match` == `valid_solution_rate` on every eval,
i.e. every valid board the model produces is exactly the stored label. With 95% of test givens seen in train, the
model is recalling the train solution for those givens rather than searching for any valid completion. Nothing
wrong with that for the comparison, but "accuracy" here is close to "recall of a memorised label".

### 2. First TreeRM sweep (queue q2, queued behind q1)
Four single-change variants off a base of K=4, 8 simulations, horizon=step, c_puct=1.25, gumbel_temp=1.0,
kl_strength=0.05 (GRAM's default), kl_grad_prior=0.8:
- `base` and `lr1e-3`: same lr sweep as the baselines.
- `resid`: candidate_residual=true (reasoning in §2.2).
- `full`: mcts_horizon=full, i.e. rollouts run to the end of all remaining supervision steps rather than just this
  forward pass. With H_cycles=3 and horizon=step, the search at the last cycle of a step has depth 1 and
  rollout 0 — it's just scoring the K candidates directly by CE. `full` gives the search something to look ahead
  into, at ~8x the rollout cost on average.

Open questions I'm tracking, to be answered by the runs:
- (a) Does the search help at all vs. GRAMDP-style learned posterior? If `base` ≈ GRAM, the search isn't adding
  information the posterior scorer couldn't learn.
- (b) 8 simulations over K=4 gives visit counts like [5,1,1,1] — a very peaked, coarse posterior. The KL to the
  prior then just pushes the prior to be peaked too. More simulations or fewer candidates may matter more than
  anything else.
- (c) The eval-time search uses the LPRM value head. If that head is badly calibrated early (lprm_loss was 0.36 at
  step 4, 0.06 at step 20), eval search is noise and eval acc will lag train acc. Worth comparing train-time
  exact-match vs eval.
