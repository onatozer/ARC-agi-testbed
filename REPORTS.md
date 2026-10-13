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

### Fast tier: 6x6 Sudoku (added after the first baseline)
A TreeRM run on 8x8 N-Queens is ~45 min (1.5–2 h with `horizon=full`), i.e. ~10 experiments a night — too slow
for a hyper-param loop. So there are now two tiers:
- **Fast tier: 6x6 Sudoku** (`data/sudoku-6`): 10k puzzles x (1 + 8 shuffle augs) = 90k train rows, 1000 test, 10
  clues of 36 (`clue_ratio=0.3`). **0% test-in-train overlap and 89,862 distinct labels**, so unlike N-Queens this
  measures generalisation. seq_len 36 vs 64 halves the per-step cost. Budget: batch 64, 1500 steps, eval every 500
  on 512 test puzzles. Target ~10 min per TreeRM run, ~3 min per TRM run. 6x6 N-Queens was rejected for this role
  because it has only 4 solutions.
  Metric: `all/exact_accuracy`. The first build (`sudoku-6`, 10 clues) turned out to have **0% unique-solution
  puzzles**, so exact match would have been a lottery among valid completions and the training label an arbitrary
  one of them. Sweeping clue counts on the same boards: 12 clues -> 5% unique, 16 -> 45%, 20 -> 75%, 22 -> 86%;
  even easy puzzles aren't reliably unique. So I added `unique_solution: true` to `build_sudoku_dataset.py`
  (commit `2899cef`): remove cells to the clue ratio, then reveal them back one at a time until a backtracking
  solver finds exactly one solution. The fast tier uses that build (`data/sudoku-6u`); with unique solutions,
  exact match is the correct metric and no validity evaluator is needed.
- **Confirmation tier: 8x8 N-Queens**, as above, for variants that win on the fast tier.

### Harness
Runs are launched from a scratch directory with `+wandb_mode=offline`; eval rows are parsed back out of the offline
`.wandb` file (`summary.py`). Nothing is uploaded. Each run's config and log are under
`scratchpad/runs/<name>/` (N-Queens) and `scratchpad/runs6/<name>/` (Sudoku).
Harness mishap worth recording: chaining queue B behind queue A with `until ! pgrep -f "queue A"` never fires,
because the waiter's own command line matches the pattern. Cost ~25 min of idle GPU once. Queues are now
launched as one sequential command instead.

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

3. **`eval_prior_mean` option for GRAM** (baseline fairness). At eval GRAM samples eps ~ N(mu_prior, var_prior)
   with var starting at softplus(0) ≈ 0.69 per dim on an rms-normalised z_H; on both tiers its eval
   `zH_rel_residual` is ~0.9 (the state is rewritten by noise every step) and it scores 2–6% exact. With
   `eval_prior_mean: true` eval uses mu_prior with no noise. Off by default (reference behaviour); run as an extra
   baseline so TreeRM isn't compared only against a handicapped GRAM.

4. **Throughput: TreeRM is Python-bound, not GPU-bound.** On the fast tier (batch 64, seq 36) TreeRM runs at the
   same 1.2 it/s as on N-Queens (batch 128, seq 64); GPU utilisation is 18% and memory 0.7 GB of 4 GB. The
   lockstep search still does per-tree PUCT selection in Python (64 trees x depth per simulation x 8
   simulations x 3 cycles per step). Rather than optimise further, I run two or three queues concurrently; they
   barely slow each other. Wall-clock in the tables is therefore only comparable within a queue.

5. **`param_lr_scales`** (pretrain.py, generic): parameter-name substring → lr multiplier, applied as a per-group
   `lr_scale` inside the lr schedule. Override as `+param_lr_scales.candidate_=0.3`. Motivation: TRM wants
   lr 1e-3 on this tier but TreeRM's candidate generator / scorer blow up there (§5), so the backbone and the
   heads may want different lrs. Unit-tested and smoke-run; queued as `heads0.3` / `heads0.1` at lr 1e-3.

6. **`value_bootstrap_weight`** (backlog item a). Every leaf the training search visits gets a rollout; the
   rollout's token accuracy is recorded as a target for the LPRM value head evaluated at that leaf's own state
   (first token, rms-normed, as the head sees in `forward`). On grad cycles the head pays
   weight × MSE summed over puzzles / averaged over the search's leaves. The head otherwise sees one bit per
   puzzle per 8 forwards (final exact-match), which is why the eval search has been inert. Queued as `vb0.5`.

7. **Search only on the grad cycles** (hard-coded, per the user). Until now the label-informed search picked z_H
   on all H_cycles, but the KL is only computed where gradients are on — so on the two no-grad cycles label
   information steered the trajectory unpaid. That is the same leak GRAM's own code comment warns about (its
   `_guidance` uses the posterior whenever `self.training`, with the `is_grad_enabled()` gate commented out),
   and the likely source of the 94% train / 6% test gap. Now: no-grad cycles sample the prior (as at eval), the
   grad cycle searches and pays the KL. Also 3× fewer searches per forward. Every run from `v2_*` on has this;
   earlier TreeRM rows are the leaky version. The three leaky-version runs relaunched after the reboot were
   cancelled at step ~500 (per the user) and replaced by v2 runs: `v2_lr3e-4`, `v2_lr1e-3`, `v2_lr1e-4`,
   `v2_lr5e-4`, `v2_lr1e-3_heads0.3`, `v2_lr1e-3_heads0.1`, then `v2_vb0.5_lr3e-4`.

8. **`rollout_policy: prior`** (backlog item d). Rollouts take the scorer's argmax candidate at each update
   instead of the plain TRM update, so a candidate's value means "take it, then keep following the policy" —
   consistent with what the no-grad cycles and eval actually do. Roughly doubles rollout cost. Queued as
   `rollprior`.

9. **`candidate_mode: flags`** (backlog item c; replaces the `candidate_residual` flag with
   `candidate_mode: full | residual | flags`). cand_k = u_t + g_k(z_L, u_t) · √H · e_k with K frozen orthonormal
   directions e_k (seeded QR) and a learned per-token gate g_k (init 1). The slots differ by construction, so the
   search always has a real choice; the network can learn what each flag means downstream; the gates can only
   shrink the flags, so the collapse point is plain TRM rather than "TRM + MLP". Adds just a 2H→K linear.
   Queued as `flags` at lr 3e-4 and 1e-3.

10. **`argmax_pick`** (backlog item e): outside the grad cycle (no-grad cycles, eval) take the argmax candidate
    instead of a Gumbel sample. TRM's recursion is deterministic; TreeRM injects a random pick at every z_H
    update at eval, which can only hurt exact match if the prior is already right. Queued with flags.

11. **`eval_mcts_simulations`**: simulations per search at eval, independent of training's. Test-time compute
    knob for once the value head ranks candidates (needs `value_bootstrap_weight` > 0 to be meaningful).

## 3. Results

`acc` = `NQueens/accuracy` on the 1000-puzzle test set (valid 8-queens board keeping every given), at 1k / 2k / 3k
steps. `exact` = exact match with the stored label at 3k. Wall-clock is the full run including evals; `*` = GPU
was shared with a fast-tier run, so wall-clock is inflated.

| run | arch | params | lr | acc @1k | acc @2k | acc @3k | exact @3k | wall | notes |
|---|---|---|---|---|---|---|---|---|---|
| trm192_lr3e-4 | TRM | 886k | 3e-4 | 0.008 | 0.132 | 0.210 | 0.210 | 12.3 min | baseline |
| trm192_lr1e-3 | TRM | 886k | 1e-3 | 0.000 | 0.019 | 0.050 | 0.050 | 14.5 min* | baseline; lr too high |
| gram112_lr3e-4 | GRAM | 865k | 3e-4 | 0.001 | 0.016 | 0.031 | 0.031 | 17.0 min* | baseline; eval zH_rel_residual 0.94 (prior noise dominates at eval) |
| gram112_lr1e-3 | GRAM | 865k | 1e-3 | 0.004 | 0.026 | 0.023 | 0.023 | 12.4 min | baseline; eval zH_rel_residual 0.95 |
| treerm128_base | TreeRM | 886k | 3e-4 | | | | | | K=4, 8 sims, horizon=step |
| treerm128_lr1e-3 | TreeRM | 886k | 1e-3 | | | | | | |
| treerm128_resid | TreeRM | 886k | 3e-4 | | | | | | candidate_residual |
| treerm128_full | TreeRM | 886k | 3e-4 | | | | | | horizon=full |

### Fast tier: 6x6 Sudoku (unique), exact match on 512 test puzzles

| run | arch | params | lr | steps | exact @500 | @1000 | @1500 | @2000 | train exact (end) | wall | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cal_u_trm128_lr1e-3 | TRM | 525k | 1e-3 | 1500 | 0 | 0 | 0.178 | – | 0.90 | 5.2 min* | calibration |
| cal_u_trm128_lr3e-4 | TRM | 525k | 3e-4 | 1500 | 0 | 0 | 0.020 | – | 0.47 | 5.5 min* | calibration |
| trm192_lr1e-3 | TRM | 888k | 1e-3 | 2000 | 0 | 0 | 0.119 | **0.738** | 0.93 | 4.4 min | parity baseline |
| trm192_lr3e-4 | TRM | 888k | 3e-4 | 2000 | 0 | 0 | 0.076 | 0.383 | 0.81 | 4.5 min | parity baseline |
| trm192_lr1e-3_s1 | TRM | 888k | 1e-3 | 2000 | 0 | 0.002 | 0.059 | 0.660 | 0.73 | 9 min† | seed 2; @1750 = 0.244. Seed band 66–74% |
| trm128_lr1e-3 | TRM | 525k | 1e-3 | 2000 | 0 | 0 | 0.146 | **0.725** | 0.76 | 12 min† | **equal width to TreeRM**: @1250 = 0.018, @1750 = 0.453. Width isn't the gap |
| trm128_lr5e-4 | TRM | 525k | 5e-4 | 2000 | 0 | 0 | 0.098 | 0.668 | 0.94 | 15 min† | equal width *and* lr to v2_lr5e-4 (0.229): lr isn't the gap either |
| trm128_lr1e-3_noact | TRM | 525k | 1e-3 | 2000 | 0 | 0 | 0.055 | 0.693 | 0.78 | 15 min† | disable_act_halt: always 8 steps, like TreeRM. Not the gap |
| gram112_lr1e-3 | GRAM | 867k | 1e-3 | 2000 | 0 | 0.002 | 0.012 | 0.061 | 0.98 | 4.8 min | parity baseline; eval zH_rel_residual 0.87 |
| gram112_evalmean | GRAM | 867k | 1e-3 | 2000 | 0 | 0 | 0.018 | 0.059 | 0.94 | 10 min† | eval_prior_mean=true: eval noise is not the cause |
| treerm128_lr1e-3 | TreeRM | 888k | 1e-3 | 2000 | 0 | 0.002 | 0.016 | **0.100** | 0.61 (label-informed) | 52 min† | K=4, 8 sims, horizon=step. token acc 82%; per-puzzle KL ≈ 0.004 nats |
| treerm128_lr3e-4 | TreeRM | 888k | 3e-4 | 2000 | 0 | 0.002 | 0.029 | **0.316** | 0.95 (label-informed) | 46 min† | unfixed search. 3x the lr-1e-3 run; TRM @3e-4 = 0.383 |
| treerm128_K2_s8 | TreeRM | ~820k | 1e-3 | 2000 | 0 | 0 | 0.014 | 0.117 | 0.66 (label-informed) | 53 min† | K=2: same as K=4 |
| treerm128_noevalsearch | TreeRM | 888k | 1e-3 | 2000 | 0 | 0.002 | 0.010 | 0.078 | 0.63 (label-informed) | 43 min† | eval_search=false: eval search worth ≤2 pts |
| treerm192_lr1e-3 | TreeRM | ~1.5M | 1e-3 | 2000 | 0 | 0.002 | 0.006 | 0.020 | 0.16 | 63 min† | width control (not parity): wider is *worse* at 1e-3 |
| **v2 runs** (search on the grad cycle only, forced root visits, eval every 250) | | | | | | | | | | | |
| treerm128_v2_lr3e-4 | TreeRM | 888k | 3e-4 | 2000 | 0 | 0 | 0.002 | 0.164 | 0.94 (label-inf.) | 29 min† | @1750 = 0.037. Below the leaky 0.316 |
| treerm128_v2_lr5e-4 | TreeRM | 888k | 5e-4 | 2000 | 0 | 0 | 0.031 | **0.229** | 0.80 | 29 min† | @1250 = 0.008, @1750 = 0.111; still steep |
| treerm128_v2_lr1e-4 | TreeRM | 888k | 1e-4 | 2000 | 0 | 0 | 0.002 | 0.016 | 0.25 | 30 min† | too low |
| treerm128_v2_flags_lr3e-4 | TreeRM | ~790k | 3e-4 | 2000 | 0 | 0 | 0.031 | **0.361** | 0.86 | 34 min† | candidate_mode=flags: @1750 = 0.133. Best TreeRM so far; cand_div grows 0.03 → 0.11 |
| treerm128_v2_lr1e-3 | TreeRM | 888k | 1e-3 | 2000 | 0 | 0 | 0 | 0.000 | 0.00 | 43 min† | full-vector candidates never learn at 1e-3 |
| treerm128_v2_lr1e-3_heads0.3 | TreeRM | 888k | 1e-3 / heads 3e-4 | 2000 | 0 | 0.002 | 0.064 | **0.564** | 0.83 | 47 min† | param_lr_scales candidate_=0.3: @1750 = 0.279 |
| treerm128_v2_lr1e-3_heads0.1 | TreeRM | 888k | 1e-3 / heads 1e-4 | 2000 | 0 | 0 | 0.025 | **0.557** | 0.95 | 48 min† | candidate_=0.1: @1750 = 0.209 |
| treerm128_v2_flags_K1_lr5e-4 | TreeRM | ~530k | 5e-4 | 2000 | 0 | 0 | 0.182 | **0.736** | 0.91 | 24 min† | **K=1 control** (no choice): = TRM-128 (0.668 at 5e-4). @1750 = 0.488 |
| treerm128_v2_flags_lr1e-3 | TreeRM | ~790k | 1e-3 | 2000 | 0 | 0.004 | 0.062 | **0.707** | 0.94 | 49 min† | **K=4 flags at uniform lr 1e-3 = TRM level** (TRM-192 band 0.66–0.74). @1750 = 0.273 |
| treerm128_v2_flags_lr5e-4 | TreeRM | ~790k | 5e-4 | (1750) | 0 | 0 | 0.086 | – | – | – | @1750 = 0.279; stalled mid-eval under 6-way contention, cancelled (superseded) |
| **treerm128_v2_flags_lr1e-3_heads0.1** | TreeRM | ~790k | 1e-3 / heads 1e-4 | 2000 | 0 | 0.002 | 0.357 | **0.811** | 0.86 | 78 min† | **above TRM** (0.725 / 0.66–0.74). @1250 = 0.062, @1750 = 0.695. Seeds pending |
| **treerm128_v2_flags_lr1e-3_heads0.3** | TreeRM | ~790k | 1e-3 / heads 3e-4 | 2000 | 0 | 0.006 | 0.244 | **0.793** | 0.72 | 78 min† | second config above TRM. @1750 = 0.527 |
| treerm128_v2_flags_kl0_lr5e-4 | TreeRM | ~790k | 5e-4 | 2000 | 0 | 0 | 0 | 0.000 | 0.00 | 73 min† | kl_strength=0: never learns, even on train. The KL is load-bearing |
| **treerm128_v2_flags_lr1e-3_heads0.1_s1** | TreeRM | ~790k | 1e-3 / heads 1e-4 | 2000 | 0 | 0.002 | 0.463 | **0.828** | 0.94 | 35 min† | seed 1 replicate. @1750 = 0.736 |
| treerm128_v2_flags_lr1e-3_heads0.03 | TreeRM | ~790k | 1e-3 / heads 3e-5 | 2000 | 0 | 0.002 | 0.207 | 0.734 | 0.88 | 37 min† | heads too slow: 0.1–0.3× is the range. @1750 = 0.562 |
| treerm128_resid | TreeRM | 888k | 1e-3 | 2000 | | | | | | | candidate_residual |
| treerm128_full | TreeRM | 888k | 1e-3 | 2000 | | | | | | | horizon=full |
| grampd120_lr1e-3 | GRAMDP | 873k | 1e-3 | 2000 | | | | | | | parity baseline |

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

All four N-Queens baselines done. **TRM lr 3e-4 at 21.0% is the confirmation-tier number to beat.** GRAM at
parity reaches only 2–3% at either lr. Its eval `zH_rel_residual` is ~0.95 (vs 0.006 for TRM): at eval GRAM
samples eps from the Gaussian prior, whose variance starts at softplus(0) ≈ 0.69 per dimension on an
rms-normalised z_H, so the noise is as large as the state itself and the recursion never settles. This is the
reference GRAM behaviour (the paper's), not a bug I introduced, but it means GRAM here is a weak baseline; a
GRAM variant with the eval noise removed (mean of the prior) would be fairer. Parked — TreeRM first.

### 2. First TreeRM sweep (queue q2, queued behind q1) — CANCELLED, see §4
Four single-change variants off a base of K=4, 8 simulations, horizon=step, c_puct=1.25, gumbel_temp=1.0,
kl_strength=0.05 (GRAM's default), kl_grad_prior=0.8:
- `base` and `lr1e-3`: same lr sweep as the baselines.
- `resid`: candidate_residual=true (reasoning in §2.2).
- `full`: mcts_horizon=full, i.e. rollouts run to the end of all remaining supervision steps rather than just this
  forward pass. With H_cycles=3 and horizon=step, the search at the last cycle of a step has depth 1 and
  rollout 0 — it's just scoring the K candidates directly by CE. `full` gives the search something to look ahead
  into, at ~8x the rollout cost on average.

### 3. Fast-tier calibration (6x6 Sudoku)
- TRM h=128, lr 1e-3, batch 64, 1500 steps on the **non-unique** 10-clue build: token acc 41% / 50% / 55% at
  500 / 1000 / 1500, **exact match 0%** throughout. 6.5 min sharing the GPU. 10 clues is 28% of the cells, so
  55% token accuracy means the model is barely past the givens. Non-unique labels make the CE target
  inconsistent, which is the likely cause — the model can't fit a label that's one of several arbitrary
  completions. Rebuilding with `unique_solution=true`; the unique build has more clues (revealed until unique),
  so it should be both better-posed and faster to learn.
- Same TRM on the **unique** build (`sudoku-6u`, mean 17.6 clues): token acc 62% / 71% / 83%, exact match
  0 / 0 / **17.8%** at 500 / 1000 / 1500, train exact 90%, 5.2 min. So the labels were the problem. Signal
  exists at this budget; the curve is still steep at 1500, so small differences between variants will be
  amplified (good for ranking, but a 2-seed check is needed before believing a small gap).
- Same at lr 3e-4: exact 0 / 0 / 2.0%, train exact 47%. lr 1e-3 is the fast-tier default — the opposite of
  N-Queens, where 1e-3 hurt TRM. (Reasonable: Sudoku h=128 with 2000 steps is a short run that benefits from a
  high lr; the N-Queens run at h=192 for 3000 steps didn't.) This means lr must be swept per arch on each tier
  rather than carried across.

### 4. First fast-tier sweep (queue q6a, 6x6 Sudoku unique, 2000 steps x batch 64)
Budget raised from 1500 to 2000 steps (eval every 500 on 512 test puzzles) so the readout isn't taken on the
steepest part of the curve. Runs, in order: TRM h=192 at lr 1e-3 and 3e-4 (parity baselines), GRAM h=112, then
TreeRM h=128 (K=4, 8 sims) at lr 1e-3 / 3e-4, `candidate_residual`, `mcts_horizon=full`, and GRAMDP h=120.
The 8x8 N-Queens TreeRM sweep (q2) was cancelled in favour of this; N-Queens is now the confirmation tier.

First result: TRM h=192 lr 1e-3 goes 0 / 0 / 11.9% / **73.8%** at 500 / 1000 / 1500 / 2000 (4.4 min). The curve
"clicks" between 1500 and 2000, so the endpoint is very sensitive to *when* a model clicks: a variant that
clicks 300 steps later will read ~10% vs 74% and look terrible while being nearly identical. Subsequent queues
eval every 250 steps (`queue6b.sh`) to get the curve; comparisons should use the step at which exact match
crosses e.g. 25% / 50% as well as the endpoint.

**GRAM diagnosis.** GRAM reaches 6% test exact either way (sampled or mean prior at eval), but its *train* exact
is 94–98%, versus TRM's 93% train / 74% test. The train forward in GRAM uses the posterior q(eps | u_t, y), which
sees the label, so "train exact" for GRAM-family models is label-informed and not comparable to TRM's. The real
finding is that the prior never learns to reproduce the posterior's choices (KL weight 0.05 is too weak here) —
the posterior encodes the answer and the prior can't. **This is the central risk for TreeRM**: its training-time
search is valued by the label CE, so the MCTS posterior is label-informed exactly like GRAM's. Whether the
scorer (prior) can learn to predict the search's choice — or, failing that, whether the LPRM-valued eval search
can recover it — is what decides the arch. Watch the train/test gap on every TreeRM run.
(† wall-clock while sharing the GPU with two other runs.)

### 5. First TreeRM results: the search adds nothing (yet)
- `treerm128_lr1e-3` (K=4, 8 sims): 0 / 0.2 / 1.6 / **10.0%**. `treerm128_K2_s8`: **11.7%**. TRM band: 66–74%.
- Smoking gun: the logged KL is 0.012 summed over 64 puzzles at weight 0.05 → **~0.004 nats per puzzle**. The
  scorer's prior and the MCTS visit distribution agree almost perfectly. If the search were extracting label
  information the posterior would have to differ from the label-free prior. So the tree is picking among
  candidates that the label CE can't tell apart — i.e. the K candidates are (nearly) interchangeable, and
  TreeRM degrades to a narrower TRM (h=128 vs 192) plus a lot of search noise. K=2 vs K=4 making no difference
  fits that.
- Why would candidates be interchangeable? The generator `cat[z_L, u_t] -> ... -> K*H` starts random, so at
  init the K outputs *are* different — but none of them is a good z_H, and the LM loss pushes every one toward
  "a z_H that decodes well" through the straight-through path; the only force separating them is the search's
  preference, which is noise when they're all equally bad. Nothing in the loss rewards diversity.
- To stop guessing I added per-puzzle diagnostics (commit above) logged in train and eval:
  `search/cand_div` (mean pairwise 1 − cosine between candidates), `search/post_entropy` (visit distribution
  entropy; max ln K = 1.39 for K=4), `search/kl_nats` (unweighted KL(post‖prior)), `search/argmax_agree`
  (fraction of searches where prior and posterior pick the same candidate). Runs started after this commit
  (s16, lr3e-4, resid, full, kl0.5, kl2, noevalsearch is too early) carry them.
**Diagnostics readout (s16 run, first 400 steps) — hypothesis revised.**
- `cand_div ≈ 1.03`: the candidates are *orthogonal*, not interchangeable. Diversity is not the problem.
- Eval: `post_entropy = 0`, `kl_nats = 0`, `argmax_agree = 1`. The eval search is one-hot on the prior's argmax
  every time: the LPRM values don't separate the children, so PUCT follows the prior. Matches
  `noevalsearch` ≈ base.
- s16 train: `kl_loss` 0.04 → 47 over 400 steps while `lm_loss` rises 1.05 → 1.89 — **diverging**. With 16
  simulations the visit distribution is near-one-hot (entropy 0.23 of max 1.39); the KL to a one-hot target
  drives the scorer's unbounded logits apart (`kl_nats` ≈ 975/puzzle = logit gaps of ~1000). Killed.
- Base (8 sims) train: `kl_loss` ≈ 0.015 flat for 2000 steps. Tracing PUCT by hand: unvisited children score
  Q̂ = 0 and the first-visited child keeps its *raw* Q (MinMaxStats can't normalise until a second distinct value
  arrives), so whichever child is picked first keeps winning; with ≤ 0 training values each child gets one visit
  and the best gets the rest. For the KL to sit at ~0.004 nats the posterior's argmax must be the *same slot
  every time*. That is a collapse: straight-through only trains the selected candidate, so it improves, so it's
  selected again, and the other K−1 slots become untrained junk (hence orthogonal). **TreeRM had collapsed to
  "TRM(h=128) + one fixed MLP on z_H"**, which explains 10% ≈ a narrower TRM. K=2 vs K=4 identical: same reason.

Fixes, applied (commit below) and to be applied:
1. **MCTS: evaluate every root child once before PUCT** (`mcts.py`). Removes the first-visit bias, and with
   K ≤ sims each child keeps ≥ 1 visit, so the posterior is never one-hot and the KL is bounded (max ln 8 at
   8 sims). This is a correctness fix to the search, not a hyper-parameter; all runs from here use it.
2. `candidate_residual` (queued): candidates = u_t + delta_k, so no slot can be junk — every candidate is at
   least the TRM update.
3. If the KL still explodes once the posterior carries information: smooth the target, π ← (1−ε)π + ε/K.

**Fixed-search run (`treerm128_fix`, 8 sims, forced root visits) — diagnostics over the first 960 steps.**
`post_entropy` ≈ 1.09 (≈ a [5,1,1,1]/8 split: the 4 free simulations all go to one child), `kl_nats` 0.2–0.3,
`argmax_agree` rising 0.2 → 0.8 — the prior learns to predict the search's pick, as the ELBO intends. But
`cand_div` goes **1.0 → 0.035** by step 200: the candidates become near-identical. Mechanism: now that every
slot gets selected sometimes, the straight-through LM gradient pulls each of them toward the same best z_H.
Old run: one trained slot + K−1 orthogonal junk slots. Fixed run: K copies. In neither case does the search
have a real choice. Test exact at 750 (0.2%) tracks the old run. Nothing in the objective rewards candidate
diversity, so the next option is an explicit penalty on pairwise candidate cosine (`candidate_diversity_weight`).

**lr matters more than anything so far.** `treerm128_lr3e-4` (unfixed search) reaches **31.6%** vs 10.0% at
lr 1e-3; its KL series is the same flat ~0.015 (same one-slot collapse), but lm_loss ends at 0.13 vs 0.46. So
the collapsed model ("TRM h=128 + one MLP on z_H") just optimises badly at 1e-3 — the scorer/generator heads
are probably what's unstable, since TRM itself prefers 1e-3. At 3e-4 TreeRM (31.6%) is close to TRM at 3e-4
(38.3%); TRM's edge is that it tolerates 1e-3 (66–74%). All further TreeRM variants run at 3e-4 (and an lr
sweep 1e-4 / 5e-4 is due). The lr-1e-3 fix-only control was killed at step 1000 (0.2%; its diagnostics had
already served their purpose).

**Reboot (2026-10-13 ~03:20).** The machine restarted minutes after the last commit. All queues died and `/tmp`
was wiped, taking the run logs / offline W&B files with it; the four in-flight runs (`resid` @1e-3,
`fix_lr3e-4`, `resid_lr3e-4`, `fix_lr1e-4`) never reached an endpoint. Interim readouts before the reboot:
`resid` @1e-3 was training faster than any earlier TreeRM (lm_loss 0.26 at step 860 vs 0.36–0.46; test 0% at
500, nothing later); `fix_lr3e-4` showed the usual candidate collapse (cand_div → 0.03). Everything in this
file was recorded before the reboot and is intact. The harness now lives in `scripts/autoresearch/` and runs go
to `~/ARC-agi-testbed/autoresearch_runs/`, both outside `/tmp`. Relaunched at 10:50: `fix_lr3e-4`,
`resid_lr3e-4`, `fix_div0.1_lr3e-4` (three concurrent).

**v2 (grad-cycle-only search) diagnostics, first 480 steps, lr 3e-4 and 5e-4:** identical shape to before —
`cand_div` 0.73 → 0.04 by step 400, posterior entropy ~1.28 (near-uniform over K=4), `kl_nats` ~0.08,
`argmax_agree` ~0.8, `cand_norm_ratio` 1.35 (3e-4) / 2.5 (5e-4) — the generator makes candidates larger than
u_t. The leak fix doesn't touch the collapse. → backlog (c), implemented as `candidate_mode: flags`.

### 6. The heads' learning rate was the dominant problem
References at equal width: TRM h=128 gets 72.5% (lr 1e-3), 66.8% (5e-4), 69.3% (no ACT). So neither width,
nor lr per se, nor the always-8-steps regime explains TreeRM's 16–36%.
With `param_lr_scales`: backbone at 1e-3, candidate generator + scorer at 0.3× or 0.1× → **56.4% / 55.7%**
(full-vector candidates, grad-cycle-only search). Same model at a uniform lr 1e-3 → 0.0%; at 5e-4 → 22.9%.
Reading: the backbone wants 1e-3 (TRM's optimum); the candidate heads diverge at 1e-3 (the earlier width
control, h=192 → 2%, pointed the same way) and at a uniform lower lr the backbone under-trains. Decoupling the
two lrs fixes most of the gap: TreeRM is now ~12 points behind TRM-128 at parity width, rather than 50.
Next: flags + scaled heads (running), argmax pick, a seed replicate, value bootstrap on top.

### 7. The K=1 control: the scaffold is free, the choice is what costs
`flags_K1_lr5e-4` (one candidate, so no choice, KL ≡ 0, but the same code path, LPRM head and losses) reaches
**73.6%** — equal to TRM-128 (66.8% at 5e-4, 72.5% at 1e-3). So nothing in the scaffold hurts. The same model
with K=4 at the same lr: 22.9% (full) / ~? (flags, running). Everything the K>1 machinery adds is what loses
the points: (a) a *random* candidate (Gumbel sample from the prior) at every no-grad cycle and at eval, which
before the flags mean anything is pure noise injected into the recursion 3× per forward; (b) the KL and the
straight-through gradient through the selection; (c) the heads' lr (now handled). Runs in flight separate
these: `kl0` (no KL), `heads0.3_argmax` (deterministic pick), flags+heads.

The deeper point. On the grad cycle the search picks by label CE, and the diagnostics show the prior learns to
predict that pick (argmax_agree 0.97, KL 0.014 nats). If the pick is predictable from the state, it carries no
information the network couldn't compute itself; if it isn't, the prior can't reproduce it at eval. That is
the ELBO's tension (GRAM's too). The way a search model can actually *beat* TRM is test-time: K branches
explored with a value head that ranks them, i.e. test-time compute. That needs (1) K>1 not to hurt (this
section), (2) a value head that ranks candidates (value bootstrapping, `vb0.5` queued), (3) then more eval
simulations. Working through those in order.

### 8. First configuration above TRM (fast tier)
`candidate_mode=flags` + search only on the grad cycle + forced root visits + backbone lr 1e-3 with the
candidate heads at 0.1× (or 0.3×): **81.1% / 79.3%** vs TRM 72.5% (h=128) and 66–74% (h=192, parity), at
~790k params. The no-KL ablation (`flags_kl0`, 0% at 1500) shows the KL is load-bearing: without it the prior
never learns to follow the search and the eval-time pick is noise. Two seed replicates of heads0.1 are running;
the config is also on the 8x8 N-Queens confirmation tier at lr 3e-4 and 1e-3 (TRM there: 21.0%).

What the recipe amounts to: the K candidates are u_t plus K fixed orthogonal "flags" with learned per-token
gates; on the grad cycle a label-valued search picks a flag and the scorer is trained (via the KL) to predict
that pick from the state; elsewhere the scorer's pick is used. The downstream network learns what each flag
means. It is a discrete latent-variable model whose posterior is a search rather than a network.

- Candidate hypotheses to test next, in order of how cheaply they're falsified:
  1. `candidate_residual` (queued): candidates = u_t + delta_k. If diversity is the issue this changes little; if
     the issue is that full-vector candidates are all *bad*, this should jump.
  2. Stronger KL (0.5, 2.0, queued): forces the prior toward the posterior — only helps if the posterior carries
     information, which the diagnostics will say.
  3. A diversity-forcing change: e.g. candidates as u_t + sigma * fixed random directions, or an explicit
     repulsion term — if `cand_div` is near 0.
  4. More simulations (16, running) — only matters if candidates differ.

(Per the user, 2026-10-13: no GRAMDP baseline — the loop is about iterating on TreeRM itself. TRM at parity
is the reference.)

**Architecture backlog** (beyond hyper-params), roughly in the order I intend to try them:
- a. **Value bootstrapping for the LPRM head.** The eval search is useless because the LPRM head is trained
  only on the final exact-match of the whole step (one bit per puzzle, every 8th forward). The training search
  computes a rollout value (label CE) for every leaf it visits — those are free regression targets for the
  value head at intermediate states, as in AlphaZero. A value head that ranks candidates at eval is the
  precondition for the search to do anything at test time.
- b. **Train-time search valued by the LPRM head** (`train_search_value: lprm`), once (a) makes the head
  useful: closes the train/eval mismatch (the GRAM failure mode) at the cost of label information entering only
  through the value head.
- c. **Candidate construction that can't collapse**: candidates as u_t plus K learned, mutually orthogonal
  directions scaled per token (so diversity is structural), or K candidates computed from K different
  stochastic views (dropout / noise on z_L) of the same update.
- d. **Rollouts that follow the policy**: rollouts currently use plain TRM updates, so a candidate's value is
  "what if I take this candidate and then act like TRM". Using prior-argmax candidates in rollouts makes the
  value consistent with the policy being trained.
- e. Deterministic argmax at eval instead of a Gumbel sample.

Open questions I'm tracking, to be answered by the runs:
- (a) Does the search help at all vs. GRAMDP-style learned posterior? If `base` ≈ GRAM, the search isn't adding
  information the posterior scorer couldn't learn.
- (b) 8 simulations over K=4 gives visit counts like [5,1,1,1] — a very peaked, coarse posterior. The KL to the
  prior then just pushes the prior to be peaked too. More simulations or fewer candidates may matter more than
  anything else.
- (c) The eval-time search uses the LPRM value head. If that head is badly calibrated early (lprm_loss was 0.36 at
  step 4, 0.06 at step 20), eval search is noise and eval acc will lag train acc. Worth comparing train-time
  exact-match vs eval.
