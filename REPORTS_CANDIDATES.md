# TreeRM: training the candidate generator (autoresearch loop, started 2026-10-16)

Branch `treerm-candidates` (off `treerm-arc`, worktree `.claude/worktrees/treerm-candidates`). Phase 1 runs under
`~/ARC-agi-testbed/autoresearch_runs/candidates6/` (W&B `cand6_<name>`, harness `scripts/candidates/{run6.sh,queue6.sh}`);
phases 2+ under `~/ARC-agi-testbed/autoresearch_runs/candidates/<tier>/` (W&B `cand_<tier>_<name>`, harness
`scripts/candidates/{run.sh,queue.sh}` with the queue files in `scripts/candidates/queues/`), all in
`ozeronatwanb/ARC-agi-testbed`. `summary.py <run dir>` prints a run's eval rows and train means; each tier's
`results.txt` has the summaries, but its `===` headers are written when a run *starts* and the `##` blocks when it
*ends*, so read it by `##` block, not by header.

## 0. Summary (read this first)

State at 2026-10-16 13:15 (24 N-Queens runs over phases 2-7, 12 Sudoku runs over phases 1, 3, 4; section 4 is the log).

1. **The candidate loss is the fix, and it is robust.** On 8x8 N-Queens (`NQueens/accuracy` at step 3000) every run
   trained with `candidate_loss` (wta or posterior) scores **44-55%** (mean 50.5 over 18 runs) and every run without
   it **30-37%** apart from one 51% seed (mean 36 over 6): +14 points, with the prior turned into a usable policy
   (eval `argmax_agree` 0.70-0.83 vs 0.42-0.47). The control (phase 4) says about half of that is the K-rollout
   training signal the loss adds per step and half the learned proposals. On 6x6 Sudoku, where one good u_t is
   enough, every variant stays in the recipe's 80-89% band (phase 1).
2. **With seeds (phase 7), the learned residual candidates at K=4 are the best configuration, by a margin inside the
   noise.** Means over seeds: residual + wta K=4 **53.5%** (55.2 / 54.1 / 51.2), flags + wta K=8 / 16 sims **53.6%**
   (55.0 / 52.2), residual + wta K=8 **48.9%** (51.7 / 46.1), flags + wta K=4 **47.4%** (44.4 / 50.3), with a seed
   spread of 4-6 points on the trained runs (and ~20 on the untrained flags: 51.0 / 32.0 at K=8). So: learned
   proposals beat the fixed flags at K=4 by ~6 points, two seeds each; the same ~6 points is what K=8 buys the
   *flags* under the loss; and K=8 costs the learned generator ~5 points. Reading: the search wants K candidates
   that differ; the flags get that by construction, so more of them helps; the learned generator's slots differ by
   0.06-0.10 nats in outcome (`cand_ce_gap`), enough to beat the flags at K=4, but its 8 slots are nearer copies than
   8 flags are, and the extra copies dilute the search (and cost 2.4x eval memory).
3. **Nothing tried makes the learned slots more different in outcome.** Load balancing, slot recoverability, the
   annealed wta (phase 3), the `flags_residual` mode (a fixed flag per slot plus a learned delta, phase 6: 44.9 /
   46.5%) and more guided simulations at K=8 (phase 5: 46.6%) all leave `cand_div` at 0.05-0.11 and `cand_ce_gap` at
   0.03-0.10 nats and the accuracy in the trained range. The recover head does learn to tell the slots apart on
   N-Queens (0.52 vs chance 0.25) and not on Sudoku, which is the direct measurement: the slots are weakly
   distinguishable where the task has alternatives and identical where it does not.
4. **What to carry forward.** `candidate_mode=residual` + `candidate_loss=wta` at K=4 / 8 sims as the TreeRM default
   candidate recipe (the config keeps `flags` as the default until this is reviewed; the N-Queens tier is the one
   that can see the difference, the Sudoku fast tier cannot). The open problem, unchanged by this loop, is a
   generator whose K slots are K different proposals; the measurements that would show it are `cand_ce_gap` well
   above 0.1 nats and `recover_acc` near 1, and the test is K=8 beating K=4 on N-Queens.
## 1. The problem

TreeRM's learned candidate generator (`candidate_mode=full`, and `residual` = u_t + generator output) collapsed in every
run of the first loop (REPORTS.md section 4): one trained slot plus junk, or K copies. The flag candidates
(`candidate_mode=flags`, u_t plus a gated fixed orthonormal direction per slot) were the workaround that produced the
recipe, but they are not learned proposals, so a larger K only adds random directions; K=8 was no better than K=4 on
Sudoku (67.8% vs 81.3%, single runs) and level on MiniARC.

The mechanism, from the code: the pick is a hard one-hot straight-through sample, so the task loss reaches only the
picked candidate; the search runs on detached candidates, so the rollout values that compare the K proposals never
reach the generator. The unpicked candidates do get a gradient, but only through the scorer's input (the KL and the
straight-through path make them *scored* right, not *better*). The picked slot improves, gets picked again, and the
others never train: the standard multiple-choice-learning collapse.

## 2. Changes

1. **`config/arch/treerm.yaml` restructured** into sections (candidates / search / training signal / runtime) with the
   recipe as the defaults (K=4, 8 training simulations; the file had kept K=8 and 16 from the first commit) and
   **`arch.param_lr_scales: {candidate_: 0.1}` as a real config key** (`pretrain.py` merges the arch's scales under
   the train config's `param_lr_scales`), so the 0.1x head lr no longer has to be remembered as a command-line flag.
2. **`candidate_loss`** (`none | wta | posterior`): on the grad cycles every candidate is rolled out with gradient
   exactly as the search's root-child rollouts are (plain TRM updates to the end of the supervision step, then the
   decode; with the default `grad_cycles=1` that is just the decode) and its label CE is weighted per candidate:
   `wta` = relaxed winner-take-all (the lowest-CE candidate gets 1 - `candidate_loss_eps`, the others share eps;
   Rupprecht et al. 2017, the standard cure for a multiple-choice head whose losing slots never train), `posterior` =
   the search's visit distribution. `candidate_loss_weight` scales it. Diagnostics: `search/cand_loss`,
   `search/cand_ce_gap` (CE of the worst minus the best candidate: how much the choice matters) and
   `search/cand_best_agree` (does the search's argmax pick the CE-best candidate).
3. **`candidate_soft_grad`**: the pick's backward reaches every candidate in proportion to its posterior mass (the
   forward is still exactly one candidate; the straight-through path to the scorer is unchanged).
5. **`candidate_balance_weight`** (the user's pick, 2026-10-16): load balancing as for a mixture-of-experts router
   (Switch Transformer): K * sum_k f_k * P_k per puzzle, f_k the fraction of the batch's picks on slot k and P_k the
   scorer's mean prior on k. A slot cannot take every puzzle, so every slot's candidate trains on its share and
   specialises on it. Diagnostics `search/balance`, `search/pick_entropy` (entropy of the batch's pick distribution).
6. **`candidate_recover_weight`**: slot recoverability (InfoGAN-style). A K-way linear classifier reads each
   candidate's decoded distribution (flattened probabilities) and must say which slot made it; its CE trains the
   generator through the decode, so candidates that decode alike are penalised in *outcome* space, not latent space.
   `search/recover_acc` is then a direct measure of outcome diversity (chance = 1/K).
7. **`candidate_loss_eps_anneal_steps`**: the wta eps anneals linearly from 1 (every candidate trains equally) to
   `candidate_loss_eps` over that many training steps, so no slot locks in before the others have trained
   (`search/cand_eps` logs it; a resumed run anneals again).
8. Tests: `tests/test_treerm.py` (with the scorer's candidate input detached, `none` gives exactly one slot a gradient
   and `wta` / `posterior` / soft grad give every slot one; the rollout path with `grad_cycles=2`),
   `tests/test_param_lr_scales.py` (arch scales under train scales).

## 3. Protocol

Fast tier of REPORTS.md: 6x6 Sudoku with unique solutions (`data/sudoku-6u`), hidden 128, batch 64 x 2000 steps, lr
1e-3, eval on 512 test puzzles every 500 steps, metric `all/exact_accuracy` at step 2000, seed 0 unless stated. The
recipe's seed spread on this tier is about +-4 points. One GPU job at a time (the shared `gpu.lock`). Every root child
is rolled out once before PUCT, so K and the simulation count are tuned together: "guided" simulations =
`mcts_simulations - K`.

Harness mishap (2026-10-16 00:41-01:16): the diverged `residual_k4_soft` run segfaulted, but its orphaned dataloader
worker kept the inherited `gpu.lock` descriptor, so the next run waited 35 min on the lock with the GPU idle. The run
scripts now use `flock -o` (the lock descriptor is closed before the command runs, so only the flock process holds it).

## 4. Log

### Phase 3 (queued 2026-10-16 01:40, both tiers): the three diversity mechanisms on residual + wta

Weights 0.1 for balance and recover, anneal over 1000 (Sudoku) / 1500 (N-Queens) steps; `_all3` combines them.
Watch `cand_div`, `cand_ce_gap`, `recover_acc`, `pick_entropy` as much as the accuracy: the question is whether the
candidates become different in outcome without losing accuracy.

The two `_bal` runs started before the diagnostic-scale fix (phase 1 caveat), so their `cand_ce_gap` and
`pick_entropy` are quoted x3. `cand_div` is the eval value, the other diagnostics are train means over the last
200 rows. Reference rows: wta alone is 80.3% on Sudoku (phase 1) and 55.2% on N-Queens (phase 2).

| tier | run | exact / NQueens acc @end | cand_div | cand_ce_gap | recover_acc | pick_entropy |
|---|---|---|---|---|---|---|
| sudoku6 | residual_k4_wta_bal | 87.9% | 0.084 | 0.050 | - | 1.30 |
| sudoku6 | residual_k4_wta_rec | 88.1% | 0.096 | 0.072 | 0.26 (chance 0.25) | - |
| sudoku6 | residual_k4_wta_anneal | **92.8%** | 0.077 | 0.047 | - | - |
| sudoku6 | residual_k4_wta_all3 | 89.7% | 0.068 | 0.058 | 0.26 | 1.30 |
| nqueens8 | residual_k4_wta_bal | 51.7% (exact 48.3) | 0.084 | 0.068 | - | 1.35 |
| nqueens8 | residual_k4_wta_rec | 52.5% (47.4) | 0.109 | 0.105 | 0.52 | - |
| nqueens8 | residual_k4_wta_anneal | 44.0% (41.9) | 0.079 | 0.061 | - | - |
| nqueens8 | residual_k4_wta_all3 | 52.0% (48.4) | 0.091 | 0.090 | 0.52 | 1.37 |

**Reading.** None of the three moves the outcome diversity. The balance loss does what a router constraint does
(the batch's picks spread evenly over the four slots: pick entropy 1.30-1.37 of ln 4 = 1.39, balance loss ~1 = its
minimum) and the recover head does learn to tell the slots apart on N-Queens (0.52 vs chance 0.25; on Sudoku it
stays at chance: the decoded boards are the same), but `cand_div` and `cand_ce_gap` are the wta-alone values on both
tiers and so are the accuracies (N-Queens 51.7-52.5% for bal / rec / all3 against 55.2% for wta alone; Sudoku
88-90% against the 80-89% band). A balanced scorer does not make a single shared generator's K outputs differ, and a
recoverability term of weight 0.1 buys separability in the decoded distribution without a CE gap. The annealed wta is
the one variant that moved accuracy, in opposite directions on the two tiers (92.8% Sudoku, 44.0% N-Queens), which
reads as seed noise until the replicate (`residual_k4_wta_anneal_s1`, queued) says otherwise.

### Phase 2 (2026-10-16 02:00-05:50, 8x8 N-Queens): the candidate loss where the search overrides the prior

Confirmation tier of REPORTS.md (`cfg_pretrain_nqueens`, batch 128 x 3000 steps, lr 3e-4, eval on the full
1000-puzzle test set every 500 steps), metric `NQueens/accuracy` at step 3000 (a board with 8 non-attacking queens;
`all/exact_accuracy` = label match in parentheses). Runs take 17-24 min. `argmax_agree` / `kl_nats` are the eval
search's agreement with the prior; `cand_ce_gap` is the train mean (nats, CE of the worst minus the best candidate).

| run | candidates | loss | K / sims (guided) | NQueens acc @3000 | argmax_agree | kl_nats | cand_div | cand_ce_gap | train lm_loss |
|---|---|---|---|---|---|---|---|---|---|
| flags_k4 | flags | none | 4 / 8 (4) | 34.9% (27.5) | 0.42 | 0.47 | 0.025 | - | 0.177 |
| flags_k4_s1 (seed 1) | flags | none | 4 / 8 (4) | 37.3% (33.6) | 0.45 | 0.33 | 0.020 | - | 0.175 |
| residual_k4_none | residual | none | 4 / 8 (4) | 30.4% (24.6) | 0.67 | 0.42 | 0.048 | - | 0.175 |
| residual_k4_wta | residual | wta 0.05 | 4 / 8 (4) | **55.2% (51.7)** | 0.80 | 0.17 | 0.091 | 0.066 | 0.135 |
| residual_k4_post | residual | posterior | 4 / 8 (4) | **52.7% (51.2)** | 0.83 | 0.25 | 0.049 | 0.033 | 0.137 |
| residual_k4_wta_s12 | residual | wta 0.05 | 4 / 12 (8) | 51.0% (49.0) | 0.76 | 0.26 | 0.104 | 0.080 | 0.131 |
| residual_k8_wta_s16 | residual | wta 0.05 | 8 / 16 (8) | 51.7% (48.0) | 0.72 | 0.18 | 0.088 | 0.089 | 0.135 |
| residual_k8_none_s16 | residual | none | 8 / 16 (8) | 32.3% (22.2) | 0.82 | 0.21 | 0.057 | - | 0.165 |
| flags_k8_s16 | flags | none | 8 / 16 (8) | **51.0% (46.3)** | 0.73 | 0.23 | 0.034 | - | 0.161 |

**Reading.** This tier separates the variants where Sudoku could not. The flags recipe lands at 35-37% on two seeds
(the first loop's 41-52% used random seeds and the Python search; the band here is what counts), residual candidates
without a candidate loss at 30%, and the same residual candidates with the wta or posterior loss at 51-55%: the loss
is worth +15-20 points, ten times the seed spread. It also changes how the search is used: with trained candidates the
eval search agrees with the prior 80% of the time and the posterior is 0.17 nats from it, against 42% and 0.47 nats
for the recipe, so the prior has become a usable policy rather than something the search has to override. The
candidates do differ in outcome here (`cand_ce_gap` 0.03-0.08 nats, recover accuracy 0.52 at chance 0.25 in phase 3),
though not by much. More guided simulations at K=4 (s12) do not add to it. What the tier does not yet separate is
whether the gain comes from the learned proposals or from the K-rollout training signal the loss adds per step (train
`lm_loss` 0.135 vs 0.177 at the same step count, 38% more wall-clock): `flags_k4_wta` (phase 4) is the control.
**K=8** (added 08:15): with 16 simulations (8 guided, as at K=4 / 8) the trained residual candidates are level-to-below
K=4 (51.7% vs 55.2 / 54.1) at 2792 MB eval memory vs 1180, and residual without the loss stays where K=4 was (32.3%:
eight copies of one vector, `cand_div` 0.057). The surprise is `flags_k8_s16` at 51.0%, +14-16 over the flags recipe
at K=4 / 8 sims, which the first loop's Sudoku runs had ruled out (K=8 no better than K=4 there, but that tier is at
ceiling). Eight frozen orthonormal directions are eight genuinely different proposals, and with the simulations
scaled the search can use them. Single seed; phase 5 replicates it.

### Phase 4 (2026-10-16 06:00-08:10): the control and the replicates

| tier | run | acc @end | argmax_agree | cand_div | cand_ce_gap | reading |
|---|---|---|---|---|---|---|
| nqueens8 | flags_k4_wta (control: wta loss, fixed flag candidates, no learned generator) | 44.4% (43.8) | 0.68 | 0.106 | 0.031 | the K-rollout training signal alone is worth +8 over the recipe (35-37%); the learned residual candidates are worth another +10 |
| nqueens8 | residual_k4_wta_s1 (seed 1) | 54.1% (50.0) | 0.70 | 0.083 | 0.064 | replicates the 55.2% |
| sudoku6 | residual_k4_wta_anneal_s1 (seed 1) | 86.1% | 0.93 | 0.086 | 0.073 | the 92.8% was seed noise; anneal is in the band |

### Phase 5 (queued 2026-10-16 08:20, N-Queens): K=8

`flags_k8_s16_s1` (seed replicate of the 51.0%), `flags_k8_wta_s16` (the wta loss on the eight flags: does the
K-rollout signal stack on the by-construction diversity), `residual_k8_wta_s24` (K=8 with 16 guided simulations in
training and 16 guided at eval, `eval_mcts_simulations=24`: the trained K=8 with the search budget K=4 had).

| run | acc @3000 | argmax_agree | kl_nats | cand_div | cand_ce_gap | eval MB | reading |
|---|---|---|---|---|---|---|---|
| flags_k8_s16_s1 (seed 1) | 32.0% (24.8) | 0.47 | 0.28 | 0.033 | - | 2786 | the 51.0% was seed noise: flags K=8 is 51 / 32, the recipe's spread on this tier is ~20 points |
| flags_k8_wta_s16 | 55.0% (51.6) | 0.83 | 0.13 | 0.053 | 0.084 | 2786 | the wta loss on eight fixed flags matches residual K=4 + wta |
| residual_k8_wta_s24 | 46.6% (44.2) | 0.72 | 0.31 | 0.101 | 0.096 | 3048 | 16 guided simulations at K=8, train and eval: no better than 8 (51.7%) |

### Phase 6 (2026-10-16 09:40-10:30, N-Queens): `candidate_mode=flags_residual`

Commit `4f4bfb6`: slot k = u_t + its fixed flag + a learned delta from the residual generator, so the slots differ by
construction and carry learned content (the per-slot conditioning the phase-3 reading asked for).

| run | acc @3000 | argmax_agree | kl_nats | cand_div | cand_ce_gap | reading |
|---|---|---|---|---|---|---|
| flagres_k4_wta | 44.9% (43.8) | 0.68 | 0.30 | 0.101 | 0.059 | level with the flags + wta control (44.4%), 10 below residual + wta |
| flagres_k8_wta_s16 | 46.5% (44.6) | 0.83 | 0.17 | 0.101 | 0.085 | K=8 no better |

**Reading of phases 5-6.** The flags' seed spread (51.0 / 32.0 at K=8, 34.9 / 37.3 at K=4) is as large as the whole
range of the trained variants, so single runs cannot rank residual (55.2 / 54.1), flags + wta (44.4 at K=4, 55.0 at
K=8) and flags_residual (44.9 / 46.5). The one robust fact is the loss itself (+14 on average over 20 runs). Phase 7
adds seeds to the key comparison.

### Phase 7 (queued 2026-10-16 10:40, N-Queens): seeds

`residual_k4_wta_s2`, `flags_k4_wta_s1`, `flags_k8_wta_s16_s1`, `residual_k8_wta_s16_s1`: does the learned
generator beat the fixed flags under the loss, and does K=8 beat K=4, once each has two or three seeds.

| configuration | seeds (acc @3000) | mean | cand_ce_gap (train) | eval MB |
|---|---|---|---|---|
| residual + wta, K=4 / 8 sims | 55.2 / 54.1 / 51.2 | **53.5** | 0.06-0.07 | 1180 |
| flags + wta, K=8 / 16 sims | 55.0 / 52.2 | **53.6** | 0.06-0.08 | 2786 |
| residual + wta, K=8 / 16 sims | 51.7 / 46.1 | 48.9 | 0.08-0.09 | 2792 |
| flags + wta, K=4 / 8 sims | 44.4 / 50.3 | 47.4 | 0.03 | 1139 |
| flags, no loss, K=4 / 8 sims (the recipe) | 34.9 / 37.3 | 36.1 | - | 1139 |
| flags, no loss, K=8 / 16 sims | 51.0 / 32.0 | 41.5 | - | 2786 |

**Reading.** See section 0, items 1-2. The seed spread of the trained runs is 4-6 points, so the ~6-point gaps
(learned over flags at K=4; K=8 over K=4 for the flags; K=4 over K=8 for the learned generator) are each about one
spread wide: consistent across the pairs, not individually conclusive.

**Reading of phase 1 so far.** The fast tier cannot answer the question. Every variant lands in the recipe's seed band
(80-89%), and the diagnostics say why: the candidates differ by ~0.02 nats in rollout CE (`cand_ce_gap`, logged 0.007), the search's
argmax agrees with the prior 98-100% of the time, and the search's argmax is the CE-best candidate ~100% of the time
(`cand_best_agree` logged 0.33 = 1.0 after the x3 correction): the search finds the best candidate, but with a 0.02-nat
gap the best is no better than the prior's pick. On unique-solution Sudoku one good u_t is enough, there is nothing for a second
proposal to be *for*, so neither a winner-take-all loss nor a bigger K has anything to buy. This is the "Sudoku is near
ceiling" caveat of REPORTS.md in its sharpest form. Phase 2 moves the question to 8x8 N-Queens, where the first loop
measured the eval search overriding the prior in ~70% of searches.

### Phase 1 (2026-10-16, 6x6 Sudoku): does any candidate loss stop the collapse at K=4, and does K=8 then pay?

Runs take 6-7 min each with the compiled C++ search (the first loop's 50 min were the Python search).
**Diagnostic scale caveat:** until commit `f615f83` (2026-10-16 02:30) the grad-cycle-only diagnostics
(`cand_loss`, `cand_ce_gap`, `cand_best_agree`, `cand_eps`, `balance`, `pick_entropy`, `recover_*`) were divided by
H_cycles=3 like the per-cycle search diagnostics, so the logged values of the runs below are 1/3 of the truth
(summary rows are quoted as logged; the readings use the corrected values). Runs started after that commit log them right.

| run | candidates | loss | K / sims (guided) | exact @2000 | notes |
|---|---|---|---|---|---|
| flags_k4 | flags | none | 4 / 8 (4) | **85.2%** (50.4 @1500) | the recipe, fresh on this branch; 337 s wall with the compiled C++ search; 593k params |
| residual_k4_none | residual | none | 4 / 8 (4) | **86.3%** (82.6 @1500) | no fix at all and no collapse: with the heads at 0.1x lr and value bootstrap (neither was on when residual collapsed in the first loop) residual candidates are at recipe level. But cand_div 0.06: the K candidates are nearly the same vector |
| residual_k4_wta | residual | wta 0.05 | 4 / 8 (4) | 80.3% | cand_ce_gap 0.0075 logged (0.02 nats): the candidates barely differ in outcome, so the winner-take-all has nothing to separate; cand_best_agree 0.33 logged = 1.0 |
| residual_k4_post | residual | posterior | 4 / 8 (4) | **88.5%** | best of the phase, inside the +-4 seed band of the baseline |
| residual_k4_soft | residual | soft grad | 4 / 8 (4) | diverged | loss NaN at step ~360 (kl_nats 13, cand_div 0.96: the candidates blew apart), then the C++ tree segfaulted on the NaN priors (now a FloatingPointError). Parked |
| full_k4_wta | full | wta 0.05 | 4 / 8 (4) | 83.2% | full-vector candidates no longer collapse either (cand_div 0.28 at eval, the most diverse of the phase) but cand_ce_gap 0.012: different vectors, same outcome |
| residual_k4_wta_s12 | residual | wta 0.05 | 4 / 12 (8) | 86.5% | cand_ce_gap 0.011; 8 guided simulations change nothing on this tier |
| residual_k8_wta_s16 | residual | wta 0.05 | 8 / 16 (8) | 85.6% | K=8 at 8 guided sims: level with K=4; cand_div 0.10, cand_ce_gap 0.024; eval memory 807 MB vs 358 (the arena scales with K x sims) |
