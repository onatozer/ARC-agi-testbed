# TreeRM: training the candidate generator (autoresearch loop, started 2026-10-16)

Branch `treerm-candidates` (off `treerm-arc`, worktree `.claude/worktrees/treerm-candidates`). Phase 1 runs under
`~/ARC-agi-testbed/autoresearch_runs/candidates6/` (W&B `cand6_<name>`, harness `scripts/candidates/{run6.sh,queue6.sh}`);
phases 2+ under `~/ARC-agi-testbed/autoresearch_runs/candidates/<tier>/` (W&B `cand_<tier>_<name>`, harness
`scripts/candidates/{run.sh,queue.sh}` with the queue files in `scripts/candidates/queues/`), all in
`ozeronatwanb/ARC-agi-testbed`. `summary.py <run dir>` prints a run's eval rows and train means; each tier's
`results.txt` has the summaries, but its `===` headers are written when a run *starts* and the `##` blocks when it
*ends*, so read it by `##` block, not by header.

## 0. Summary (read this first)

Interim (2026-10-16 06:00; the K=8 N-Queens runs, a control and two seed replicates are still queued, section 4).

1. **The candidate loss is the change that matters, and it shows only where the search has something to decide.** On
   8x8 N-Queens (section 4, phase 2) residual candidates trained with the wta or posterior loss reach **51-55%**
   (`NQueens/accuracy` at step 3000) against **35% and 37%** for the flags recipe on two seeds and **30%** for residual
   candidates without the loss: +15-20 points, far outside the 2-point seed spread. The trained candidates also make
   the prior better calibrated (eval `argmax_agree` 0.80 vs 0.42, `kl_nats` 0.17 vs 0.47: the search overrides the
   prior in 20% of searches instead of 58%). On 6x6 Sudoku, where one good u_t is enough, every variant stays in the
   recipe's 80-89% band (phase 1) and the diagnostics say the candidates are identical in outcome.
   *Open confound* (phase 4, queued): the loss rolls out all K candidates with gradient, so a step trains the TRM
   core on K rollouts instead of one (train `lm_loss` 0.135 vs 0.177). `flags_k4_wta` (the loss on the fixed-direction
   candidates, no learned generator) separates "learned proposals" from "more gradient per step".
2. **The three diversity mechanisms do not make the candidates more different in outcome** (phase 3, both tiers).
   Balancing works as a router constraint (pick entropy 1.35 of the maximum 1.39, balance loss at its minimum) and
   recoverability trains to 0.52 on N-Queens (chance 0.25) and chance on Sudoku, but `cand_div` (0.07-0.11) and
   `cand_ce_gap` (0.05-0.10 nats) are the same with and without them, and the accuracies are the wta-alone numbers
   (N-Queens 52-53% for bal / rec / all3; Sudoku 88-90%). The annealed wta is the one outlier in both directions:
   92.8% on Sudoku (the best number of the loop, +7.6 over the recipe) and 44.0% on N-Queens (the worst of the
   trained-candidate runs); a Sudoku seed replicate is queued.
3. **Still unanswered: does K=8 pay once the candidates are trained?** `residual_k8_wta_s16` (K=8, 16 simulations, so
   8 guided as at K=4 / 12) is running, with `residual_k8_none_s16` and `flags_k8_s16` behind it. On Sudoku it was
   level with K=4 (85.6%).

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
| residual_k8_wta_s16 | residual | wta 0.05 | 8 / 16 (8) | running | | | | | |
| residual_k8_none_s16 | residual | none | 8 / 16 (8) | queued | | | | | |
| flags_k8_s16 | flags | none | 8 / 16 (8) | queued | | | | | |

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

### Phase 4 (queued 2026-10-16 06:00): the control and the replicates

`flags_k4_wta` (N-Queens: the wta loss on the fixed-direction flag candidates, which have no learned generator; if it
also reaches ~55% the gain is the K-rollout signal, not the proposals), `residual_k4_wta_s1` (seed replicate of the
55.2%), `residual_k4_wta_anneal_s1` (Sudoku: seed replicate of the 92.8%).

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
