# TreeRM: training the candidate generator (autoresearch loop, started 2026-10-16)

Branch `treerm-candidates` (off `treerm-arc`, worktree `.claude/worktrees/treerm-candidates`). Runs under
`~/ARC-agi-testbed/autoresearch_runs/candidates6/` (`results.txt` has every run's summary), W&B online as `cand6_<name>`
in `ozeronatwanb/ARC-agi-testbed`. Harness: `scripts/candidates/{run6.sh,queue6.sh,summary.py}`.

## 0. Summary (read this first)

Pending: filled in as the loop produces results (section 4 is the log).

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

| tier | run | exact / NQueens acc | cand_div | cand_ce_gap | recover_acc | pick_entropy |
|---|---|---|---|---|---|---|
| sudoku6 | residual_k4_wta_bal | | | | | |
| sudoku6 | residual_k4_wta_rec | | | | | |
| sudoku6 | residual_k4_wta_anneal | | | | | |
| sudoku6 | residual_k4_wta_all3 | | | | | |
| nqueens8 | residual_k4_wta_bal | | | | | |
| nqueens8 | residual_k4_wta_rec | | | | | |
| nqueens8 | residual_k4_wta_anneal | | | | | |
| nqueens8 | residual_k4_wta_all3 | | | | | |

**Reading of phase 1 so far.** The fast tier cannot answer the question. Every variant lands in the recipe's seed band
(80-89%), and the diagnostics say why: the candidates differ by 0.007 nats in rollout CE (`cand_ce_gap`), the search's
argmax agrees with the prior 98-100% of the time, and the search's pick is the CE-best candidate only at chance rate
(`cand_best_agree` 0.33 for K=4). On unique-solution Sudoku one good u_t is enough, there is nothing for a second
proposal to be *for*, so neither a winner-take-all loss nor a bigger K has anything to buy. This is the "Sudoku is near
ceiling" caveat of REPORTS.md in its sharpest form. Phase 2 moves the question to 8x8 N-Queens, where the first loop
measured the eval search overriding the prior in ~70% of searches.

### Phase 1 (2026-10-16, 6x6 Sudoku): does any candidate loss stop the collapse at K=4, and does K=8 then pay?

Runs take 6-7 min each with the compiled C++ search (the first loop's 50 min were the Python search).

| run | candidates | loss | K / sims (guided) | exact @2000 | notes |
|---|---|---|---|---|---|
| flags_k4 | flags | none | 4 / 8 (4) | **85.2%** (50.4 @1500) | the recipe, fresh on this branch; 337 s wall with the compiled C++ search; 593k params |
| residual_k4_none | residual | none | 4 / 8 (4) | **86.3%** (82.6 @1500) | no fix at all and no collapse: with the heads at 0.1x lr and value bootstrap (neither was on when residual collapsed in the first loop) residual candidates are at recipe level. But cand_div 0.06: the K candidates are nearly the same vector |
| residual_k4_wta | residual | wta 0.05 | 4 / 8 (4) | 80.3% | cand_ce_gap 0.0075 nats: the candidates barely differ in outcome, so the winner-take-all has nothing to separate; cand_best_agree 0.33 (chance 0.25) |
| residual_k4_post | residual | posterior | 4 / 8 (4) | **88.5%** | best of the phase, inside the +-4 seed band of the baseline |
| residual_k4_soft | residual | soft grad | 4 / 8 (4) | diverged | loss NaN at step ~360 (kl_nats 13, cand_div 0.96: the candidates blew apart), then the C++ tree segfaulted on the NaN priors (now a FloatingPointError). Parked |
| full_k4_wta | full | wta 0.05 | 4 / 8 (4) | | |
| residual_k4_wta_s12 | residual | wta 0.05 | 4 / 12 (8) | | matched guided sims for the K=8 run |
| residual_k8_wta_s16 | residual | wta 0.05 | 8 / 16 (8) | | |
