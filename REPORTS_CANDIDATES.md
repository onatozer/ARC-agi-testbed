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
4. Tests: `tests/test_treerm.py` (with the scorer's candidate input detached, `none` gives exactly one slot a gradient
   and `wta` / `posterior` / soft grad give every slot one; the rollout path with `grad_cycles=2`),
   `tests/test_param_lr_scales.py` (arch scales under train scales).

## 3. Protocol

Fast tier of REPORTS.md: 6x6 Sudoku with unique solutions (`data/sudoku-6u`), hidden 128, batch 64 x 2000 steps, lr
1e-3, eval on 512 test puzzles every 500 steps, metric `all/exact_accuracy` at step 2000, seed 0 unless stated. The
recipe's seed spread on this tier is about +-4 points. One GPU job at a time (the shared `gpu.lock`). Every root child
is rolled out once before PUCT, so K and the simulation count are tuned together: "guided" simulations =
`mcts_simulations - K`.

## 4. Log

### Phase 1 (queued 2026-10-16): does any candidate loss stop the collapse at K=4, and does K=8 then pay?

| run | candidates | loss | K / sims (guided) | exact @2000 | notes |
|---|---|---|---|---|---|
| flags_k4 | flags | none | 4 / 8 (4) | | the recipe, fresh on this branch |
| residual_k4_none | residual | none | 4 / 8 (4) | | the collapse control |
| residual_k4_wta | residual | wta 0.05 | 4 / 8 (4) | | |
| residual_k4_post | residual | posterior | 4 / 8 (4) | | |
| residual_k4_soft | residual | soft grad | 4 / 8 (4) | | |
| full_k4_wta | full | wta 0.05 | 4 / 8 (4) | | |
| residual_k4_wta_s12 | residual | wta 0.05 | 4 / 12 (8) | | matched guided sims for the K=8 run |
| residual_k8_wta_s16 | residual | wta 0.05 | 8 / 16 (8) | | |
