# TreeRM: does accuracy scale with K, simulations and depth?

Branch `treerm-scaling` (off `treerm-drop-generator`, PR #65). The question (2026-10-17): an MCTS should improve
monotonically with more candidates (K), more simulations and more depth; TreeRM's logs do not show that. Two ideas
to test: (1) MCTS over the GRAM/GRAMDP prior as a test-time scaling procedure; (2) flags plus the unperturbed z_H as
a candidate, so a larger K can never make the candidate set worse. The research notes behind this file are under
`research_notes/TreeRM K and depth scaling/` and the synthesis in `reports/TreeRM K and depth scaling.md` (both
untracked). One GPU job at a time (4 GB laptop GPU, `gpu.lock`).

## 0. Scoreboard

| stage | what | result |
|---|---|---|
| 1 | eval-sims sweep, reference Sudoku-6u checkpoint (old K=4 flags, 1000 puzzles), 2 eval seeds | flat: 86.5-87.1% at 4/8/16/32 sims; sims=K (uniform random pick) is as good as 32 |

## 1. Diagnosis from the existing runs and code (no GPU)

Measured so far (REPORTS*.md, `autoresearch_runs/candidates/*/results.txt`): every K / sims difference is inside
the seed spread (4-6 pts on trained N-Queens runs, about 4 on Sudoku, 2-7 tasks on MiniARC). No accuracy-vs-depth
measurement exists: eval depth is `H_cycles - cycle` (3/2/1) and `mcts_horizon` was stripped.

Why the knobs cannot matter in the current code:
1. **Training search is depth 1.** Search runs on the grad cycle only with `remaining = 1`, so root children are never
   expanded; every simulation beyond the K forced visits re-scores an already-scored child with the same label CE.
   The visit vector is `[sims-K+1, 1, ...]` (predicted train posterior entropies 1.07 / 1.54 / 1.17 for K4-8 / K8-16 /
   K8-24 sims match the logged 1.14-1.19 / 1.61-1.71 / 1.40). `mcts_simulations` only sets the KL target's sharpness.
2. **The pick is a sample.** The picked candidate is a Gumbel sample from the visit distribution, which has a `K/sims`
   uniform floor from the forced root visits: P(pick = search favourite) is about 0.81 at K=4 / 16 sims but 0.56 at
   K=8 / 16 sims. Doubling K doubles the random floor (eval posterior entropy 1.55-1.72 at K=8 vs 0.70-0.97 at K=4).
3. **Candidates are near-duplicates.** `cand_ce_gap` 0.03-0.10 nats, `cand_div` 0.02-0.11, not rising with K.
4. **The eval value head is unmeasured.** No eval `lprm_loss`; the bootstrap MSE is folded into `kl_loss`. Across the
   24 N-Queens runs, more overrides go with lower accuracy (argmax_agree 0.42-0.47: 32-37%; 0.68-0.83: 44-55%).

Literature (Gumbel MuZero, Grill et al. 2020, Sampled MuZero): visit-count targets plus a sampled pick are not a
policy improvement at 4-8 guided simulations; the guarantee needs a deterministic Q-informed root pick and a target
that falls back to the prior for unvisited candidates.

Harness bugs found and fixed on this branch:
- `evaluate.py`: `param_iter` arch overrides reached `loss_head.model.config` but not `inner.config` (GRAM/TreeRM
  build the inner from a dict), so `arch.eval_mcts_simulations` sweeps from a checkpoint were silently inert.
  `_apply_runtime_arch_overrides` now applies them to the inner config too.
- `scripts/candidates/run.sh` deleted every final `.pth`; `KEEP_CKPT=1` keeps it.
- New: `config/eval/treerm_sims.yaml` + `scripts/candidates/sims_sweep.sh <ckpt> '[sims...]' [seed] [batch] [data]`
  (one model build, one eval per budget) and `sims_table.py` to read the result JSONs.

## 2. Stage 1: eval-time simulations on one checkpoint (Sudoku-6u)

Reference checkpoint `ref_sudoku6u` (REPORTS_MCTS, K=4 flags, trained before the candidate loss), 1000 test puzzles,
batch 250, `eval_mcts_simulations` swept from one build. Two eval seeds (they change the Gumbel pick and the batch
composition only; the weights are the same).

| eval sims | exact acc s0 | exact acc s1 | argmax_agree | post_entropy | eval s |
|---|---|---|---|---|---|
| 4 (= K: uniform random pick) | 0.870 | 0.865 | 1.00 | 1.386 (= ln 4) | 36-71 |
| 8 | 0.871 | 0.870 | 0.56 | 1.27 | 15-23 |
| 16 | 0.871 | 0.869 | 0.46 | 1.21 | 15 |
| 32 | 0.870 | 0.867 | 0.48 | 1.07 | 24 |

Reading: the test-time search budget changes nothing on Sudoku, to within 0.5 pt, although the search overrides the
prior in about half the puzzles at 8+ sims. At sims = K the pick is uniform over the four candidates and accuracy is
the same, so the four candidates lead to the same answer: finding 3 above (near-duplicate candidates) measured
directly. The Sudoku fast tier cannot show search scaling; the search question has to be asked on N-Queens, where
the override rate is 0.2-0.6 and tracks accuracy.
