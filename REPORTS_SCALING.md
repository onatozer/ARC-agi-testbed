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
| 2 | Sudoku fast tier, identity candidate, K = 1/2/4/8 (train sims 2K), seed 0 | 91.4 / 92.2 / 92.4 / 85.7%; K=1 (plain TRM, no search) is already at the recipe's level |
| 3 | N-Queens 8x8, identity candidate, K = 1/2/4/8, seed 0 | 51.9 / 54.2 / 50.4 / 55.9%; non-monotone, all within the seed spread; K=1 = the old recipe's best (44-55%) |
| 4 | control: K=1 with candidate_loss=none, seed 0 | Sudoku 87.9, N-Queens 40.8: the candidate loss at K=1 (a doubled lm loss) is +3.5 / +11; the max-steps loss head is the other +20 over TRM |
| 5 | eval-sims sweeps from the N-Queens K=8 and K=4 checkpoints | K=8: 51.9 / 53.1 / 53.6 at 8 / 16 / 32 sims; K=4: 49.9 / 49.4 / 49.8 / 49.6 at 4-32: the search is worth at most ~2 pts over a random pick among its candidates |
| 6 | TreeGRAM (search over GRAM's prior), K=4 and K=1, seed 0 | Sudoku 81.3 / 88.7, N-Queens 30.7 / 41.6: searching prior samples is worse than the prior mean; the mean is below TreeRM K=1 |
| 9 | round 2: argmax eval pick + pick diagnostics on the saved N-Queens checkpoints | +1 pt at K=8, 0 at K=4: the candidates' CEs differ by 0.01-0.02 nats and the value head ranks them at 0.2-0.3 (0 for TreeGRAM) |
| 10 | round 2: MiniARC 5x5, TreeRM identity K=4 vs K=1, pick x sims from the K=4 checkpoint | pass@2 4.7 vs 4.7; candidates differ more (CE gap 0.067) but the value head is blind (0.04-0.07) and the pick is at chance; sims inert |
| 11 | round 2: TreeGRAM K=4 with the visit-weighted prior target (after fixing it) | worse everywhere: Sudoku 59.2, N-Queens 13.6, MiniARC pass@2 2.7-4.0; pick sweeps flat, value head 0.06-0.09 / negative |

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

## 3. Stages 2-3: the identity candidate, K = 1 / 2 / 4 / 8 (seed 0)

Code (commit b5c8fed): slot 0 of the flag candidates is `u_t` itself (`directions[0] = 0`), so the plain update is
always among the K proposals, candidate sets are nested in K, and K=1 is plain TRM bit-exactly (asserted in
`tests/test_treerm.py`). Training sims = 2K (guided sims = K, the convention of every earlier run), in-training eval
sims 16 (1 at K=1), `candidate_loss=wta`, `value_bootstrap_weight=0.5`, the tiers' COMMON settings
(`scripts/candidates/queue.sh`, queue `queues/id_k.txt`, `KEEP_CKPT=1`).

| tier | K=1 (sims 2) | K=2 (4) | K=4 (8) | K=8 (16) | wall K=1 / K=8 | eval GPU K=1 / K=8 |
|---|---|---|---|---|---|---|
| Sudoku-6u, 2000 steps, 512 puzzles | 91.4 | 92.2 | 92.4 | 85.7 | 256 / 387 s | 213 / 644 MB |
| N-Queens 8x8, 3000 steps, 1000 puzzles | 51.9 | 54.2 | 50.4 | 55.9 | 665 / 1294 s | 678 / 2226 MB |

Search diagnostics at the end (N-Queens): argmax_agree 0.77 / 0.64 / 0.76 at K=2/4/8, post_entropy 0.36 / 0.85 / 1.61
(the K/sims floor of the sampled pick: ln 8 = 2.08), cand_div 0.04 / 0.11 / 0.05.

Reading:
- **K=1 is the recipe's level.** With no candidates and no search, the TreeRM wrapper scores 51.9% on N-Queens,
  where TRM-192 scored 20-21% and every TreeRM recipe 44-55% (REPORTS.md, REPORTS_CANDIDATES.md), and 91.4% on Sudoku
  against the 66-74% TRM band. What differs from TRM at K=1 is the recipe around the search: GRAM's max-steps loss
  head (no halting), the value head, and the candidate loss, which at K=1 is a second copy of the lm loss. The gains
  credited to the search and to the candidate loss were mostly this. Control queued: K=1 with `candidate_loss=none`.
- **K does not pay, in either direction.** The four K points span 5.5 pts on N-Queens and 6.7 on Sudoku, in no
  order, with one seed; the trained-run seed spread is 4-6 pts. The identity candidate removes the "more flags are
  worse" failure (K=8 is the best N-Queens point) but, as predicted in section 1, it cannot make K monotone while the
  pick is a sample over visit counts.

## 4. Control: K=1 without the candidate loss (seed 0)

At K=1 `candidate_loss=wta` rolls out the single candidate to the end of the step and adds its CE: a second copy of
the lm loss. `queues/control_k1.txt` drops it (`candidate_loss=none`); everything else as in section 3.

| tier | K=1, wta | K=1, none | TRM baseline (REPORTS.md) |
|---|---|---|---|
| Sudoku-6u | 91.4 | 87.9 | 66-74 (TRM-128, ACT head) |
| N-Queens 8x8 | 51.9 | 40.8 | 20-21 (TRM-192, ACT head) |

Reading (one seed each): the doubled lm term is worth ~3.5 pts on Sudoku and ~11 on N-Queens, which is most of the
"+14 from candidate_loss" of REPORTS_CANDIDATES.md; the max-steps GRAM loss head without halting (plus the value
head, inert at K=1) is worth the other ~20 pts over TRM on N-Queens. Neither has anything to do with the search. The
search's own contribution, K=8 vs K=1 at equal recipe, is +4 on N-Queens and -6 on Sudoku at one seed: noise.

## 5. Eval-time simulations on the N-Queens checkpoints (seed 0 weights, eval seed 0)

`sims_sweep.sh` on the section 3 checkpoints (1000 test puzzles, one model build per checkpoint). At sims = K every
simulation is a forced root visit, so the visit distribution is uniform and the pick is a uniform random candidate.

| checkpoint | sims = K (random pick) | 2K | 4K | 8K | in-training eval (16 sims) |
|---|---|---|---|---|---|
| K=8 (train sims 16) | 51.9 | 53.1 | 53.6 | - | 55.9 |
| K=4 (train sims 8) | 49.9 | 49.4 | 49.8 | 49.6 | 50.4 |

Reading: flat at K=4 (0.5 pt span); a weak upward trend of 1.7 pts from 8 to 32 sims at K=8, inside the eval-seed
noise (the same weights scored 55.9 in the in-training eval at 16 sims). The search is worth at most ~2 pts over a
random pick among its own candidates, which agrees with section 2 (Sudoku, 0 pts) and with the candidates being
near-duplicates (cand_div 0.05-0.11): there is little for a better pick to choose between. Eval cost: 21-86 s per
budget on 1000 puzzles (the first budget includes the compile).

## 6. TreeGRAM: the search over GRAM's learned prior (seed 0)

Idea 1: candidates from GRAM's prior p(eps | u_t) instead of a candidate head, MCTS as the test-time procedure.
Code (commit 49422d0): `TreeRMInner._prior_logits` hook (the scorer's logits; TreeRM unchanged, 22 tests),
`models/recursive_reasoning/treegram.py`: `TreeGRAMInner` overrides `_candidates` with slot 0 = `u_t + mu` and
slots 1..K-1 = `u_t + mu + sqrt(var) * noise`, and `_prior_logits` with a uniform prior (Sampled MuZero: the
samples carry the prior), so there is no scorer and no generator; `config/arch/treegram.yaml` inherits every
TreeRM key. K=1 is GRAM's `eval_prior_mean` inference bit-exactly (`tests/test_treegram.py`). The prior head
trains pathwise through the picked sample, the candidate loss (every sample) and the lm loss; nothing else was
added (the visit-weighted Gaussian NLL target of the design notes is the next option, not built). 756k params vs
TreeRM's 593k (the Gaussian head's SwiGLU). Same recipe as section 3 (`queues/treegram.txt`).

| tier | TreeGRAM K=4 (8 sims) | TreeGRAM K=1 (prior mean) | TreeRM identity K=4 | TreeRM identity K=1 |
|---|---|---|---|---|
| Sudoku-6u | 81.3 | 88.7 | 92.4 | 91.4 |
| N-Queens 8x8 | 30.7 | 41.6 | 50.4 | 51.9 |

Diagnostics at the end (N-Queens K=4): eval cand_div 0.025 (the prior's variance collapses, so the samples are
near-duplicates too), post_entropy 1.07, argmax_agree 0.41 (the search picks a noise sample over the mean 59% of
the time); train cand_div 0.21 on Sudoku. Wall / eval GPU: Sudoku 252-387 s, 220 MB; N-Queens 687-933 s, 1.1 GB.
Eval-sims sweep of the N-Queens K=4 checkpoint (eval seed 0): 31.7 / 30.8 / 31.4 / 31.5 at 4 / 8 / 16 / 32 sims,
the random pick (sims = K) best of all: the budget is as inert here as for the flag candidates.

Reading: searching over prior samples is worse than taking the prior mean by 7-11 pts at one seed, and the prior
mean itself is below TreeRM's K=1 by 3-10 pts. With the pick a sample over visit counts and the value head unable
to rank near-duplicates, each noise sample that is picked is a small random perturbation of z_H that the lm loss
then has to absorb; more candidates mean more such picks. Not a test-time scaling procedure as built: the prior
would need a target that uses the search (the Gaussian NLL of the search-weighted samples) or a deterministic
Q-based pick before K or the budget could pay. Memory and wall are at TreeRM's level, as designed.

## 7. Where this leaves the question

1. **The non-monotonicity is not noise around a working search; the search is inert.** On every checkpoint, picking
   a candidate uniformly at random (sims = K) scores the same as the full search, to within 2 pts. More K, more
   simulations or more depth cannot help a search that has nothing to choose between (cand_div 0.03-0.11) and picks
   by sampling. The identity candidate (idea 2) removes the "more flags are worse" failure mode and makes K=1 an
   exact TRM control, which is what exposed the rest; it does not, and cannot by itself, make K pay.
2. **TreeRM's gains over TRM are the recipe.** K=1 with no search reproduces the old recipe's best numbers; the
   max-steps loss head and the doubled lm loss (the candidate loss at K=1) account for the gap to the TRM baselines
   on both tiers. Any future TreeRM claim needs the K=1 identity run as its control, not TRM with the ACT head.
3. **MCTS over the GRAM prior (idea 1) as built does not scale either**: noise samples are worse than the mean and
   the prior's variance collapses. What would be needed, in order of cost: (a) a deterministic, Q-informed root pick
   at eval (Gumbel MuZero's argmax of logits + sigma(Q); `argmax_pick` was stripped in #64), (b) a training target for
   the proposals that uses the search (the visit-weighted Gaussian NLL for TreeGRAM; completed-Q / sequential halving
   for the visit targets), (c) a measured value head (eval `lprm_loss`, rank correlation with the true CE), since the
   whole chain rests on it. None of these was asked for; they are the companions the research notes rank first.
4. **Not measured here:** more than one training seed per cell (the trained-run spread is 4-6 pts, so single-seed
   differences under ~6 are not differences); the TRM-with-ACT-head control inside this harness (the 20-21 / 66-74
   baselines are the earlier reports'); depth as an axis (it is not a config knob after the strip).

## 8. Round 2 (user, 2026-10-17 evening): deterministic pick, measured value head, proposal target, MiniARC 5x5

Built (commit a558245), per the user's choice of the two companions in section 7 and the point that Sudoku may be too
easy to show a search gain while MiniARC 5x5's low accuracy says the latent's basin is often wrong:
- `arch.eval_pick: argmax` (default; `sample` = the old draw). At eval the search takes its most-visited candidate.
  Training keeps the straight-through draw (the scorer's gradient).
- The value head measured: at eval every candidate is rolled out to the end of the step and scored with the label
  (the eval loss head now hands the inner its loss_fn like the training head), giving `search/pick_ce_regret`
  (the picked candidate's CE above the CE-best one), `search/pick_is_best`, `search/cand_ce_gap` and
  `search/value_rank_corr` (Spearman of the LPRM values with -CE over the K candidates: 1 = the value head ranks
  the candidates as the label would, 0 = no information).
- `TreeRMInner._prior_loss` hook (the balanced KL) and TreeGRAM's `arch.prior_target: visits` (default): the
  search's visit-weighted Gaussian NLL of its own samples under the prior, Sampled MuZero's target, so mu follows
  the samples the search favoured and var their spread instead of collapsing (section 6). `none` = section 6.
- `treerm_sims.yaml` sweeps pick x sims from one checkpoint; `scripts/miniarc5/queue_scaling.sh` runs the MiniARC
  tier (standard build `miniarc5-aug-31`, rebuilt: 149 tasks, 4768 test rows; recipes.sh plus `puzzle_emb_lr=3e-2`).

Runs, in order on the one GPU: (a) the section 3 / 6 N-Queens checkpoints under both picks at 2K / 4K / 8K sims
with the new diagnostics; (b) TreeGRAM K=4 with the target on Sudoku and N-Queens against section 6; (c) MiniARC
5x5: TreeRM identity K=4, its K=1 control, TreeGRAM K=4 with the target (4000 steps each), then both picks from
the K=4 checkpoints.

## 9. Round 2 (a): the deterministic pick and the value head, on the saved N-Queens checkpoints (seed 0 weights)

`sims_sweep.sh` with `pick=[argmax,sample]`; the sample rows reproduce section 5 within eval noise. The pick
diagnostics roll every candidate out to the end of the step: `pick_is_best` = the picked candidate is the CE-best
(chance 1/K), `cand_ce_gap` = CE of the worst minus the best candidate, `value_rank_corr` = Spearman of the LPRM
value head's ranking of the K candidates with the label's (1 = perfect, 0 = none).

| checkpoint | sims | sample pick | argmax pick | pick_is_best (argmax) | value_rank_corr | cand_ce_gap (nats) |
|---|---|---|---|---|---|---|
| TreeRM identity K=8 | 16 / 32 | 53.0 / 53.9 | 54.0 / 54.1 | 0.33 (chance 0.125) | 0.22 | 0.019 |
| TreeRM identity K=4 | 8 / 16 / 32 | 49.3 / 49.8 / 49.4 | 49.0 / 48.8 / 48.8 | 0.47-0.49 (chance 0.25) | 0.26-0.27 | 0.011 |
| TreeGRAM K=4 (no target) | 8 / 16 / 32 | 27.3 / 27.9 / 27.7 | 28.2 / 28.5 / 29.0 | 0.26-0.27 | 0.03-0.06 | 0.018 |

Reading: the deterministic pick is worth about 1 pt at K=8 and nothing at K=4, as the diagnostics predict: the K
candidates' final CEs differ by 0.01-0.02 nats (the best and the worst candidate are the same answer), and the
value head ranks them at 0.2-0.3 correlation (0 for TreeGRAM). The pick does pick better than chance (0.33 vs
0.125 at K=8), it just has nothing to gain from. Order of business confirmed: proposals that differ first.

Round 2 (b), the divergences: every TreeGRAM run with the target went NaN within ~100 steps (Sudoku, N-Queens,
MiniARC; the C++ search then segfaults on the NaN values at the first eval, exit 139, and each crash left an
orphaned python holding GPU memory, one of them the lock, which stalled the queue for 11 hours overnight). Two
reformulations (draws only; mean by MSE, variance by the likelihood with detached residuals) did not help. A
per-step trace on the test model (`debug_tgv.py`: loss components, var, |mu| every 10 steps) showed |mu| at 20x its
normal value by step 20 and var growing 700x by step 100. The bug: a candidate is `u_t + mu + sqrt(var) * noise`,
and the target compared the candidate itself to mu instead of the draw's offset `cand - u_t`, so the mean chased
u_t plus its own output, which runs away at any variance. With the offset the trace is as stable as the run without
the target (var 0.60 vs 0.53, cand_div 0.11 vs 0.09 at step 150). Lesson for the harness: the lock must be
`flock -o` everywhere (`scripts/miniarc5/run.sh` is not), and a NaN loss should stop a run before the search sees it.

## 10. Round 2 (c): MiniARC 5x5 (standard build, 4000 steps, puzzle_emb_lr 3e-2, seed 0)

`scripts/miniarc5/queue_scaling.sh`; metrics from `scripts/miniarc5/summarize.py <run dir>`; REPORTS_ARC.md's
references at this recipe: TreeRM (flags, no candidate loss) pass@2 7.4 / 4.7 over 2 seeds, TRM 8.1.

| run | pass@1 | pass@2 | pass@5 | exact acc | wall | eval GPU |
|---|---|---|---|---|---|---|
| TreeRM identity K=4 (wta, argmax pick) | 2.7 | 4.7 | 6.7 | 1.5 | 1298 s | 550 MB |
| TreeRM identity K=1 (control) | 3.4 | 4.7 | 7.4 | 1.2 | 764 s | 364 MB |
| TreeGRAM K=4, prior target (fixed, section 11) | 1.3 | 2.7 | 6.0 | 0.6 | 1326 s | 544 MB |

Pick x sims sweep of the K=4 checkpoint (4768 test rows, batch 149, eval seed 0):

| pick | sims | pass@1 | pass@2 | pick_is_best (chance 0.25) | value_rank_corr | cand_ce_gap | argmax_agree |
|---|---|---|---|---|---|---|---|
| argmax | 8 / 16 / 32 | 2.7 / 2.7 / 2.7 | 4.7 / 4.7 / 4.0 | 0.31-0.32 | 0.04-0.05 | 0.067 | 0.31-0.35 |
| sample | 8 / 16 / 32 | 2.7 / 2.7 / 2.7 | 3.4 / 3.4 / 3.4 | 0.29-0.31 | 0.06-0.07 | 0.068 | 0.33-0.36 |

Reading: K=1 equals K=4 on MiniARC as on the other tiers (pass@2 4.7 both; pass@1 and pass@5 higher at K=1).
The candidates do differ more here (cand_ce_gap 0.067 nats, 3-6x the N-Queens gap: the user's point about wrong
basins holds), and the search overrides the prior two times in three, but the value head cannot rank them
(0.04-0.07) and the pick is at chance (0.31 vs 0.25), so the argmax pick's +1.3 pass@2 over the sample pick (two
tasks of 149) is within noise. More search budget changes nothing. Diverse candidates plus a blind value head is
the MiniARC failure mode; the sims/pick axis cannot pay until the value head does.

## 11. Round 2 (b): TreeGRAM with the visit-weighted prior target (fixed; seed 0)

`arch=treegram` with `prior_target: visits` (commit d942e1c), otherwise the section 6 recipe; MiniARC per section 10.

| tier | TreeGRAM K=4 + target | TreeGRAM K=4, no target (s. 6) | TreeGRAM K=1 (prior mean) | TreeRM identity K=4 (s. 3/10) |
|---|---|---|---|---|
| Sudoku-6u, 2000 steps | 59.2 | 81.3 | 88.7 | 92.4 |
| N-Queens 8x8, 3000 steps | 13.6 | 30.7 | 41.6 | 50.4 |
| MiniARC 5x5, 4000 steps (pass@2) | 2.7 (in-training eval) / 2.7-4.0 (sweep) | - | - | 4.7 |

Training was finite throughout (no NaN in any log; the loss goes negative because of the log-variance term). Eval
diagnostics at the end: N-Queens cand_div 0.05 (0.03 at step 3000 in-training), cand_ce_gap 0.013, pick_is_best
0.27-0.29 (chance 0.25), value_rank_corr 0.06-0.09, argmax vs sample pick 14.5-14.9 vs 13.8-14.8 at 8-32 sims;
MiniARC cand_ce_gap 0.050, pick_is_best 0.19-0.25, value_rank_corr -0.02 to -0.04, both picks 2.7-4.0 pass@2.
Sudoku kept its draws apart (cand_div 0.18) and still lost 22 pts.

Reading: pulling the prior toward the draws the label-valued training search favoured is the same thing GRAM's
posterior KL does, and REPORTS.md found that label-informed guidance is what GRAM's prior fails to learn (it fits
the training set and does not transfer). Here it is worse than GRAM's version because the "posterior" is a handful of
visit counts over 3 random draws, so the mean is dragged by sampling noise that happened to score well on the label,
and the lm loss then has to undo it. Candidate diversity does not rise where it matters (N-Queens 0.03-0.05), the
value head stays blind, and the pick and budget axes stay flat.

## 12. Where round 2 leaves the question

- Both companion mechanisms the user chose are built, tested and measured: the deterministic pick is worth at most
  1 pt and only where the candidates already differ a little (K=8 N-Queens); the proposal target, once its bug was
  fixed, lowers accuracy on all three tiers. Neither makes K or the search budget pay.
- The diagnostics are the lasting result: `pick_is_best`, `cand_ce_gap` and `value_rank_corr` say, per checkpoint,
  whether there is anything to choose between and whether the value head can see it. Today: TreeRM's flag candidates
  are the same answer (gap 0.01-0.02 nats, head at 0.2-0.3); MiniARC's candidates differ (0.05-0.07) but the head is
  at 0 (its single-token, short-rollout view does not predict which perturbation lands in the right basin); TreeGRAM's
  draws are both near-duplicates and unrankable.
- What a working search would need, in the order the evidence suggests: (1) a value head that can rank candidates
  on MiniARC (train it on the eval-time question, i.e. on rolled-out candidates' correctness with the label, and read
  more than token 0 of the state); only then (2) proposals that differ in outcome and are not label-steered (a
  diversity objective on the rollouts' predictions rather than a visit-count target); then the pick and budget axes.
  None of this is built. Single seeds throughout; the N-Queens trained-run spread is 4-6 pts, so nothing here
  under ~6 pts is a difference, but the target's -17 to -22 and the K=1 equivalences are well outside it.

Harness for whoever continues: `queues/id_k.txt`, `queues/control_k1.txt`, `queues/treegram.txt` with
`KEEP_CKPT=1 bash scripts/candidates/queue.sh <tier> <queue>`; `sims_sweep.sh <ckpt> '[sims]' [seed] [batch] [data]`
and `sims_table.py <run dir>`; results under `~/ARC-agi-testbed/autoresearch_runs/candidates/{sudoku6,nqueens8,sims_ref,sims_nq}/`.
