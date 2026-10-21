# First-order MAML on MiniARC 5x5: report

Branch `l-reinit-ttt`, worktree `.claude/worktrees/treerm-candidates`. Sections 1-7 of the MiniARC work are in
REPORTS_ARC.md; this file holds the MAML loop and is updated as each run lands.

## Setup

**Idea (user, 2026-10-19).** Train the weights *for* test-time adaptation: MAML (Finn et al. 2017, arXiv 1703.03400;
reference code cbfinn/maml). Per task, a few SGD steps on its support examples from the shared weights θ give θ'_i,
and θ is trained by the query loss at θ'_i summed over a meta-batch of tasks (Adam). The user chose the paper's
first-order variant (section 5.2: drop the second derivatives; MiniImagenet 48.1 vs 48.7% 1-shot, 63.2 vs 63.1%
5-shot, a third less compute; `stop_grad=True` in the reference code).

**Mapping to this codebase** (`pretrain.py`: `task_grads`, `maml_adapt`, `maml_batch`; `maml.enabled=true
maml.inner_steps maml.inner_lr`). A task is one augmented copy of a MiniARC task; a training batch already holds whole
copies (rows contiguous per copy, each copy with its own puzzle identifier), so the meta-batch is the copies in one
loader batch (≈ `global_batch_size / 5`). Support / query = leave-one-out within the copy (one random row is the
query, the paper's separate D'_i; copies with a single row are skipped). The task loss runs the puzzle from a fresh
carry through every supervision step as eval does, each step's gradient taken on its own (the inner reads the previous
step's state with graph and `_reset_rows` writes in place into saved tensors, so one summed backward fails) and
normalised per row and step so α is on the paper's scale. Inner loop: plain SGD in place; meta update: the recipe's
AdamAtan2 and schedule, everything after the gradient shared with plain training (logging keys
`train/fprl/query_loss`, `train/fprl/support_loss`). No puzzle embeddings (`arch.puzzle_emb_ndim=0`: the task identity
comes from the adaptation; the SignSGD-updated embedding buffer cannot sit in an SGD inner loop), eager mode
(`DISABLE_COMPILE=1`: in-place parameter updates under a compiled module are untested and the per-task row counts
would recompile). Test time (`evaluate_ttt.py maml_steps=N`): the same inner loop, N SGD steps on all of the task's
demos (every aux copy's, 48 rows) at once; `maml_steps=0` runs the ordinary `ttt_mini` loop on the same checkpoint.

**Step size probe** (`maml_adapt`, 3 steps, 2 held-out tasks, from the shared `trm_ho2` checkpoint, so only the SGD
scale is tested): support loss before step 1 -> before step 3, per task: α 0.01: 0.87 -> 0.84, 1.22 -> 1.13; α 0.1:
0.87 -> 0.71, 1.22 -> 0.56; α 1.0: 0.87 -> 0.43, 1.22 -> 0.76. α = 0.1 (the paper's Omniglot 20-way value) for the
runs. Before the per-step normalisation the same probe at α 0.1 diverged (20 -> 511).

**Cost.** The per-task inner loops are sequential and eager: the 40-step smoke at batch 32 (~6 tasks) took 373 s
including start-up, ≈ 7 s per meta-step, so the runs use batch 16 (~3 tasks per meta-step, as the paper's
MiniImagenet meta-batch of 4) and 2000 meta-steps (`scripts/miniarc5/queue_maml.sh`).

**Runs:** `maml_trm` (shared block), `maml_trm_sep` (`arch.H_layers=2`, section 7's best), `plain_noemb` (the paper's
baseline: the same no-embedding model trained plainly at the section-1 recipe, 4000 steps). TTT over the 30 held-out
tasks: `mm_s1` / `mm_s5` / `mm_s10` (1 / 5 / 10 inner SGD steps at α 0.1) and `mm_adam` (the ordinary `ttt_mini`
AdamAtan2 loop, `maml_steps=0`), the plain model with the same SGD adaptation (`+train_overrides.maml.*`).

## Results

### Progress log (updated as runs land)
- 2026-10-20 00:30, `maml_trm` at meta-step 280 of 2000 (4.9 s per step, no errors). Block means over ~55 meta-steps
  (~3 tasks each): support loss before adaptation 2.2 / 1.8 / 2.4 / 1.8 / 2.1 (noisy: 2-5 rows per task), query
  loss after the one adapted step 1.53 / 1.53 / 1.51 / 1.51 / 1.45: the post-adaptation query loss is drifting down
  slowly. `maml_trm_sep`, `plain_noemb` and the TTT runs are queued.
- 2026-10-20 02:20, `maml_trm` at meta-step 969 (4.9 s per step after a slower hour at a lower GPU clock following a
  laptop restart; the process was not interrupted). Block means over ~193 meta-steps: query loss after the adapted
  step 1.54 / 1.45 / 1.42 / 1.41 / 1.44, flat since step ~400; support loss before adaptation 2.07 / 1.85 / 1.79 /
  2.04 / 2.13, not improving: the shared initialisation is not getting better at a task unadapted, only (slightly)
  after its one SGD step, which is what the meta-objective asks for.
- 2026-10-20 04:35, `maml_trm` finished: 2000 meta-steps in 11,679 s (3.2 h, GPU clock 600-750 MHz most of the way),
  no errors. Final blocks: query loss after adaptation 1.35 (from 1.49), support loss before adaptation 1.64 (from
  1.95): both moved in the second half. `maml_trm_sep` (separate H block) is training next, then `plain_noemb`, then
  the TTT variants.
- 2026-10-20 07:20, `maml_trm_sep` finished: 2000 meta-steps in 9,370 s, no errors; query loss after adaptation 1.38 ->
  1.03, flat over the last 800 steps. `plain_noemb` (13 min) is training, then the 11 TTT variants.
- 2026-10-20 07:45, `plain_noemb` finished (4000 steps, 463 s; no-adaptation eval 0 of 30, token acc 0.65 at step
  4000, i.e. the same as the meta-trained separate-H model unadapted). First TTT variant, `maml_trm mm_s1` (the
  training-time inner loop, one SGD step at α 0.1 on all 48 demo rows): 0 of 30, token accuracy 0.598, support
  loss before the step 1.48 on average: one step moves the shared-block model's prediction nowhere off the
  background. The 5- and 10-step variants are running.
- 2026-10-20 08:00, `maml_trm mm_s5`: 0 of 30, token accuracy 0.613; mean support loss 1.48 before step 1 -> 1.27 before
  step 5. Five steps fit the demos somewhat but the test predictions stay near the background.
- 2026-10-20 08:15, `maml_trm mm_s10`: 0 of 30, token accuracy 0.614, support loss 1.48 -> 1.23: ten steps are no
  better than five. The shared-block MAML model does not adapt to a held-out task by its own inner loop at any
  tested step count; the ordinary AdamAtan2 loop on it (`mm_adam`) runs next, then the separate-H model's variants.
- 2026-10-20 09:05, `maml_trm mm_adam` stopped after 7 tasks: the queue runs every TTT eager (`DISABLE_COMPILE=1`, needed
  for the MAML inner loop), and the ordinary 160-step AdamAtan2 loop takes ~5 min per task eager against 12 s
  compiled. The three `mm_adam` variants (shared, separate H, plain) are re-run compiled after the queue
  (`scripts/miniarc5/queue_maml_adam.sh`; the suffix resumes the 7 finished tasks). The SGD variants on the
  separate-H checkpoint are running.
- 2026-10-20 09:25, `maml_trm_sep mm_s1`: 0 of 30 but token accuracy 0.718 after one SGD step (the shared model: 0.598;
  the unadapted separate-H model: 0.654), mean support loss before the step 1.09 (shared: 1.48). The meta-trained
  separate-H model does move toward the task in one step; no grid is complete yet.
- 2026-10-20 09:40, `maml_trm_sep mm_s5`: 0 of 30, token accuracy 0.728, support loss 1.09 -> 0.91 before step 5: four
  more steps fit the demos further but add only one point of token accuracy and no solved task.
- 2026-10-20 09:55, `maml_trm_sep mm_s10`: 0 of 30, token accuracy 0.730, support loss 1.09 -> 0.87. The meta-trained
  inner loop saturates at ~0.73 token accuracy on the held-out tasks whatever the step count; the section-7 full
  fine-tune of the embedding model reached 0.80 and 7 solved tasks. Its eager `mm_adam` was stopped at start (to be
  re-run compiled); the plain control's SGD variants run next.
- 2026-10-20 11:05, the queue's `plain_noemb mm_s5` sat 54 min with an empty log and no task directory (full CPU, no
  output at all); stopped, together with the queued `mm_s10` / `mm_adam`, and the main queue ended. The same command
  in the foreground finishes a task in seconds (5 SGD steps, demo loss 0.67 -> 0.64, token accuracy 0.80 on that
  task), so the hang was in that process, cause unknown. `scripts/miniarc5/queue_maml_rest.sh` re-runs the plain
  control's 5 / 10-step variants and then the three compiled `mm_adam` loops.
- 2026-10-20 11:40, cause found: not a hang. The `maml_trm_sep mm_adam` python process had survived its stop (a wrapper
  was killed instead) and kept the GPU lock for 1.5 h running the eager loop at 5 min per task, so the plain
  control's run was waiting on the lock with an empty log. Killed; the follow-up queue is now running.
- 2026-10-20 12:05, `plain_noemb mm_s5`: 0 of 30, token accuracy 0.683, support loss 1.09 -> 1.00. Under the same 5-step
  SGD adaptation the meta-trained separate-H model gets 0.728 and fits the demos further (0.91): a MAML gain in
  token accuracy of ~4.5 points over the plainly trained model, and none in solved tasks (both 0).
- 2026-10-20 12:30, `plain_noemb mm_s10`: 0 of 30, token accuracy 0.698, support loss 1.09 -> 0.93 (the separate-H MAML
  model at 10 steps: 0.730, 0.87). The gap holds at 10 steps (~3 points); still no task solved by SGD adaptation on
  any checkpoint. The three compiled `mm_adam` loops run next.
- 2026-10-20 13:20, the `mm_adam` runs on the MAML checkpoints were wrong and slow: with `maml_steps=0` the ordinary loop
  called `train_batch`, which still saw the checkpoint's `maml.enabled` and ran a meta-step per TTT batch (minutes
  per task). Fixed (`evaluate_ttt.py` disables `maml` for that loop); their partial results are discarded and the
  loop re-runs as `mm_adam2`. `plain_noemb mm_adam` (unaffected): 0 of 30, token accuracy 0.723, so the plainly
  trained no-embedding model under the full 160-step AdamAtan2 fine-tune reaches the same token accuracy as the
  meta-trained separate-H model under 5-10 SGD steps (0.73) and also solves nothing; the embedding models of
  section 7 solved 2 (shared) and 7 (separate H) under the same loop.
- 2026-10-20 13:40, `maml_trm mm_adam2`: 0 of 30, token accuracy 0.617: the full AdamAtan2 fine-tune on the shared-block
  MAML checkpoint does no better than its own inner loop (0.61), and far worse than the plainly trained model
  under the same loop (0.72). The separate-H checkpoint's `mm_adam2` is the last run.
- 2026-10-20 13:55, `maml_trm_sep mm_adam2`: 1 of 30 (pass@1/2/5 all 1), token accuracy 0.739: the only task solved from
  a MAML checkpoint, by the ordinary 160-step AdamAtan2 loop rather than the meta-trained inner loop. The grid is
  complete.

## Verdict

1. **MAML's inner loop does not solve held-out MiniARC tasks at this scale.** Both meta-trained checkpoints solve 0 of
   30 under 1, 5 or 10 SGD steps of the meta-trained adaptation. What meta-training buys is token accuracy under the
   same adaptation: separate-H MAML 0.728 / 0.730 (5 / 10 steps) against the plainly trained no-embedding model's
   0.683 / 0.698, about 3-4.5 points, and a lower demo loss after adaptation (0.91 vs 1.00 at 5 steps). The
   shared-block MAML model adapts worse than the plain model (0.61).
2. **Against the existing test-time training, MAML loses.** The reference is section 7's full `ttt_mini` fine-tune of
   the embedding models: 2 of 30 (shared) and 7 of 30 (separate H). The MAML checkpoints under that same loop:
   0 (shared, token accuracy 0.617) and 1 (separate H, 0.739); the plain no-embedding model under it: 0 (0.723).
   Dropping the puzzle embedding costs more than meta-training gives back: section 3 found the embedding carries
   most of TRM's adaptation, and every MAML run here had none.
3. **The separate H block helps again** (query loss 1.03 vs 1.35 in meta-training, 0.73 vs 0.61 token accuracy at test),
   consistent with section 7.

Caveats: first-order MAML with one inner step at a fixed α 0.1, 2000 meta-steps of ~3 tasks (6k task episodes,
against 4000 x 128 rows for plain training), eager and sequential (3 h per run on the laptop GPU), one seed, 30
tasks, no puzzle embeddings in any MAML run. Untried and cheap to add: keep the puzzle embeddings in the MAML model
(the inner loop adapts the weights, the embedding trains in the outer loop as usual), 3 inner steps at train,
longer meta-training, or a smaller α with more steps; second-order MAML would need a functional forward of the
recursive model.

### Meta-training curve, `maml_trm` (block means over ~400 meta-steps, ~3 tasks each; 2000 steps, 11,679 s)

| meta-steps | support loss before adaptation | query loss after the adapted step |
|---|---|---|
| 1-399 | 1.95 | 1.49 |
| 400-798 | 1.98 | 1.41 |
| 799-1197 | 1.86 | 1.42 |
| 1198-1596 | 1.67 | 1.35 |
| 1597-1995 | 1.64 | 1.35 |

In-training eval (no adaptation, the held-out test pairs): pass@2 0 of 30 at steps 1000 and 2000, token accuracy
0.600 (the background colour's share: without an embedding or an adaptation step the model has nothing to tell the
tasks apart, as expected).

### Meta-training curve, `maml_trm_sep` (separate H block; 2000 steps, 9,370 s)

| meta-steps | support loss before adaptation | query loss after the adapted step |
|---|---|---|
| 1-399 | 1.80 | 1.38 |
| 400-798 | 1.35 | 1.13 |
| 799-1197 | 1.35 | 1.06 |
| 1198-1596 | 1.40 | 1.03 |
| 1597-1995 | 1.39 | 1.03 |

In-training eval (no adaptation): pass@2 0 of 30 at steps 1000 and 2000, token accuracy 0.612 / 0.654 (above the
0.600 background share: with its own H block the unadapted model already gets some cells beyond the background).
The separate-H model meta-trains clearly better than the shared one: post-adaptation query loss 1.03 vs 1.35, and
its unadapted support loss 1.39 vs 1.64.

### Held-out TTT results (pass@k counts of 30; filled in as each variant lands)

| checkpoint | TTT variant | pass@1 | pass@2 | pass@5 | token acc |
|---|---|---|---|---|---|
| TRM shared, embeddings (section 7 of REPORTS_ARC.md) | full fine-tune, `ttt_mini` | 1 | 2 | 3 | 0.776 |
| TRM separate H, embeddings (section 7) | full fine-tune, `ttt_mini` | 3 | 7 | 8 | 0.803 |
| `maml_trm` (shared block, MAML) | `mm_s1`: 1 inner SGD step, α 0.1 | 0 | 0 | 0 | 0.598 |
| `maml_trm` | `mm_s5`: 5 inner SGD steps | 0 | 0 | 0 | 0.613 |
| `maml_trm` | `mm_s10`: 10 inner SGD steps | 0 | 0 | 0 | 0.614 |
| `maml_trm` | `mm_adam2`: the ordinary `ttt_mini` loop | 0 | 0 | 0 | 0.617 |
| `maml_trm_sep` (separate H, MAML) | `mm_s1`: 1 inner SGD step, α 0.1 | 0 | 0 | 0 | 0.718 |
| `maml_trm_sep` | `mm_s5`: 5 inner SGD steps | 0 | 0 | 0 | 0.728 |
| `maml_trm_sep` | `mm_s10`: 10 inner SGD steps | 0 | 0 | 0 | 0.730 |
| `maml_trm_sep` | `mm_adam2`: the ordinary `ttt_mini` loop | 1 | 1 | 1 | 0.739 |
| `plain_noemb` (no MAML, no embeddings) | `mm_s5`: 5 SGD steps, α 0.1 (the paper's fine-tune baseline) | 0 | 0 | 0 | 0.683 |
| `plain_noemb` | `mm_s10`: 10 SGD steps | 0 | 0 | 0 | 0.698 |
| `plain_noemb` | `mm_adam`: the ordinary `ttt_mini` loop (160 AdamAtan2 steps) | 0 | 0 | 0 | 0.723 |

## Re-run: puzzle-level episodes, base TRM with its puzzle embeddings (2026-10-20)

**Why.** The first grid was not comparable to the TRM recipe: no puzzle embeddings (the recipe's task identifiers,
and the carrier of most of TRM's adaptation per section 3), copy-level episodes with 2-5 support rows against 48 at
test, and 2000 meta-steps of ~3 tasks (~30k example rows) against the plain run's 4000 x 128.

**What changed** (`pretrain.EpisodeSampler`, `_maml_targets`; same first-order MAML otherwise).
- A task is a puzzle. An episode takes 16 of its augmented copies (`maml.support_copies`, the test-time
  `num_aux_tasks`), holds one random row of each copy out as the query and uses the copies' other rows as the
  support: ~45 support rows and 16 query rows per task, the structure the test-time adaptation sees (every aux
  copy's demos, then the copies' test pairs). 4 puzzles per meta-step (`maml.meta_batch_size`).
- The puzzle-embedding table stays and its rows adapt in the inner loop with the weights (the task's copies each
  have a row; at test the new aux copies' rows start from the table mean and adapt the same way). The outer loop
  updates the touched rows with the recipe's SignSGD (`puzzle_emb_lr` 3e-2, the schedule's lr), as plain training
  does. The TTT log line now counts the rows moved (16 per task, verified on `trm_ho2`).
- Budget: 2000 meta-steps x 4 puzzles x ~61 rows ≈ 490k example rows, the plain recipe's 4000 x 128 = 512k.
  Still eager and sequential over puzzles: 5.8 s per meta-step in the smoke, ≈ 3.2 h.
- Base TRM only: shared block (`H_layers=0`), hidden 128, the section-1 recipe. The control is section 7's
  `trm_ho2` (same recipe, build and budget, plain training): full `ttt_mini` fine-tune 1 / 2 / 3 of 30 at
  pass@1/2/5, token accuracy 0.776; and here also under the same SGD adaptation (`mm2_s5` / `mm2_s10`).
- Step-size probe with the embedding rows adapted (3 steps, 2 tasks, from `trm_ho2`): α 0.1: 0.87 -> 0.71,
  1.22 -> 0.56; α 1.0: 0.87 -> 0.42, 1.22 -> 0.76 (the same numbers as without the rows: over 2 steps the rows'
  contribution to the loss is below the print precision). α stays 0.1.

**Runs** (`scripts/miniarc5/queue_maml2.sh`): `maml_trm_emb`; TTT suffixes `mm2_s1` / `mm2_s5` / `mm2_s10`
(meta-trained inner loop, embedding rows included) and `mm2_adam` (the ordinary loop, compiled); `trm_ho2 mm2_s5` /
`mm2_s10` (plain checkpoint under the same adaptation).

### Results (re-run)

| checkpoint | TTT variant | pass@1 | pass@2 | pass@5 | token acc |
|---|---|---|---|---|---|
| `trm_ho2` (plain, section 7) | `lr_full`: ordinary `ttt_mini` loop | 1 | 2 | 3 | 0.776 |
| `trm_ho2` | `mm2_s5`: 5 SGD steps, α 0.1, rows adapted (plain init) | 0 | 0 | 0 | 0.731 |
| `trm_ho2` | `mm2_s10`: 10 SGD steps (plain init) | 0 | 0 | 1 | 0.740 |
| `maml_trm_emb` (MAML, embeddings) | `mm2_s1`: 1 inner SGD step, α 0.1, 16 rows adapted | 0 | 0 | 0 | 0.724 |
| `maml_trm_emb` | `mm2_s5`: 5 inner SGD steps | 0 | 0 | 0 | 0.732 |
| `maml_trm_emb` | `mm2_s10`: 10 inner SGD steps | 0 | 0 | 0 | 0.722 |
| `maml_trm_emb` | `mm2_adam`: ordinary `ttt_mini` loop (160 AdamAtan2 steps) | 2 | 3 | 4 | 0.771 |

Progress log:
- 2026-10-20 16:40, `maml_trm_emb` at meta-step 597 of 2000 (6.7 s per step, no errors). Block means over ~120
  meta-steps: support loss before adaptation 2.88 / 2.26 / 1.30 / 1.22 / 1.15, query loss after the adapted step
  1.46 / 1.15 / 1.01 / 0.90 / 0.82. Unlike the first grid (query loss flat at 1.35-1.45 after step 400), both are
  still falling steadily: with the embeddings and puzzle-level episodes the meta-objective has something to work
  with. Below 0.9 already, against the first grid's best of 1.03 after 2000 steps.
- 2026-10-20 18:10, step 1168 of 2000 (6.9 s per step). Block means over ~233 meta-steps: support loss before
  adaptation 2.56 / 1.29 / 1.11 / 1.04 / 1.00, query loss after the adapted step 1.31 / 0.96 / 0.83 / 0.77 / 0.73,
  still falling. In-training eval at step 1000 (held-out test pairs, no adaptation, untrained identifiers): 0 of 30,
  token accuracy 0.617, as expected for the no-adaptation control.
- 2026-10-20 19:25, step 1798 of 2000 (7.6 s per step, the GPU clock low again). Block means over ~356 meta-steps:
  support loss before adaptation 2.15 / 1.14 / 1.02 / 0.96 / 0.93, query loss after the adapted step 1.21 / 0.85 /
  0.75 / 0.71 / 0.67, still falling at the end of the budget.
- 2026-10-20 20:00, `maml_trm_emb` finished: 2000 meta-steps in 13,349 s (3.7 h), no errors; last block query loss
  0.67 (first grid's shared model: 1.35, separate H: 1.03). In-training eval at step 2000 (no adaptation): 0 of 30,
  token accuracy 0.651. First TTT variant `mm2_s1` (one SGD step on the 48 demo rows, the 16 aux copies' embedding
  rows adapted too): 0 of 30, token accuracy 0.724, support loss before the step 1.20 on average. One step: level
  with the first grid's separate-H model (0.718), above its shared one (0.598).
- 2026-10-20 20:20, `mm2_s5`: 0 of 30, token accuracy 0.732, support loss 1.20 -> 0.85 before step 5. The demos are
  fitted further than in the first grid (0.91) but the held-out test pairs stay at 0.73.
- 2026-10-20 20:40, `mm2_s10`: 0 of 30, token accuracy 0.722, support loss 1.20 -> 0.81: ten steps fit the demos a
  little more and give nothing on the test pairs. The meta-trained inner loop, with embeddings and matched
  episodes, saturates at ~0.73 token accuracy exactly as the first grid's separate-H model did. The plain control
  under the same adaptation (`trm_ho2 mm2_s5` / `mm2_s10`) runs next, then the ordinary loop on this checkpoint.
- 2026-10-20 21:00, `trm_ho2 mm2_s5` (the plainly trained checkpoint under the same 5-step adaptation): 0 of 30, token
  accuracy 0.731, support loss 1.60 -> 0.91. Level with the meta-trained checkpoint (0.732) on the test pairs; the
  meta-trained one starts from a lower demo loss (1.20 vs 1.60) and ends lower (0.85 vs 0.91), the gain the
  meta-objective bought, but it does not reach the held-out pairs.
- 2026-10-20 21:20, `trm_ho2 mm2_s10`: 0 / 0 / 1 of 30 at pass@1/2/5, token accuracy 0.740, support loss 1.60 -> 0.76.
  The plain checkpoint under ten SGD steps edges past the meta-trained one (0.722, 0 at pass@5). The last run,
  the ordinary loop on the MAML checkpoint, is going.
- 2026-10-20 21:40, `maml_trm_emb mm2_adam` (the ordinary 160-step AdamAtan2 loop on the meta-trained checkpoint):
  2 / 3 / 4 of 30 at pass@1/2/5, token accuracy 0.771, demo loss 1.19 -> 0.48 over the tasks. The grid is complete.

### Verdict (re-run)

1. **The meta-trained inner loop still solves nothing.** With the embeddings adapted and train/test episodes
   matched, 1 / 5 / 10 SGD steps give 0 of 30 at token accuracy 0.72-0.73, and the plainly trained checkpoint
   under the identical adaptation gets the same (0.731 at 5 steps, 0.740 and one task at pass@5 at 10 steps).
   Meta-training did what its objective asks, the demo loss before and after the step is lower (1.20 -> 0.85 vs
   1.60 -> 0.91 at 5 steps) and the post-adaptation query loss fell from 1.18 to 0.67 over training, but the
   adapted model's gain on the demos does not transfer to the held-out test pairs.
2. **As an initialisation for the ordinary fine-tune, MAML is at best a small plus.** The meta-trained checkpoint
   under the recipe's `ttt_mini` loop solves 2 / 3 / 4 against the plain checkpoint's 1 / 2 / 3 (same recipe,
   build and row budget), with token accuracy 0.771 vs 0.776. One task at pass@2 is inside the 2-task seed spread
   of section 3b, so this is "not worse", not a demonstrated gain; a second seed would decide.
3. **Against the first grid**, keeping the embeddings and matching the episodes fixed the meta-training (query loss
   0.67 vs 1.35 for the same architecture) and lifted the ordinary-loop result from 0 to 3, i.e. the first grid's
   failure was the setup, as argued; but the paper's mechanism, adaptation by a few SGD steps from a meta-learned
   start, does not produce solved MiniARC tasks at this scale either way.

Caveats: first-order, one inner step at a fixed uniform α 0.1 (no learned step sizes), 2000 meta-steps of 4
puzzles (row budget matched to the plain recipe, puzzle count not: each meta-step sees 4 puzzles against the
plain batch's ~25), eager and sequential (3.7 h), one seed, 30 tasks. Untried: learned per-parameter step sizes
(Meta-SGD), 3 inner steps at train, a second seed of both checkpoints, and a batched functional inner loop to
afford more meta-steps.

### Meta-training curve, `maml_trm_emb` (embeddings, puzzle-level episodes; block means over ~400 meta-steps)

| meta-steps | support loss before adaptation | query loss after the adapted step |
|---|---|---|
| 1-399 | 2.08 | 1.18 |
| 400-798 | 1.12 | 0.84 |
| 799-1197 | 1.01 | 0.75 |
| 1198-1596 | 0.95 | 0.70 |
| 1597-1995 | 0.93 | 0.67 |

## Partial MAML: meta-learning the initialisation of one part of the network (2026-10-20)

**Idea (user).** Use MAML for a subset of the parameters only: the puzzle embeddings, the reasoning block, or GRAM's
guidance heads (prior, posterior, LPRM value). Reading taken: the inner loop adapts only that part at train and at
test, while the outer loop still trains everything (ANIL, Raghu et al. 2020, where the inner loop adapts the head
alone). `maml.adapt_substrings` selects the part by parameter-name substring (`puzzle_emb` = the embedding table);
the meta-gradient is still the query gradient w.r.t. every parameter at the adapted point. Everything else is the
re-run's setup: base architecture (shared block), puzzle-level episodes of 16 copies, 4 puzzles per meta-step,
2000 meta-steps, eager.

**Step-size probes** (3 inner steps, 2 held-out tasks, from the plain `trm_ho2` checkpoint; support loss before step 1
-> before step 3, per task, then the test token accuracy):

| part adapted | α 0.1 | α 1.0 | α 10 |
|---|---|---|---|
| embedding rows only | 0.87 -> 0.87, 1.22 -> 1.21 (0.69) | 0.87 -> 0.85, 1.22 -> 1.12 (0.70) | 0.87 -> 0.62, 1.22 -> 0.60 (0.81) |
| L block only | 0.87 -> 0.72, 1.22 -> 0.56 (0.81) | 0.87 -> 0.44, 1.22 -> 0.76 (0.80) | |

The embedding rows' gradient is two orders of magnitude smaller than the block's, so the embedding-only inner loop
needs α 10 to move at all. This also says that in the re-run above (everything adapted at α 0.1) the embedding
rows contributed nothing to the adaptation: a per-part step size (Meta-SGD) would be the principled fix. GRAM's
heads diverged at α 0.1 on a 40-step smoke checkpoint (support loss 2.2 -> 6.5), so their α is probed on the
plain `gram_ho` control once it is trained (α 0.01 / 0.1 / 1).

GRAM heads probe (3 steps, 2 tasks, from the plain base-GRAM `gram_ho`, 4000 steps; support loss before step 1 ->
before step 3 per task, then test token accuracy):

| part adapted | α 0.01 | α 0.03 | α 0.1 | α 1 |
|---|---|---|---|---|
| prior + posterior + LPRM heads (494k params) | 1.07 -> 1.07, 1.51 -> 1.54 (0.70) | 1.07 -> 2.87, 1.51 -> 33.7 (0.74) | 1.07 -> 147, 1.51 -> 2.8e6 (0.00) | 1e22, 3e32 (0.00) |
| prior + LPRM heads (no posterior) | 1.07 -> 1.07, 1.51 -> 1.44 (0.70) | 1.07 -> 1.99, 1.51 -> 20.8 (0.76) | 1.07 -> 98, 1.51 -> 4.8e5 (0.01) | |

Plain SGD on the Gaussian guidance heads is unstable above α 0.01 (the KL and the variance outputs amplify a step;
the posterior is not the cause), and at 0.01 the step barely moves the demo loss. The queue runs at α 0.01 with all
three heads, as specified: the question becomes whether meta-training can find head weights for which that small
step is useful, since a larger one cannot be taken from the plain start.

**Queues** (`scripts/miniarc5/queue_maml_{emb,lh,gramheads}.sh`, runner `queue_maml_parts.sh`): `maml_emb`
(embedding rows, α 10), `maml_lh` (the block, α 0.1), `maml_gramheads` (GRAM heads, α from the probe), each with
the meta-trained inner loop at 1 / 5 / 10 steps (`mm3e_*`, `mm3l_*`, `mm3g_*`), the same subset adaptation on the
plain checkpoint (`trm_ho2`, `gram_ho`) at 5 / 10 steps, and the ordinary `ttt_mini` loop. `gram_ho` is a new plain
base-GRAM pretrain on the held-out build (same recipe), also run under the ordinary loop as its full-TTT reference.

### Results (partial MAML)

| checkpoint | TTT variant | pass@1 | pass@2 | pass@5 | token acc |
|---|---|---|---|---|---|
| `trm_ho2` (plain) | `lr_full`: ordinary loop (section 7) | 1 | 2 | 3 | 0.776 |
| `maml_trm_emb` (everything adapted, re-run) | `mm2_adam`: ordinary loop | 2 | 3 | 4 | 0.771 |
| `maml_emb` (embedding rows) | `mm3e_s1` / `s5` / `s10` / `mm3e_adam` | (queued) | | | |
| `trm_ho2` | `mm3e_s5` / `s10`: embedding rows, α 10 | (queued) | | | |
| `maml_lh` (the block) | `mm3l_s1` / `s5` / `s10` / `mm3l_adam` | (queued) | | | |
| `trm_ho2` | `mm3l_s5` / `s10`: the block, α 0.1 | (queued) | | | |
| `gram_ho` (plain GRAM) | `mm3g_adam`: ordinary loop | (queued) | | | |
| `maml_gramheads` (GRAM heads) | `mm3g_s1` / `s5` / `s10` / `mm3g_adam` | (queued) | | | |
| `gram_ho` | `mm3g_s5` / `s10`: the heads | (queued) | | | |

Progress log (partial MAML):
- 2026-10-20 23:40, the plain `gram_ho` control trained (4000 steps) and the heads probe ran (table above); `maml_emb`
  had started. The user then replaced the partial-MAML runs with Meta-SGD ones (next section); the MAML runs were
  stopped before any result. `gram_ho` and the probes are reused.

## Partial Meta-SGD: learned step sizes per element (2026-10-21)

**Why.** The probes showed the parts of the network need inner step sizes two orders of magnitude apart (embedding
rows ~10, the block 0.1, GRAM's heads 0.01), so one fixed α makes the adaptation a weights-only one, and the
partial ablations pick the partition by hand. Meta-SGD (Li et al. 2017, arXiv 1707.09835) learns a step size per
element of the adapted tensors together with the initialisation; MAML is its special case with one fixed scalar.

**Implementation** (`pretrain.MetaSGD`, `maml.learn_alpha`). α = exp(ρ), one ρ per element of every adapted tensor
(the puzzle-embedding table gets one per column, shared by its rows, since a test task's rows are new), initialised
at the probed α of the part. First-order: the inner step is θ' = θ − α ∘ g_support; the query gradient at θ'
trains θ as before, and ρ gets dL_query/dρ = −(g_query ∘ g_support) ∘ α, which needs no second derivatives
(g_support does not depend on α). ρ sits on the loss head (saved in the checkpoint, used at test time), trained by
its own Adam at 1e-3 under the schedule, kept out of AdamAtan2 (whose weight decay would pull α to 1), and clamped
to log α0 ± 5 after every meta-step against runaway steps (a shortcut; the meta-objective alone should bound α).
The outer loop trains everything (ANIL); the inner loop adapts the named part. Logged: `train/fprl/alpha_mean`
(geometric mean over the adapted elements) and `alpha_max`; the TTT line prints them too.

**Runs** (`scripts/miniarc5/queue_msgd_parts.sh`, one GPU in sequence): `msgd_emb` (embedding rows, α0 10), `msgd_lh`
(the block, α0 0.1), `msgd_gramheads` (GRAM's prior / posterior / LPRM heads, α0 0.01); each with the learned inner
loop at 1 / 5 / 10 steps (`ms3e_*`, `ms3l_*`, `ms3g_*`), the plain checkpoint (`trm_ho2`, `gram_ho`) under the same
part at the fixed α0 (5 / 10 steps), and the ordinary `ttt_mini` loop; plus `gram_ho` under the ordinary loop.
Same budget as the re-run: puzzle-level episodes of 16 copies, 4 puzzles per meta-step, 2000 meta-steps, eager.

### Results (partial Meta-SGD)

Counts of 30 held-out tasks. pass@k: the task's test grid is among the top-k voted predictions; ARC's rule is
pass@2. pass@100 / @1000 count the tasks whose correct grid appears anywhere in the vote (16 augmentation copies
give at most 16 distinct candidates, so pass@100 and pass@1000 coincide: "solved by some copy").

| checkpoint | TTT variant | pass@1 | pass@2 | pass@5 | pass@100 | pass@1000 | token acc | step sizes (geo. mean / max) |
|---|---|---|---|---|---|---|---|---|
| `trm_ho2` (plain) | `lr_full`: ordinary loop (section 7) | 1 | 2 | 3 | 6 | 6 | 0.776 |  |
| `maml_trm_emb` (MAML, everything, re-run) | `mm2_adam`: ordinary loop | 2 | 3 | 4 | 4 | 4 | 0.771 |  |
| `msgd_emb` (embedding rows) | `ms3e_s1`: 1 learned step on the 16 rows | 0 | 0 | 0 | 0 | 0 | 0.725 | 12.5 / 14.3 |
| `msgd_emb` | `ms3e_s5`: 5 learned steps on the rows | 0 | 0 | 0 | 0 | 0 | 0.729 | 12.5 / 14.3 |
| `msgd_emb` | `ms3e_s10`: 10 learned steps on the rows | 0 | 0 | 1 | 1 | 1 | 0.739 | 12.5 / 14.3 |
| `msgd_emb` | `ms3e_adam`: ordinary loop | 1 | 1 | 1 | 1 | 1 | 0.743 |  |
| `trm_ho2` (plain) | `ms3e_s5`: 5 fixed steps (α 10) on the rows | 0 | 1 | 1 | 1 | 1 | 0.738 | fixed 10.0 |
| `trm_ho2` | `ms3e_s10`: 10 fixed steps on the rows | 0 | 1 | 1 | 1 | 1 | 0.757 | fixed 10.0 |
| `msgd_lh` (the block) | `ms3l_s1`: 1 learned step on the block | 0 | 0 | 0 | 0 | 0 | 0.727 | 0.106 / 0.242 |
| `msgd_lh` | `ms3l_s5`: 5 learned steps | (queued) | | | | | | |
| `msgd_lh` | `ms3l_s10`: 10 learned steps | (queued) | | | | | | |
| `msgd_lh` | `ms3l_adam`: ordinary loop | (queued) | | | | | | |
| `trm_ho2` (plain) | `ms3l_s5`: 5 fixed steps (α 0.1) on the block | (queued) | | | | | | |
| `trm_ho2` | `ms3l_s10`: 10 fixed steps on the block | (queued) | | | | | | |
| `varc_ho2` (plain VARC, rebuilt build) | `ms3v_adam`: ordinary loop | 4 | 4 | 4 | 6 | 6 | 0.811 |  |
| `msgd_varc_emb` (VARC, task-token rows) | `ms3ve_s1`: 1 learned step on the rows | 0 | 0 | 0 | 0 | 0 | 0.680 | 1.71 / 2.48 |
| `msgd_varc_emb` | `ms3ve_s5`: 5 learned steps | 0 | 0 | 0 | 0 | 0 | 0.730 | 1.71 / 2.48 |
| `msgd_varc_emb` | `ms3ve_s10`: 10 learned steps | 0 | 0 | 0 | 0 | 0 | 0.729 | 1.71 / 2.48 |
| `msgd_varc_emb` | `ms3ve_adam`: ordinary loop | 1 | 2 | 2 | 2 | 2 | 0.750 |  |
| `varc_ho2` (plain) | `ms3ve_s5`: 5 fixed steps (α 1) on the rows | 0 | 0 | 0 | 1 | 1 | 0.744 | fixed 1.0 |
| `varc_ho2` | `ms3ve_s10`: 10 fixed steps on the rows | 0 | 0 | 1 | 1 | 1 | 0.752 | fixed 1.0 |
| `msgd_varc_lh` (VARC, transformer layers) | `ms3vl_s1`: 1 learned step on the layers | 0 | 0 | 0 | 0 | 0 | 0.731 | 0.0108 / 0.0265 |
| `msgd_varc_lh` | `ms3vl_s5`: 5 learned steps | 0 | 0 | 0 | 0 | 0 | 0.722 | 0.0108 / 0.0265 |
| `msgd_varc_lh` | `ms3vl_s10`: 10 learned steps | 0 | 0 | 0 | 0 | 0 | 0.722 | 0.0108 / 0.0265 |
| `msgd_varc_lh` | `ms3vl_adam`: ordinary loop | 1 | 1 | 1 | 2 | 2 | 0.748 |  |
| `varc_ho2` (plain) | `ms3vl_s5`: 5 fixed steps (α 0.01) on the layers | 0 | 0 | 0 | 0 | 0 | 0.715 | fixed 0.01 |
| `varc_ho2` | `ms3vl_s10`: 10 fixed steps on the layers | 0 | 0 | 1 | 1 | 1 | 0.726 | fixed 0.01 |
| `gram_ho` (plain GRAM) | `ms3g_adam`: ordinary loop | (queued) | | | | | | |
| `msgd_gramheads` (GRAM heads) | `ms3g_s1`: 1 learned step on the heads | (queued) | | | | | | |
| `msgd_gramheads` | `ms3g_s5`: 5 learned steps | (queued) | | | | | | |
| `msgd_gramheads` | `ms3g_s10`: 10 learned steps | (queued) | | | | | | |
| `msgd_gramheads` | `ms3g_adam`: ordinary loop | (queued) | | | | | | |
| `gram_ho` (plain) | `ms3g_s5`: 5 fixed steps (α 0.01) on the heads | (queued) | | | | | | |
| `gram_ho` | `ms3g_s10`: 10 fixed steps on the heads | (queued) | | | | | | |

Progress log (partial Meta-SGD):
- 2026-10-21 01:50, the user put VARC at the head of the chain. `msgd_lh` (the TRM block) was stopped at step ~700 and
  will rerun from scratch after the VARC runs; `msgd_emb`'s grid is complete. New runner
  `scripts/miniarc5/queue_msgd_varc_parts.sh`: the plain VARC control `varc_ho2` on the rebuilt held-out build (the
  MiniARC VARC recipe of section 1: hidden 128, 4 layers, mlp 256, canvas 10, 606,768 params, batch 128, 4000
  steps, `puzzle_emb_lr` 3e-2 as the TRM runs here; section 3's `varc_ho` trained on the deleted build), step-size
  probes on it (embedding rows at α 1 / 10 / 100, the transformer layers `inner.layers.` at 0.01 / 0.1 / 1; 3 steps,
  2 tasks) with `pick_alpha.py` choosing the largest α whose demo loss fell ≥ 5% on every probed task, then Meta-SGD of
  the VARC embedding rows (`msgd_varc_emb`, grid `ms3ve_*`) and of its layers (`msgd_varc_lh`, `ms3vl_*`), with
  `varc_ho2` under the same parts at the fixed α and under the ordinary loop (`ms3v_adam`). VARC's forward is one
  pass (`halt_max_steps` 1) so a meta-step costs ~0.7 s: 2000 meta-steps ≈ 25 min against TRM's 3.8 h. Section 3's
  VARC reference: Full-TTT 3 / 4 / 4 of 30 (flat lr: 4 / 4 / 7), Embed-TTT 1 / 1 / 2.
- 2026-10-21 02:10, `varc_ho2` trained (4000 steps, 264 s) and probed (support loss before step 1 -> before step 3,
  per task):

  | part adapted | α 0.01 | α 0.1 | α 1 | α 10 | α 100 |
  |---|---|---|---|---|---|
  | embedding rows (the task token) | | | 0.90 -> 0.62, 0.96 -> 0.69 | 0.90 -> 1.20, 0.96 -> 0.69 | 0.90 -> 1.50, 0.96 -> 0.88 |
  | transformer layers | 0.90 -> 0.68, 0.96 -> 0.85 | 0.90 -> 2.77, 0.96 -> 0.81 | 0.90 -> 2.76, 0.96 -> 1.18 | | |

  Picked: embedding rows α 1 (VARC's task token has a 10x larger gradient than TRM's zero-initialised embedding
  slot, so a 10x smaller step), layers α 0.01 (0.1 already diverges on one task). `msgd_varc_emb` is meta-training.
- 2026-10-21 02:35, `msgd_varc_emb` finished (2000 meta-steps in 537 s) and its three learned-loop TTT runs landed.
  Meta-training: query loss after the adapted step 1.48 -> 1.07 (block means; support before the step 1.56 ->
  1.11), the task-token step sizes rising 1 -> 2.2 (max 4.9). Much less adaptation than the TRM embedding rows
  (0.67) : VARC's single task token has less to give than TRM's embedding slot. TTT: 1 / 5 / 10 learned steps on the
  rows 0 of 30 each (0 at pass@100), token accuracy 0.680 / 0.730 / 0.729, demo loss 1.68 -> 1.03 / 0.98. The plain
  VARC under the same rows and the ordinary loops are running, then the layers ablation.
- 2026-10-21 02:55, `varc_ho2 ms3ve_s5` / `s10` (plain VARC, fixed α 1 on the rows): 0 / 0 / 0 (1 at pass@100), 0.744
  and 0 / 0 / 1 (1 at pass@100), 0.752. As for TRM, the plain start under fixed steps on the rows is level with or
  above the Meta-SGD checkpoint's learned loop (0.730 / 0.729, nothing produced). The two ordinary loops run next.
- 2026-10-21 03:10, `msgd_varc_emb ms3ve_adam` (the ordinary `ttt_mini` loop on the VARC rows-Meta-SGD checkpoint):
  1 / 2 / 2 of 30 (2 at pass@100), token accuracy 0.750. The plain VARC under the same loop on this build
  (`ms3v_adam`) runs next and is the number this compares to (section 3's old build: 3 / 4 / 4).
- 2026-10-21 03:35, `msgd_varc_lh` finished (560 s): query loss after the adapted step 1.48 -> 1.07, the layers' step
  sizes barely moving on average (geometric mean 0.010 -> 0.011) with a few elements climbing to 0.05: the
  meta-objective keeps the layers' steps small. TTT: 1 / 5 learned steps 0 of 30 (0 at pass@100), token accuracy
  0.731 / 0.722; the 10-step run, the ordinary loop, and the plain VARC under the layers are going.
- 2026-10-21 03:55, `msgd_varc_lh ms3vl_s10`: 0 of 30, 0.722; the plain VARC under the layers at fixed α 0.01:
  5 steps 0 of 30, 0.715; 10 steps 0 / 0 / 1 (1 at pass@100), 0.726. Learned and fixed layer steps are level on the
  held-out pairs, as everywhere else in this grid. Left for VARC: the two ordinary loops (`ms3vl_adam`, `ms3v_adam`).
- 2026-10-21 04:15, `msgd_varc_lh ms3vl_adam` (ordinary loop on the VARC layers-Meta-SGD checkpoint): 1 / 1 / 1 of 30
  (2 at pass@100), token accuracy 0.748. The plain VARC's own ordinary loop (`ms3v_adam`) is the last VARC run.
- 2026-10-21 04:30, `varc_ho2 ms3v_adam` (the plain VARC under the ordinary `ttt_mini` loop, rebuilt build): 4 / 4 / 4 of
  30 (6 at pass@100), token accuracy 0.811, which replicates section 3's VARC Full-TTT (3 / 4 / 4). The VARC part of
  the chain is complete; the TRM block (`msgd_lh`) is meta-training.

**VARC verdict.** The two VARC Meta-SGD checkpoints lose on every count against the plainly trained VARC. Under the
ordinary fine-tune: plain 4 / 4 / 4 (6 produced) at 0.811 vs 1 / 2 / 2 (2) at 0.750 for the task-token Meta-SGD start
and 1 / 1 / 1 (2) at 0.748 for the layers one. Under their own learned inner loops: 0 of 30 at 1 / 5 / 10 steps
(nothing produced), while the plain VARC under the same parts at the fixed probed α gets 0.74-0.75 and produces a
task. Meta-training the VARC start for a one-step adaptation of a part costs it most of what plain training gives
the ordinary fine-tune (4 -> 1-2 at pass@2), a larger loss than for TRM (2 -> 1), and VARC's single task token
adapts far less in meta-training (post-step query loss 1.07 vs TRM's 0.67). Same single-seed, 30-task caveat.
- 2026-10-21 06:40, `msgd_lh` (the TRM block, rerun) at meta-step 1180 of 2000 (5.7 s per step). Block means over ~360
  meta-steps: support loss before the step 2.43 / 1.68 / 1.05 / 1.16, query loss after the adapted step 1.34 / 1.06 /
  0.82 / 0.73, the block's step sizes nearly flat on average (0.10 -> 0.107) with the largest elements up to 0.3. The
  in-training eval at step 1000 (no adaptation, untrained identifiers) shows one held-out task solved (pass@2 0.033,
  token accuracy 0.615), the first unadapted solve in this whole grid.
- 2026-10-21 08:30, `msgd_lh` finished: 2000 meta-steps in 11,737 s (3.3 h). Last block: query loss after the adapted
  step 0.65 (support before it 1.24), step sizes 0.109 geometric mean, max 0.39, so the meta-objective leaves the
  block's steps near the probed 0.1 but lets a small set of elements grow 3-4x. The step-2000 eval no longer has the
  unadapted solve (0 of 30, token accuracy 0.651). (The run dir's W&B history also holds the ~260 steps of the
  stopped first attempt, so block boundaries above are approximate.) First TTT run `ms3l_s1`: 0 of 30 (0 at
  pass@100), token accuracy 0.727, the checkpoint's sizes 0.106 / 0.24.

### Meta-training curve, `msgd_lh` (the block adapted, learned per-element step sizes)

| meta-steps (approx.) | support loss before the step | query loss after the adapted step | step size, geometric mean / max |
|---|---|---|---|
| 1-450 | 2.40 | 1.33 | 0.100 / 0.107 |
| 450-900 | 1.30 | 0.93 | 0.103 / 0.15 |
| 900-1350 | 1.08 | 0.76 | 0.106 / 0.26 |
| 1350-1800 | 1.25 | 0.69 | 0.108 / 0.33 |
| 1800-2000 | 1.24 | 0.65 | 0.109 / 0.39 |
- 2026-10-20 20:00, `msgd_emb` at meta-step 520 of 2000 (7 s per step, no errors). Block means over 130 meta-steps:
  support loss before the step 1.57 / 1.22 / 1.19 / 1.09, query loss after the adapted step 1.56 / 1.07 / 0.99 /
  0.86. The learned embedding step sizes drift up from 10 (geometric mean 10.3 -> 12.2, max 13.5): the
  meta-objective wants the rows to move a little more than the probe's α, not less. Adapting the rows alone, the
  post-adaptation query loss is already below the everything-adapted MAML re-run's at the same step (0.86 vs ~0.95).
- 2026-10-20 22:20, `msgd_emb` finished: 2000 meta-steps in 13,523 s (3.8 h), no errors. In-training eval (no
  adaptation): 0 of 30, token accuracy 0.65 at both evals. Its six TTT runs are going.

### Meta-training curve, `msgd_emb` (embedding rows adapted, learned per-column step sizes; block means over ~400 meta-steps)

| meta-steps | support loss before the step | query loss after the adapted step | step size, geometric mean / max |
|---|---|---|---|
| 1-399 | 1.33 | 1.20 | 11.3 / 11.8 |
| 400-798 | 1.04 | 0.84 | 12.3 / 13.8 |
| 799-1197 | 0.93 | 0.75 | 12.6 / 14.3 |
| 1198-1596 | 0.83 | 0.70 | 13.1 / 15.4 |
| 1597-1995 | 0.77 | 0.67 | 13.6 / 16.9 |

Adapting the embedding rows alone reaches the same post-adaptation query loss as the everything-adapted MAML re-run
(0.67 at the end of both) with a lower unadapted support loss (0.77 vs 0.93), and the step sizes keep drifting up
(10 -> 13.6, max 17), still rising when the budget ends: the meta-objective asks for larger embedding steps
throughout.
- 2026-10-20 22:45, `msgd_emb ms3e_s1` (one learned step on the 16 aux copies' rows, backbone untouched): 0 of 30,
  token accuracy 0.725, support loss before the step 1.16 (the checkpoint's learned sizes: geometric mean 12.5, max
  14.3 at the end of training). Level with the everything-adapted MAML re-run's one step (0.724) while moving only
  the 16 embedding rows.
- 2026-10-20 23:05, `ms3e_s5`: 0 of 30, token accuracy 0.729, support loss 1.16 -> 0.63 before step 5. Five learned
  steps on the rows alone fit the demos far better than any earlier inner loop did (the everything-adapted MAML:
  1.20 -> 0.85; the plain checkpoint: 1.60 -> 0.91), and the held-out pairs still sit at 0.73.
- 2026-10-20 23:25, `ms3e_s10`: 0 / 0 / 1 of 30 at pass@1/2/5, token accuracy 0.739, support loss 1.16 -> 0.59. Ten
  learned steps on the rows: one task reaches the top 5, the same as the plain checkpoint under ten fixed SGD steps on
  everything (`mm2_s10`: 0 / 0 / 1, 0.740). The plain checkpoint under the rows alone at α 10 runs next, then the
  ordinary loop on this checkpoint.
- 2026-10-20 23:45, `trm_ho2 ms3e_s5` (the plainly trained checkpoint, 5 fixed steps at α 10 on the 16 rows only): 0 / 1
  / 1 of 30, token accuracy 0.738, support loss 1.60 -> 0.61. The plain start under embedding-only SGD is level with
  the Meta-SGD checkpoint (0.729, 0 solved) and even solves one task at pass@2. So far meta-learning the
  initialisation and the rows' step sizes buys nothing on the held-out pairs that a large fixed step on the rows
  from the plain start does not already give.
- 2026-10-21 00:10, `trm_ho2 ms3e_s10` (plain, 10 fixed steps at α 10 on the rows): 0 / 1 / 1 of 30 (1 at pass@100),
  token accuracy 0.757, above the Meta-SGD checkpoint's ten learned steps (0.739). The embedding-rows ablation's
  last run, the ordinary loop on `msgd_emb`, is going; the block ablation follows.
- 2026-10-21 00:40, `msgd_emb ms3e_adam` (the ordinary 160-step AdamAtan2 loop on the embedding-rows Meta-SGD
  checkpoint): 1 / 1 / 1 of 30 (1 at pass@100), token accuracy 0.743, against 1 / 2 / 3 (6 at pass@100), 0.776 for
  the plain checkpoint under the same loop. The embedding-rows ablation is complete: as a start for the ordinary
  fine-tune this checkpoint is worse than plain training (1 vs 2 at pass@2, 1 vs 6 tasks ever produced), and its own
  learned inner loop is level with fixed large steps on the rows from the plain start. `msgd_lh` (the block) is
  meta-training.
