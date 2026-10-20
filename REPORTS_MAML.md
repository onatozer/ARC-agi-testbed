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
| `maml_trm` | `mm_s10` / `mm_adam` | (running) | | | |
| `maml_trm_sep` | `mm_s1` / `mm_s5` / `mm_s10` / `mm_adam` | (queued) | | | |
| `plain_noemb` | `mm_s5` / `mm_s10` / `mm_adam` | (queued) | | | |
