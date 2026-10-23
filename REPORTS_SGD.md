# Second-order Meta-SGD on MiniARC 5x5: report

Branch `meta-sgd-step` (stacked on `miniarc-heldout-ttt`, PR #68). Updated as each run lands; the table is regenerated with
`python scripts/miniarc5/msgd2_table.py`. The previous, first-order attempt is `REPORTS_MAML.md` on branch `l-reinit-ttt`.

## Setup

**Idea (user, 2026-10-22).** Re-implement Meta-SGD (Li et al. 2017, arXiv 1707.09835) properly and re-run the previous
experiments, VARC first; TRM may be reduced or skipped if the second-order step is intractable. Meta-SGD trains the shared
weights θ and a learned per-parameter step size α: per task, θ'_i = θ − α ⊙ ∇θ L_support(θ) from one gradient step on
the task's demos, and (θ, α) are updated by ∇ L_query(θ'_i) summed over the tasks of a batch, differentiating through the
inner step.

**What is new against the previous attempt** (first-order MAML / Meta-SGD on `l-reinit-ttt`, which never solved a held-out
task at pass@2):

- *Second order.* The outer gradient for θ goes through the inner step (`torch.func.functional_call` with substituted
  weights, `create_graph=True`, attention on the math kernel for its double backward). The previous code was first-order
  and updated the weights in place.
- *Query = the task's real test pair.* The dataset build routes the non-held-out tasks' test pairs into training next to
  their demos, flagged per row (`all__validation.npy`, `batch["validation"]`, PR #68). Previously the query was a random
  leave-one-out demo.
- *Every puzzle in the batch is a task.* Episodes come from the ordinary loader batch (rows grouped by puzzle id, split by
  the flag); a 16-row batch holds 3-5 tasks. Previously a separate sampler drew 4 tasks of 16 copies each.
- *Each loss runs the row's ACT trajectory.* Support and query losses start from a fresh carry and run every supervision
  step until the model halts the row (or `halt_max_steps`), as `train_batch` does across batches. For VARC that is one
  step; for TRM up to 16, which is why TRM is expensive.
- *α on the puzzle-embedding table is per column* `(1, D)`, shared by rows (a held-out task's rows are new), as before.
  The table's outer gradient goes to its sign-SGD optimizer per batch row; α has its own Adam (`alpha_lr`), no weight decay,
  clamped to `inner_lr · exp(±5)`.

Code: `pretrain.py` (`meta_sgd_batch`, `_episodes_from_batch`, `_supervised_loss`, `_adapt`, `_task_outer_backward`,
`MetaSGD`), `evaluate_ttt.py` (`meta_sgd_steps`, `meta_sgd_adapt`), config `meta_sgd.*`, tests `tests/test_meta_sgd.py`.

## Protocol (VARC, the previous loop's)

- **Data.** `data/miniarc5-aug-31-ho30`: 149 MiniARC tasks at 5x5, 31 augmentations, 30 tasks held out (`heldout_seed=0`,
  the same 30 as before). 119 tasks train with their demos and their test pair (one flagged row per augmented copy, 3808 in all);
  the 30 held-out tasks' test pairs are the test split.
- **Model.** VARC as in the previous loop: 10x10 canvas, patch 2, hidden 128, 4 layers, 8 heads, mlp 256, one task token per
  puzzle id (`arch.image_size=10 arch.hidden_size=128 arch.num_layers=4 arch.mlp_dim=256 arch.num_heads=8`, the current
  `cfg_pretrain_miniarc_varc.yaml` has since grown).
- **Baseline `varc_ho2`.** Plain VARC, 4000 steps, batch 128, lr 1e-3, from the previous loop; its checkpoint is reused.
- **Meta-training.** `scripts/miniarc5/msgd2_varc.sh`: batch 16 (the meta-batch), 2000 meta-steps, AdamAtan2 lr 1e-3
  (warmup 100, constant), `puzzle_emb_lr` 3e-2, `alpha_lr` 1e-3, seed 0. Three adapt sets, as before plus the paper's default:
  `msgd2_varc_emb` (rows only, α0 = 1), `msgd2_varc_lh` (the transformer layers only, α0 = 0.01), `msgd2_varc_all`
  (everything, α0 = 0.01; α is learned per element).
- **Evaluation**, on the 30 held-out tasks, 16 aux copies per task (the task + 15 dihedral/color augmentations with fresh
  embedding rows initialised to the table mean), votes pooled across copies:
  - *learned loop* `s1/s5/s10`: `evaluate_ttt.py meta_sgd_steps=N`, N inner steps with the checkpoint's learned α on all
    copies' demos (~48 rows), nothing else;
  - *ttt_mini* (`_adam`): the ordinary per-task fine-tune (16 aux, 20 epochs, batch 8, lr 3e-4, ~160 AdamAtan2 steps)
    from the same checkpoint;
  - *fixed-α controls* on the plain checkpoint: the same inner steps with α fixed at the previous loop's picks (1 on the rows,
    0.01 on the layers), via `+train_overrides.meta_sgd.*`.

## Results (30 held-out tasks; pass@k = tasks solved; token acc over test cells)

| checkpoint | test-time procedure | pass@1 | pass@2 | pass@5 | pass@100 | token acc |
|---|---|---|---|---|---|---|
| varc_ho2 | plain VARC, ttt_mini (ordinary fine-tune) | 4 | 4 | 4 | 6 | 0.811 |
| varc_ho2 | plain VARC, fixed alpha 1 on the rows, 5 steps | 0 | 0 | 0 | 1 | 0.744 |
| varc_ho2 | plain VARC, fixed alpha 1 on the rows, 10 steps | 0 | 0 | 1 | 1 | 0.752 |
| varc_ho2 | plain VARC, fixed alpha 0.01 on the layers, 5 steps | 0 | 0 | 0 | 0 | 0.715 |
| varc_ho2 | plain VARC, fixed alpha 0.01 on the layers, 10 steps | 0 | 0 | 1 | 1 | 0.727 |
| msgd2_varc_emb | Meta-SGD rows (alpha0 1), learned loop 1 step | 0 | 0 | 0 | 0 | 0.676 |
| msgd2_varc_emb | Meta-SGD rows, learned loop 5 steps | 0 | 0 | 0 | 0 | 0.721 |
| msgd2_varc_emb | Meta-SGD rows, learned loop 10 steps | 0 | 0 | 0 | 0 | 0.721 |
| msgd2_varc_emb | Meta-SGD rows, then ttt_mini | 1 | 1 | 1 | 1 | 0.754 |
| msgd2_varc_lh | Meta-SGD layers (alpha0 0.01), learned loop 1 step | 0 | 0 | 0 | 0 | 0.682 |
| msgd2_varc_lh | Meta-SGD layers, learned loop 5 steps | 0 | 0 | 0 | 0 | 0.705 |
| msgd2_varc_lh | Meta-SGD layers, learned loop 10 steps | 0 | 0 | 0 | 0 | 0.712 |
| msgd2_varc_lh | Meta-SGD layers, then ttt_mini | 1 | 1 | 1 | 1 | 0.758 |
| msgd2_varc_all | Meta-SGD everything (alpha0 0.01), learned loop 1 step | 0 | 0 | 0 | 0 | 0.688 |
| msgd2_varc_all | Meta-SGD everything, learned loop 5 steps | 0 | 0 | 0 | 0 | 0.717 |
| msgd2_varc_all | Meta-SGD everything, learned loop 10 steps | 0 | 0 | 0 | 0 | 0.721 |
| msgd2_varc_all | Meta-SGD everything, then ttt_mini | 1 | 1 | 1 | 1 | 0.758 |
| varc_ho2_cont | plain VARC continued 2000 steps (batch 16), then ttt_mini | 1 | 1 | 1 | 1 | 0.750 |
| msgd2_varc_all_ft | Meta-SGD everything from the plain checkpoint, 2000 meta-steps, then ttt_mini | 1 | 1 | 1 | 1 | 0.747 |
| msgd2_varc_all_ft | Meta-SGD everything from the plain checkpoint, learned loop 10 steps | 0 | 0 | 0 | 0 | 0.667 |
| msgd2_varc_all_full | Meta-SGD everything from scratch, 4000 x batch 128 (baseline budget), then ttt_mini | 1 | 2 | 2 | 2 | 0.787 |
| msgd2_varc_all_full | Meta-SGD everything from scratch at the baseline budget, learned loop 10 steps | 0 | 0 | 0 | 0 | 0.727 |
| varc_ho2_cont2 | plain VARC continued 1000 steps (batch 256, lr 1e-4), then ttt_mini | 3 | 3 | 5 | 6 | 0.822 |
| msgd3_varc_ft | round 3: meta-fine-tuned, 16-copy episodes, fresh rows, alpha0 rows 1 / weights 0.01; learned loop 1 step | 0 | 0 | 0 | 0 | 0.673 |
| msgd3_varc_ft | round 3, learned loop 5 steps | 0 | 0 | 0 | 0 | 0.737 |
| msgd3_varc_ft | round 3, learned loop 10 steps | 0 | 0 | 0 | 0 | 0.751 |
| msgd3_varc_ft | round 3, then ttt_mini | 3 | 3 | 3 | 4 | 0.809 |
| msgd3_varc_ft | round 3, learned loop 30 steps | 0 | 1 | 1 | 1 | 0.772 |
| varc_ho2 | plain VARC, fixed alpha 0.01 weights / 1 rows, 30 steps (round 6 control) | 1 | 2 | 2 | 2 | 0.777 |
| msgd3_varc_ft | round 3, learned loop 120 steps (ttt_mini's step count) | 0 | 2 | 3 | 4 | 0.791 |
| varc_ho2 | plain VARC, fixed alpha 0.01 weights / 1 rows, 120 steps (round 6 control) | 2 | 3 | 4 | 4 | 0.801 |
| varc_ho2 | plain VARC, no adaptation (0 epochs) | 0 | 0 | 0 | 0 | 0.641 |
| msgd5_pt | round 5: alpha per tensor; learned loop 1 / 5 / 10 steps | 0 | 0 | 0 | 0 | .673 / .738 / .753 |
| msgd5_pt | round 5: alpha per tensor; then ttt_mini | 3 | 3 | 3 | 4 | 0.809 |
| msgd5_alr1e-3 | round 5: alpha_lr 1e-3; learned loop 1 / 5 / 10 steps | 0 | 0 | 0 | 0 | .675 / .734 / .748 |
| msgd5_alr1e-3 | round 5: alpha_lr 1e-3; then ttt_mini | 3 | 3 | 3 | 4 | 0.809 |
| msgd5_alr1e-1 | round 5: alpha_lr 1e-1; learned loop 1 / 5 / 10 steps | 0 | 0 | 0 | 0 | .675 / .738 / .754 |
| msgd5_alr1e-1 | round 5: alpha_lr 1e-1; then ttt_mini | 3 | 3 | 3 | 4 | 0.807 |
| msgd5_lr3e-4 | round 5: outer lr 3e-4; learned loop 1 / 5 / 10 steps | 0 | 0 | 0 | 0 | .671 / .737 / .751 |
| msgd5_lr3e-4 | round 5: outer lr 3e-4; then ttt_mini | 3 | 3 | 3 | 5 | 0.808 |
| msgd5_cos | round 5: cosine outer lr (min 0.1); learned loop 1 / 5 / 10 steps | 0 | 0 | 0 | 0 | .671 / .736 / .750 |
| msgd5_cos | round 5: cosine outer lr (min 0.1); then ttt_mini | 3 | 3 | 3 | 5 | 0.808 |
| msgd5_pp | round 5: MAML++ (5 inner steps, multi-step loss, alpha per tensor and step, first order 300 steps); learned loop 1 / 5 / 10 steps | 0 | 0 | 0 | 0 | .687 / .740 / .754 |
| msgd5_pp | round 5: MAML++; then ttt_mini | 3 | 3 | 3 | 5 | 0.810 |
| msgd7_range10 | round 7: alpha per tensor, clamp e^10 (alpha up to 220); learned loop 1 / 5 / 10 steps | 0 | 0 | 0 | 0 | .673 / .738 / .752 |
| msgd7_range10 | round 7: clamp e^10; then ttt_mini | 3 | 3 | 3 | 4 | 0.809 |
| msgd7_init1 | round 7: alpha per tensor, weights' alpha init 1; learned loop 1 / 5 / 10 steps (diverges: demo loss 1.8 -> 2238 at 10) | 0 | 0 | 0 | 0 | .584 / .526 / .472 |
| msgd7_init1 | round 7: alpha init 1; then ttt_mini | 3 | 3 | 3 | 5 | 0.807 |
| msgd7_norm | round 7: RMS-normalised inner step, alpha per tensor (init 1e-3 weights / 3e-2 rows); learned loop 1 / 5 / 10 steps | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 1 / 1 | .662 / .697 / .738 |
| msgd7_norm | round 7: normalised step; then ttt_mini | 3 | 3 | 3 | 3 | 0.806 |
| msgd7_norm | round 8: normalised learned loop, 30 steps | 2 | 3 | 4 | 6 | 0.795 |
| msgd7_norm | round 8: normalised learned loop, 120 steps (ttt_mini's step count) | 7 | 7 | 8 | 9 | 0.833 |
| msgd7_norm | round 9: normalised learned loop, 120 steps, eval seed 1 | 5 | 7 | 9 | 10 | 0.836 |
| varc_ho2 | round 9 control: plain VARC, RMS-normalised step, hand-set alpha 1e-3 weights / 3e-2 rows, 10 steps | 1 | 1 | 1 | 1 | 0.744 |
| varc_ho2 | round 9 control: plain VARC, normalised step, hand-set alpha, 30 steps | 2 | 2 | 3 | 3 | 0.774 |
| varc_ho2 | round 9 control: plain VARC, normalised step, hand-set alpha, 120 steps | 3 | 5 | 8 | 10 | 0.828 |
| varc_ho2 | round 10: plain VARC, normalised step, hand-set alpha, 120 steps, eval seed 1 | 4 | 7 | 8 | 11 | 0.829 |
| varc_ho2 | round 10: plain VARC, normalised step, hand-set alpha, 120 steps, eval seed 2 | 5 | 8 | 8 | 11 | 0.824 |
| msgd7_norm | round 10: normalised learned loop, 120 steps, eval seed 2 | 6 | 8 | 9 | 9 | 0.839 |
| varc_ho2 | plain VARC, ttt_mini, eval seed 1 | 3 | 4 | 6 | 7 | 0.812 |
| varc_ho2 | plain VARC, ttt_mini, eval seed 2 | 2 | 5 | 6 | 6 | 0.805 |
| varc_ho2 | plain VARC, ttt_mini, 4 canvas views pooled | 4 | 4 | 6 | 7 | 0.808 |
| varc_ho2 | plain VARC, ttt_mini, 2 runs pooled | 4 | 4 | 6 | 7 | 0.811 |
| varc_ho2 | plain VARC, ttt_mini, 40 epochs (4 warmup) | 5 | 5 | 5 | 6 | 0.831 |
| varc_ho2 | plain VARC, ttt_mini, 32 aux tasks | 4 | 6 | 7 | 7 | 0.823 |

Previous loop, for reference (first-order, leave-one-out query, same baseline and eval): Meta-SGD rows s1/s5/s10 0/0/0
(tok .680/.730/.729), then ttt_mini 1/2/2; Meta-SGD layers s1/s5/s10 0/0/0, then ttt_mini 1/1/1.

## Conclusions (VARC, 2026-10-23)

1. **The second-order implementation reproduces the first-order null result.** With the real test pair as query, every
   puzzle in the batch as a task and gradients through the inner step, the learned inner loop solves 0 of 30 held-out tasks
   at 1, 5 and 10 steps for all three adapt sets (rows, layers, everything), token accuracy .68-.72. The previous loop's
   numbers were the same to within a few thousandths. The fixed-α controls on the plain checkpoint match the previous loop to
   0.001, so the eval path is identical; what changed is the training step, and it did not matter.
2. **α does not learn.** After 2000 meta-steps the step sizes sit at their initialisation: rows 1.04 (start 1), layers
   0.011 and everything 0.011 (start 0.01), with no element more than 4x from the start. The meta-gradient on α is not
   pushing anywhere consistently.
3. **The inner step barely fits the demos.** On a held-out task the support loss goes from ~1.35 per row to ~1.16 after 10
   learned steps; the plain checkpoint with a fixed α of 1 on the rows starts lower (~0.90) and drops further (~0.6). The
   meta-trained checkpoints are worse models of new tasks' demos before any adaptation.
4. **Meta-trained weights are a worse start for ordinary test-time training**: 1/1/1 (tok .754-.758) from each of the three,
   against 4/4/4 (.811) from the plain checkpoint. This is also the previous loop's finding.
5. **The "worse start" is the continuation recipe, not the meta step.** Follow-up (user): Meta-SGD for 2000 meta-steps
   *from* the plain checkpoint, against the plain checkpoint continued for 2000 ordinary steps at the same batch 16 and lr
   1e-3. Both land at 1/1/1 after ttt_mini (tok .747 vs .750), down from the plain checkpoint's 4/4/4 (.811), and the
   meta-trained one's support loss on a new task before adaptation is 1.35, the same as the from-scratch meta runs'. So 2000
   steps at batch 16 / lr 1e-3 degrade the checkpoint whichever step is used, and the meta step is neutral against ordinary
   training at that recipe. The learned loop from it: 0/0/0, tok .667. Meta-SGD from scratch at the baseline's own recipe
   (4000 steps x batch 128, ~30 tasks per meta-step, 3.5 h) closes most but not all of the gap: ttt_mini 1/2/2 (tok .787)
   against the plain 4/4/4 (.811), learned loop 10 steps 0/0/0 (.727). α again did not move (weights geometric mean 0.0104,
   max 0.17, 0.3% of elements above 0.02) in 4000 steps, so the data budget was part of the gap and the meta-step the rest.
6. **TRM not run.** Its meta-step unrolls each row's ACT trajectory (up to 16 supervision steps) inside both losses, about
   85 s per 32-row batch on this GPU, i.e. days for the previous loop's recipe; with VARC null across the board there is no
   signal that would justify it. The user allowed skipping it.

## Round 4a: the test-time recipe itself (2026-10-23)

The held-out accuracy comes entirely from test-time training: the plain checkpoint solves 0/30 before it (token accuracy
.641) and 4/30 after ttt_mini (.811). So the recipe is the lever; `ttt_mini` is `ttt_varc` scaled down ~100x for this GPU
(16 aux tasks, 20 epochs, 1 run, 1 view). Measured on the plain checkpoint, each eval 2.5-5 min:

- **Noise floor.** The eval seed (aux augmentations, training order) alone moves pass@1 over 2-4, pass@2 over 4-5, pass@5
  over 4-6, token accuracy .805-.812. A one-task difference in this table is nothing; two is suggestive.
- **Views and runs do nothing** (4/4/6 .808 and 4/4/6 .811): the VARC canvas view and a second run's votes are redundant
  with the 16 aux tasks already pooled.
- **More training helps beyond the noise**: 40 epochs 5/5/5 (.831), 32 aux tasks 4/6/7 (.823). Both are the first results
  above .812 on this build. (Round 4b, scaling these further, was cancelled by the user: more test-time data obviously
  helps; the question is the Meta-SGD recipe.)

## Round 3 and the comparison that matters (2026-10-23)

The user's framing: the goal is a checkpoint that adapts to held-out tasks better than the plain VARC *under the same
test-time procedure*; ttt_mini out-doing a 10-step loop is expected (120 Adam steps with a schedule against 10 plain SGD
steps). Equal-procedure comparisons so far, meta-fine-tuned (round 3) against plain:

- 10 inner steps: .751 against the plain checkpoint's best fixed alpha (.752 with alpha 1 on the rows): a tie. Round 3's
  loop fits the held-out demos far better than earlier rounds (support loss 1.51 -> 0.95 in 10 steps; before, 1.35 ->
  1.16) and is the first learned loop above .73, but still 0/30.
- ttt_mini: 3/3/3 (.809) against 4/4/4 (.811) and the same-recipe control 3/3/5 (.822), all within the eval-seed spread.
  So the "worse TTT start" of rounds 1-2 was the data budget (user's reading), and meta-training from the plain
  checkpoint is now neutral for TTT.

Measured on the budget-matched run's curves (W&B): the support loss at theta stays 1.4-1.9 for all 4000 steps while the
query loss after the step is 0.85-1.1 (plain training: 0.54 on the same rows), i.e. theta is a pre-adaptation point, not
a model, as MAML intends. Alpha on the weights never moved in any run (per-element meta-gradient sign at chance); the
rows' alpha, one per column, did (to 1.8 in round 3).

Round 5 (`scripts/miniarc5/msgd5_varc.sh`, on round 3's recipe from the plain checkpoint) tries "How to train your MAML"
(Antoniou et al., arXiv 1810.09502) and the meta-parameter lrs round 3 never varied: alpha per tensor (LSLR); alpha_lr
1e-3 / 1e-1; outer lr 3e-4; cosine outer lr; and the full MAML++ step (5 inner steps, multi-step loss annealed over the
first 30%, alpha per tensor and per step, first order for the first 300 steps). New options `meta_sgd.inner_steps`,
`msl_anneal_fraction`, `alpha_per_tensor`, `alpha_per_step`, `first_order_steps`, all off by default.

## Round 5 complete (2026-10-23 13:09): six variants, one result

The MAML++ run (37 min: 1.9 s/step first-order, ~2.5 s/step second-order through 5 steps) is the same as the rest: learned
loop .687 / .740 / .754 at 1 / 5 / 10 steps (the one-step number is the best one-step so far, +.014 over round 3, as a
loop trained for 5 steps should be), ttt_mini 3/3/3 (.810), demo fit 1.50 -> 0.94. Its per-step alphas learned a
structure (layer-2 MLP input at the clamp for steps 1-3; layer-0 MLP and patch_embed large at step 3 only; rows 1.1 ->
1.6 -> 0.75 across steps; layer 3 ~0.01 throughout), with 5 tensors at the clamp in step 3. Round 5's six variants
(per-tensor alpha, alpha_lr 1e-3 / 1e-1, outer lr 3e-4, cosine, MAML++) span .748-.754 at 10 steps and .807-.810 after
ttt_mini: the meta-step's optimiser settings and the MAML++ machinery are irrelevant while the inner step cannot move the
weights (section above). Round 7 is the test of that.

## Round 6 (2026-10-23 13:16): the equal-procedure comparison, up to ttt_mini's step count

Same rule on both checkpoints (full-batch SGD on the 48 aux demo rows, all weights + the fresh rows; the meta-trained one
with its learned alpha, the plain one with the hand-set init values alpha 0.01 weights / 1 rows):

| steps | meta-trained (round 3), learned alpha | plain checkpoint, hand-set alpha |
|---|---|---|
| 10 | 0/0/0, .751, demos 1.51 -> 0.95 | 0/0/1, .752 (rows only, alpha 1) / 0/0/1, .727 (layers only, alpha 0.01) |
| 30 | 0/1/1, .772, demos 1.51 -> 0.87 | 1/2/2, .777, demos 1.74 -> 0.78 |
| 120 | 0/2/3, .791, demos 1.51 -> 0.78 | 2/3/4, .801, demos 1.74 -> 0.72 |
| ttt_mini (120 Adam steps) | 3/3/3, .809 | 4/4/4, .811 (same-recipe control 3/3/5, .822) |

Both loops keep improving with steps and at 120 are within .01-.02 of ttt_mini, so the SGD rule itself is not the gap.
But at every step count the plain checkpoint adapts at least as well as the meta-trained one under the same rule, and at
120 steps better (2/3/4 vs 0/2/3; it also fits the demos further, 0.72 vs 0.78, from a worse start). Meta-fine-tuning
has not made the network adapt better; the learned alpha has not beaten the hand-set init it started from. (Note the
plain checkpoint's rows start at the table mean of a table trained on 119 tasks either way.)

## Round 7 (2026-10-23 14:00): letting the weights move

Three ways to let the inner step move the weights, each on round 3's recipe with alpha per tensor:

- **Wider clamp (alpha up to 220).** Alpha used little of it (layer-0 MLP input 2.75, from the old clamp 1.48; the rest
  unchanged) and the result is round 3's again: .673/.738/.752, ttt_mini 3/3/3 (.809). The clamp was not the constraint.
- **Weights' alpha init 1.** Unstable: meta-training loss ~2.0 throughout, alpha stayed at 0.45-2.0, the test loop diverges
  (demo loss 1.80 -> 2238 in 10 steps, tok .584/.526/.472 < the no-adaptation .641). ttt_mini from it is intact (3/3/3,
  .807): the outer step did not damage the weights, the inner step is just unusable at that size on raw gradients.
- **RMS-normalised inner step** (`meta_sgd.normalize_grad`: a step moves every element of a tensor by ~alpha, as Adam
  does; alpha init 1e-3 weights / 3e-2 rows, wide clamp). Meta-training loss 0.9-1.3 like the others. Alpha settled small
  (weights geometric mean 0.0004, the largest 0.005-0.01 on the MLP input matrices and pos_embed; rows 0.013-0.098, mean
  0.043). Learned loop: 0/0/0 .662, 0/0/0 .697, **1/1/1 .738** at 1 / 5 / 10 steps, demo fit 1.53 -> 0.93: the first
  learned loop to solve a held-out task at pass@1, at a lower token accuracy than the raw-gradient loops (.752), and still
  climbing with steps. ttt_mini from it: 3/3/3 (0.806). Round 8 (`msgd8_eval.sh`) runs this loop for 30 and
  120 steps against the plain checkpoint's 1/2/2 (.777) and 2/3/4 (.801).

## Round 8 (2026-10-23 14:05): the normalised learned loop at ttt_mini's step count

| steps, same 48 aux rows | normalised learned loop (round 7 checkpoint) | raw-gradient learned loop (round 3) | plain, raw gradient, hand-set alpha | plain, ttt_mini |
|---|---|---|---|---|
| 10 | 1/1/1, .738, demos -> 0.93 | 0/0/0, .751 | 0/0/1, .752 | |
| 30 | 2/3/4, .795, demos -> 0.75 | 0/1/1, .772 | 1/2/2, .777 | |
| 120 | **7/7/8** (p@100 9), **.833**, demos -> 0.53 | 0/2/3, .791 | 2/3/4, .801 | 4/4/4, .811 (seeds 2-4 / 4-5 / 4-6) |

The first result on this build above the plain checkpoint's test-time training, by 3 tasks at pass@1 and 4 at pass@5,
with the token accuracy .833 above anything measured (the 40-epoch ttt_mini reached .831, 5/5/5). It fits the demos to
0.53, where the raw-gradient loops stop at 0.72-0.78 and ttt_mini's lm_loss after training is ~0.46. Pending controls
(round 9, `msgd9_eval.sh`): the plain checkpoint under the same normalised rule with the hand-set alphas the meta-run
started from (rule or meta-training?), and a second eval seed of the 120-step result.

## Round 9 (2026-10-23 14:12): the gain is mostly the update rule

| steps, RMS-normalised rule, same 48 aux rows | meta-trained (round 7), learned alpha | plain checkpoint, hand-set alpha (1e-3 / 3e-2) |
|---|---|---|
| 10 | 1/1/1, .738, demos -> 0.93 | 1/1/1, .744, demos -> 0.93 |
| 30 | 2/3/4, .795, demos -> 0.75 | 2/2/3, .774, demos -> 0.78 |
| 120, seed 0 | 7/7/8, .833, demos -> 0.53 | 3/5/8, .828, demos -> 0.57 |
| 120, seed 1 | 5/7/9, .836, demos -> 0.54 | 4/7/8, .829, demos -> 0.57 |
| 120, seed 2 | 6/8/9, .839, demos -> 0.55 | 5/8/8, .824, demos -> 0.58 |
| 120, mean of 3 seeds | **6.0 / 7.3 / 8.7, .836** | **4.0 / 6.7 / 8.0, .827** |

- The 120-step result **replicates** on a second eval seed (5/7/9, .836), so the normalised loop at ttt_mini's step
  count is genuinely above ttt_mini (4/4/4, .811; seeds 2-4 at pass@1) on this build.
- But the plain checkpoint under the **same rule with the hand-set alphas** reaches 3/5/8 (.828): the RMS-normalised SGD
  step itself, full-batch on the 48 aux rows for 120 steps, beats ttt_mini's Adam recipe from the plain checkpoint. That
  is a test-time-training finding, not a meta-learning one.
- What meta-training adds on top is small: +2 to +4 at pass@1, +2 at pass@2, ~0 at pass@5, +.005 to +.008 token
  accuracy, a slightly better demo fit (0.53 vs 0.57), and a clearer lead at 30 steps (.795 vs .774). pass@1 differences
  of 2-4 are the size of the eval-seed spread, so round 10 (`msgd10_eval.sh`) adds seeds of both 120-step loops before
  this is called a meta-learning gain.

## Round 10 (2026-10-23 14:21) and where this leaves Meta-SGD

Three eval seeds each, 120 RMS-normalised steps on the same 48 aux rows: the meta-trained checkpoint with its learned
alpha averages 6.0 / 7.3 / 8.7 at pass@1/2/5 and .836 token accuracy; the plain checkpoint with the hand-set alphas
4.0 / 6.7 / 8.0 and .827. The token accuracy is higher on every seed pair (.833/.836/.839 against .828/.829/.824, no
overlap) and the demo fit deeper (0.54 against 0.57), so meta-training does make the network adapt better under the same
procedure, by ~2 tasks at pass@1 and ~1 point of token accuracy. Both are far above the plain checkpoint's ttt_mini
(4/4/4, .811, and 3-5 / 4-5 / 4-6 over seeds).

In order of size, the ingredients of 6/7/9 against 4/4/4 are:

1. **The inner update rule** (RMS-normalised gradient per tensor, i.e. a step moves every element by ~alpha, as Adam does;
   full batch on all 48 aux rows): the plain checkpoint goes from 2/3/4 (.801, raw gradient) to 4/7/8 (.827) at 120
   steps, and above its own Adam ttt_mini. Round 5's post-mortem was right about the mechanism: the weights could not
   move under alpha x raw gradient (0.26% in 10 steps against Adam's 7.6%).
2. **Steps**: every loop keeps improving to 120 steps (the meta-trained normalised one .738 -> .795 -> .836 at 10 / 30 /
   120); the one-step regime of the paper is nowhere near enough here.
3. **Meta-training** (1000 meta-fine-tuning steps from the plain checkpoint with the normalised inner step, round 3's
   episodes): +2 pass@1, +0.9 token accuracy, consistent over seeds; with the learned per-tensor alphas 5-10x apart
   across tensors (MLP inputs and pos_embed highest, layer 3 lowest).

Everything tried before the normalised step (rounds 2-5, 7a/b: alpha granularity, alpha_lr, outer lr and schedule,
multi-step loss, per-step alpha, the clamp, a larger init) was equal to round 3 within noise, because none of it let the
weights move. Not done: the deferred from-scratch run with this recipe at the baseline budget (user: keep in mind);
meta-training with more inner steps under the normalised rule; longer test loops than 120.

## Queued runs (as of 2026-10-23 11:30; one GPU job at a time)

All training runs start from the plain checkpoint `varc_ho2` on round 3's recipe (`FT3`/`EPISODE3` in
`scripts/miniarc5/msgd_common.sh`: batch 256, lr 1e-4, 1000 steps; task = 16 augmented copies, fresh rows, alpha init rows
1 / weights 0.01, alpha_lr 1e-2) and get the four evaluations of `grid` (learned loop 1/5/10 steps, then ttt_mini). A
single-step run is ~9 min of training + ~5 min of evaluations; round 3's per-eval times were 0.5 / 0.7 / 0.6 / 2.7 min.

| queue | run | change against round 3 | eval prefix | ETA |
|---|---|---|---|---|
| 5 (`msgd5_varc.sh`, started 11:22) | msgd5_pt | alpha per tensor (LSLR) | ms7pt | 11:40 |
| 5 | msgd5_alr1e-3 | alpha_lr 1e-3 | ms7a3 | 11:55 |
| 5 | msgd5_alr1e-1 | alpha_lr 1e-1 | ms7a1 | 12:10 |
| 5 | msgd5_lr3e-4 | outer lr 3e-4 | ms7lr | 12:25 |
| 5 | msgd5_cos | cosine outer lr, lr_min_ratio 0.1 | ms7cos | 12:40 |
| 5 | msgd5_pp | MAML++: 5 inner steps, multi-step loss annealed over the first 30%, alpha per tensor and per step, first order for the first 300 steps | ms7pp | ~13:15 (started 12:28; 1.9 s/step in the first-order phase, slower after step 300) |
| 6 (`msgd6_eval.sh`, chained) | msgd3_varc_ft | learned loop 30 and 120 steps (ttt_mini's step count), evaluations only | ms5va_s30, ms5va_s120 | ~14:30 |
| 6 | varc_ho2 | the same loop with hand-set alpha 0.01 weights / 1 rows, 30 and 120 steps | ms4vfix_s30, ms4vfix_s120 | ~13:45 |
| 7 (`msgd7_varc.sh`, chained; user) | msgd7_range10 | alpha per tensor, clamp e^10 (alpha up to 220) | ms8r | ~14:00 |
| 7 | msgd7_init1 | alpha per tensor, weights' alpha init 1 | ms8i | ~14:15 |
| 7 | msgd7_norm | RMS-normalised inner step (`meta_sgd.normalize_grad`), alpha per tensor, init 1e-3 weights / 3e-2 rows, clamp e^10 | ms8n | ~14:30 |

Queue logs: `autoresearch_runs/miniarc5/msgd5_varc_queue.log`, `msgd6_eval_queue.log`. The table script
(`scripts/miniarc5/msgd2_table.py`) has rows for every one of these and prints `pending` until the result exists.

## Round 5, first three variants (2026-10-23 12:10): alpha learns, the result does not move

Per-tensor alpha (LSLR), alpha_lr 1e-3 and alpha_lr 1e-1 all reproduce round 3 to a few thousandths: learned loop 10 steps
.753 / .748 / .754 (round 3 .751), ttt_mini 3/3/3 .807-.809, support loss after 10 steps 0.94-0.96 (round 3 0.95).

Yet alpha did learn this time. Per tensor (init 0.01, clamp 0.01·e^±5 = [7e-5, 1.48]): the MLP input matrices of layers
0-2 went to 1.48 (the clamp), 0.38 and 0.93, pos_embed 0.87, patch_embed 0.27, the qkv matrices of layers 0-1 0.05;
nearly everything else shrank to 0.002-0.005, layer 3 entirely. Per element at alpha_lr 1e-1: 1st percentile 0.0004,
99th 1.48 (the clamp), rows 0.16-7.5. So the meta-gradient has a clear, consistent preference (big steps on the early MLPs
and the position/patch embeddings, none on the last layer), and acting on it changes nothing at test.

The plain checkpoint explains why: alpha 1 on the rows alone, 10 steps, gives .752 and fits the demos *better* (1.74 ->
0.86) than any learned loop (1.51 -> 0.95); alpha 0.01 on the layers alone gives .727 and 1.74 -> 1.14. Every learned
loop is, in effect, the rows-only loop: the weight updates contribute nothing measurable even at alpha 1.48, because a
plain SGD step moves a weight by alpha times its gradient, and the per-element gradient of the mean per-row loss on 48
demos is tiny. ttt_mini's Adam moves every weight by ~lr per step whatever the gradient, 120 times, with a consistent
sign: orders of magnitude more weight change. The clamp binding on the MLP matrices says the meta-gradient wanted to go
further still.

Measured (`scratchpad/weight_change.py`, CPU, held-out task `1_3_5_l6aejqqqc1b47pjr5g4`, the per-tensor checkpoint, the
same 48 aux rows both ways), relative weight change ||theta' - theta|| / ||theta||:

| | learned loop, 10 steps | ttt_mini, ~100-120 Adam steps |
|---|---|---|
| all weights | 0.26% | 7.6% |
| layer-0 MLP input matrix (alpha 1.48, the clamp) | 0.67% | 11.9% |
| layer-1 / layer-2 MLP input matrices (alpha 0.38 / 0.93) | 0.22% / 0.27% | 13.4% / 11.1% |
| attention qkv matrices (alpha 0.004-0.05) | 0.04-0.17% | 5.4-7.7% |
| pos_embed (alpha 0.87) | 7.0% | 23.8% |
| puzzle-embedding rows | the adaptation | sign-SGD at 0.03 |

The learned loop moves the weights 30x less overall and 50-100x less in the matrices; only pos_embed and a few biases
move at all. Its demo fit on this task (0.86 -> 0.63) comes from the rows and those few tensors.

So the missing piece is not alpha's learning rate or its granularity but its *range*, or the update rule: a learned step
size on raw gradients cannot reach Adam-sized weight updates within alpha <= 1.48. Candidates (not started; user
present): alpha_range 10 (alpha up to 220) with per-tensor alpha; a higher weights' init (0.1 or 1); or a normalised inner
step (sign / RMS of the gradient, as Adam does), which leaves Meta-SGD proper.

## Post-mortem (2026-10-23, after the 2000-step VARC runs)

Measured on the all-parameters checkpoint over 45 training-task episodes (CPU, train mode): support loss per row at θ
1.54 on its *own* training tasks (the plain checkpoint: 0.90 on unseen ones); the inner step lowers the query loss
1.47 -> 1.19; replacing the task's trained row by the table mean changes that to 1.51 -> 1.23, i.e. the rows carry nothing;
the sign of ∂L_query/∂ρ agrees across 12 batches at chance level (0.26 vs 0.23 for coin flips; 1.7% of elements consistent).

- α's gradient is noise, so Adam random-walks ρ by ~lr·√T = 0.045 nats (the measured 4-10% drift); even a consistent
  gradient could only move it lr·T = 2 nats in this budget, while the rows need a factor 100 from the 0.01 start.
- The gradient is noisy because an episode is one augmented copy: 2-3 demo rows and one test pair, ~4 tasks per step; α's
  gradient per element is a product of two such small gradients. The previous loop averaged 16 copies per task.
- Under-training dominates: 2000 x 16 rows is under two passes; the baseline had ~29. Rows got ~2 sign-SGD updates each.
- Train/test mismatch for the row: at test a held-out task's rows are fresh (table mean); in training the outer loop trains
  them, so a fitted model would put the task in the row and leave the inner step nothing to learn.

Fixes (user: implement all): (1) meta-fine-tune the plain checkpoint at its batch size and a tenth of its lr; (2) α
initialised per group (rows 1, weights 0.01) with a 10x larger lr; (3) the episode is the task with 16 augmented copies
(`meta_sgd.copies_per_group`, sampler draws k copies of a group and tags `batch["group_ids"]`); (4) fresh rows
(`meta_sgd.fresh_rows`: the episode's rows start from the table mean, the table is not trained). Round 3 =
`scripts/miniarc5/msgd3_varc.sh`: `msgd3_varc_ft` (all four) against `varc_ho2_cont2` (the plain checkpoint continued with
the same recipe and the ordinary step), batch 256 (~3.5 tasks x 16 copies), lr 1e-4, 1000 steps.

## Log

- **2026-10-22 23:30.** Baseline re-evaluated through the new eval path: 4/4/4, tok .811, identical to the previous loop's
  `ms3v_adam`, so the eval path reproduces. Fixed α = 1 on the rows, 5 steps: 0/0/0 (p@100 1), tok .744, also identical
  to the previous `ms3ve_s5`. Meta-training queue started (`autoresearch_runs/miniarc5/msgd2_varc_queue.log`).
- **2026-10-22 23:51. Rows-only Meta-SGD (`msgd2_varc_emb`) done.** Meta-training 12 min (2.8 meta-steps/s), query loss
  2.0 -> 1.6; the learned α on the rows stayed at its start (geometric mean 1.04, range 0.79-1.25). Learned loop 1/5/10
  steps: 0/0/0 solved, tok .676/.721/.721, the same as the previous first-order run (.680/.730/.729). Support loss after 10
  learned steps: median 1.16 per row (range 0.65-1.72), i.e. the demos are far from fitted. Ordinary ttt_mini from this
  checkpoint: 1/1/1, tok .754 (previous loop 1/2/2, .750), against the plain checkpoint's 4/4/4, .811. The fixed-α controls
  on the plain checkpoint reproduce the previous loop to 0.001 (rows 10 steps .752, layers 5/10 .715/.727). Layers-only
  run started.
- **2026-10-23 00:09. Layers-only Meta-SGD (`msgd2_varc_lh`) done.** Meta-training 13.5 min, query loss 2.0 -> 1.30 (lower
  than rows-only's 1.60: the inner step on 530k layer weights fits the demos better than one row). Learned α stayed near
  0.01 (geometric mean 0.0112, max 0.039, 0.5% of elements above 0.02). Learned loop 1/5/10 steps: 0/0/0, tok
  .682/.705/.712 (previous first-order run .731/.722/.722); support loss after 10 steps median 1.20 per row. Ordinary
  ttt_mini from this checkpoint: 1/1/1, tok 0.758 (previous 1/1/1, .748). All-parameters run started.
- **2026-10-23 00:28. All-parameters Meta-SGD (`msgd2_varc_all`) done; queue finished (57 min in all).** Meta-training
  14 min, query loss 2.0 -> 1.21 (the lowest of the three). α stayed at 0.01 everywhere (weights geometric mean 0.0109, max
  0.033; the embedding row 0.0116, max 0.014, so the row never got the step of 1 the rows-only run was given). Learned loop
  1/5/10: 0/0/0, tok .688/.717/.721; support loss after 10 steps median 1.16 per row. Ordinary ttt_mini from it: 1/1/1,
  tok .758. Conclusions written above.
- **2026-10-23 00:55. Continued-run pair done.** `msgd2_varc_all_ft` (Meta-SGD everything from the plain checkpoint,
  2000 meta-steps at batch 16, 14.5 min; α stayed at 0.01, max 0.022): ttt_mini 1/1/1 tok .747, learned loop 10 steps
  0/0/0 tok .667, support loss 1.35 -> 1.24. Control `varc_ho2_cont` (plain, continued 2000 steps at batch 16, 2 min):
  ttt_mini 1/1/1 tok .750. Conclusion 5 rewritten. `msgd2_varc_all_full` (from scratch, 4000 x batch 128) started.
- **2026-10-23 04:31. Baseline-budget run (`msgd2_varc_all_full`) done; round-2 queue finished.** 3 h 33 min for 4000
  meta-steps at batch 128 (3.2 s/step, 2.3 GB resident). ttt_mini from it: 1/2/2, tok .787; learned loop 10 steps: 0/0/0,
  tok .727. α unchanged from 0.01 (geometric mean 0.0104, max 0.17). Conclusion 5 extended. Round 3
  (`scripts/miniarc5/msgd3_varc.sh`) started on the GPU; `msgd2_varc.sh` now sources `msgd_common.sh`.
- **2026-10-23 04:37. Round 3 crashed at meta-step 257** (CUDA index assert in the query forward). Cause: with 16 copies
  per group the batch tail can cut a copy's demo rows and keep its validation row; the query ids are remapped into the
  support ids by `searchsorted`, so the orphan pointed past the rows (the assert) or, when a larger id existed, at another
  copy's row (silent; batches 19 and 246 of the same stream). Fixed in `_episodes_from_batch` (such query rows are dropped,
  test added); the failed log is `msgd3_varc_ft/train_failed_step257.log`. Control `varc_ho2_cont2` finished (1000 steps,
  1 min). Round 4a (`msgd4_varc.sh`: the plain checkpoint's test-time recipe, no-adaptation and seed controls, canvas
  views, runs, epochs, aux tasks) took the GPU; round 3 is re-queued behind it with the Meta-SGD tests first.
- **2026-10-23 05:00. Round 4a done (23 min).** Rows and section above. No adaptation 0/0/0 (.641); seeds 1/2 3/4/6 (.812)
  and 2/5/6 (.805); views 4/4/6 (.808); runs 4/4/6 (.811); 40 epochs 5/5/5 (.831); 32 aux 4/6/7 (.823). Round 4b queued
  behind the round-3 re-run.
- **2026-10-23 10:45. GPU idle 05:00-10:45 by mistake.** The round-3 re-run wrapper waited on `pgrep -f msgd4_varc.sh`,
  which matched the wrapper's own command line, and round 4b waited on that wrapper. Both stopped; the pgrep waits are
  removed from the queue scripts and the three jobs (tests, round 3, round 4b) now run chained in one shell.
- **2026-10-23 11:00. Round 3 done** (8 min meta-training at 2 it/s, 3.5 tasks x 16 copies per step). Learned loop
  1/5/10: 0/0/0, tok .673/.737/.751; then ttt_mini 3/3/3 (.809); control `varc_ho2_cont2` then ttt_mini 3/3/5 (.822).
  Alpha: weights 0.010 (unchanged), rows up to 1.8. Section above; round 5 queued (`msgd5_varc_queue.log`).
- **2026-10-23 12:10. Round 5, first three variants done** (per tensor, alpha_lr 1e-3, 1e-1): all equal to round 3; alpha
  learned a clear structure and hit the clamp on the early MLPs; see the section above. lr 3e-4 run training.
- **2026-10-23 12:14. Round 5, outer lr 3e-4 done:** learned loop .671/.737/.751, ttt_mini 3/3/3 (.808), demo fit
  1.53 -> 0.96, alpha weights 0.011 (max at the clamp), rows 1.5. Same as round 3. Cosine-lr run training.
- **2026-10-23 12:28. Round 5, cosine outer lr done:** .671/.736/.750, ttt_mini 3/3/3 (.808), demo fit 1.51 -> 0.95. All
  five single-step variants of round 5 equal round 3. MAML++ run training (1.9 s/step first-order). Round 7 queued (user's
  three fixes for the immobile weights), ETAs in the queued-runs table.
- **2026-10-23 13:09. Round 5 done (MAML++ run):** .687/.740/.754, ttt_mini 3/3/3 (.810). Section "Round 5 complete".
  Round 6 (30/120-step loops) running.
- **2026-10-23 13:10. Round 6, round-3 checkpoint at 30 learned steps:** 0/1/1 (p@100 1), tok .772, demo fit 1.51 ->
  0.87. First held-out task a learned loop has solved (at pass@2), and still improving with steps (.751 at 10). 120 steps
  and the plain checkpoint's fixed-alpha loops running.
- **2026-10-23 13:11. Round 6, plain checkpoint under the same rule, 30 steps** (alpha 0.01 weights / 1 rows, hand-set):
  1/2/2, tok .777, demo fit 1.74 -> 0.78. Equal-procedure comparison at 30 steps: plain 1/2/2 .777 against meta-trained
  0/1/1 .772, i.e. no advantage for the meta-trained checkpoint (differences within noise). 120-step runs next.
- **2026-10-23 13:13. Round 6, round-3 checkpoint at 120 learned steps** (2.3 min): 0/2/3 (p@100 4), tok .791, demo fit
  1.51 -> 0.78. Still climbing with steps (.751 / .772 / .791 at 10 / 30 / 120) and now within .02 of ttt_mini's .809,
  with pass@5 3 against ttt_mini's 3, but pass@1 0 against 3. Plain checkpoint's 120-step loop running.
- **2026-10-23 13:16. Round 6 done.** Plain checkpoint, hand-set alpha, 120 steps: 2/3/4, tok .801, demos 1.74 -> 0.72:
  better than the meta-trained checkpoint's 0/2/3 (.791) under the same rule. Section "Round 6" above. Round 7 started
  (`msgd7_range10`).
- **2026-10-23 13:30. Round 7, wide clamp done** (10 min training): learned loop .673/.738/.752, ttt_mini 3/3/3
  (0.809), demo fit 1.51 -> 0.95. Alpha used the range only a little: layer-0 MLP input 2.75 (was at the 1.48 clamp),
  layer-2 MLP 0.89, pos_embed 0.86, the rest as in round 5. Same result as every run since round 3. Alpha-init-1 run
  training.
- **2026-10-23 13:45. Round 7, alpha init 1 done:** the inner step of a full raw gradient on every weight is unstable: the
  meta-training loss stayed ~2.0 throughout (others 0.9-1.1), alpha stayed near 1 (0.45-2.0; the meta-gradient did not
  pull it down), and at test the loop diverges (demo loss 1.80 -> 2238 in 10 steps; tok .584/.526/.472, below the
  no-adaptation .641). ttt_mini from it: 3/3/3 (0.807). Normalised-step run training.
- **2026-10-23 14:01. Round 7 done (normalised step):** loop .662/.697/.738 with 1/1/1 at 10 steps, ttt_mini
  3/3/3 (0.806). Section "Round 7" above. Round 8 (30/120 steps of this loop) started.
- **2026-10-23 14:05. Round 8 done:** normalised learned loop 30 steps 2/3/4 (.795), 120 steps 7/7/8 (.833), demos 1.53
  -> 0.53. Section "Round 8" above. Round 9 controls started.
- **2026-10-23 14:12. Round 9 done:** learned loop 120 steps seed 1: 5/7/9 (.836), replicates. Plain checkpoint under the
  same normalised rule: 1/1/1 (.744), 2/2/3 (.774), 3/5/8 (.828) at 10/30/120. Section "Round 9" above. Round 10 (seeds)
  started.
- **2026-10-23 14:21. Round 10 done:** plain normalised 120 steps seeds 1/2: 4/7/8 (.829), 5/8/8 (.824); learned loop seed
  2: 6/8/9 (.839). Three-seed means 6.0/7.3/8.7 (.836) meta-trained against 4.0/6.7/8.0 (.827) plain. Section "Round 10"
  above. GPU idle.
