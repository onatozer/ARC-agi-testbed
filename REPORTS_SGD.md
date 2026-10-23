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
5. **Confound, carried over from the previous loop.** The meta-trained checkpoints saw far less data than the baseline:
   2000 meta-steps at batch 16 (one query row per task, ~3-5 tasks per step) against 4000 steps at batch 128. Point 3 is
   consistent with under-training as much as with meta-learning failing. The clean test is to start Meta-SGD from the plain
   checkpoint (`load_checkpoint=<varc_ho2>`, ~13 min) and compare against the plain checkpoint continued for the same
   steps. Not run.
6. **TRM not run.** Its meta-step unrolls each row's ACT trajectory (up to 16 supervision steps) inside both losses, about
   85 s per 32-row batch on this GPU, i.e. days for the previous loop's recipe; with VARC null across the board there is no
   signal that would justify it. The user allowed skipping it.

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
