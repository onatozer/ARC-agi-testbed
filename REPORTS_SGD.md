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

Bold: the best value in its column; underlined: the rest of the column's top 10% (`scripts/miniarc5/highlight_results.py`).
TRM rows use only checkpoints trained on this branch and build (trm_ho3 and its fine-tunes); the trm_ho2-based ones
(round 16; trm_ho2 was trained on another worktree's build) are removed from this table (user). VARC rows from varc_ho2
(trained on l-reinit-ttt's build) remain; varc_ho3-based rows are marked "this branch" or "from varc_ho3".

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
| msgd7_norm | round 11: normalised learned loop, 300 steps | 8 | 9 | 9 | 13 | 0.862 |
| varc_ho2 | round 11: plain VARC, normalised step, hand-set alpha, 300 steps | 7 | 9 | 10 | 11 | 0.853 |
| msgd7_norm | round 11: normalised learned loop, 1000 steps | <u>10</u> | <u>13</u> | **15** | <u>16</u> | <u>0.887</u> |
| varc_ho2 | round 11: plain VARC, normalised step, hand-set alpha, 1000 steps | <u>10</u> | <u>13</u> | **15** | <u>16</u> | <u>0.879</u> |
| varc_ho3 | round 13: plain VARC retrained on this branch and build (varc_ho2's recipe), ttt_mini | 3 | 4 | 5 | 7 | 0.793 |
| varc_ho3 | round 13: retrained plain VARC, normalised step, hand-set alpha, 120 steps | 2 | 5 | 5 | 5 | 0.811 |
| varc_ho4 | round 14: plain VARC, this branch's code on varc_ho2's build, ttt_mini | 4 | 4 | 4 | 7 | 0.812 |
| varc_ho4 | round 14: this branch's code on varc_ho2's build, normalised 120 steps | 5 | 6 | 8 | 10 | 0.821 |
| varc_ho5 | round 14: plain VARC, this build, training seed 1, ttt_mini | 4 | 4 | 5 | 6 | 0.795 |
| varc_ho5 | round 14: this build, training seed 1, normalised 120 steps | 4 | 6 | 8 | 9 | 0.817 |
| msgd11_ms5 | round 11: meta-trained with 5 normalised inner steps (alpha per tensor and step, MSL); learned loop 10 / 30 / 120 steps, last alpha held | 1 / 2 / 7 | 1 / 2 / 7 | 1 / 3 / 8 | 1 / 5 / 10 | .753 / .797 / .837 |
| msgd11_ms5 | round 11: 5-step meta-trained; then ttt_mini | 3 | 3 | 3 | 5 | 0.804 |
| msgd11_ms5 | round 12: 5-step meta-trained; 120 steps, per-step alphas stretched over the loop | 7 | 8 | 9 | 9 | 0.845 |
| msgd11_ms5 | round 12: 5-step meta-trained; 300 steps, stretched | <u>10</u> | <u>12</u> | <u>13</u> | 14 | <u>0.872</u> |
| msgd11_ms5 | round 12: 5-step meta-trained; 1000 steps, stretched | **12** | <u>12</u> | <u>14</u> | <u>16</u> | <u>0.888</u> |
| msgd11_ms5 | round 12: 5-step meta-trained; 300 steps, last alpha held | <u>10</u> | <u>12</u> | <u>13</u> | 14 | <u>0.865</u> |
| msgd11_ms5 | round 12: 5-step meta-trained; 1000 steps, last alpha held | **12** | <u>13</u> | **15** | <u>16</u> | <u>0.885</u> |
| msgd11_ms10 | round 11: meta-trained with 10 normalised inner steps (alpha per tensor and step, MSL); learned loop 10 / 30 / 120 steps, last alpha held | 1 / 2 / 4 | 1 / 3 / 5 | 2 / 4 / 6 | 2 / 7 / 8 | .782 / .803 / .833 |
| msgd11_ms10 | round 11: 10-step meta-trained; then ttt_mini | 3 | 3 | 3 | 5 | 0.806 |
| msgd11_ms10 | round 12: 10-step meta-trained; 120 / 300 / 1000 steps, stretched | 7 / <u>10</u> / **12** | 7 / <u>11</u> / <u>13</u> | 10 / <u>13</u> / <u>14</u> | 10 / 14 / **18** | .845 / <u>.870</u> / <u>.888</u> |
| msgd11_ms10 | round 12: 10-step meta-trained; 300 / 1000 steps, last alpha held | 9 / <u>10</u> | 10 / <u>11</u> | <u>13</u> / <u>14</u> | 14 / <u>16</u> | .862 / <u>.884</u> |
| msgd15_n1_fo | round 15: one-step normalised recipe, entirely first order; 120 steps | 5 | 6 | 8 | 9 | 0.832 |
| msgd15_n1_fo | round 15: one-step normalised recipe, first order; 300 steps | 6 | 10 | 11 | 12 | <u>0.863</u> |
| msgd15_n1_fo | round 15: one-step normalised recipe, first order; 1000 steps | <u>10</u> | <u>11</u> | <u>13</u> | <u>15</u> | <u>0.884</u> |
| msgd15_ms5_fo | round 15: 5-step normalised recipe, first order; 120 steps, stretched | 7 | 7 | 8 | 10 | 0.841 |
| msgd15_ms5_fo | round 15: 5-step first order; 300 steps, stretched | 8 | 9 | 12 | 12 | <u>0.872</u> |
| msgd15_ms5_fo | round 15: 5-step first order; 1000 steps, stretched | **12** | <u>12</u> | <u>14</u> | <u>17</u> | **0.890** |
| msgd15_ms5_fo | round 15: 5-step first order; 120 steps, held | 5 | 7 | 8 | 9 | 0.837 |
| varc_ho3 | round 17: plain VARC (this branch), normalised step, hand-set alpha, 120 / 300 / 1000 steps | 2 / 8 / <u>11</u> | 5 / 9 / <u>13</u> | 5 / 10 / <u>13</u> | 5 / 11 / **18** | .811 / .847 / <u>.878</u> |
| msgd17_v_a1e-3 | round 17: VARC (from varc_ho3) 5-step first order, alpha init 1e-3 / 3e-2; 120 / 300 / 1000 steps, stretched | 6 / 8 / **12** | 7 / 9 / <u>13</u> | 7 / 11 / <u>14</u> | 11 / 14 / <u>16</u> | .844 / .861 / <u>.885</u> |
| msgd17_v_a0.1 | round 17: same, alpha init 0.1; 120 / 300 / 1000 steps, stretched | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | .087 / .000 / .000 |
| msgd17_v_a1 | round 17: same, alpha init 1; 120 / 300 / 1000 steps, stretched | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | .000 / .000 / .000 |
| trm_ho3 | round 18: plain TRM retrained on this branch, ttt_mini | 1 | 2 | 3 | 3 | 0.779 |
| trm_ho3 | round 18: plain TRM (this branch), normalised segment steps, ACT off, hand-set alpha; 120 / 300 / 1000 steps | 5 / 7 / 8 | 7 / 10 / 9 | 8 / 11 / 12 | 10 / 12 / <u>15</u> | .799 / .822 / .815 |
| msgd18_trm | round 18: TRM (from trm_ho3) meta-fine-tuned, 8 segment steps, ACT off, alpha init 1e-3; 120 / 300 / 1000 steps, segment schedule | 6 / 8 / <u>11</u> | 6 / 9 / <u>13</u> | 7 / 9 / **15** | 10 / 13 / <u>17</u> | .798 / .817 / .829 |
| trm_ho3 | round 19: plain TRM (this branch), normalised segment steps, ACT ON, hand-set alpha, 1000 steps | **12** | **14** | <u>14</u> | <u>16</u> | 0.835 |
| msgd18_trm | round 20: TRM meta-fine-tuned (ACT off in training), ACT ON at test, segment schedule, 1000 steps | 9 | <u>11</u> | 12 | 13 | 0.825 |
| msgd21_trm_act | round 21: TRM meta-fine-tuned with ACT on, ACT on at test, segment schedule, 1000 steps | <u>11</u> | <u>11</u> | <u>13</u> | 14 | 0.832 |
| trm_ho3 | round 22: plain TRM, ACT on, halt-and-wait, hand-set alpha, 1000 steps | 5 | 6 | 7 | 13 | 0.816 |
| msgd22_trm_wait | round 22: TRM meta-fine-tuned with ACT on and halt-and-wait, segment schedule, 1000 steps | 9 | <u>11</u> | <u>13</u> | <u>17</u> | 0.816 |
| varc_ho3 | round 23: plain VARC (this branch), Embed-TTT (embedding only, arXiv 2609.21181), ttt_mini length | 2 | 3 | 3 | 3 | 0.760 |
| varc_ho3 | round 23: plain VARC (this branch), Embed-Full-TTT (embedding, then backbone), ttt_mini length | 3 | 3 | 3 | 3 | 0.779 |
| trm_ho3 | round 23: plain TRM (this branch), Embed-TTT, ttt_mini length | 1 | 1 | 1 | 1 | 0.751 |
| trm_ho3 | round 23: plain TRM (this branch), Embed-Full-TTT, ttt_mini length | 1 | 1 | 1 | 1 | 0.755 |
| varc_ho3 | round 24: plain VARC, embedding first (500 rows-only + 500 weights-only steps), hand-set alpha | 9 | <u>11</u> | <u>13</u> | <u>15</u> | 0.858 |
| msgd17_v_a1e-3 | round 24: VARC meta-fine-tuned, embedding first, stretched, 1000 steps | 9 | <u>12</u> | 12 | <u>17</u> | <u>0.876</u> |
| trm_ho3 | round 24: plain TRM, ACT on, embedding first, hand-set alpha, 1000 steps | 9 | <u>11</u> | 11 | 12 | 0.817 |
| msgd18_trm | round 24: TRM meta-fine-tuned (ACT off), embedding first, segment schedule, 1000 steps | 9 | 10 | 11 | 12 | 0.806 |
| msgd11_scratch | round 11: normalised recipe from scratch, 4000 x batch 128 (baseline budget); learned loop 10 / 30 / 120 steps | 1 / 1 / 1 | 3 / 2 / 1 | 3 / 2 / 2 | 4 / 3 / 3 | .746 / .751 / .775 |
| msgd11_scratch | round 11: from scratch; then ttt_mini | 2 | 3 | 3 | 3 | 0.772 |
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

## Round 11a (2026-10-23 16:26): longer loops

RMS-normalised rule, full batch on the 48 aux rows, same checkpoints as round 9:

| steps | meta-trained (round 7), learned alpha | plain checkpoint, hand-set alpha | eval time / 30 tasks |
|---|---|---|---|
| 120 (3 seeds) | 6.0 / 7.3 / 8.7, .836, demos -> 0.54 | 4.0 / 6.7 / 8.0, .827, demos -> 0.57 | 2.5 min |
| 300 | 8 / 9 / 9, .862, demos -> 0.43 | 7 / 9 / 10, .853, demos -> 0.46 | 5.6 min |
| 1000 | 10 / 13 / 15, .887, demos -> 0.36 | 10 / 13 / 15, .879, demos -> 0.37 | 20 min |

Neither curve has flattened by 1000 steps: pass@1 goes 6 -> 8 -> 10 and pass@5 9 -> 9 -> 15. At 1000 steps the two
checkpoints solve the same tasks counts (10/13/15); the meta-trained one keeps a .008 token-accuracy lead (.887 vs .879)
and a slightly deeper demo fit. So the meta-training increment is a constant small offset in token accuracy that shows up
as 1-2 tasks at 120-300 steps and as nothing in task counts at 1000, where both reach half the held-out tasks at pass@5.
The large effect is the test-time procedure: normalised full-batch SGD for 1000 steps (20 min / 30 tasks) takes the plain
checkpoint from 4/4/4 (.811, ttt_mini) to 10/13/15 (.879).

## Round 11b (2026-10-23 17:28): meta-training the 5-step loop

`msgd11_ms5`: round 3's recipe from the plain checkpoint with the normalised inner step, 5 inner steps, alpha per tensor
and per step, multi-step loss annealed over the first 30%, first order for the first 300 meta-steps; 54 min (2.1 s/step
first-order, 3-4.9 s/step second-order). Meta-training query loss 0.77-1.0 at the end (the one-step runs: 0.9-1.3).

The per-step alphas learned a **decaying schedule**: rows 0.058 / 0.039 / 0.020 / 0.016 / 0.010 over the five steps,
weights ~0.0006 -> 0.0002 (pos_embed 0.001 -> 0.004 -> 0.002, layer-2 MLP input 0.0025 -> 0.0005), i.e. a learned
learning-rate decay with the same layer structure as before.

Evaluated with the last alpha held past step 5 (the smallest): 1/1/1 (.753), 2/2/3 (.797), 7/7/8 (.837) at 10 / 30 /
120 steps, demo fit 1.58 -> 0.86 / 0.73 / 0.54; ttt_mini 3/3/3 (0.804). Level with the one-step normalised
checkpoint (.738 / .795 / .836), slightly ahead at 10 steps. Round 12 evaluates the same checkpoint with the alphas
stretched over the loop (user's idea) and at 300 / 1000 steps.

## Round 11c (2026-10-23 18:08): the deferred from-scratch run

`msgd11_scratch`: the normalised recipe (16-copy episodes, fresh rows, alpha per tensor, init 1e-3 / 3e-2, alpha_lr 1e-2)
from random init at the baseline's budget, 4000 steps x batch 128, lr 1e-3; 31 min (under two tasks per step). Plain
control = the baseline `varc_ho2` (same build, budget, seed). Query loss noisy, 0.7-1.8 at the end; the weights' alpha
collapsed (geometric mean 0.0001; only the final norm 0.33 and a few norm / embedding tensors ~0.01), rows 0.08.

| 120 normalised steps | pass@1/2/5 | token acc |
|---|---|---|
| meta-trained from scratch (this run), learned alpha | 1 / 1 / 2 | .775 |
| plain from scratch (`varc_ho2`), hand-set alpha, same rule | 4 / 7 / 8 (3 seeds: 3-5 / 5-8 / 8) | .827 |
| plain checkpoint meta-fine-tuned (round 7), learned alpha | 6 / 7 / 9 (3 seeds) | .836 |

Loops 10 / 30 / 120: 1/3/3 (.746), 1/2/2 (.751), 1/1/2 (.775), demo fit 1.58 -> 0.95 / 0.83 / 0.70; ttt_mini
2/3/3 (0.772) against the plain 4/4/4 (.811). So **Meta-SGD from scratch does not beat plain training from
scratch** on this build, even with the normalised step: the gain of rounds 7-11 belongs to meta-fine-tuning a trained
model (and mostly to the test-time rule). Likely reasons, untested: ~1.75 tasks per meta-step (the baseline sees ~30
puzzles per step), the fresh-row regime from random init (the puzzle table is never trained, so theta must carry every
task), and 1000 meta-fine-tuning steps on top of 4000 plain being simply more training than 4000 meta-steps.

## Round 12a (2026-10-23 20:36): the 5-step checkpoint, stretched vs held alphas

RMS-normalised rule, full batch on the 48 aux rows, one eval seed each:

| steps | 5-step, stretched | 5-step, last alpha held | one-step checkpoint | plain, hand-set alpha |
|---|---|---|---|---|
| 120 | 7 / 8 / 9, .845, demos -> 0.49 | 7 / 7 / 8, .837, demos -> 0.54 | 6.0 / 7.3 / 8.7, .836 (3 seeds) | 4.0 / 6.7 / 8.0, .827 (3 seeds) |
| 300 | 10 / 12 / 13, .872, demos -> 0.41 | 10 / 12 / 13, .865, demos -> 0.45 | 8 / 9 / 9, .862 | 7 / 9 / 10, .853 |
| 1000 | 12 / 12 / 14, .888, demos -> 0.35 | 12 / 13 / 15, .885, demos -> 0.36 | 10 / 13 / 15, .887 | 10 / 13 / 15, .879 |

- **Meta-training the 5-step loop is the gain.** Either schedule beats the one-step and plain checkpoints by 2-3 tasks
  at pass@1 at 300 and 1000 steps (10 vs 7-8; 12 vs 10), and by 3-4 at pass@2/5 at 300. This is the first meta-learning
  increment larger than the eval-seed spread (pass@1 2-4).
- **Stretching the alphas (user's idea) helps a little and early.** Stretched beats held in token accuracy at every
  length (+.008 / +.007 / +.003) and fits the demos further, with equal task counts at 300 and 1000 (one task either way
  at pass@2/5). Its value shrinks as the loop lengthens, as expected: both schedules converge once the demos are fitted.
- 12/30 at pass@1 (40%) is the best held-out result in the project; ttt_mini from the plain checkpoint was 4/30.
- One seed per cell outside the 120-step plain/one-step rows; seeds for the 300-step cells would firm up the 5-step vs
  one-step gap. The 10-step run (restarted 20:36) tests whether more trained steps widen it.

## Round 12b (2026-10-23 23:10): the 10-step checkpoint, stretched vs held

| steps | 10-step, stretched | 10-step, held | 5-step, stretched | 5-step, held | one-step | plain, hand-set |
|---|---|---|---|---|---|---|
| 10 | | 1 / 1 / 2, .782 | | 1 / 1 / 1, .753 | 1 / 1 / 1, .738 | 1 / 1 / 1, .744 |
| 120 | 7 / 7 / 10, .845 | 4 / 5 / 6, .833 | 7 / 8 / 9, .845 | 7 / 7 / 8, .837 | 6.0 / 7.3 / 8.7, .836 | 4.0 / 6.7 / 8.0, .827 |
| 300 | 10 / 11 / 13, .870 | 9 / 10 / 13, .862 | 10 / 12 / 13, .872 | 10 / 12 / 13, .865 | 8 / 9 / 9, .862 | 7 / 9 / 10, .853 |
| 1000 | 12 / 13 / 14, .888 | 10 / 11 / 14, .884 | 12 / 12 / 14, .888 | 12 / 13 / 15, .885 | 10 / 13 / 15, .887 | 10 / 13 / 15, .879 |

- **Stretching matters more the more steps were trained.** On the 10-step checkpoint the held schedule spends almost every
  step at the smallest alpha and falls behind (4/5/6 at 120); stretched it recovers fully (7/7/10 at 120, 12/13/14 at
  1000, demos 0.49 / 0.40 / 0.35). Stretched beats held on it at every length, by 3 / 1 / 2 at pass@1.
- **5 and 10 trained steps end up level once stretched** (.845 / .870-.872 / .888 at 120 / 300 / 1000; task counts
  within one). Ten trained steps buy the best short loop (.782 at 10 steps) and nothing at 120+.
- Stretched multi-step meta-trained checkpoints at 1000 steps: 12/30 pass@1 (neither solves the duplicated task, so also
  12/29 on the uncontaminated ones), against 10/13/15 for the plain checkpoint under the same
  rule. Single seed per cell.

## Round 13 (2026-10-23 23:31): the plain baseline retrained on this branch

`varc_ho3`: varc_ho2's recipe (resolved configs diffed) on this session's build, 3.4 min.

| procedure | varc_ho2 (l-reinit-ttt, old build) | varc_ho3 (this branch, this build) |
|---|---|---|
| ttt_mini | 4 / 4 / 4, .811 (eval seeds: 2-4 / 4-5 / 4-6, .805-.812) | 3 / 4 / 5, .793 |
| normalised step, hand-set alpha, 120 steps | 4.0 / 6.7 / 8.0, .827 (3 seeds: .824-.829) | 2 / 5 / 5, .811 (demos -> 0.61, vs 0.57) |

The code reproduces the baseline's behaviour in kind (test-time training lifts it from no-adaptation, the normalised
120-step loop beats ttt_mini in token accuracy) but the retrained checkpoint is weaker: token accuracy .015-.018 below
varc_ho2 under both procedures, outside the eval-seed spread. Neither solves the duplicated task. A weaker model trained
on data with no more held-out leakage is not a contamination signal (the content check above found none beyond the
MiniARC duplicate); the gap is training-run variance (augmentation draws, non-deterministic kernels), which was never
measured. Not yet separated: this branch's code on the *old* build (code replication alone) and a second training seed
on this build (training variance).

## Round 14 (2026-10-23 23:50): what round 13's gap was

| plain checkpoint | build | training seed | ttt_mini | normalised 120 steps |
|---|---|---|---|---|
| varc_ho2 (l-reinit-ttt code) | old (treerm-candidates) | 0 | 4 / 4 / 4, .811 | 4.0 / 6.7 / 8.0, .827 (3 eval seeds) |
| varc_ho4 (this branch) | old | 0 | 4 / 4 / 4, .812 | 5 / 6 / 8, .821 |
| varc_ho3 (this branch) | this session's | 0 | 3 / 4 / 5, .793 | 2 / 5 / 5, .811 |
| varc_ho5 (this branch) | this session's | 1 | 4 / 4 / 5, .795 | 4 / 6 / 8, .817 |

- **The code replicates.** This branch's code on the original build reproduces the baseline: ttt_mini 4/4/4 at .812
  against .811, and the normalised loop within the eval-seed spread (.821 vs .824-.829; demo fit 0.58 vs 0.57).
- **The gap is the build's augmentation draw.** Both training seeds on this session's build sit at .793-.795 under
  ttt_mini and .811-.817 under the normalised loop: ~.017 and ~.01 below the old build, and consistent across seeds.
  The same 119 tasks with different random dihedral / colour copies give a slightly weaker baseline.
- **Checkpoint-to-checkpoint spread of the plain baseline** under the normalised 120-step loop: pass@1 2-5, token
  accuracy .811-.829 over four checkpoints. The meta-fine-tuned checkpoints (one-step 6.0 / 5-step stretched 7 / 10-step
  stretched 7 at pass@1, .836-.845) are above every plain checkpoint in token accuracy and at or above the best one in
  pass@1, but their pass@1 lead over the original baseline (+2 to +3) is about the size of this spread. The meta
  checkpoints all descend from varc_ho2, so the like-for-like comparison remains against varc_ho2.
- The from-scratch Meta-SGD run (this build, 1/1/2, .775) is still below both plain checkpoints on the same build
  (2/5/5 .811, 4/6/8 .817): round 11c's conclusion holds with the matched baseline.

## Learned trajectory targets: diagnostic (2026-10-26 00:06)

New option `meta_sgd.query_loss=trajectory` (user's idea after DIS, arXiv 2511.16886, whose per-step targets are a fixed
corruption schedule of the answer): with the adapted weights frozen, the query rows' latent states after each of the 8
segments are optimised as leaves (Adam on CE(decode(s~_8)) + lambda * mean_t ||s~_{t+1} - F(s~_t)||^2, from the model's own
rollout, every H-cycle carrying gradient), and the outer loss matches the adapted model's free-running per-segment states
to them, instead of the final CE. Under it the output and halt heads get no outer gradient.

Diagnostic (`scratchpad/traj_diag.py`): trm_ho3 frozen, 601 query rows of 10 training batches in 16-row episodes, Adam lr
1e-2. Delta_t = RMS distance of the target from the rollout at segment t; R = RMS transition residual; the rollout's own
per-segment step is .716.

| lambda | K | CE before -> after | exact match before -> after | R | Delta_1 | Delta_7 | Delta_8 | s / episode |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 5 | .630 -> .553 | 4.3% -> 5.2% | .017 | .0004 | .011 | .049 | 4.9 |
| 0.1 | 20 | .630 -> .369 | 4.3% -> 9.4% | .058 | .0025 | .046 | .183 | 18.1 |
| 1 | 5 | .630 -> .553 | 4.3% -> 5.2% | .018 | .0004 | .011 | .049 | 4.4 |
| 1 | 20 | .630 -> .369 | 4.3% -> 9.4% | .053 | .0032 | .045 | .170 | 16.7 |
| 10 | 5 | .630 -> .553 | 4.3% -> 5.2% | .016 | .0004 | .011 | .043 | 5.4 |
| 10 | 20 | .630 -> .374 | 4.3% -> 9.4% | .036 | .0034 | .042 | .127 | 18.7 |

- **Accuracy:** 20 iterations double the exact-match rate of decode(s~_8) (4.3% -> 9.4%) and lower its CE by 40%.
- **Consistency:** the residuals stay at 2-8% of the rollout's own step; lambda mostly trades residual for final-state
  movement (lambda 10: R .036, Delta_8 .127 vs lambda 0.1: .058, .183) at nearly equal CE (Adam normalises the step).
- **Where the change sits:** concentrated at the end, as the multiple-shooting analysis predicted. Delta_7 / Delta_8 is
  .25-.33, but Delta_1 / Delta_8 is only .014-.027 even at K = 20: in 20 iterations the signal reaches the early
  segments only weakly, so the targets are mostly a correction of the last two or three segments.
- Cost: ~0.9 s per iteration per 16-row episode; at K = 20 ~+63 s per meta-step if recomputed every step.

## Round 24 (2026-10-25 11:18): embedding first, then weights, inside the best recipes

1000 normalised steps; `meta_sgd_embed_first=0.5` = 500 steps on the puzzle-embedding rows only, then 500 on the weights
only with the rows frozen; everything else as the single-phase run it is paired with:

| configuration | single phase | embedding first | single phase, clean 29 | embedding first, clean 29 |
|---|---|---|---|---|
| VARC plain, hand-set alpha | 11 / 13 / 13, .878 | 9 / 11 / 13, .858 | 10 / 12 / 12 | 9 / 11 / 13 |
| VARC meta-fine-tuned, stretched | 12 / 13 / 14, .885 | 9 / 12 / 12, .876 | 12 / 13 / 14 | 9 / 12 / 12 |
| TRM plain, ACT on, hand-set alpha | 12 / 14 / 14, .835 | 9 / 11 / 11, .817 | 11 / 13 / 13 | 9 / 11 / 11 |
| TRM meta-fine-tuned, ACT off, segment | 11 / 13 / 15, .829 | 9 / 10 / 11, .806 | 11 / 13 / 15 | 8 / 9 / 10 |

**The two-phase loop is worse in all four configurations**: -2 to -3 at pass@1 and -.009 to -.023 token accuracy, also on
the 29 tasks without the MiniARC duplicate. The loops were still improving at 1000 steps, so halving the steps that
reach the weights (500 instead of 1000) is the likely cost; the embedding-only phase does not make up for it. A fairer
test at matched weight steps (e.g. 500 rows-only + 1000 weights-only) was not run. Together with round 23, the idea of
arXiv 2609.21181 does not carry over to these recipes at the budgets tried.

## Round 23 (2026-10-25 09:10): arXiv 2609.21181's TTT protocols, at ttt_mini's length

| plain checkpoint (this branch) | Full-TTT (= ttt_mini) | Embed-TTT (embedding only) | Embed-Full-TTT (embedding, then backbone) |
|---|---|---|---|
| VARC varc_ho3 | 3 / 4 / 5, .793 | 2 / 3 / 3, .760 | 3 / 3 / 3, .779 |
| TRM trm_ho3 | 1 / 2 / 3, .779 | 1 / 1 / 1, .751 | 1 / 1 / 1, .755 |

At ttt_mini's ~120 steps the paper's protocols do not help: the embedding alone already recovers most of Full-TTT at
pass@1 (2 of 3 on VARC, 1 of 1 on TRM) with <0.01% of the parameters, as the paper reports, but the two-step version
ends at or below Full-TTT, with its backbone phase only ~60 updates at the decayed half of a single schedule. Round 24
puts the two phases into the 1000-step normalised loop instead.

## Round 22 (2026-10-25 08:52): halt-and-wait

| TRM (this branch), normalised segment steps, 1000 steps | plain trm_ho3, hand-set alpha | meta-fine-tuned to match, segment schedule |
|---|---|---|
| ACT off (rows always run 8 segments) | 8 / 9 / 12, .815 | 11 / 13 / 15, .829 (round 18) |
| ACT on, halted rows restart at once | **12 / 14 / 14, .835** | 11 / 11 / 13, .832 (round 21) |
| ACT on, halt-and-wait | 5 / 6 / 7, .816 | 9 / 11 / 13, .816 (round 22; 3 h 9 min) |

- **Halt-and-wait hurts, the plain checkpoint most** (12/14/14 -> 5/6/7), i.e. it is worse than turning ACT off. With
  halting on, letting a halted row restart at once is clearly the best of the three adaptation loops for the plain TRM.
- With halt-and-wait the meta-fine-tuned checkpoint gains the most over its plain pair of any ACT setting (+4 / +5 / +6),
  but the absolute level stays below plain with immediate restarts.
- Not measured: how often rows halt during the adaptation loop. The 6% early halts were measured at the starting weights
  on training tasks; as the loop fits a held-out task's demos the halt head plausibly fires earlier and more often, so
  halt-and-wait would idle many rows for most of each trajectory, while immediate restarts turn the same halts into extra
  early-segment training. That would explain the ordering, but it is a hypothesis.
- Across all six TRM cells, the best remains the plain checkpoint with ACT on and immediate restarts. On TRM, Meta-SGD
  fine-tuning has not beaten it in any configuration tried.

## Round 21 (2026-10-25 04:39): TRM with ACT on throughout

| TRM (this branch), normalised segment steps, 1000 steps, ACT on at test | pass@1/2/5 | token acc | clean 29 tasks |
|---|---|---|---|
| plain trm_ho3, hand-set alpha | 12 / 14 / 14 | .835 | 11 / 13 / 13 |
| meta-fine-tuned with ACT off (round 18), segment schedule | 9 / 11 / 12 | .825 | 8 / 10 / 11 |
| meta-fine-tuned with ACT on (round 21, 3 h 19 min), segment schedule | 11 / 11 / 13 | .832 | 10 / 10 / 12 |

Meta-fine-tuning with ACT on beats meta-fine-tuning with ACT off when the test loop has ACT on (+2 / 0 / +1), but it does
not beat the plain checkpoint under the same rule (-1 / -3 / -1). Halted rows still restart at once here, so its
per-segment alphas see mixed positions; round 22 (halt-and-wait) removes that.

## Round 20 (2026-10-24 23:56): the meta-fine-tuned TRM with ACT on at test time

| TRM (this branch), normalised segment steps, 1000 steps | ACT off | ACT on |
|---|---|---|
| plain trm_ho3, hand-set alpha | 8 / 9 / 12, .815 | 12 / 14 / 14, .835 |
| meta-fine-tuned msgd18_trm (trained with ACT off), learned alpha, segment schedule | 11 / 13 / 15, .829 | 9 / 11 / 12, .825 |

On the 29 tasks without the MiniARC duplicate (which both ACT-on runs solve and both ACT-off runs do not): plain 8 / 9 / 12
(off) and 11 / 13 / 13 (on); meta-fine-tuned 11 / 13 / 15 (off) and 8 / 10 / 11 (on).

- **The meta-fine-tuned TRM wants the ACT setting it was trained with.** With ACT on at test it drops 2 / 2 / 3 tasks
  (rows leave lockstep, so the segment-indexed alphas misalign, and the trajectories differ from meta-training's).
- **Best TRM per checkpoint pair: plain with ACT on (12 / 14 / 14, .835) vs meta-fine-tuned with ACT off (11 / 13 / 15,
  .829).** On the clean 29 tasks: 11 / 13 / 13 vs 11 / 13 / 15, i.e. tied at pass@1 and pass@2, +2 for meta at pass@5.
  So on TRM, meta-fine-tuning with trajectory-indexed alphas recovers what turning ACT off cost, and no more.
- Not run: meta-fine-tuning with ACT on (the alphas would be trained on desynchronised rows; ~3.5 h).

## Round 19 (2026-10-24 21:32): ACT off was the handicap

| plain TRM, normalised segment steps, hand-set alpha, 1000 steps | pass@1/2/5 | token acc |
|---|---|---|
| trm_ho2 (other build), ACT on | 14 / 16 / 17 | .825 |
| trm_ho3 (this branch and build), ACT on | 12 / 14 / 14 | .835 |
| trm_ho3, ACT off | 8 / 9 / 12 | .815 |
| trm_ho3 meta-fine-tuned with ACT off (round 18), segment schedule | 11 / 13 / 15 | .829 |

- **Turning ACT halting off costs the plain TRM ~4 tasks at 1000 steps** (12/14/14 -> 8/9/12) on the same checkpoint. The
  build accounts for less (trm_ho2 vs trm_ho3 with ACT on: 14/16/17 vs 12/14/14, token accuracy .825 vs .835).
- So round 18's meta-learning gain (+3 / +4 / +3 at 1000 steps) was measured against a baseline handicapped by ACT-off;
  against the plain checkpoint with ACT on, the ACT-off meta-fine-tuned TRM is about level (11/13/15 vs 12/14/14, .829 vs
  .835). Why halting helps the long loop is not established: only ~6% of rows halt before the cap, but exploration noise
  in training mode (which the test-time loop runs in) also randomises trajectory lengths when ACT is on, desynchronising
  and restarting rows more often.
- Not run: the meta-fine-tuned checkpoint evaluated with ACT on (its rows would no longer be in lockstep, so the segment
  schedule would be approximate), or meta-fine-tuning with ACT on.

## Round 18 (2026-10-24 20:43): TRM, one trajectory of segment steps, ACT off, segment-indexed alphas

All from trm_ho3 (plain TRM retrained on this branch and build), ACT halting off throughout:

| steps | meta-fine-tuned (8 segment steps, alpha init 1e-3), segment schedule | plain trm_ho3, hand-set alpha |
|---|---|---|
| ttt_mini | | 1 / 2 / 3, .779 |
| 120 | 6 / 6 / 7, .798 | 5 / 7 / 8, .799 |
| 300 | 8 / 9 / 9, .817 | 7 / 10 / 11, .822 |
| 1000 | **11 / 13 / 15, .829** | 8 / 9 / 12, .815 |

- Meta-fine-tuning (3 h 20 min first order, ~25 s / meta-step) now helps TRM at long loops: +3 / +4 / +3 tasks and
  +.014 token accuracy at 1000 steps, where the plain checkpoint stalls after 300; level at 120 / 300 (within a task).
  Round 16's 5-step version, trained on a fraction of one trajectory and evaluated on ACT-on rows, was behind its plain
  checkpoint at every length.
- Learned alphas by segment position (weights, geometric mean): 0.00033 0.00024 0.00031 0.0002 0.0004 0.00031 0.00031 0.00018; rows: 0.022 0.026 0.023 0.014 0.02 0.014 0.012 0.01.
- Against round 16's TRM (other build, ACT on: plain 14/16/17 at 1000), this branch's plain TRM with ACT off is far
  lower (8/9/12); round 19 shows that is mostly ACT-off, not the build.

## Round 17 (2026-10-24 14:15): very large alpha inits, VARC

All from varc_ho3 (this branch and build), 5-step normalised first-order recipe, stretched schedule:

| alpha init (weights / rows) | 120 steps | 300 steps | 1000 steps | learned alpha (weights, per step) | meta-training query loss |
|---|---|---|---|---|---|
| plain varc_ho3, hand-set 1e-3 / 3e-2 | 2 / 5 / 5, .811 | 8 / 9 / 10, .847 | 11 / 13 / 13, .878 | | |
| 1e-3 / 3e-2 | 6 / 7 / 7, .844 | 8 / 9 / 11, .861 | 12 / 13 / 14, .885 | 4e-4 .. 6e-4 .. 3e-4 | ~0.8-1.2 |
| 0.1 / 0.1 | 0 / 0 / 0, .087 | 0 / 0 / 0, .000 | 0 / 0 / 0, .000 | 0.10 -> 0.08 | ~2.0-2.4 |
| 1 / 1 | 0 / 0 / 0, .000 | 0 / 0 / 0, .000 | 0 / 0 / 0, .000 | 1.0 -> 0.83 | ~12-50 |

- **Large inits destroy the model at test.** A normalised step moves every element of a tensor by about alpha, so at
  0.1 or 1 each step rewrites the weights (their typical magnitude is ~0.05). The demos are fitted perfectly (loss 1.52 ->
  0.26 at init 0.1, -> 0 at init 1 in 120 steps) while the test outputs become all wrong (token accuracy ~0).
- **Meta-training barely pulls them back.** Alpha moved only to 0.08 and 0.83 in 1000 meta-steps; the meta-loss stayed 2x
  (0.1) and 20x (1) the control's. The control init (1e-3) is the one round 18 uses (`pick_alpha_init.py`).
- On this branch's checkpoint the meta-learning gain over plain is clearest at 120 steps (6 vs 2 at pass@1, .844 vs
  .811) and small at 1000 (12/13/14 vs 11/13/13, .885 vs .878), the same pattern as on varc_ho2.

## Rounds 17-18 (queued 2026-10-24 11:25; user)

No foreign checkpoints from here on (user): every run starts from a checkpoint trained on this branch and this build.
VARC uses `varc_ho3` (round 13); TRM is retrained as `trm_ho3` with trm_ho2's recipe (resolved configs diffed: equal).

- **Round 17, very large alpha inits on VARC** (`msgd17_varc.sh`): the 5-step normalised first-order recipe from
  varc_ho3 with alpha initialised at 1e-3 / 3e-2 (control), 0.1 and 1 (weights and rows alike), evaluated stretched at
  120 / 300 / 1000; plain varc_ho3 under the same rule at 300 / 1000. ~2.5 h, ETA ~14:00.
- **Round 18, TRM with segment-indexed alphas and ACT halting off** (`msgd18_trm.sh`, chained): trm_ho3 retrained; meta-
  fine-tune with 8 segment inner steps (one whole trajectory), `+arch.disable_act_halt=true` (rows run all 8 segments in
  lockstep, in meta-training and at test), alpha per tensor and per step, MSL, first order, normalised; one meta run, at
  the alpha init that did best on VARC in round 17 (user; `pick_alpha_init.py`: highest pass@1 at 1000 stretched steps,
  ties by token accuracy). Evaluated with the new `meta_sgd_schedule=segment` (step n uses the alpha of segment
  (n mod 8) + 1, trained trajectories stretched) at 120 / 300 / 1000. Controls on trm_ho3: ttt_mini, and the segment-step
  rule with ACT off and the hand-set alphas. ~4 h meta run, ETA ~19:30.
- `evaluate_ttt.py` nested train_overrides (e.g. `+train_overrides.arch.disable_act_halt=true`) now set single fields; they
  used to replace the whole block they named. Past runs are unaffected (their only nested overrides were meta_sgd blocks
  on plain checkpoints, where replacing defaults equals merging).

## Round 16 (2026-10-24 07:47): TRM

Plain TRM `trm_ho2` against the same checkpoint meta-fine-tuned on the VARC headline recipe (`msgd16_trm_ms5`: inner step =
one ACT segment continuing the carry, query gradient through the final segment only, first order throughout after the
second-order phase ran out of memory; 1 h 55 min):

| test-time procedure | plain TRM | meta-fine-tuned TRM | plain VARC | best meta-trained VARC |
|---|---|---|---|---|
| ttt_mini | 1 / 2 / 3, .776 | 1 / 3 / 3, .766 | 4 / 4 / 4, .811 | 3 / 3 / 3, .804 |
| normalised, 120 steps | 4 / 7 / 9, .808 (hand-set alpha) | 4 / 5 / 5, .801 (stretched); 3 / 4 / 5, .795 (held) | 4.0 / 6.7 / 8.0, .827 | 7 / 8 / 9, .845 |
| normalised, 300 steps | 10 / 12 / 13, .812 | 8 / 10 / 12, .809 | 7 / 9 / 10, .853 | 10 / 12 / 13, .872 |
| normalised, 1000 steps | **14 / 16 / 17, .825** | 11 / 12 / 13, .836 | 10 / 13 / 15, .879 | 12 / 12 / 14, .888 |

- **The normalised segment-step rule is the big result on TRM**: with no meta-training it takes the plain TRM from 1/2/3
  (ttt_mini) to 4/7/9 at the same compute and to **14/16/17 at 1000 steps, the best held-out result in the project**
  (VARC's best: 12/12/14). The duplicated MiniARC task is among them (1/1/1 at 300 and 1000 steps), so on the 29 clean tasks it is 13/15/16
  (meta-fine-tuned TRM 11/12/13 and best VARC 12/12/14 do not solve it, so theirs are unchanged). Each
  segment step costs about what a ttt_mini step does (120 steps: 5.7 min vs 5.9 for 30 tasks; 1000 steps: 43 min).
- **Meta-fine-tuning does not help TRM.** It is behind the plain checkpoint under the same rule at 120 / 300 / 1000 steps
  (by 0-4, 2, 3 tasks at pass@1), though its token accuracy is higher at 1000 (.836 vs .825). It learned smaller step sizes
  than the hand-set ones (weights ~3e-4 vs 1e-3, rows .015-.028 vs .03) for a 5-segment loop, i.e. less than one
  8-segment trajectory, so the long loops adapt more slowly. Not tried: more inner steps (a full trajectory or more), a
  larger alpha init, second order with smaller episodes.
- TRM and VARC differ in kind: TRM gets more grids exactly right with lower token accuracy (14 tasks at .825 vs 12 at
  .888). Single seeds throughout; the 1000-step cells are one evaluation each.

## TRM: ACT halting inside the inner loop (2026-10-24 08:40)

Question (user): does ACT halting desynchronise the rows, so that a per-step alpha must cover several trajectory positions?
Measured (`scratchpad/halt_stats.py`, `halt_stats16.py`): 10 batches of 16-copy training episodes, 1959 demo rows, training
mode as in the meta step, segments from a fresh carry, weights fixed.

| first halt | plain TRM | meta-fine-tuned TRM |
|---|---|---|
| within the 5 meta-training segments | 6.2% (mostly segment 2) | 0.0% |
| at the 8-segment cap | 93.8% | 100.0% |

The halt head almost never fires (never, after meta-fine-tuning), so rows stay in lockstep: in meta-training alpha k was
"segment k of the first trajectory", and in a test loop inner step n has every row at segment (n mod 8) + 1, with all rows
halting and restarting together. Mixing across rows is negligible. The mixing that does happen is over time, through how
the alphas are indexed: with 5 trained alphas stretched over 1000 steps, alpha 1 covers 200 steps, i.e. 25 trajectories
through all 8 segment positions, and alphas trained only on segments 1-5 of one trajectory are used for 125 of them.

Proposed (not run): alphas indexed by segment position. Meta-train with 8 inner steps (one trajectory: 8 alphas, one per
segment) or 16 (two trajectories: 16 alphas). At test, apply them by segment: step n uses the alpha of segment
(n mod 8) + 1 and, with several trained trajectories, of trained trajectory t * K // T for test trajectory t of T (K
trained), i.e. exact per-segment, stretched across trajectories. Cost ~3-6 h first order, plus the evaluation mapping.
(Comparing against the plain checkpoint does not bear on this: it learns no alpha, so it says only that one constant step
size suffices at test, not whether learning alpha is hampered by mixed positions.)

## Round 15 (2026-10-24 02:23): second order off, VARC

Meta-fine-tuning entirely first order, otherwise identical:

| recipe, eval | first order | second order | plain, same rule |
|---|---|---|---|
| one step, 120 | 5 / 6 / 8, .832 | 6.0 / 7.3 / 8.7, .836 (3 seeds: 5-7 / 7-8 / 8-9) | 4.0 / 6.7 / 8.0, .827 |
| one step, 300 | 6 / 10 / 11, .863 | 8 / 9 / 9, .862 | 7 / 9 / 10, .853 |
| one step, 1000 | 10 / 11 / 13, .884 | 10 / 13 / 15, .887 | 10 / 13 / 15, .879 |
| 5 steps, stretched 120 | 7 / 7 / 8, .841 | 7 / 8 / 9, .845 | |
| 5 steps, stretched 300 | 8 / 9 / 12, .872 | 10 / 12 / 13, .872 | |
| 5 steps, stretched 1000 | 12 / 12 / 14, .890 | 12 / 12 / 14, .888 | |
| 5 steps, held 120 | 5 / 7 / 8, 0.837 | 7 / 7 / 8, .837 | |
| meta-training time | 7 / 22 min | 12 / 54 min | |

**The second-order term buys nothing measurable here.** Token accuracy is within .004 of second order at every length
(and above plain everywhere); task counts differ by 0-3 in both directions, the size of the eval-seed spread; at 1000
stretched steps the two 5-step checkpoints are identical in task counts. First order halves to quarters the meta-training
time. Consistent with the round-7+ picture: the gain comes from learning the step sizes (whose gradient is first-order
either way) and from the meta-fine-tuned theta's starting point, not from curvature through the inner loop.

## Rounds 15-16 (queued 2026-10-24 01:10; user)

- **Round 15, second order off on VARC** (`msgd15_varc_fo.sh`, started 00:56): the one-step normalised recipe
  (`msgd15_n1_fo`, against `msgd7_norm`) and the 5-step one (`msgd15_ms5_fo`, against `msgd11_ms5`), meta-fine-tuned
  entirely first order. First order still trains alpha (its gradient, -dL_query/dtheta' x g, has no second derivative);
  it drops the curvature term in theta's. Evaluated at 120 / 300 / 1000 steps (5-step: stretched, plus held at 120).
  ETA ~02:30.
- **Round 16, Meta-SGD on TRM** (`msgd16_trm.sh`, chained): new options `meta_sgd.segment_inner_steps` (an inner step is
  one ACT segment continuing the previous step's carry, as train_batch and ttt_mini do; before, every inner step ran
  every row's full trajectory from a fresh carry) and `meta_sgd.last_segment_grad` (the query trajectory backpropagates
  only through each row's final segment). Both are no-ops on one-segment models (tested). Meta-fine-tune of the plain TRM
  `trm_ho2` (previous loop: 8 segments, H 3 / L 4, hidden 128, 4000 x 128; this branch rebuilds it with
  `arch.activation_function=conv_swiglu arch.puzzle_emb_len=1`, configs diffed) on the VARC headline recipe, round 3's
  budget (1000 x 256). Smoke: ~10 s / meta-step first order, ~22 s second order with the GPU shared, against 2-5 s for
  VARC, so ~4-5 h. Controls: trm_ho2 under ttt_mini and under the same segment-step normalised rule with the hand-set
  alphas (120 / 300 / 1000). ETA ~09:00. Single seeds.

## Contamination check (2026-10-23 22:00)

`scratchpad/contamination.py` searches every training-split row for each held-out pair, up to the 8 dihedral transforms
and any colour relabelling (verified on a training row under a random dihedral + colour permutation). Both builds give the
same answer: **one held-out task is duplicated in the training set by MiniARC itself.** The training task
`find_the_total_shape_revealed_by_gray_pixels_l6bl8wmnt49h58bhbzm` contains the held-out task
`reveal_the_original_gray_pixel_l6blhmgkypxwgqpqsv`'s test pair and one of its demos; the other 29 held-out tasks' test
pairs and 97 of 99 demos appear nowhere in training. This is a dataset duplicate, not a builder or loader bug, and it
affects the plain and meta-trained models equally.

That task is almost never solved: 0/0/0 under every procedure on the plain checkpoint (including no adaptation, so the
model did not memorise the answer), 1/1/1 only for the one-step meta checkpoint at 1000 steps, 0/1/1 for the stretched
5-step loop at 300. Without it (29 tasks) the headline numbers are: plain 1000 steps 10/13/15; one-step meta 1000 steps
9/12/14; 5-step meta stretched 300 10/11/12 and 1000 12/12/14; 5-step held 1000 12/13/15. No conclusion changes. A clean
build would drop either task from the pool before the held-out draw.

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
| 11 (`msgd11_varc.sh`, launched 14:35, process started 15:33; user's next steps 1-3) | msgd7_norm, varc_ho2 | normalised loops for 300 and 1000 steps (evaluations only) | ms8n_s300/1000, ms4vnorm_s300/1000 | ~16:25 |
| 11 | msgd11_ms5 | meta-trained with 5 normalised inner steps (alpha per tensor and step, MSL 30%, first order 300), from the plain checkpoint | ms9m5_s10/30/120, _adam | ~17:20 |
| 11 | msgd11_scratch | normalised recipe from scratch, 4000 x batch 128, lr 1e-3 (plain control = varc_ho2) | ms9sc_* | ~18:10 (step time unmeasured) |
| 11 | msgd11_ms10 | as msgd11_ms5 with 10 inner steps | ms9m10_* | ~22:45 (killed at ~step 450 by a laptop restart at 18:48; restarted 20:36 after round 12's 5-step evals) |
| 12 (`msgd12_eval.sh`, chained; user) | msgd11_ms5, msgd11_ms10 | the per-step alphas as a learned schedule shape: `meta_sgd_schedule=stretch` (k-th trained alpha for the k-th fraction of the loop) vs hold, at 120 / 300 / 1000 steps | ms9m5_st*, ms9m5_s300/1000, ms9m10_st*, ms9m10_s300/1000 | 5-step part done 20:36, 10-step part ~23:45 |

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
- **2026-10-23 14:35. Round 11 queued** (user: next steps 1-3): long loops, 5/10-step normalised meta-training, from
  scratch. ETAs in the queued-runs table.
- **2026-10-23 15:39. Round 11, learned loop 300 steps:** 8/9/9 (p@100 13), tok .862, demos 1.53 -> 0.43 (5.6 min for 30
  tasks). Still climbing past 120 (6/7/9, .836). The queue's first job started only at 15:33, 58 min after launch
  (the launched process itself started at 15:33, i.e. the launch was held, probably for an approval; ETAs moved +1 h); plain 300-step control running.
- **2026-10-23 15:44. Round 11, plain checkpoint, normalised rule, 300 steps:** 7/9/10 (p@100 11), tok .853, demos 1.74
  -> 0.46. Against the meta-trained 8/9/9 (.862): the gap at 300 steps is +1 pass@1, -1 pass@5, +.009 tok; the plain
  checkpoint keeps improving as fast (120 -> 300: +3 pass@1 both). 1000-step loops running.
- **2026-10-23 16:04. Round 11, learned loop 1000 steps** (20 min for 30 tasks): 10/13/15 (p@100 16), tok .887, demos
  1.53 -> 0.36. Half the held-out tasks solved at pass@5; still climbing (120 / 300 / 1000: 6/7/9 -> 8/9/9 -> 10/13/15).
  Plain checkpoint's 1000-step control running.
- **2026-10-23 16:26. Round 11, plain checkpoint, normalised rule, 1000 steps:** 10/13/15 (p@100 16), tok .879, demos
  1.74 -> 0.37: the same task counts as the meta-trained checkpoint, .008 lower token accuracy. Section "Round 11a".
  5-step normalised meta-training (`msgd11_ms5`) started.
- **2026-10-23 16:50. Round 12 queued** (user's idea): the per-step alphas of the 5/10-step checkpoints stretched over
  longer loops (`meta_sgd_schedule=stretch`, new evaluate_ttt option, tested) against holding the last one.
- **2026-10-23 17:28. Round 11, 5-step meta-training done:** hold-schedule loops 1/1/1 (.753), 2/2/3 (.797), 7/7/8
  (.837); ttt_mini 3/3/3 (0.804); learned decaying per-step alphas. Section "Round 11b". From-scratch run started.
- **2026-10-23 18:08. Round 11, from-scratch run done:** loops 1/3/3 (.746), 1/2/2 (.751), 1/1/2 (.775); ttt_mini
  2/3/3 (0.772). Below plain-from-scratch under the same rule. Section "Round 11c". 10-step meta-training started.
- **2026-10-23 19:40. Laptop restart at ~18:48 killed the 10-step meta-training** (`msgd11_ms10`, ~step 450, no
  intermediate checkpoint; log kept as `train_killed_by_restart.log`). Relaunched in one chain: round 12's 5-step-checkpoint
  evaluations first (stretch vs hold), then the 10-step run from step 0, then round 12's 10-step evaluations.
- **2026-10-23 19:45. Round 12, 5-step checkpoint, stretched schedule, 120 steps:** 7/8/9, tok .845, demos 1.58 -> 0.49,
  against the same checkpoint with the last alpha held, 7/7/8 (.837, demos -> 0.54). Best 120-step token accuracy so far
  (one-step normalised checkpoint .836 over 3 seeds, plain .827). 300 and 1000 steps running.
- **2026-10-23 19:51. Round 12, 5-step checkpoint, stretched, 300 steps:** 10/12/13 (p@100 14), tok .872, demos -> 0.41.
  At 300 steps the one-step checkpoint gets 8/9/9 (.862) and the plain one 7/9/10 (.853): the stretched 5-step loop at
  300 steps matches the others at 1000 (10/13/15, .879-.887) on pass@1. 1000 steps running.
- **2026-10-23 20:09. Round 12, 5-step checkpoint, stretched, 1000 steps:** 12/12/14 (p@100 16), tok .888, demos -> 0.35.
  Against 10/13/15 for both the one-step (.887) and plain (.879) checkpoints at 1000 steps: +2 pass@1, -1 pass@2 and
  pass@5. The stretched schedule's lead is largest at 300 steps and mostly closes by 1000. Held-schedule 300/1000 running.
- **2026-10-23 20:15. Round 12, 5-step checkpoint, held schedule, 300 steps:** 10/12/13 (p@100 14), tok .865, demos ->
  0.45: the same task counts as stretched (10/12/13), .007 lower token accuracy and a shallower demo fit (0.45 vs 0.41).
  So at 300 steps the gain over the one-step (8/9/9) and plain (7/9/10) checkpoints is mostly the 5-step meta-training;
  stretching adds the token-accuracy margin. Held 1000 steps running.
- **2026-10-23 20:36. Round 12, 5-step checkpoint done:** held 1000 steps 12/13/15 (.885). Section "Round 12a". 10-step
  meta-training started (ETA ~22:45 with its evaluations), round 12 on it after.
- **2026-10-23 21:55. Provenance of the plain checkpoint `varc_ho2`** (the start of every fine-tuning run and the plain
  control throughout): trained on 2026-10-20 by the previous loop (`l-reinit-ttt`, commit e4c1284), 4000 x batch 128,
  adam_atan2 lr 1e-3, wd 1.0, EMA, seed 0, on a `miniarc5-aug-31-ho30` build in the `treerm-candidates` worktree.
  Compared with this session's build: same held-out list, same 30 test puzzles (so every held-out evaluation is
  unaffected), same 119 training tasks and array shapes, but **different augmentation draws** (identifiers, inputs and
  labels differ; probably PR #68's builder changes moved the random stream). So fine-tuning runs saw new augmentations of
  the same tasks, the non-fresh-row controls loaded embedding rows that index other augmentations (equally for meta and
  control), and the from-scratch comparison pits a meta run on this build against a plain baseline on the old one.
  Verification of that build (user asked how the held-out set is known to match): its files were all written 2026-10-19
  19:37, before the checkpoint's run (2026-10-20 23:18-23:22), and not since; its held-out list is byte-identical to ours;
  its train split covers 119 base tasks with zero overlap with the 30 held-out ones, and its test split is exactly them.
  That the run read this path rests on the data_path in its all_config.yaml (written at launch); the re-evaluation in this
  session reproduced the previous loop's 4/4/4 (.811) exactly.

- **2026-10-23 22:00. Contamination check:** one held-out task duplicated by MiniARC in a training task; rarely solved,
  conclusions unchanged without it (section above). Round 13 queued (user): retrain the plain baseline on this branch's
  build with varc_ho2's exact recipe (resolved configs diffed), then ttt_mini and the 120-step normalised loop on it.
- **2026-10-23 22:02. Round 11 done: 10-step normalised meta-training** (1 h 22 min, ~5-7 s/step second-order).
  Held-schedule loops 1/1/2 (.782), 2/3/4 (.803), 4/5/6 (.833) at 10/30/120 steps, demos 1.68 -> 0.90 / 0.72 / 0.54;
  ttt_mini 3/3/3 (0.806). Best 10-step result of any checkpoint (.782 vs the 5-step's .753), behind the 5-step one
  at 120 (7/7/8, .837). Per-step alphas decay again (rows .053 -> .009, weights 8e-4 -> 2e-4). Round 12 on it (stretched
  vs held at 120/300/1000) running.
- **2026-10-23 23:25. Round 12 done on the 10-step checkpoint** (section "Round 12b"). Round 13's retrain was killed 12 s
  in by the previous session ending at 23:10 (log `varc_ho3/train_killed_session_end.log`); relaunched 23:23.
- **2026-10-23 23:31. Round 13 done:** retrained baseline ttt_mini 3/4/5 (.793), normalised 120 steps 2/5/5 (.811);
  .015-.018 below varc_ho2. Section "Round 13".
- **2026-10-23 23:50. Round 14 done:** code replicates on the original build (4/4/4, .812); the retrain gap is the build's
  augmentation draw (two seeds agree). Section "Round 14". GPU idle.
- **2026-10-24 01:07. Round 15, one-step recipe first order, 120 steps:** 5/6/8 (.832), against second order 6.0/7.3/8.7
  (.836, 3 seeds; seed range 5-7 / 7-8 / 8-9, .833-.839) and plain 4.0/6.7/8.0 (.827). Within or at the edge of the
  second-order seed range. Meta-training took 7 min (second order: ~12).
- **2026-10-24 01:13. Round 15, one-step first order, 300 steps:** 6/10/11 (.863), against second order 8/9/9 (.862) and
  plain 7/9/10 (.853): token accuracy equal to second order, task counts mixed (-2 pass@1, +1 / +2 at pass@2 / 5).
- **2026-10-24 01:31. Round 15, one-step first order, 1000 steps:** 10/11/13 (0.884), against second order 10/13/15
  (.887) and plain 10/13/15 (.879). 5-step first-order meta-training started.
- **2026-10-24 01:56. Round 15, 5-step first order** (22 min meta-training vs 54 second order), stretched 120 steps:
  7/7/8 (0.841), against second order 7/8/9 (.845).
- **2026-10-24 02:01. Round 15, 5-step first order, stretched 300 steps:** 8/9/12 (0.872), against second order
  10/12/13 (.872).
- **2026-10-24 02:20. Round 15, 5-step first order, stretched 1000 steps:** 12/12/14 (0.890), against second order
  12/12/14 (.888).
- **2026-10-24 02:23. Round 15 done** (5-step first order, held 120: 5/7/8, 0.837). Section "Round 15". TRM queue
  (round 16) started.
- **2026-10-24 02:29. Round 16, plain TRM under ttt_mini:** 1/2/3 (p@100 6), tok .776, 5.9 min for 30 tasks; identical to
  the previous loop's evaluation of the same checkpoint (`lr_full`, 1/2/3 .776), so the eval path reproduces on TRM. TRM
  is the weaker baseline on this build (VARC: 4/4/4, .811).
- **2026-10-24 02:35. Round 16, plain TRM, normalised segment-step loop, 120 steps** (hand-set alpha, no meta-training):
  4/7/9 (p@100 12), tok .808, against its ttt_mini 1/2/3 (.776), in 5.7 min (ttt_mini 5.9). The normalised rule transfers
  to TRM, and by more than on VARC (+3 / +5 / +6 tasks; VARC plain: 4/4/4 -> 4/7/8). Per-segment demo loss 1.24 -> 0.34
  (one segment's loss, not comparable to VARC's per-trajectory numbers).
- **2026-10-24 02:48. Round 16, plain TRM, normalised segment loop, 300 steps:** 10/12/13 (0.812). TRM
  meta-fine-tune (`msgd16_trm_ms5`) started.
- **2026-10-24 03:55. Round 16's TRM meta-fine-tune ran out of GPU memory at meta-step 301**, the first second-order step
  (`msgd16_trm_ms5/train_oom_step301.log`; the smoke had survived 3 second-order steps on smaller episodes). Relaunched
  entirely first order, everything else unchanged: round 15 found first order equal to second order on VARC at every
  length, and the first 300 steps were first order anyway. No intermediate checkpoint, so it restarts at step 0; ~13 s per
  step, training ~3.6 h, evaluations after (~09:00).
- **2026-10-24 06:10. Round 16, TRM meta-fine-tuned** (first order, 1 h 55 min; per-step alphas decay, rows .028 -> .015,
  largest weight alphas on the L-level MLPs), stretched: 120 steps 4/5/5 (0.801), 300 steps
  8/10/12 (0.809); plain TRM under the same rule with the hand-set alphas: 4/7/9 (.808) and 10/12/13 (.812).
- **2026-10-24 07:47. Round 16 done.** Plain TRM, normalised segment loop, 1000 steps: 14/16/17 (.825), the project's best;
  meta-fine-tuned TRM 11/12/13 (.836). Section "Round 16". GPU idle.
- **2026-10-24 08:40. TRM halting measured:** 94% (plain) and 100% (meta-fine-tuned) of rows run to the 8-segment cap,
  so rows stay in lockstep; segment-indexed alphas proposed. Section "TRM: ACT halting inside the inner loop".
- **2026-10-24 14:15. Round 17 results** (section "Round 17"): large alpha inits (0.1, 1) collapse to 0/30 and ~0 token
  accuracy; the 1e-3 control on this branch's checkpoint reproduces the earlier pattern. Round 18 will use init 1e-3.
- **2026-10-24 16:35. Round 18 so far.** trm_ho3 (plain TRM retrained on this branch and build, 13.5 min): ttt_mini 1/2/3
  (.779), the same as trm_ho2 (1/2/3, .776). Under the normalised segment rule with ACT off and the hand-set alphas:
  5/7/8 (.799), 7/10/11 (.822), 8/9/12 (.815) at 120 / 300 / 1000 steps, well below trm_ho2's 4/7/9, 10/12/13,
  14/16/17 (ACT on, other build) at 300 and 1000, and no longer improving from 300 to 1000. Build and ACT-off are
  confounded in that comparison. The meta-fine-tune picked init 1e-3 / 3e-2 (round 17) and runs at ~27 s / meta-step
  (8 segment steps), so ~7.5 h: training ~23:40, evaluations ~00:45.
- **2026-10-24 16:45. Round 19 queued** (user, `msgd19_trm_act.sh`, after round 18): trm_ho3 under the normalised segment
  rule with ACT halting on, 1000 steps, to separate the build from ACT-off in round 18's plain-TRM drop. ~45 min, ETA ~01:30.
- **2026-10-24 20:43. Round 18 done:** meta-fine-tuned TRM 6/6/7, 8/9/9, 11/13/15 at 120 / 300 / 1000 (segment schedule)
  against plain 5/7/8, 7/10/11, 8/9/12. Section "Round 18". Round 19 (ACT on) running.
- **2026-10-24 21:32. Round 19 done:** trm_ho3 with ACT on, 1000 steps: 12/14/14 (.835) vs ACT off 8/9/12 (.815). Section
  "Round 19". GPU idle.
- **2026-10-24 21:55. Round 20 queued** (user, `msgd20_trm_act.sh`): msgd18_trm with ACT on at test time, segment
  schedule, 1000 steps, against plain trm_ho3 with ACT on (12/14/14, .835). ~50 min.
- **2026-10-24 23:56. Round 20 done:** meta-fine-tuned TRM with ACT on at test: 9/11/12 (0.825). Section "Round 20".
- **2026-10-25 00:15. Round 21 queued** (user, `msgd21_trm_act.sh`): TRM with ACT on throughout: round 18's recipe from
  trm_ho3 without `arch.disable_act_halt`, evaluated with ACT on and the segment schedule at 1000 steps, against plain
  trm_ho3 with ACT on (12/14/14, .835). ~3.5 h training + ~50 min evaluation, ETA ~04:45.
- **2026-10-25 00:45. Round 22 queued** (user, `msgd22_trm_wait.sh`, after round 21): new option `meta_sgd.halt_wait`
  (with segment steps, a row the ACT head halts sits out until its trajectory ends, every 8 inner steps, then all rows
  restart together; before, a halted row restarted at once, so later inner steps mixed rows at different trajectory
  positions under one per-step alpha). Round 21's recipe plus halt_wait, and the plain trm_ho3 control under the same
  rule, both at 1000 steps. ~3.5 h + 2 x 50 min, ETA ~10:00.
- **2026-10-25 01:00. Round 23 queued** (user, `msgd23_embed_ttt.sh`, after round 22): arXiv 2609.21181's protocols on
  varc_ho3 and trm_ho3 at ttt_mini's length. Embed-TTT = embedding only (`train_overrides.freeze_weights=true`);
  Embed-Full-TTT, the paper's best = embedding for epochs 1-10, then the backbone with the embedding frozen for 11-20
  (`train_overrides.freeze_alternation_epochs=10`; the lr schedule spans all 20 epochs, the paper's per-step schedule
  is not given in its text). Full-TTT = ttt_mini, already run. ~20 min, ETA ~10:20.
- **2026-10-25 01:30. Round 24 queued** (user, `msgd24_embed_first.sh`, after round 23): arXiv 2609.21181's two-step idea
  inside the best recipes' test-time loop. New `evaluate_ttt.py` option `meta_sgd_embed_first` (fraction of the inner
  steps that adapt only the puzzle-embedding rows; the rest adapt only the weights, rows frozen; tested). At 0.5 and 1000
  steps on varc_ho3 (hand-set), msgd17_v_a1e-3 (stretched), trm_ho3 (ACT on, hand-set) and msgd18_trm (ACT off, segment),
  each against its single-phase result. ~2.5 h, ETA ~13:00.
- **2026-10-25 04:39. Round 21 done:** TRM meta-fine-tuned with ACT on throughout, 1000 steps: 11/11/13 (.832), against
  plain with ACT on 12/14/14 (.835). Section "Round 21". Round 22 (halt-and-wait) started.
- **2026-10-25 08:52. Round 22 done:** halt-and-wait: plain 5/6/7 (.816), meta-fine-tuned 9/11/13 (.816). Section
  "Round 22". Round 23 (the paper's TTT protocols) started.
- **2026-10-25 09:10. Round 23 done** (section "Round 23"): Embed-TTT / Embed-Full-TTT at ttt_mini's length at or
  below Full-TTT on both models. Round 24 started.
- **2026-10-25 11:18. Round 24 done:** embedding-first loop worse than single phase in all four configurations (section
  "Round 24"). GPU idle.
- **2026-10-26 00:40. Round 25 queued** (user, `msgd25_trm_traj.sh`): TRM Meta-SGD with the trajectory query loss, lambda
  10, from trm_ho3 on round 21's recipe (MSL off). First targets re-optimised per episode at the adapted weights, K = 10
  (smoke: 60-100 s / meta-step, ~22 h), then targets cached from the unadapted weights with each puzzle's own embedding
  row (new option `meta_sgd.traj_refresh=0.25`: rebuilt every 25% of training, K = 20, ~75 min per rebuild, ~8.5 h).
  Smoke matching losses: ~1e-3 per-step (targets start at the adapted rollout) vs ~0.8 cached (the adapted model, from a
  fresh row, starts far from the base model's trajectory). Each evaluated at 1000 steps, ACT on, segment schedule.
  ETA ~09:00 on 2026-10-27.
