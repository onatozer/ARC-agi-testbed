#!/bin/bash
# Runner for the partial-MAML ablations (one GPU): the plain base-GRAM control on the held-out build and the
# heads-only step-size probe on it (3 steps, 2 tasks, alphas 0.01 / 0.1 / 1), then the embedding-only and
# block-only queues. queue_maml_gramheads.sh is launched by hand with ALPHA from the probe (probe lines in
# gram_ho's run dir: ttt_pg_a*.log).
source "$(dirname "$0")/maml_parts.sh"
run_trm gram_ho "$DATA_HO" arch=gram puzzle_emb_lr=3e-2
for A in 0.01 0.1 1.0; do
  DISABLE_COMPILE=1 bash "$HERE/ttt.sh" "$(ckpt_of gram_ho)" pg_a${A//./p} task_ids_file="$HO" max_tasks=2 wandb_mode=disabled \
    +train_overrides.maml.enabled=true +train_overrides.maml.inner_lr=$A "+train_overrides.maml.adapt_substrings=[prior_head,posterior_head,v_head]" maml_steps=3
done
bash "$HERE/queue_maml_emb.sh"
bash "$HERE/queue_maml_lh.sh"
