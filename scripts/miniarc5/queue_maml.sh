#!/bin/bash
# REPORTS_ARC.md section 8: first-order MAML on the held-out build. Meta-trains TRM (shared block and separate H block)
# without puzzle embeddings, eager (DISABLE_COMPILE), batch 16 (~3 tasks per meta-step), 2000 meta-steps, inner step
# alpha 0.1 (the probe of section 8), and the paper's baseline (the same model trained plainly, then fine-tuned with
# the same SGD adaptation). Then TTT over the 30 held-out tasks: maml_steps 1 / 5 / 10 (the meta-trained inner loop)
# and the ordinary ttt_mini AdamAtan2 loop (maml_steps=0). One GPU job at a time; suffixes are new.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/recipes.sh"
HO="$DATA_HO/heldout_tasks.json"
NOEMB="arch.puzzle_emb_ndim=0 optimizer=adam_atan2"
MAML="maml.enabled=true maml.inner_steps=1 maml.inner_lr=0.1"
# COMMON with the MAML budget: batch 16, 2000 meta-steps, eval every 1000
MAML_COMMON="global_batch_size=16 optimizer=adam_atan2 lr=1e-3 lr_warmup_steps=100 lr_min_ratio=1.0 epochs=100000 train_epochs_per_iter=100000 \
max_steps=2000 eval_interval_steps=1000 max_eval_steps=null seed=0 +project_name=ARC-agi-testbed +entity=ozeronatwanb"

run_maml() { local name=$1; shift; DISABLE_COMPILE=1 bash "$HERE/run.sh" "$name" cfg_pretrain_miniarc dataset.data_path="$DATA_HO" $MAML_COMMON $BACKBONE $NOEMB $MAML +run_name="$name" "$@"; }
run_maml maml_trm
run_maml maml_trm_sep arch.H_layers=2
run_trm plain_noemb "$DATA_HO" $NOEMB

ttt() { local run=$1 suffix=$2; shift 2; DISABLE_COMPILE=1 bash "$HERE/ttt.sh" "$(ckpt_of "$run")" "$suffix" task_ids_file="$HO" "$@"; }
for run in maml_trm maml_trm_sep; do
  ttt $run mm_s1 maml_steps=1
  ttt $run mm_s5 maml_steps=5
  ttt $run mm_s10 maml_steps=10
  ttt $run mm_adam maml_steps=0
done
ttt plain_noemb mm_s5 +train_overrides.maml.enabled=true +train_overrides.maml.inner_lr=0.1 maml_steps=5
ttt plain_noemb mm_s10 +train_overrides.maml.enabled=true +train_overrides.maml.inner_lr=0.1 maml_steps=10
ttt plain_noemb mm_adam
