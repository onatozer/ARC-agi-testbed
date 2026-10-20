#!/bin/bash
# Partial MAML (REPORTS_MAML.md "Partial MAML"): the inner loop adapts one part of the network, the outer loop trains
# everything (ANIL). Shared definitions for the queue_maml_{emb,lh,gramheads}.sh queues (source this file).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/recipes.sh"
HO="$DATA_HO/heldout_tasks.json"
MAML_COMMON="global_batch_size=16 optimizer=adam_atan2 lr=1e-3 lr_warmup_steps=100 lr_min_ratio=1.0 epochs=100000 train_epochs_per_iter=100000 \
max_steps=2000 eval_interval_steps=1000 max_eval_steps=null seed=0 +project_name=ARC-agi-testbed +entity=ozeronatwanb puzzle_emb_lr=3e-2"
MAML_BASE="maml.enabled=true maml.inner_steps=1 maml.meta_batch_size=4 maml.support_copies=16"

# maml_pretrain <name> <alpha> <adapt_substrings e.g. [puzzle_emb]> <extra overrides...>
maml_pretrain() { local name=$1 alpha=$2 parts=$3; shift 3; DISABLE_COMPILE=1 bash "$HERE/run.sh" "$name" cfg_pretrain_miniarc dataset.data_path="$DATA_HO" $MAML_COMMON $BACKBONE $MAML_BASE maml.inner_lr=$alpha "maml.adapt_substrings=$parts" +run_name="$name" "$@"; }
# ttt_maml <run> <suffix> <overrides...>: the meta-trained inner loop (eager)
ttt_maml() { local run=$1 suffix=$2; shift 2; DISABLE_COMPILE=1 bash "$HERE/ttt.sh" "$(ckpt_of "$run")" "$suffix" task_ids_file="$HO" "$@"; }
# ttt_plain_subset <run> <suffix> <alpha> <parts> <steps>: the same subset adaptation on a plainly trained checkpoint
ttt_plain_subset() { local run=$1 suffix=$2 alpha=$3 parts=$4 steps=$5; DISABLE_COMPILE=1 bash "$HERE/ttt.sh" "$(ckpt_of "$run")" "$suffix" task_ids_file="$HO" \
  +train_overrides.maml.enabled=true +train_overrides.maml.inner_lr=$alpha "+train_overrides.maml.adapt_substrings=$parts" maml_steps=$steps; }
# ttt_adam <run> <suffix> <overrides...>: the ordinary ttt_mini loop, compiled (maml disabled for it on MAML checkpoints)
ttt_adam() { local run=$1 suffix=$2; shift 2; bash "$HERE/ttt.sh" "$(ckpt_of "$run")" "$suffix" task_ids_file="$HO" maml_steps=0 "$@"; }

# grid <maml run> <plain run> <prefix> <alpha> <parts>: the TTT grid of one ablation
grid() {
  local run=$1 plain=$2 pfx=$3 alpha=$4 parts=$5
  ttt_maml "$run" ${pfx}_s1 maml_steps=1
  ttt_maml "$run" ${pfx}_s5 maml_steps=5
  ttt_maml "$run" ${pfx}_s10 maml_steps=10
  ttt_plain_subset "$plain" ${pfx}_s5 "$alpha" "$parts" 5
  ttt_plain_subset "$plain" ${pfx}_s10 "$alpha" "$parts" 10
  ttt_adam "$run" ${pfx}_adam
}
