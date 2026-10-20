#!/bin/bash
# REPORTS_MAML.md, re-run: first-order MAML of the base TRM (shared block, puzzle embeddings kept and adapted in the
# inner loop) on the held-out build with puzzle-level episodes (16 augmented copies of a puzzle; their demos are the
# support, one held-out row per copy the query), 4 puzzles per meta-step, 2000 meta-steps (~512k example rows, the
# plain recipe's 4000 x 128), eager. Then TTT over the 30 held-out tasks: the meta-trained inner loop at 1 / 5 / 10
# steps and the ordinary ttt_mini loop; the same SGD adaptation on the plainly trained trm_ho2 (section 7's control,
# same recipe and build) as the paper's fine-tune baseline. One GPU job at a time; suffixes are new (mm2_*).
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/recipes.sh"
HO="$DATA_HO/heldout_tasks.json"
MAML="maml.enabled=true maml.inner_steps=1 maml.inner_lr=0.1 maml.meta_batch_size=4 maml.support_copies=16"
MAML_COMMON="global_batch_size=16 optimizer=adam_atan2 lr=1e-3 lr_warmup_steps=100 lr_min_ratio=1.0 epochs=100000 train_epochs_per_iter=100000 \
max_steps=2000 eval_interval_steps=1000 max_eval_steps=null seed=0 +project_name=ARC-agi-testbed +entity=ozeronatwanb puzzle_emb_lr=3e-2"

DISABLE_COMPILE=1 bash "$HERE/run.sh" maml_trm_emb cfg_pretrain_miniarc dataset.data_path="$DATA_HO" $MAML_COMMON $BACKBONE $MAML +run_name=maml_trm_emb

ttt() { local run=$1 suffix=$2; shift 2; DISABLE_COMPILE=1 bash "$HERE/ttt.sh" "$(ckpt_of "$run")" "$suffix" task_ids_file="$HO" "$@"; }
ttt maml_trm_emb mm2_s1 maml_steps=1
ttt maml_trm_emb mm2_s5 maml_steps=5
ttt maml_trm_emb mm2_s10 maml_steps=10
ttt trm_ho2 mm2_s5 +train_overrides.maml.enabled=true +train_overrides.maml.inner_lr=0.1 maml_steps=5
ttt trm_ho2 mm2_s10 +train_overrides.maml.enabled=true +train_overrides.maml.inner_lr=0.1 maml_steps=10
# the ordinary loop on the MAML checkpoint, compiled (evaluate_ttt disables maml for it)
bash "$HERE/ttt.sh" "$(ckpt_of maml_trm_emb)" mm2_adam task_ids_file="$HO" maml_steps=0
