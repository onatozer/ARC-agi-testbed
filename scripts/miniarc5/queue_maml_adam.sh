#!/bin/bash
# REPORTS_MAML.md: the ordinary ttt_mini AdamAtan2 loop (maml_steps=0) on the two MAML checkpoints, suffix mm_adam2
# (the first mm_adam attempt ran a MAML meta-step per batch: evaluate_ttt.py now disables maml for this loop) and on
# the plain no-embedding control (mm_adam). One GPU job at a time.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/recipes.sh"
HO="$DATA_HO/heldout_tasks.json"
for run in maml_trm maml_trm_sep; do
  bash "$HERE/ttt.sh" "$(ckpt_of "$run")" mm_adam2 task_ids_file="$HO" maml_steps=0
done
bash "$HERE/ttt.sh" "$(ckpt_of plain_noemb)" mm_adam task_ids_file="$HO"
