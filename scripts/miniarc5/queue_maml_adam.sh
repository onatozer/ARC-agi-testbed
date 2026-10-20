#!/bin/bash
# REPORTS_MAML.md: the ordinary ttt_mini AdamAtan2 loop (maml_steps=0) on the MAML checkpoints and the plain
# no-embedding control, compiled (queue_maml.sh ran its TTTs eager for the MAML inner loop, ~5 min per task for this
# loop). Resumes the tasks queue_maml.sh finished under the same suffix.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/recipes.sh"
HO="$DATA_HO/heldout_tasks.json"
for run in maml_trm maml_trm_sep; do
  bash "$HERE/ttt.sh" "$(ckpt_of "$run")" mm_adam task_ids_file="$HO" maml_steps=0
done
bash "$HERE/ttt.sh" "$(ckpt_of plain_noemb)" mm_adam task_ids_file="$HO"
