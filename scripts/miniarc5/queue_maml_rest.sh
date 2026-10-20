#!/bin/bash
# REPORTS_MAML.md: the rest of the MAML TTT grid after queue_maml.sh. The plain no-embedding control under the
# meta-trained inner loop (5 / 10 SGD steps at alpha 0.1, eager), then the ordinary ttt_mini AdamAtan2 loop on the
# two MAML checkpoints and the control, compiled (queue_maml_adam.sh). One GPU job at a time.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/recipes.sh"
HO="$DATA_HO/heldout_tasks.json"
CK=$(ckpt_of plain_noemb)
for n in 5 10; do
  DISABLE_COMPILE=1 bash "$HERE/ttt.sh" "$CK" mm_s$n task_ids_file="$HO" +train_overrides.maml.enabled=true +train_overrides.maml.inner_lr=0.1 maml_steps=$n
done
bash "$HERE/queue_maml_adam.sh"
