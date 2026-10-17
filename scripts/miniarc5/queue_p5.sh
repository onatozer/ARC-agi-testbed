#!/bin/bash
# Phase 5: seed replicates of the standard baselines and the augmentation-count sensitivity run for TreeRM.
source "$(dirname "$0")/recipes.sh"
DATA_AUG100=$WT/data/miniarc5-aug-100-t31   # 100 augmentations in training, the test split capped at 32 copies per task
run_treerm treerm_std_s1 "$DATA_STD" seed=1
run_trm    trm_std_s1    "$DATA_STD" seed=1
run_treerm treerm_std_aug100 "$DATA_AUG100"
for name in treerm_std_s1 trm_std_s1 treerm_std_aug100; do "$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/$name"; done
