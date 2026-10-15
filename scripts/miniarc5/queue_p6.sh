#!/bin/bash
# Phase 6, the open loop: TreeRM variants on the standard build, judged on pass@2 against treerm_std. Waits for phase 5.
source "$(dirname "$0")/recipes.sh"
until grep -q "^exit=" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/treerm_std_aug100/train.log" 2>/dev/null; do sleep 60; done
run_treerm treerm_hfull   "$DATA_STD" arch.mcts_horizon=full          # rollouts to the end of the last supervision step: ARC rules span steps
run_treerm treerm_k8      "$DATA_STD" arch.num_candidates=8           # more candidates per z_H update
run_treerm treerm_halt16  "$DATA_STD" arch.halt_max_steps=16          # twice the supervision steps (and twice the eval searches)
run_treerm treerm_emblr3  "$DATA_STD" puzzle_emb_lr=3e-2              # faster task embeddings
run_treerm treerm_sims16  "$DATA_STD" arch.mcts_simulations=16 arch.eval_mcts_simulations=32   # deeper search, train and eval
for name in treerm_hfull treerm_k8 treerm_halt16 treerm_emblr3 treerm_sims16; do "$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/$name"; done
