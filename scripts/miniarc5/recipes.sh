#!/bin/bash
# The MiniARC 5x5 recipes shared by the smoke and experiment queues (source this file). See REPORTS_ARC.md section 1.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
DATA_STD=$WT/data/miniarc5-aug-31          # every task's demos in training, its 32 test copies in the test split
DATA_HO=$WT/data/miniarc5-aug-31-ho30      # 119 tasks in training; the 30 held-out tasks' test copies are the test split

# Same optimizer, schedule and budget for every architecture; W&B online to the user's project
COMMON="global_batch_size=128 optimizer=adam_atan2 lr=1e-3 lr_warmup_steps=100 lr_min_ratio=1.0 epochs=100000 train_epochs_per_iter=100000 \
max_steps=4000 eval_interval_steps=2000 max_eval_steps=null seed=0 +project_name=ARC-agi-testbed +entity=ozeronatwanb"
# Recurrent backbones (TRM, TreeRM): hidden 128, one puzzle-embedding slot (seq 25 + 1)
BACKBONE="arch.hidden_size=128 arch.num_heads=4 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=8 arch.puzzle_emb_len=1"
# TreeRM: the REPORTS.md recipe (no candidate loss) the MiniARC runs used; K=4, 8/16 simulations, value bootstrap,
# the 0.1x head lr and the compiled C++ search are the arch defaults
TREERM="arch=treerm arch.candidate_loss=none"

# run_trm <name> <data_path> <extra overrides...>
run_trm()    { local name=$1 data=$2; shift 2; bash "$HERE/run.sh" "$name" cfg_pretrain_miniarc dataset.data_path="$data" $COMMON $BACKBONE +run_name="$name" "$@"; }
run_treerm() { local name=$1 data=$2; shift 2; bash "$HERE/run.sh" "$name" cfg_pretrain_miniarc dataset.data_path="$data" $COMMON $BACKBONE $TREERM +run_name="$name" "$@"; }
run_varc()   { local name=$1 data=$2; shift 2; bash "$HERE/run.sh" "$name" cfg_pretrain_miniarc_varc dataset.data_path="$data" $COMMON +run_name="$name" "$@"; }

# newest checkpoint of a run
ckpt_of() { ls -t "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/$1"/outputs/*/*/*/checkpoints/*.pth 2>/dev/null | head -1; }
