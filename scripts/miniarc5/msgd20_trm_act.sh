#!/usr/bin/env bash
# Round 20 (user): the meta-fine-tuned TRM (msgd18_trm: 8 segment steps, trained with ACT off) evaluated with ACT halting
# ON at test time, trajectory-indexed alphas (meta_sgd_schedule=segment), at VARC's best setting, 1000 steps; against the
# plain trm_ho3 under the same rule with ACT on and the hand-set alphas (round 19: 12/14/14, .835).
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
ttt msgd18_trm ms14m_act_sg1000 meta_sgd_steps=1000 meta_sgd_schedule=segment +train_overrides.arch.disable_act_halt=false
echo "[done20] $(date +%H:%M:%S)"
