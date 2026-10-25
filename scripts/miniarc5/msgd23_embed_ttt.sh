#!/usr/bin/env bash
# Round 23 (user): the TTT protocols of arXiv 2609.21181 ("Implicit Rule Induction with Test-Time Task Embeddings in
# ARC-like Tasks") on this branch's plain checkpoints, at ttt_mini's length (20 epochs, lr 3e-4 cosine, batch 8):
#   Embed-TTT       only the task (puzzle) embedding is finetuned       train_overrides.freeze_weights=true
#   Embed-Full-TTT  embedding first (epochs 1-10), then the backbone with the embedding frozen (11-20)
#                                                                         train_overrides.freeze_alternation_epochs=10
# Full-TTT (embedding and backbone together) is ttt_mini itself, already run: varc_ho3 ms10v_adam, trm_ho3 ms14t_adam.
# Run after round 22, in one shell.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
for run in varc_ho3 trm_ho3; do
  ttt $run ms17embed +train_overrides.freeze_weights=true
  ttt $run ms17embedfull +train_overrides.freeze_alternation_epochs=10
done
echo "[done23] $(date +%H:%M:%S)"
