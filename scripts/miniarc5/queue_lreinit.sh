#!/bin/bash
# REPORTS_ARC.md section 7: L-module test-time training on the held-out build. Pretrains with a separate H block
# (arch.H_layers=2) for TRM and GRAM (the sc_gram_pf recipe: pooled LPRM head, final target, prior sample fraction,
# draw bootstrap; the PR #67 KL), a shared-block TRM control, then the TTT variants on each: full fine-tune, L block
# only, L block re-initialised and trained alone, and for GRAM the label-free LPRM phase on top. One GPU job at a time;
# every TTT suffix is new (the script resumes finished tasks).
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/recipes.sh"
EMB="puzzle_emb_lr=3e-2"
GRAM="arch=gram arch.H_layers=2 arch.prior_sample_frac=0.25 arch.lprm_prior_rollouts=4 arch.lprm_pool=mean arch.lprm_target=final"
HO="$DATA_HO/heldout_tasks.json"
L='reinit_substrings=[inner.L_level.]'
T='train_only_substrings=[inner.L_level.]'

run_trm trm_sep_ho "$DATA_HO" arch.H_layers=2 $EMB
run_trm trm_ho2 "$DATA_HO" $EMB
run_trm gram_sep_ho "$DATA_HO" $GRAM $EMB

# ttt <run> <suffix> <evaluate_ttt.py overrides...>
ttt() { local run=$1 suffix=$2; shift 2; bash "$HERE/ttt.sh" "$(ckpt_of "$run")" "$suffix" task_ids_file="$HO" "$@"; }
ttt trm_sep_ho lr_full
ttt trm_sep_ho lr_lonly "$T"
ttt trm_sep_ho lr_lreinit "$L" "$T"
ttt trm_ho2 lr_full
ttt gram_sep_ho lr_full
ttt gram_sep_ho lr_lonly "$T"
ttt gram_sep_ho lr_lreinit "$L" "$T"
ttt gram_sep_ho lr_lonly_lprm "$T" lprm_steps=20
ttt gram_sep_ho lr_lreinit_lprm "$L" "$T" lprm_steps=20
