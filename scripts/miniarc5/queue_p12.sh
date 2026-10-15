#!/bin/bash
# Phases 1-2: the three architectures on the standard build, then on the held-out build (sequential, locked).
source "$(dirname "$0")/recipes.sh"
for arch in trm varc treerm; do echo "== $arch standard"; run_$arch "${arch}_std" "$DATA_STD"; done
for arch in trm varc treerm; do echo "== $arch held-out"; run_$arch "${arch}_ho" "$DATA_HO"; done
for name in trm_std varc_std treerm_std trm_ho varc_ho treerm_ho; do "$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/$name"; done
