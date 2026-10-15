# MCTS benchmark

Measures one `arch.mcts_impl` implementation of TreeRM's search; see `REPORTS_MCTS.md` at the repo root for the
results and `tests/test_mcts_contract.py` for the correctness contract. Everything runs from the repo root with
`venv/bin/python`; the datasets `data/sudoku-6u` and `data/nqueens-6` must exist (worktrees symlink them).

```
# once: the reference checkpoint (the recipe, 2000 steps, ~5 min), kept under ~/ARC-agi-testbed/autoresearch_runs/mcts_bench/ref_sudoku6u/
scripts/mcts_bench/train_reference.sh sudoku6u

# benchmark an implementation (micro search timings + training it/s + full test-set eval from the checkpoint)
venv/bin/python scripts/mcts_bench/bench.py --impl python --env sudoku6u --out scripts/mcts_bench/results/baseline_python.json
venv/bin/python scripts/mcts_bench/bench.py --impl <name> --env sudoku6u

# compare against the baseline (markdown tables)
venv/bin/python scripts/mcts_bench/compare.py scripts/mcts_bench/results/baseline_python.json scripts/mcts_bench/results/<name>_sudoku6u.json
```

Useful flags: `--skip eval` (no checkpoint needed), `--eval-steps 4` (256 puzzles instead of 1000), `--sims 8`,
`--depths 3`, `--no-compile`, `--override arch.num_candidates=8` (any Hydra override, repeatable). The run holds an
exclusive lock on `gpu.lock` next to the reference runs so two benchmarks never share the GPU.
