# Reference checkpoints

Trained with `scripts/mcts_bench/train_reference.sh <env>` (the autoresearch recipe: TreeRM, hidden 128, K=4 flag
candidates, 8 training / 16 eval simulations, value bootstrap 0.5, candidate heads at 0.1x lr, batch 64). They live
outside the repo, under `~/ARC-agi-testbed/autoresearch_runs/mcts_bench/ref_<env>/`; `bench.py --checkpoint auto`
picks the newest `.pth` there. Every implementation's eval accuracy is measured from these weights, so a
semantics-changing search is judged on a trained model rather than on a noisy short training run.

| env | steps | lr | wall (train + final eval) | test puzzles | exact accuracy | token accuracy | checkpoint |
|---|---|---|---|---|---|---|---|
| sudoku6u | 2000 | 1e-3 | 378 s | 1000 | 0.866 | 0.970 | `ref_sudoku6u/outputs/34827415-.../checkpoints/step_2000_34827415-4f22-4169-83b0-d452bb958280.pth` (2026-10-14, commit `bd24ad6`) |

Training ran alone on the GPU at ~7 it/s (the 4.4 min of training; the 1000-puzzle eval with 16 simulations took
~1.9 min of the wall-clock). The eval's `search/argmax_agree` was 0.46: the eval search overrode the prior's pick in
more than half of the searches, so the search is doing real work on this checkpoint.
