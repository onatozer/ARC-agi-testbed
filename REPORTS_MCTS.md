# TreeRM MCTS runtime optimization: report

Branch `treerm-mcts-opt` (off `origin/TreeRM`, the autoresearch result; see `REPORTS.md` for how TreeRM got there).
Goal: make TreeRM's per-puzzle MCTS (`models/mcts.py`, driven from `TreeRMInner._search` in
`models/recursive_reasoning/treerm.py`) as fast as possible without changing what it computes, with a test suite and
a benchmark that say so. Sections: 0 scoreboard, 1 decisions, 2 where the time goes, 3 the test suite and benchmark,
4 the optimization log (one entry per change, with measured numbers, kept or dropped).

## 0. Scoreboard

Measured on the reference setting (6x6 Sudoku, unique solutions; the recipe: hidden 128, K=4 flag candidates,
8 training / 16 eval simulations, batch 64; one RTX 3050 Ti laptop GPU, 4 GB). `scripts/mcts_bench/compare.py`
prints these tables from the JSON results under `scripts/mcts_bench/results/`.

| impl | search ms, train 8 sims depth 3 | search ms, eval 16 sims depth 3 | train it/s | eval s (1000 puzzles) | exact acc | status |
|---|---|---|---|---|---|---|
| python (baseline) | pending | pending | pending | pending | pending | reference |

## 1. Decisions

- **Sequential, one GPU job at a time.** The previous session (2026-10-14 15:30) launched three worker agents in
  parallel worktrees, each building a model, running pytest and training on the same 4 GB GPU / 15 GB laptop; the
  machine crashed minutes later. Everything after that is done in this one worktree, sequentially, with one training
  or benchmark process on the GPU at a time (the benchmark holds a file lock, `autoresearch_runs/mcts_bench/gpu.lock`,
  so two benchmarks cannot overlap). Only the contract tests (a 32-hidden model) run next to a training job.
- **Salvaged from the crashed session:** the implementation registry (`models.mcts.MCTS_IMPLS`, config
  `arch.mcts_impl`), its contract docstring, and the factory line in `TreeRMInner.__init__`. The two other workers
  had produced nothing.
- **Registry, not conditionals.** Every optimized search is a new class in `MCTS_IMPLS` (new file, one registry line,
  at most a few impl-guarded lines in `treerm.py`); the baseline `MCTS` stays untouched so parity tests always have a
  reference. Classes set `exact = True` when they reproduce the baseline's visit counts bit-for-bit given the same
  callback values; a semantics-changing search (e.g. WU-UCT) sets `exact = False` and only the invariants apply.
- **Merging main broke the TreeRM tests.** `origin/TreeRM` had merged main (#59 added GRAM's `prior_sample_frac`,
  `prior_sample_per_puzzle`, `posterior_nograd_cycles`); `tests/test_treerm.py`'s fixture did not supply them and 10
  of 18 tests failed. Fixed by adding the three keys to the fixture, as `tests/test_gram.py` does (commit `fab383b`).
- **Reference setting = the recipe at its real budget.** The plan said "800 steps" for the reference checkpoints, but
  at 800 steps the recipe's test accuracy is ~0 (REPORTS.md section 3: 0.002 at 1000, 0.5 at 1500, 0.93 at 2000), and
  a checkpoint with no accuracy cannot tell a correct search from a broken one. Training turned out to run at
  ~7 it/s here (the old report's 1.2 it/s was measured with several runs sharing the GPU), so the full 2000-step
  recipe costs under 5 minutes and is used as-is: `scripts/mcts_bench/train_reference.sh sudoku6u`.
- **Environments.** 6x6 Sudoku (`data/sudoku-6u`, seq 36, vocab 8, 10k/1k puzzles, 0% test-in-train) is the primary
  benchmark, because it has real held-out accuracy. 6-queens (`data/nqueens-6`) is kept as a second shape only;
  REPORTS.md section 1 shows it has 4 solutions and its test set is in train, so its accuracy means nothing.
  `data/4x4-sudoku` actually holds 9x9 boards (seq 81) and is not used.
- **The report file.** The task named it `MCTS_REPORTS.md`; the user later asked for `REPORTS_MCTS.md`, which is this
  file.

## 2. Where the search's time goes (reading the code)

Per z_H update, `_search` builds B=64 roots (one per puzzle), each already expanded with its K=4 candidates, and
runs `mcts_simulations` strictly sequential rounds. Each round, for the 64 trees in lockstep:

1. `select`: a pure-Python PUCT walk per tree (depth ≤ 3 with `mcts_horizon=step` and H_cycles=3).
2. `evaluate_fn` = `_rollout_values`: `torch.stack` of the 64 leaf states, then up to `remaining` plain TRM updates
   (each `L_cycles + 1 = 5` eager `L_level` passes at batch 64), then `_leaf_values` (decode + CE in training, the
   value head at eval), `.tolist()` sync at the end.
3. `expand_fn` = `_expand`: another stack of the growing leaves, **the same 5 passes again from the same (z_H, z_L)**
   (its `u_t` is exactly the rollout's first update), the K candidates, the scorer, `.tolist()` of the priors.
4. Python backprop per tree, min-max stats update.

So one training search (8 sims, depth up to 3) is up to 8 × (15 + 5) = 160 eager `L_level` passes versus 15 for the
TRM step itself, every pass a handful of tiny kernels at batch 64 × seq 36 × hidden 128, plus two host syncs per
round. Each `Blocks.forward` also draws `randn_like` noise although `noise_scale = 0`, and `_seq_dict()` (RoPE
tables) is rebuilt on every callback. The whole search sits behind `@torch.compiler.disable`, so none of it is
compiled while the rest of the model is. Eval searches on all 3 cycles × 8 supervision steps with 16 sims, training
only on the one grad cycle: that is why eval is the expensive phase (REPORTS.md measured 18% GPU utilisation).

Levers, in the order the code and the three papers suggest (MCTS-NC: struct-of-arrays trees on the GPU; Chaslot
et al.: leaf/root/tree parallelism; WU-UCT / BU-UCT: batch W leaves per round with unobserved-visit counts):

- fuse expand + rollout (the rollout's first update is the expand's `u_t`): fewer passes, identical semantics;
- batch the K forced root-child visits into one round (they are independent by construction);
- replace the per-node Python tree with tensor arrays or a C++ tree so select/backprop stop being Python loops;
- make each pass cheaper: skip the zero noise draw, cache `_seq_dict`, compile or CUDA-graph the fixed-shape rollout
  step;
- WU-UCT-style W leaves per tree per round (semantics-changing): 16 sims become 16/W rounds at 4x the batch, on a
  GPU that is mostly idle.

## 3. Test suite and benchmark

### Contract tests: `tests/test_mcts_contract.py` (commit `bd24ad6`)
Parametrized over every `MCTS_IMPLS` entry. For all implementations: visit counts sum to the simulations and every
root child is visited; leaves at `allowed_depth` 0 are evaluated but never expanded and no state goes below the
root's depth; rows with different depths search independently; an affine rescaling of one tree's values changes
nothing (min-max normalisation) and one tree never moves another's counts; determinism. For `exact`
implementations additionally: the first K simulations visit root children 0..K-1 in order; a hand-computed depth-2
trace (counts, the exact states passed to `evaluate_fn`/`expand_fn` each round, their depths); PUCT ties go to the
first maximal child; priors order unvisited children; exactly one `evaluate_fn` call per simulation over every tree;
visit counts equal the baseline's on random trees; and, on the tiny TreeRM model, the posterior logits of `_search`
equal the baseline's in train and eval mode for both horizons. Plus: TreeRM trains and evals with every impl, and
an unknown `mcts_impl` is rejected. Runtime: 8 s on the GPU.

### Benchmark: `scripts/mcts_bench/bench.py`
One command per implementation, three sections, JSON out, `compare.py` turns two JSONs into the tables above.
- **micro**: `TreeRMInner._search` alone at the recipe's shapes (batch 64, seq 36, hidden 128, K=4) in train and eval
  mode, 8 and 16 simulations, remaining depth 1 and 3; warmup 2, median of 5 synchronized runs; a breakdown into
  rollout (`evaluate_fn`), expand (`expand_fn`) and "tree" (everything else: selection, backprop, host syncs) by
  timing the callbacks; peak memory; and, for any non-baseline impl, the posterior's max abs difference and argmax
  agreement with the baseline on the same inputs.
- **train**: the real `train_batch` from `pretrain.py` on `data/sudoku-6u`, model compiled exactly as pretrain
  compiles it (the search stays eager, as in training), 15 warmup + 50 timed steps, it/s and samples/s.
- **eval**: a full test-set eval (1000 puzzles, 16 eval sims) through `evaluators.eval_fn.evaluate` from the
  reference checkpoint (EMA weights, as `evaluate.py` loads them), after a one-batch warmup for the compile;
  wall-clock, `all/exact_accuracy`, peak memory. The eval samples the next z_H from the posterior, so it is seeded;
  an implementation that issues a different number of random draws (e.g. fewer `randn_like` calls) will still see
  different draws, so eval accuracy is a statistical check, while the micro parity and the contract tests are exact.
- The run holds `gpu.lock`; results are only comparable when nothing else uses the GPU.

### Reference checkpoints: `scripts/mcts_bench/train_reference.sh`, `scripts/mcts_bench/REFERENCE.md`
The recipe for 2000 steps on `data/sudoku-6u`, checkpoint kept under `~/ARC-agi-testbed/autoresearch_runs/mcts_bench/ref_sudoku6u/`.

## 4. Optimization log

Entries are in order; each records the change, the measured effect against the baseline, and whether it stays.

### 4.0 Baseline (`mcts_impl=python`), commit `029b7d9`
`scripts/mcts_bench/results/baseline_python.json`. Search alone (batch 64, median of 5):

| mode | sims | depth | ms | rollout ms | expand ms | tree ms |
|---|---|---|---|---|---|---|
| train | 8 | 1 | 24.7 | 21 | 0 | 4 |
| train | 8 | 3 | 245.9 | 132 | 118 | 8 |
| train | 16 | 1 | 35.0 | 30 | 0 | 5 |
| train | 16 | 3 | 435.9 | 243 | 204 | 12 |
| eval | 8 | 1 | 14.9 | 11 | 0 | 3 |
| eval | 8 | 3 | 264.0 | 134 | 93 | 8 |
| eval | 16 | 1 | 27.8 | 25 | 0 | 5 |
| eval | 16 | 3 | 538.5 | 238 | 207 | 14 |

Training: 9.04 it/s (109 ms/step, compiled model, 50 steps, peak 282 MB). Eval: 96.4 s for the 1000-puzzle test set at
16 sims, exact accuracy 0.865 (the training run's own final eval: 0.866), peak 354 MB.

What this says:
- **The Python tree is not the problem.** Selection + backprop + bookkeeping ("tree") is 3-5% of a search. A C++
  or tensor tree, on its own, could not give more than that; the user's C++ idea is only worth it combined with a
  cheaper domain step.
- **The domain callbacks are, and not because of their GEMMs.** `profile_search.py` (train, 8 sims, depth 3): one
  search issues ~8,900 kernel launches; CPU-side time 260 ms vs 97 ms of GPU time, i.e. the GPU waits for Python
  and the launch queue most of the time. Of the GPU time itself only ~18% is matmul and 9% attention; 22% is
  `aten::mul`, 11% `add`, 15% `copy_`/`_to_copy` (every `CastedLinear` casts its fp32 weight to bf16 on every call),
  6% `cat` (the per-round `torch.stack`s), and the rest is rms-norm's pow/mean/rsqrt and SiLU: small elementwise
  kernels that an eager loop cannot fuse.
- The expand costs almost as much as a 3-update rollout (118 vs 132 ms) although it runs a third of the passes:
  it carries its own stack, the scorer's cross-attention, 256 candidate slices into Python tuples and a second host
  sync per round.
- Depth matters more than simulations: 16 sims at depth 1 cost 35 ms, 8 sims at depth 3 cost 246 ms. Training
  searches run at depth 3, 2, 1 across the three cycles (only the grad cycle searches); eval searches at every cycle.

So the order of attack is: fewer passes per round (fuse), then make a round one launch instead of a thousand
(CUDA graph / `torch.compile` of the fixed-shape round, which also fuses the elementwise chains), then the tree.

**Can a compiled function run inside the compiler-disabled search?** Yes. Probe (`tmp/compile_probe.py`): a
`torch.compile(mode="reduce-overhead")` callee invoked from a `@torch.compiler.disable` frame, itself called from a
compiled outer function, ran compiled with CUDA graphs: 1.3-1.5 ms vs 13.9 ms eager for 8 x 20 elementwise ops. The
only trap is that CUDA-graph outputs are overwritten by the next replay, so anything kept across rounds (candidates,
bootstrap targets) must be cloned out.

### 4.1 Fused expand + rollout, batched forced root visits (`mcts_impl=fused`)
`models/mcts.py::FusedMCTS`, `TreeRMInner._step`. Two changes, both exact:
- **One domain call per round.** `run` takes a `step_fn(states, depths) -> (values, children, priors)`; TreeRM's
  `_step` runs the rollout once and, after its first TRM update (which is exactly what `_expand` recomputed), takes the
  candidates and priors of the leaves with depth left from that update. Saves the 5 expand passes and one stack per
  round, and merges the two host syncs (`values.tolist()`, `priors.tolist()`) into one. Without a `step_fn` the class
  composes the two baseline callbacks, so the toy contract tests cover it like any other impl.
- **The forced root visits in one batch.** The baseline spends its first K rounds evaluating root child k of every
  tree, by construction without any selection in between; they are independent, so `FusedMCTS` evaluates all K x B
  of them in a single call (K rounds become one; at batch 256 instead of 64 a launch-bound pass costs about the same).
  Backprop and min-max updates commute, so the tree state after the batch is identical to the baseline's.
- **Loss normalisation made batching-proof.** The value-bootstrap term divided by `len(self._bootstrap)`, i.e. the
  number of rollout *calls*, which equals the number of simulations only when every round is its own call. It now
  divides by `mcts.num_simulations` (the same number for the baseline; `tests/test_treerm.py` still asserts one
  bootstrap entry per simulation for it).
- **Contract tests adjusted.** "One evaluate call per simulation" was a baseline property, not part of the contract;
  the tests now compare the flattened order of evaluated (tree, state) pairs, which a batched round must still follow.
- Floating point: the fused candidates come from a `u_t` computed at the rollout's batch (64) instead of at the
  growing subset's batch, and the forced round runs at batch 256; the kernels can differ in the last bf16 bit, so
  posterior parity on the real model is checked empirically (micro `parity` field, contract test on the tiny model).

**Measured** (`results/fused_sudoku6u.json`, the first version of `_step`: masked leaf values, candidates only for
growing rows):

| mode | sims | depth | baseline ms | fused ms | speedup | posterior argmax agreement |
|---|---|---|---|---|---|---|
| train | 8 | 1 | 24.7 | 9.2 | 2.7x | 1.000 |
| train | 8 | 3 | 245.9 | 113.0 | 2.2x | 1.000 |
| train | 16 | 1 | 35.0 | 27.7 | 1.3x | 1.000 |
| train | 16 | 3 | 435.9 | 205.7 | 2.1x | 0.984 (1 row of 64) |
| eval | 8 | 1 | 14.9 | 7.7 | 1.9x | 1.000 |
| eval | 8 | 3 | 264.0 | 122.9 | 2.1x | 1.000 |
| eval | 16 | 1 | 27.8 | 15.0 | 1.9x | 1.000 |
| eval | 16 | 3 | 538.5 | 235.3 | 2.3x | 1.000 |

Training 9.04 -> 10.79 it/s (109 -> 87 ms/step). Full eval 96.4 -> 53.7 s (1.8x), exact accuracy 0.865 -> 0.868
(seed noise: the sampled picks differ once any value differs in the last bit), peak memory 354 -> 564 MB (the
batch-256 forced round). The one disagreeing row is the bf16 last-bit effect above flipping one PUCT decision; all
57 GPU tests pass, including exact posterior parity on the tiny model. **Kept.** The depth-1 case shows the batching
of the forced round alone: with 8 sims, 4 rounds become 1, hence 2.7x on a search that never expands.

### 4.2 Static-shape round + `torch.compile(mode="reduce-overhead")` (`search_compile=true`)
`TreeRMInner._step_core`. The round's tensor work is rewritten so nothing depends on data: leaf values are computed
for every row at every stage and selected by `torch.where(depth == stage)` instead of boolean indexing (which
synced the host twice per stage), and candidates/priors are computed for every row of the round and only used for
the rows with depth left. That makes the round a pure function of (batch, max depth, train/eval mode), so it can be
`torch.compile`d with CUDA graphs: one replay per round instead of ~1,000 launches, with the elementwise chains
(rms-norm, residual mixes, SiLU, the fp32->bf16 weight casts) fused by inductor. Outputs that outlive the round
(candidates, the next z_L, bootstrap tokens/targets) are cloned out of graph memory. Specialisations: 2 batch sizes
(64, 256 for the forced round) x 3 depths x 2 modes = 12 graphs, so `torch._dynamo.config.cache_size_limit` is
raised to 64 when the option is on. `search_compile` requires `mcts_impl=fused` (the baseline callbacks run on
variable-size subsets and would recompile per size).
**Measured, rewrite without compile** (`results/fused_static_sudoku6u.json`): training search depth 3 109.8 ms
(fused 113.0), eval 16 sims depth 3 248.8 ms (235.3), training 9.62 it/s (10.79; one 50-step sample each, so
within noise), full eval 50.6 s (53.7), accuracy 0.868, parity as before. The two syncs per stage it removes are
paid back by the all-row leaf values and candidates, so on its own it is a wash; it exists to make the round
compilable. The breakdown now attributes ~95% of a depth-3 search to the single step callback and 5-7% to the tree.
**Measured, with compile:** pending.
