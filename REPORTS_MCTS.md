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

Search times are medians of 5 at batch 64 (`bench.py` micro section); "eval s" is the test-set eval from the
reference checkpoint over its 15 full batches (960 puzzles; see 4.5b for why the partial 16th batch is excluded);
memory is the eval section's peak GPU allocated. Exact = same visit counts as the baseline given the same values.

| config (`arch.` keys) | search ms, train 8 sims depth 3 | search ms, eval 16 sims depth 3 | train it/s | eval s | exact acc | eval peak MB | exact | status |
|---|---|---|---|---|---|---|---|---|
| `mcts_impl=python` (baseline) | 245.9 | 538.5 | 9.0 | 88.0 | 0.870 | 354 | yes | reference |
| `mcts_impl=fused` | 106.3 | 228.4 | 9.4 | – | 0.868 | 265 | yes | kept |
| `mcts_impl=fused search_compile=true` | 50.7 | 97.1 | 9.8 | 27.0* | 0.868 | 270 | yes (bf16-level) | kept |
| `mcts_impl=cpp search_compile=true` | 50.1 | 93.9 | 9.8 | **18.9** | 0.869 | 121 (eval-only run) | yes (bf16-level) | **kept, recommended** |
| `mcts_impl=cpp search_compile=true mcts_leaves_per_round=2` | 47.4 | 77.3 | 9.4 | 18.9 | 0.870 | 125 (eval-only run) | no (WU-UCT) | opt-in, no gain here |
| `mcts_impl=wu search_compile=true mcts_leaves_per_round=4` | 51.1 | 89.3 | 10.8 | – | 0.870 | 283 | no | not worth it |

Training it/s is within its noise band (9.0-10.8 across runs) for every variant: training searches run only on
the grad cycle at depth 1 and are ~5-10% of a 100 ms step. The search speedups are what eval feels: **the full
test-set eval went from 88.0 s to 18.9 s (4.7x) with identical search semantics** (exact visit counts given the
same values; `search/argmax_agree` 0.457 vs 0.462, accuracy 0.869 vs 0.870 on the 960 puzzles). `*` the fused
27.0 s was measured in an earlier run before the memory/eval protocol change and includes the test-set's reload
ordering; the two 18.9 s runs are directly comparable with the 88.0 s baseline run (same protocol, same day).
WU-2 adds nothing to the full eval over the exact search: the round is GPU-bound, so halving the rounds at twice
the batch is a wash.

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
**First compiled run was wrong, and the harness caught it.** The micro parity field showed 0% posterior argmax
agreement in eval mode and the eval section crashed with the training-only "needs rollout_loss_fn" error inside an
eval-mode model. Cause: `__init__` stored `torch.compile(self._step_core)`, a compiled *bound* method; the EMA eval
model is a `copy.deepcopy` of the training model, and deepcopy keeps function objects by reference, so the copy's
attribute still called the original instance: its weights, and its train mode. Fix (commit `3b4adaa`): compile the
unbound `TreeRMInner._step_core` once per process and pass the module as its first argument, so dynamo guards on the
caller's mode and parameters; a regression test deep-copies a compiled model and checks the copy's round against
its own eager round. Lesson for every future variant: the parity field and the eval section are the test, not the
micro speed.
**Measured, with compile (fixed, on arena states; `results/arena_compile_sudoku6u.json`):**

| mode | sims | depth | baseline ms | arena (eager) ms | arena + compile ms | vs baseline | argmax agreement |
|---|---|---|---|---|---|---|---|
| train | 8 | 1 | 24.7 | 9.3 | 6.1 | 4.0x | 1.000 |
| train | 8 | 3 | 245.9 | 106.3 | 65.4 | 3.8x | 0.984 |
| train | 16 | 1 | 35.0 | 19.9 | 16.4 | 2.1x | 1.000 |
| train | 16 | 3 | 435.9 | 210.0 | 94.5 | 4.6x | 1.000 |
| eval | 8 | 1 | 14.9 | 7.8 | 11.7 | 1.3x | 1.000 |
| eval | 8 | 3 | 264.0 | 112.9 | 63.4 | 4.2x | 1.000 |
| eval | 16 | 1 | 27.8 | 13.5 | 16.7 | 1.7x | 1.000 |
| eval | 16 | 3 | 538.5 | 228.4 | 96.7 | 5.6x | 1.000 |

Training 9.17 it/s (eager arena 9.44: within noise, and training searches are depth 1, where the compiled round is
only a value-head call and the graph replay's fixed cost shows). Full eval 48.6 s vs 49.9 s eager, accuracy 0.868,
`search/argmax_agree` 0.464 (baseline 0.462): the search makes the same decisions. The eval barely moved although
its searches are 2.4-5.6x faster: hypothesis, the 1000-puzzle test set ends with a 40-row batch, and every new
batch size means new specialisations (40 and 160 rows x 3 depths) compiled inside the timed eval.
**Confirmed:** the same eval over the 15 full batches only takes 27.0 s (960 puzzles, accuracy 0.869), so the
last 40-puzzle batch alone cost ~21 s of compilation. Fix: the round's batch dimension is marked dynamic
(`torch._dynamo.maybe_mark_dynamic(t, 0)` on the round's inputs), so each (depth, mode) is compiled once with a
symbolic batch and CUDA graphs are merely recorded per batch size (milliseconds), which also covers any other batch
size a user evals with.
**First attempt did nothing** (`results/arena_compile_dyn_sudoku6u.json`: micro unchanged, full eval 54.6 s): with
`torch.compile(dynamic=False)` the soft marks (`maybe_mark_dynamic`) are overridden and every size still
specialises. Second attempt, `dynamic=None` plus hard `mark_dynamic`: dynamo raises a constraint violation from
`Blocks.forward` in the shared backbone, which specialises the batch dimension (so a symbolic batch would need
changes to shared code). Third attempt, kept: the round's batch is **padded** to a multiple of 64
(`SEARCH_BATCH_QUANTUM`) with copies of its first leaf and the outputs sliced back, so in practice only the two
shapes 64 and 256 (the forced round) ever compile and a partial last batch costs a few wasted rows instead of
~20 s. Measured: `arena_compile_pad`, below.

### 4.3 Arena-indexed states (`fused` impls, commit `3b4adaa`)
A fused search's state was (row, z_H view, z_L view): every round `torch.stack`ed 64-256 views and built K views
per leaf (256 to 1,024 small tensor objects) for the children. Now `_search` allocates two arenas
`[B*K*(sims+1), S, H]` per search (reused across searches: 40 MB each at the recipe's shapes and 16 sims) with the
root children written in; a state is (row, arena index); a round gathers its leaves with one `index_select` and
writes all its children with one copy, and child states are integers. With `search_compile` the compiled round's
outputs are written straight into the arena instead of being cloned. Memory is the same as before (the per-round
candidate tensors were kept alive by the node views anyway), just preallocated.
**Measured** (`results/arena_sudoku6u.json`, eager): training search depth 3 106.3 ms (static 109.8), eval 16 sims
depth 3 228.4 ms (248.8), "tree" 2-11 ms (3-13), full eval 49.9 s (50.6), accuracy 0.868, peak memory 265 MB
(564: the batch-256 forced round's candidate tensor is no longer held next to its views). A few percent on its own;
its point is the fixed-shape, no-clone round for the compiled path. **Kept.**

### 4.4 WU-UCT batching (`mcts_impl=wu`, `mcts_leaves_per_round=W`)
`models/mcts.py::WUMCTS`, from the user's third paper (Liu et al., ICLR 2020; arXiv 2006.08785 is its BU-UCT
follow-up). Each round selects W leaves per tree one after another; every selection adds an *unobserved* visit
along its path, and PUCT's exploration term uses N + O in both the parent's sqrt and the child's denominator, so the
next selection is steered away from paths whose value is still pending. The W x B leaves are then evaluated in one
domain call (one compiled replay at batch B x W) and the unobserved visits become real backups, in selection order.
Design choices: a leaf selected twice in a round is evaluated twice rather than de-duplicated, so every round has
the same batch size (the compiled round specialises on it) and the duplicate costs only idle GPU; the K forced
root visits stay one batched round; W = 1 reproduces the baseline exactly (the unobserved count of a path is always
released before the next selection reads it), which is what the contract tests' exact parity checks for `wu`; at
W > 1 the semantics change and the reference checkpoint's eval accuracy is the judge. Arena capacity gains one
round of slack (`B*K*(sims + W + 1)`).

**Measured, W = 2 with compile** (`results/wu2_compile_sudoku6u.json`, dynamic-batch build):

| mode | sims | depth | arena + compile ms | WU-2 ms | rounds | argmax agreement vs baseline |
|---|---|---|---|---|---|---|
| train | 8 | 3 | 65.4 | 57.8 | 5 -> 3 | 0.828 |
| train | 16 | 3 | 94.5 | 84.6 | 13 -> 7 | 0.953 |
| eval | 8 | 3 | 63.4 | 45.2 | 5 -> 3 | 0.922 |
| eval | 16 | 3 | 96.7 | 85.7 | 13 -> 7 | 0.984 |
| eval | 16 | 1 | 16.7 | 10.0 | 13 -> 7 | 1.000 |

Training 10.66 it/s. Full eval 45.5 s, exact accuracy **0.871** (baseline 0.865, fused 0.868): the batched
selections cost nothing measurable in accuracy at W = 2; `search/argmax_agree` 0.419 (0.464), i.e. the search
overrides the prior a little more often. The speedup is far below the halving of rounds, which is the important
finding: **the compiled round is now close to GPU-bound.** A depth-3 round at batch 64 costs ~6.5 ms (15 passes:
~0.4 ms per pass, i.e. ~2.4 GFLOP of small GEMMs at roughly 6 TFLOP/s, near what this laptop GPU sustains), and
the same round at batch 128 costs ~10 ms, so doubling the batch only amortises the fixed ~3 ms. Further speedups
therefore need fewer passes per search (algorithmic), not better batching: see 4.6.

**Measured, W = 4 with compile** (`results/wu4_compile_sudoku6u.json`): eval 16 sims depth 3 89.3 ms (13 rounds
-> 4, batch 256 each: ~18 ms a round, i.e. the GPU is saturated), train 8 sims depth 3 51.1 ms; full eval 42.0 s,
accuracy 0.870, `search/argmax_agree` 0.364 (the search overrides the prior noticeably more often than the
baseline's 0.464: with 4 leaves chosen blind per round the visit distribution is flatter). Verdict: **W = 2 is the
useful setting** (same accuracy, 10-30% off the search); W = 4 buys little more speed and visibly changes the
search's behaviour. Neither is the default; `mcts_leaves_per_round` stays 1 unless a user opts in.

### 4.5 C++ search forest (`mcts_impl=cpp`, commit `da64895`)
The user's first idea. `models/mcts_cpp/forest.cpp` (pybind11 through `torch.utils.cpp_extension.load`, built on
first use into `build/mcts_cpp`, needs g++ and ninja; `ninja` was added to the venv) holds every tree of a search
as one node array: parent, children range, prior, visits, value sum, unobserved count, depth left; `select(width)`
does the PUCT walks (with WU-UCT's unobserved counts, so it is also the `wu` search at W > 1), `backup(...)`
expands, releases unobserved visits, updates the per-tree min-max stats and backs up, all with the baseline's
arithmetic and tie rule. Python keeps the domain state of node i at index i (the arena index for TreeRM, a tuple in
the toy tests) and still makes one batched domain call per round, so the latents never cross into C++, which is
what keeps the implementation 150 lines and generic. Visit counts are the baseline's: it passes every exact
contract test (hand-computed traces, random-tree parity, posterior parity on the tiny model).

**Measured, C++ forest + WU-2 + compile** (`results/cpp_wu2_compile_sudoku6u.json`, padded build):

| mode | sims | depth | python tree, WU-2 ms (tree ms) | C++ forest, WU-2 ms (tree ms) |
|---|---|---|---|---|
| train | 8 | 3 | 57.8 (5) | 47.4 (2) |
| train | 16 | 3 | 84.6 (12) | 76.4 (3) |
| eval | 8 | 3 | 45.2 (6) | 43.1 (2) |
| eval | 16 | 3 | 85.7 (12) | 77.3 (3) |
| eval | 16 | 1 | 10.0 (4) | 6.8 (1) |

Full eval **37.7 s** (WU-2 with the Python tree: 45.5 s, part of that gap being the padded last batch), accuracy
0.871 and `search/argmax_agree` 0.419, both identical to the Python WU-2 run: the forest reproduces that search's
decisions on the real model. The tree's share of a search is now 1-3 ms. **Kept** (opt-in: `mcts_impl=cpp`).
**C++ forest + compile at W = 1** (`results/cpp_compile_sudoku6u.json`, exact semantics): eval 16 sims depth 3
93.9 ms (tree 3 ms), train 8 sims depth 3 50.1 ms (tree 2 ms); full eval 47.0 s, accuracy 0.868 and
`search/argmax_agree` 0.464, i.e. the baseline's decisions exactly.

### 4.5b The eval's partial last batch (harness finding, commit pending)
Padding the compiled round did not move the full eval (53.9 s) although 15 full batches take 27 s: the ~20 s goes
to the **outer** model, which `evaluate.py`/the bench compile with `dynamic=False`, recompiling the whole forward
for the 40-row last batch (1000 = 15 x 64 + 40). It is the same for every variant (and for the baseline: its 96 s
include it), so it hid the search speedups. The bench now evaluates full batches only by default (960 puzzles;
`--eval-steps all` restores the partial batch) and the kept variants are re-measured that way below.

### Memory (the user asked for it to be tracked)
Every bench section records peak GPU memory allocated; from here on also GPU reserved (the allocator's footprint,
what `nvidia-smi` shows) and the process's peak host RSS, and `compare.py` prints them. So far (eval section, peak
GPU allocated, MB): baseline 354 | fused 564 (the batch-256 forced round's candidate tensor held next to its
views) | arena 265 | arena + compile 266-270 | WU-2 274 | WU-4 283 | C++ 313 | C++ + WU-2 274. Training section:
282 (baseline, fused, arena) | 279-326 (compiled variants: CUDA-graph pools) | 364 (C++ + WU-2). The arenas are
`2 x B*K*(sims+W+1) x S x H` bf16 = 2 x 40 MB at the recipe's shapes with 16 sims, preallocated once and reused;
nothing here approaches the 4 GB card, but the numbers are tracked so a wider search (W, K, sims) or a larger
board (8x8 N-Queens: seq 64, batch 128 -> 2 x 142 MB arenas) stays visible.

### 4.7 A profiler artefact, caught by the harness
`profile_eval.py` (which synchronizes the GPU before and after every search to time it) showed the compiled round
2-7x slower inside the compiled outer model than in the micro bench (depth 1: 106 vs 15 ms; two batches 7.6 s
vs 3.0 s with the outer model eager), while the eager round was unaffected. Two uninstrumented full-batch evals
settle it the other way: outer model compiled 18.9 s, outer model eager 23.9 s. The forced synchronization
inside the outer compiled region was what made inductor's CUDA-graph trees stall (every replay then waits for the
whole queue), not the nesting itself. Lesson recorded: time sections with the uninstrumented protocol, use the
profiler only for the op mix. A `search_compile_mode=graphs` variant (inductor without graph trees + a plain
`torch.cuda.CUDAGraph` per round shape) was written during the investigation; inductor's default-mode code is
not stream-capturable as is ("operation not permitted when stream is capturing"), and with the artefact explained
there was nothing to gain, so it was removed. What the detour did measure: `search_compile_mode=default` (inductor
fusion, no CUDA graphs) costs 20-30% over `reduce-overhead` (train 8 sims depth 3 66 vs 50 ms, eval 16 sims
depth 3 113 vs 94 ms; `results/micro_compile_default.json`), i.e. the graph replay is worth that much on top of
the kernel fusion.

### 4.8 Skip the zero-scale noise draw in the block the search replays (`NoiselessBlocks`, commit `4440642`)
`Blocks.forward` (merged_arch.py) adds `randn_like(x) * noise_scale` on every pass although every config here
sets `noise_scale = 0`; inside the compiled round that is a philox kernel plus a wider fused epilogue per pass,
15 times a round. First version: a 4-line guard in the shared block. It broke `test_trm_root_memory_exact`
(TRMRoot's memory savers align their RNG streams on that draw), so the shared block is untouched and TreeRM
wraps the same layers in a `Blocks` subclass whose forward returns the mix directly at zero noise (same state-dict
keys; bitwise the same value, x + 0 is x). Search micro (`results/micro_noise_guard.json`, same code path): eval 16
sims depth 3 93.9 -> 81.6 ms (-13%), train 16 sims depth 3 89.2 -> 84.5, the depth-1 rows unchanged (no passes).
Full-batch eval: pending (the first measurement, 26.7 s, ran while the whole test directory was on the GPU and is
discarded).

### 4.6 What is left, and the next levers
With the round compiled the search is GPU-bound at ~0.45 ms per pass (batch 64), and the Python tree is a few ms.
In an eval batch the searches themselves are roughly 8 steps x (93 + 50 + 16) ms ~ 1.3 s, but a batch takes ~2.8 s
(45 s / 16): **about half of the eval wall-clock is outside the searches**, so `profile_eval.py` is next.
Candidates after that, in order of expected value:
1. **Subtree value memo across a step's cycles (exact).** The children of root child k* in cycle c's search are,
   up to bf16 noise, exactly cycle c+1's root children (same TRM update), with the same allowed depth and hence the
   same rollout values; keeping the forest/arena across cycles and re-rooting at k* (counts reset, values and
   expansions kept) makes cycle c+1's forced round free when k*'s children were visited and skips every repeated
   rollout. Applies only where consecutive searches happen, i.e. eval (training searches on the grad cycle alone).
   Estimate ~20% of eval search time.
2. **Inductor max-autotune** for the round's small GEMMs (K = 128) and epilogue fusion. Tried
   (`search_compile_mode=max-autotune`, `results/micro_autotune.json`): train 8 sims depth 3 58.5 ms vs 50.1,
   eval 16 sims depth 3 86.7 vs 93.9: noise either way, with a much slower first compile. The option stays
   (default `reduce-overhead`) but buys nothing here: cuBLAS already handles these GEMM shapes about as well.
3. **Semantics-changing eval knobs**, each judged on the reference checkpoint: cap the rollout length and let the
   value head finish early; fewer eval simulations at depth 3 (the one that costs); W = 2 batching.
