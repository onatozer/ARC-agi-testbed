"""Benchmarks one MCTS implementation (arch.mcts_impl) of TreeRM on a small environment, with the autoresearch recipe:
  micro  the search alone (TreeRMInner._search) at the recipe's shapes, train and eval mode, with a wall-clock
         breakdown into the domain callbacks (rollout = evaluate_fn, expand = expand_fn) and the tree itself, plus the
         posterior's parity with the baseline 'python' search on the same inputs;
  train  training throughput (it/s) of the real train step, compiled as pretrain.py compiles it;
  eval   wall-clock and accuracy of a test-set eval from a checkpoint (the reference checkpoints of REFERENCE.md).
The whole run holds an exclusive lock on the GPU (gpu.lock next to the reference runs) so two benchmarks never overlap.

usage (from the repo root):
  venv/bin/python scripts/mcts_bench/bench.py --impl python --env sudoku6u --checkpoint auto --out scripts/mcts_bench/results/python.json
  venv/bin/python scripts/mcts_bench/compare.py scripts/mcts_bench/results/baseline_python.json scripts/mcts_bench/results/python.json
"""
import argparse
import fcntl
import glob
import json
import os
import random
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from config.schema import PretrainConfig
from evaluate import _summarize
from evaluators.eval_fn import evaluate
from models.mcts import MCTS_IMPLS
from pretrain import create_dataloader, create_evaluators, init_train_state, train_batch
from utils.checkpoint import load_ema, load_model_state_dict

RUNS = Path(os.environ.get("MCTS_BENCH_RUNS", Path.home() / "ARC-agi-testbed/autoresearch_runs/mcts_bench"))
ENVS = {
    "sudoku6u": ("cfg_pretrain", [f"dataset.data_path={REPO}/data/sudoku-6u", "dataset.grid_size=6", "dataset.name=Sudoku-6u"], "all/exact_accuracy"),
    "nqueens6": ("cfg_pretrain_nqueens", [f"dataset.data_path={REPO}/data/nqueens-6", "dataset.board_size=6"], "NQueens/accuracy"),
}
# The autoresearch recipe (REPORTS.md section 0); keep in sync with train_reference.sh
RECIPE = [
    "arch=treerm", "arch.hidden_size=128", "arch.num_heads=4", "arch.L_layers=2", "arch.H_cycles=3", "arch.L_cycles=4",
    "arch.halt_max_steps=8", "arch.candidate_mode=flags", "arch.candidate_loss=none", "lr=1e-3",
    "lr_warmup_steps=50", "global_batch_size=64", "epochs=100000", "train_epochs_per_iter=100000",
    "checkpoint_interval_steps=null", "heavy_metrics_log_interval=100000", "steps_hist_log_interval_steps=100000",
    "flops_log_interval_steps=100000", "grad_norm_log_interval_steps=100000", "seed=0", "+wandb_mode=disabled",
]


def build_config(env: str, impl: str, overrides: List[str]) -> PretrainConfig:
    cfg_name, dataset, _ = ENVS[env]
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        hydra_cfg = compose(config_name=cfg_name, overrides=[*dataset, *RECIPE, f"arch.mcts_impl={impl}", *overrides])
    config = PretrainConfig(**OmegaConf.to_container(hydra_cfg, resolve=True))
    if config.seed is None:
        config.seed = 0
    return config


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def find_checkpoint(env: str, spec: Optional[str]) -> Optional[str]:
    if spec in (None, "", "none"):
        return None
    if spec != "auto":
        return str(Path(spec).expanduser())
    paths = sorted(glob.glob(str(RUNS / f"ref_{env}" / "**" / "*.pth"), recursive=True), key=os.path.getmtime)
    if not paths:
        raise FileNotFoundError(f"no reference checkpoint under {RUNS / f'ref_{env}'}; run train_reference.sh {env}")
    return paths[-1]


def train_loader(config: PretrainConfig):
    # A fresh loader per use: the shared sampler state (for resuming training) makes a second iterator start past the epoch
    return create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1, enable_shared_sampler_state=False)


def eval_loader(config: PretrainConfig):
    return create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1)


def load_weights(state, checkpoint: Optional[str], config: PretrainConfig):
    """Loads the checkpoint's (EMA) weights into the train state's model, as evaluate.py does."""
    if checkpoint is None:
        return state
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=False)
    load_model_state_dict(state.model, ckpt.get("model", ckpt), strict=False, assign=True)
    state.step = int(ckpt.get("step", 0))
    return load_ema(ckpt, state, config, 0) if config.ema and "ema" in ckpt else state


def memory_mb() -> Dict[str, float]:
    """Peak memory since the last reset_peak_memory_stats: GPU allocated (tensors) and reserved (the caching allocator's
    footprint, what nvidia-smi sees), plus the process's peak host RSS (never resets: the high-water mark so far)."""
    import resource

    return {"gpu_alloc": torch.cuda.max_memory_allocated() / 2**20, "gpu_reserved": torch.cuda.max_memory_reserved() / 2**20,
            "host_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10}


class Timed:
    """Wraps a search callback: accumulates its synchronized wall-clock, calls and states."""

    def __init__(self, fn):
        self.fn, self.seconds, self.calls, self.states = fn, 0.0, 0, 0

    def __call__(self, states, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.fn(states, *args, **kwargs)
        torch.cuda.synchronize()
        self.seconds += time.perf_counter() - start
        self.calls += 1
        self.states += len(states)
        return out


def search_inputs(inner, labels: torch.Tensor, seed: int):
    torch.manual_seed(seed)
    shape = (labels.shape[0], inner.config.seq_len, inner.config.hidden_size)
    u_t, z_L, x = (torch.randn(shape, device="cuda", dtype=inner.forward_dtype) for _ in range(3))
    with torch.no_grad():
        cand = inner._candidates(z_L, u_t)
        prior = inner.candidate_scorer(u_t, cand, inner._cos_sin())
    return cand, prior, z_L, x


def run_search(inner, inputs, labels, depth: int):
    cand, prior, z_L, x = inputs
    with torch.no_grad():
        return inner._search(cand, prior, z_L, x, labels, [depth] * cand.shape[0])


def bench_micro(model, labels: torch.Tensor, impl: str, sims_list: List[int], depths: List[int], repeats: int, warmup: int) -> List[Dict[str, Any]]:
    inner = model.model.inner
    rows = []
    for training in (True, False):
        model.train(training)
        inputs = search_inputs(inner, labels, seed=0)
        for sims in sims_list:
            inner.config.mcts_simulations = sims
            inner.config.eval_mcts_simulations = sims
            for depth in depths:
                for _ in range(warmup):
                    run_search(inner, inputs, labels, depth)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    posterior = run_search(inner, inputs, labels, depth)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
                peak_mb = torch.cuda.max_memory_allocated() / 2**20

                # Breakdown (instrumented, so its total is a little above the uninstrumented median): the baseline's two
                # callbacks, the fused impl's one step callback, and "tree" = everything else (selection, backprop, syncs)
                inner._rollout_values, inner._expand, inner._step = Timed(inner._rollout_values), Timed(inner._expand), Timed(inner._step)
                torch.cuda.synchronize()
                start = time.perf_counter()
                run_search(inner, inputs, labels, depth)
                torch.cuda.synchronize()
                total = time.perf_counter() - start
                rollout, expand, step = inner._rollout_values, inner._expand, inner._step
                del inner._rollout_values, inner._expand, inner._step

                domain = rollout.seconds + expand.seconds + step.seconds
                row = {
                    "mode": "train" if training else "eval", "sims": sims, "depth": depth, "batch": labels.shape[0],
                    "ms_median": 1e3 * statistics.median(times), "ms_min": 1e3 * min(times), "peak_mem_mb": peak_mb, "memory_mb": memory_mb(),
                    "breakdown_ms": {"rollout": 1e3 * rollout.seconds, "expand": 1e3 * expand.seconds, "step": 1e3 * step.seconds, "tree": 1e3 * (total - domain)},
                    "calls": {"rollout": rollout.calls, "expand": expand.calls, "step": step.calls},
                    "states": {"rollout": rollout.states, "expand": expand.states, "step": step.states},
                }
                if impl != "python":
                    # Parity with the baseline on the same inputs (exact impls must agree; the contract tests check it too)
                    own = inner.mcts
                    inner.mcts = MCTS_IMPLS["python"](num_simulations=sims, c_puct=inner.config.mcts_c_puct)
                    reference = run_search(inner, inputs, labels, depth)
                    inner.mcts = own
                    row["parity"] = {
                        "max_abs_diff": float((posterior - reference).abs().max()),
                        "argmax_agree": float((posterior.argmax(-1) == reference.argmax(-1)).float().mean()),
                    }
                rows.append(row)
                print(f"micro {row['mode']:5s} sims={sims:2d} depth={depth}: {row['ms_median']:8.1f} ms  "
                      f"(rollout {row['breakdown_ms']['rollout']:.0f} / expand {row['breakdown_ms']['expand']:.0f} / step {row['breakdown_ms']['step']:.0f} / tree {row['breakdown_ms']['tree']:.0f})"
                      + (f"  parity argmax {row['parity']['argmax_agree']:.3f}" if "parity" in row else ""), flush=True)
    return rows


def bench_train(config: PretrainConfig, steps: int, warmup: int, compile_: bool) -> Dict[str, Any]:
    config = config.model_copy(deep=True)
    config.max_steps = steps + warmup + 10
    seed_all(config.seed)
    loader, train_meta = train_loader(config)
    state = init_train_state(config, train_meta, rank=0, world_size=1)
    if compile_:
        state.model = torch.compile(state.model, dynamic=False)
        if state.ema_helper is not None:
            state.ema_helper.register(state.model)
    state.model.train()
    times, batch_size = [], None
    for _, batch, global_batch_size in loader:
        batch_size = global_batch_size
        torch.cuda.synchronize()
        start = time.perf_counter()
        train_batch(config, state, batch, global_batch_size, rank=0, world_size=1)
        if state.ema_helper is not None:
            state.ema_helper.update(state.model)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        if len(times) == warmup:
            torch.cuda.reset_peak_memory_stats()
        if len(times) >= warmup + steps:
            break
    timed = times[warmup:]
    mean = statistics.mean(timed)
    result = {"steps": len(timed), "warmup": warmup, "compiled": compile_, "batch": batch_size, "step_ms_median": 1e3 * statistics.median(timed),
              "step_ms_mean": 1e3 * mean, "it_s": 1 / mean, "samples_s": batch_size / mean, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "memory_mb": memory_mb()}
    print(f"train: {result['it_s']:.2f} it/s ({result['step_ms_median']:.0f} ms/step median, {result['steps']} steps, peak {result['peak_mem_mb']:.0f} MB alloc / "
          f"{result['memory_mb']['gpu_reserved']:.0f} MB reserved, host RSS {result['memory_mb']['host_rss']:.0f} MB)", flush=True)
    return result


def bench_eval(config: PretrainConfig, train_meta, eval_loader, eval_meta, checkpoint: Optional[str], max_eval_steps: Optional[int], compile_: bool, metric: str) -> Dict[str, Any]:
    config = config.model_copy(deep=True)
    config.arch.loss.name = config.arch.loss.inference_name
    seed_all(config.seed)
    state = load_weights(init_train_state(config, train_meta, rank=0, world_size=1), checkpoint, config)
    state.model.eval()
    if compile_:
        state.model = torch.compile(state.model, dynamic=False)
    evaluators = create_evaluators(config, eval_meta)
    if max_eval_steps is None:
        # Full batches only: the eval harness compiles the whole model with static shapes, so the test set's partial last
        # batch (1000 = 15 x 64 + 40) costs ~20 s of recompilation that has nothing to do with the search
        max_eval_steps = eval_meta.total_groups // config.global_batch_size
    evaluate(config, state, eval_loader, eval_meta, evaluators, rank=0, world_size=1, cpu_group=None, max_eval_steps=1)  # warmup (compile)
    seed_all(config.seed)  # the eval samples from the posterior, so its draws are seeded for repeatability
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    metrics, elapsed = evaluate(config, state, eval_loader, eval_meta, evaluators, rank=0, world_size=1, cpu_group=None, max_eval_steps=max_eval_steps)
    _, flat = _summarize(metrics or {})
    result = {"checkpoint": checkpoint, "trained": checkpoint is not None, "step": state.step, "max_eval_steps": max_eval_steps, "compiled": compile_,
              "seconds": elapsed, "puzzles": flat.get("all/count", flat.get("count")), "metric": metric, "accuracy": flat.get(metric),
              "metrics": {k: v for k, v in flat.items() if k.split("/")[-1] in ("exact_accuracy", "accuracy", "count", "lm_loss", "token_accuracy") or k.startswith("search/") or "search/" in k},
              "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "memory_mb": memory_mb()}
    print(f"eval: {elapsed:.1f} s, {metric} = {result['accuracy']}, peak {result['peak_mem_mb']:.0f} MB alloc / {result['memory_mb']['gpu_reserved']:.0f} MB reserved, "
          f"host RSS {result['memory_mb']['host_rss']:.0f} MB" + ("" if checkpoint else " (untrained weights)"), flush=True)
    return result


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", default="python", choices=sorted(MCTS_IMPLS))
    parser.add_argument("--env", default="sudoku6u", choices=sorted(ENVS))
    parser.add_argument("--checkpoint", default="auto", help="path, 'auto' (newest under the env's reference run) or 'none'")
    parser.add_argument("--out", default=None, help="JSON result path (default scripts/mcts_bench/results/<impl>_<env>.json)")
    parser.add_argument("--label", default=None, help="free-text label stored in the result")
    parser.add_argument("--skip", default="", help="comma-separated sections to skip: micro,train,eval")
    parser.add_argument("--sims", default="8,16")
    parser.add_argument("--depths", default="1,3")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--train-warmup", type=int, default=15)
    parser.add_argument("--eval-steps", type=lambda s: None if s == "all" else int(s), default=None,
                        help="eval batches; default: every full batch of the test set (a partial last batch recompiles the outer model: ~20 s); 'all' includes it")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--override", action="append", default=[], help="extra hydra override (repeatable)")
    args = parser.parse_args()
    skip = {s for s in args.skip.split(",") if s}
    out = Path(args.out) if args.out else REPO / "scripts/mcts_bench/results" / f"{args.impl}_{args.env}.json"

    torch.set_float32_matmul_precision("high")
    config = build_config(args.env, args.impl, args.override)
    metric = ENVS[args.env][2]
    checkpoint = find_checkpoint(args.env, args.checkpoint) if "eval" not in skip else None
    result: Dict[str, Any] = {
        "impl": args.impl, "env": args.env, "label": args.label, "commit": git_commit(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0), "overrides": args.override, "config": {"K": config.arch.num_candidates, "hidden": config.arch.hidden_size,
        "H_cycles": config.arch.H_cycles, "L_cycles": config.arch.L_cycles, "batch": config.global_batch_size, "train_sims": config.arch.mcts_simulations,
        "eval_sims": config.arch.eval_mcts_simulations},
    }
    with gpu_lock(RUNS / "gpu.lock"):
        test_loader, eval_meta = eval_loader(config)
        _, train_meta = train_loader(config)
        if "micro" not in skip:
            seed_all(config.seed)
            state = load_weights(init_train_state(config, train_meta, rank=0, world_size=1), checkpoint, config)
            labels = next(iter(test_loader))[1]["labels"].cuda()
            result["micro"] = bench_micro(state.model, labels, args.impl, [int(s) for s in args.sims.split(",")], [int(d) for d in args.depths.split(",")], args.repeats, args.warmup)
            del state
            torch.cuda.empty_cache()
        if "train" not in skip:
            result["train"] = bench_train(config, args.train_steps, args.train_warmup, not args.no_compile)
            torch.cuda.empty_cache()
        if "eval" not in skip:
            result["eval"] = bench_eval(config, train_meta, test_loader, eval_meta, checkpoint, args.eval_steps, not args.no_compile, metric)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
