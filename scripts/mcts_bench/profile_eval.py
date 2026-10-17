"""Where does one eval batch go? Times a full inference pass (all supervision steps) over one test batch from the
reference checkpoint, split into: inside TreeRMInner._search (the searches, by remaining depth), the rest of the
model forward, and the eval loop's bookkeeping; then the torch.profiler top ops of one batch.
usage: venv/bin/python scripts/mcts_bench/profile_eval.py [--impl fused] [--override arch.search_compile=true] [--rows 25]"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.profiler import ProfilerActivity, profile

import bench
from evaluators.eval_fn import evaluate
from pretrain import create_evaluators


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", default="fused")
    parser.add_argument("--rows", type=int, default=25)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = bench.build_config("sudoku6u", args.impl, args.override)
    config.arch.loss.name = config.arch.loss.inference_name
    checkpoint = bench.find_checkpoint("sudoku6u", args.checkpoint)
    test_loader, eval_meta = bench.eval_loader(config)
    _, train_meta = bench.train_loader(config)
    bench.seed_all(config.seed)
    state = bench.load_weights(bench.init_train_state(config, train_meta, rank=0, world_size=1), checkpoint, config)
    state.model.eval()
    inner = state.model.model.inner
    if not args.no_compile:
        state.model = torch.compile(state.model, dynamic=False)
    evaluators = create_evaluators(config, eval_meta)

    # Instrument the search: wall-clock per remaining depth (synchronized), and the model call as a whole
    search_time = defaultdict(float)
    search_calls = defaultdict(int)
    original_search = inner._search

    def timed_search(cand, prior, z_L, x, y, remaining):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = original_search(cand, prior, z_L, x, y, remaining)
        torch.cuda.synchronize()
        search_time[max(remaining)] += time.perf_counter() - start
        search_calls[max(remaining)] += 1
        return out

    inner._search = timed_search

    evaluate(config, state, test_loader, eval_meta, evaluators, rank=0, world_size=1, cpu_group=None, max_eval_steps=1)  # warmup
    search_time.clear(); search_calls.clear()
    torch.cuda.synchronize()
    start = time.perf_counter()
    metrics, elapsed = evaluate(config, state, test_loader, eval_meta, evaluators, rank=0, world_size=1, cpu_group=None, max_eval_steps=2)
    torch.cuda.synchronize()
    total = time.perf_counter() - start
    searched = sum(search_time.values())
    print(f"\n2 eval batches: {total:.2f} s total ({elapsed:.2f} s inside evaluate), searches {searched:.2f} s = {100 * searched / total:.0f}%")
    for depth in sorted(search_time):
        print(f"  remaining depth {depth}: {search_calls[depth]} searches, {1e3 * search_time[depth] / search_calls[depth]:.1f} ms each, {search_time[depth]:.2f} s")
    print(f"  everything else: {total - searched:.2f} s")

    inner._search = original_search
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        evaluate(config, state, test_loader, eval_meta, evaluators, rank=0, world_size=1, cpu_group=None, max_eval_steps=1)
        torch.cuda.synchronize()
    print("\n=== one eval batch, by self CPU time ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.rows, max_name_column_width=70))
    print("\n=== by CUDA time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.rows, max_name_column_width=70))


if __name__ == "__main__":
    main()
