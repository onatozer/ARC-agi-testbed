"""Profiles one TreeRM search (torch.profiler) at the recipe's shapes and prints where the time goes, by op.
usage: venv/bin/python scripts/mcts_bench/profile_search.py [--impl python] [--mode train|eval] [--sims 8] [--depth 3] [--rows 30]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.profiler import ProfilerActivity, profile

import bench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", default="python")
    parser.add_argument("--mode", default="train", choices=["train", "inference"], help="inference = the model's eval mode (value-head rollouts)")
    parser.add_argument("--sims", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = bench.build_config("sudoku6u", args.impl, args.override)
    checkpoint = bench.find_checkpoint("sudoku6u", args.checkpoint)
    test_loader, _ = bench.eval_loader(config)
    _, train_meta = bench.train_loader(config)
    bench.seed_all(config.seed)
    state = bench.load_weights(bench.init_train_state(config, train_meta, rank=0, world_size=1), checkpoint, config)
    model = state.model
    model.train(args.mode == "train")
    inner = model.model.inner
    inner.config.mcts_simulations = args.sims
    inner.config.eval_mcts_simulations = args.sims
    labels = next(iter(test_loader))[1]["labels"].cuda()
    inputs = bench.search_inputs(inner, labels, seed=0)
    for _ in range(3):
        bench.run_search(inner, inputs, labels, args.depth)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        bench.run_search(inner, inputs, labels, args.depth)
        torch.cuda.synchronize()
    print(f"\n=== {args.impl} {args.mode} sims={args.sims} depth={args.depth}: by self CPU time ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.rows, max_name_column_width=60))
    print("\n=== by CUDA time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.rows, max_name_column_width=60))


if __name__ == "__main__":
    main()
