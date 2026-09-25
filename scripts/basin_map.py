"""Basin maps (Lai et al., "Fractal basins trap latent reasoning") for a trained run.

Example:
    python scripts/basin_map.py --run-dir outputs/<id>/<date>/<time> --puzzles 0,1,2 --grid 64 --span 1.0

Writes <out-dir>/basin_step<step>_ex<idx>.png and .json per puzzle. Run from the repo root so relative dataset
paths in the saved config (e.g. data/...) resolve.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from omegaconf import OmegaConf

from config.schema import PretrainConfig
from dataset.common import PuzzleDatasetMetadata
from models.ema import EMAHelper
from models.losses import IGNORE_LABEL_ID
from utils.checkpoint import load_model_state_dict
from utils.dynamics.basin import basin_map, plot_basin
from utils.dynamics.probe import unwrap_inner


def latest_checkpoint(run_dir: Path) -> Path:
    ckpts = list((run_dir / "checkpoints").glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"no *.pth in {run_dir / 'checkpoints'}")
    step = lambda p: int(m.group(1)) if (m := re.match(r"step_(\d+)", p.name)) else -1
    return max(ckpts, key=lambda p: (step(p), p.stat().st_mtime))


def load_config(run_dir: Path, ckpt: dict) -> PretrainConfig:
    """Same precedence as evaluate.py: the config stored in the checkpoint, else run_dir/all_config.yaml."""
    cfg = ckpt.get("config")
    if cfg is None:
        cfg = OmegaConf.to_container(OmegaConf.load(run_dir / "all_config.yaml"), resolve=True)
    config = PretrainConfig(**cfg)
    path = os.path.expandvars(os.path.expanduser(config.dataset.data_path))
    config.dataset.data_path = path if os.path.isabs(path) else os.path.abspath(path)
    return config


def load_metadata(data_path: str, split: str) -> PuzzleDatasetMetadata:
    with open(os.path.join(data_path, split, "dataset.json")) as f:
        return PuzzleDatasetMetadata(**json.load(f))


def build_model(config: PretrainConfig, ckpt: dict, use_ema: bool, device: torch.device) -> torch.nn.Module:
    from pretrain import create_model  # heavy import (wandb/hydra); keep it out of --help

    # The model is built from the *train* split metadata exactly as in training (vocab, seq_len, #puzzle ids).
    train_meta = load_metadata(config.dataset.data_path, "train")
    model, _, _ = create_model(config, train_meta, rank=0, world_size=1)
    load_model_state_dict(model, ckpt["model"], strict=bool(config.load_strict), assign=True)

    ema = ckpt.get("ema")
    if use_ema and ema:
        # Mirrors utils.checkpoint.load_ema, but applies the shadow in place instead of deep-copying a TrainState.
        shadow = ema.get("shadow")
        if shadow is None:
            shadow = {k: v for k, v in ema.items() if isinstance(v, torch.Tensor)}
        shadow = {
            k.replace("_orig_mod.", ""): v.to(device)
            for k, v in shadow.items()
            if not (k.endswith("puzzle_emb.local_weights") or k.endswith("puzzle_emb.local_ids"))
        }
        if shadow:
            helper = EMAHelper(mu=ema.get("mu", config.ema_rate))
            helper.load_state_dict(shadow)
            helper.ema(model)
            print(f"applied EMA weights ({len(shadow)} tensors)")
    return model.to(device).eval()


def load_test_example(data_path: str, idx: int, device: torch.device) -> dict:
    """Example `idx` of the test split, counting through sets in metadata order (same order as the test loader)."""
    meta = load_metadata(data_path, "test")
    offset = idx
    for set_name in meta.sets:
        load = lambda f, mm="r": np.load(os.path.join(data_path, "test", f"{set_name}__{f}.npy"), mmap_mode=mm)
        inputs = load("inputs")
        if offset < len(inputs):
            puzzle_indices = load("puzzle_indices", None)
            pid = int(np.searchsorted(puzzle_indices, offset, side="right") - 1)
            labels = np.array(load("labels")[offset : offset + 1], dtype=np.int32)
            if meta.ignore_label_id is not None:
                labels[labels == meta.ignore_label_id] = IGNORE_LABEL_ID
            row = {
                "inputs": np.array(inputs[offset : offset + 1], dtype=np.int32),
                "labels": labels,
                "puzzle_identifiers": np.array(load("puzzle_identifiers", None)[[pid]], dtype=np.int32),
            }
            return {k: torch.from_numpy(v).to(device) for k, v in row.items()}
        offset -= len(inputs)
    raise IndexError(f"test example {idx} out of range")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path, help="run dir containing all_config.yaml and checkpoints/")
    p.add_argument("--checkpoint", type=Path, default=None, help="default: highest step_*.pth in run_dir/checkpoints")
    p.add_argument("--puzzles", default="0", help="comma-separated test-set example indices")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--span", type=float, default=1.0, help="half-width of the slice, in per-coordinate RMS units")
    p.add_argument("--depth", type=int, default=None, help="outer steps per rollout (default 4 * halt_max_steps)")
    p.add_argument("--chunk", type=int, default=256, help="pixels per forward batch (lower if OOM)")
    p.add_argument("--box", type=int, default=5, help="box size for basin entropy")
    p.add_argument("--seed", type=int, default=0, help="seed for the slice directions (and random z0)")
    p.add_argument("--random-z0", action="store_true", help="centre the slice on a random Gaussian latent")
    p.add_argument("--no-ema", action="store_true", help="use raw weights even if the checkpoint stores EMA")
    p.add_argument("--out-dir", type=Path, default=None, help="default: run_dir/basin_maps")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or latest_checkpoint(args.run_dir)
    print(f"loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = load_config(args.run_dir, ckpt)
    use_ema = not args.no_ema and bool(config.ema)
    model = build_model(config, ckpt, use_ema, device)
    inner = unwrap_inner(model)
    depth = args.depth or 4 * int(inner.config.halt_max_steps)
    step = int(ckpt.get("step", 0))
    out_dir = args.out_dir or args.run_dir / "basin_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    for idx in (int(s) for s in args.puzzles.split(",") if s.strip()):
        row = load_test_example(config.dataset.data_path, idx, device)
        t0 = time.time()
        r = basin_map(
            inner, row, grid=args.grid, span=args.span, depth=depth, chunk=args.chunk, seed=args.seed,
            random_z0=args.random_z0, box=args.box,
        )
        elapsed = time.time() - t0
        name = f"basin_step{step}_ex{idx}"
        fig = plot_basin(r, f"{config.run_name or args.run_dir.name} step {step} | test ex {idx} | D={depth}")
        fig.savefig(out_dir / f"{name}.png", dpi=130)
        plt.close(fig)
        meta = {
            "checkpoint": str(ckpt_path), "step": step, "example": idx, "seed": args.seed, "random_z0": args.random_z0,
            "ema": use_ema, "seconds": elapsed,
        }
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump({**meta, **{k: np.asarray(v).tolist() for k, v in r.items()}}, f)
        print(
            f"ex {idx}: S_b={r['basin_entropy']:.3f} S_bb={r['boundary_basin_entropy']:.3f} "
            f"alpha={r['uncertainty_exponent']:.3f} correct={r['correct'].mean():.2%} capped={r['capped'].mean():.2%} "
            f"({elapsed:.1f}s) -> {out_dir / name}.png"
        )


if __name__ == "__main__":
    main()
