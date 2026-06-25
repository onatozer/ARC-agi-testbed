"""Trainer for TRMDEQ — TRM with the inner L-cycles loop replaced by Anderson
fixed-point iteration + phantom-gradient backward + Jacobian regulariser.

This is a sibling of ``train_trm.py`` (vanilla TRM trainer); it does NOT
modify any other file. The training loop, optimisers (AdamW + SignSGD for
puzzle embeddings), EMA, deep supervision via ACT carry, and eval halting are
identical -- only the model class and loss head differ.

Usage:
    torchrun --standalone --nproc_per_node=2 \\
        -m experiments.eqlm_sudoku.trm.train_trm_deq \\
        --data_dir /scratch1/feinashl/data/sudoku-extreme-1k-aug-1000 \\
        --out_dir  /scratch1/feinashl/eqlm_sudoku/run-trm-deq-30m \\
        --L_layers 8                       # bumps to ~30M params \\
        --deq_max_iter 8 --deq_min_iter 4 --deq_tol 1e-3 \\
        --bptt_through 2 --jacobian_reg_lambda 1e-3
"""
import json
import math
import os
import random
import socket
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from puzzle_dataset import (  # noqa: E402
    IGNORE_LABEL_ID,
    PuzzleDataset,
    PuzzleDatasetConfig,
)
from models.recursive_reasoning.trm_deq import TRMDEQ  # noqa: E402
from models.losses_deq import ACTLossHeadDEQ  # noqa: E402
from models.ema import EMAHelper  # noqa: E402
from models.sparse_embedding import (  # noqa: E402
    CastedSparseEmbeddingSignSGD_Distributed,
)


@dataclass
class TrainTRMDEQConfig:
    data_dir: str = ""
    out_dir: str = ""
    variant: str = "att"  # "att" only (mlp_t variant not supported here)
    evaluator: str = ""   # "arc" to enable ARC-AGI voting evaluator

    # Model arch (defaults are the ~30M HRM-scale config)
    H_cycles: int = 3
    L_cycles: int = 6   # used only as info; deq_max_iter takes over forward
    L_layers: int = 8   # 8 layers per L_level → ~28M params at hidden=512
    H_layers: int = 0
    hidden_size: int = 512
    expansion: float = 4.0
    num_heads: int = 8
    pos_encodings: str = "rope"
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True
    forward_dtype: str = "bfloat16"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # DEQ
    deq_inner: bool = True
    deq_max_iter: int = 8
    deq_min_iter: int = 4
    deq_tol: float = 1e-3
    deq_anderson_m: int = 5
    deq_anderson_beta: float = 1.0
    bptt_through: int = 2          # K last solver iterates with grad
    jacobian_reg_lambda: float = 1e-3
    jacobian_reg_n_samples: int = 1

    # Optim
    epochs: int = 50_000
    eval_interval: int = 5_000
    global_batch_size: int = 768
    seed: int = 0
    lr: float = 1e-4
    lr_min_ratio: float = 1.0
    lr_warmup_steps: int = 2_000
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 1.0
    puzzle_emb_lr: float = 1e-2
    puzzle_emb_weight_decay: float = 1.0

    ema: bool = True
    ema_rate: float = 0.999

    loss_type: str = "stablemax_cross_entropy"
    log_interval: int = 50
    keep_last_only: bool = True
    eval_max_batches: int = 200  # 0 = all test batches; default caps to ~200 for speed


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _rank_world():
    if _is_dist():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _seed_all(seed: int, rank: int):
    s = seed + rank
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def _build_dataloader(cfg: TrainTRMDEQConfig, split: str, rank: int, world: int) -> DataLoader:
    pcfg = PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=[cfg.data_dir],
        global_batch_size=cfg.global_batch_size,
        test_set_mode=(split == "test"),
        epochs_per_iter=cfg.eval_interval if split == "train" else 1,
        rank=rank,
        num_replicas=world,
    )
    ds = PuzzleDataset(pcfg, split=split)
    return DataLoader(
        ds, batch_size=None, num_workers=1, prefetch_factor=8,
        pin_memory=True, persistent_workers=True,
    )


def _make_model(cfg: TrainTRMDEQConfig, metadata, rank: int, world: int):
    cfg_dict = dict(
        batch_size=cfg.global_batch_size // world,
        seq_len=int(metadata.seq_len),
        vocab_size=int(metadata.vocab_size),
        num_puzzle_identifiers=int(metadata.num_puzzle_identifiers),
        H_cycles=cfg.H_cycles,
        L_cycles=cfg.L_cycles,
        H_layers=cfg.H_layers,
        L_layers=cfg.L_layers,
        hidden_size=cfg.hidden_size,
        expansion=cfg.expansion,
        num_heads=cfg.num_heads,
        pos_encodings=cfg.pos_encodings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        halt_max_steps=cfg.halt_max_steps,
        halt_exploration_prob=cfg.halt_exploration_prob,
        forward_dtype=cfg.forward_dtype,
        mlp_t=False,
        puzzle_emb_len=cfg.puzzle_emb_len,
        no_ACT_continue=cfg.no_ACT_continue,
        puzzle_emb_ndim=cfg.hidden_size,
        # DEQ-specific
        deq_inner=cfg.deq_inner,
        deq_max_iter=cfg.deq_max_iter,
        deq_min_iter=cfg.deq_min_iter,
        deq_tol=cfg.deq_tol,
        deq_anderson_m=cfg.deq_anderson_m,
        deq_anderson_beta=cfg.deq_anderson_beta,
        bptt_through=cfg.bptt_through,
        jacobian_reg_lambda=cfg.jacobian_reg_lambda,
        jacobian_reg_n_samples=cfg.jacobian_reg_n_samples,
    )
    with torch.device("cuda"):
        inner = TRMDEQ(cfg_dict)
        if rank == 0:
            print(inner)
        model = ACTLossHeadDEQ(
            inner, loss_type=cfg.loss_type,
            jacobian_reg_lambda=cfg.jacobian_reg_lambda,
        )
    return model, cfg_dict


def _build_optimizers(cfg: TrainTRMDEQConfig, model, world: int):
    inner = model.model  # ACTLossHeadDEQ -> TRMDEQ
    optim_signsgd = CastedSparseEmbeddingSignSGD_Distributed(
        inner.puzzle_emb.buffers(),
        lr=0.0,
        weight_decay=cfg.puzzle_emb_weight_decay,
        world_size=world,
    )
    optim_adamw = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        fused=torch.cuda.is_available(),
    )
    return [optim_signsgd, optim_adamw], [cfg.puzzle_emb_lr, cfg.lr]


def _cosine_warmup_lr(step: int, base: float, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return base * float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return base * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def _train_step(cfg: TrainTRMDEQConfig, model, batch, carry, optimizers,
                optimizer_lrs, step: int, total: int, world: int, rank: int):
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    if carry is None:
        with torch.device("cuda"):
            carry = (model.module if hasattr(model, "module") else model).initial_carry(batch)

    new_carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    ((1 / cfg.global_batch_size) * loss).backward()

    lr_this = None
    for opt, base in zip(optimizers, optimizer_lrs):
        lr_this = _cosine_warmup_lr(step, base, cfg.lr_warmup_steps, total, cfg.lr_min_ratio)
        for pg in opt.param_groups:
            pg["lr"] = lr_this
        opt.step()
        opt.zero_grad()

    reduced = None
    if metrics:
        keys = sorted(metrics.keys())
        vals = torch.stack([metrics[k] for k in keys])
        if world > 1:
            dist.reduce(vals, dst=0)
        if rank == 0:
            v = vals.cpu().numpy()
            d = {k: v[i] for i, k in enumerate(keys)}
            count = max(d.get("count", 1), 1)
            reduced = {
                f"train/{k}": (val / (cfg.global_batch_size if k.endswith("loss") else count))
                for k, val in d.items()
            }
            reduced["train/lr"] = lr_this

    # DEQ solver iters from inner module (rank 0 only is fine)
    inner = (model.module if hasattr(model, "module") else model).model.inner
    if hasattr(inner.inner_deq, "_last_info") and rank == 0 and reduced is not None:
        info = inner.inner_deq._last_info or {}
        reduced["train/deq_iters"] = float(info.get("iters", 0))
        reduced["train/deq_resid"] = float(info.get("rel_residual", 0.0))

    return new_carry, reduced


@torch.inference_mode()
def _eval(cfg: TrainTRMDEQConfig, model, eval_loader: DataLoader, world: int, rank: int,
          max_batches: int = 0, arc_evaluator=None):
    inner = model.module if hasattr(model, "module") else model
    max_steps = int(cfg.halt_max_steps)
    metric_sums: dict = {}
    n_batches = 0
    n_inference_steps_total = 0
    t0 = time.time()

    if arc_evaluator is not None:
        arc_evaluator.begin_eval()

    for set_name, batch, _ in eval_loader:
        n_batches += 1
        if max_batches > 0 and n_batches > max_batches:
            break
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.device("cuda"):
            carry = inner.initial_carry(batch)
        steps = 0
        while steps < max_steps:
            carry, _loss, metrics, _preds, all_finish = model(
                carry=carry, batch=batch,
                return_keys=["preds", "q_halt_logits"] if arc_evaluator is not None else [],
            )
            steps += 1
            if all_finish:
                break
        n_inference_steps_total += steps
        for k, v in metrics.items():
            metric_sums[k] = metric_sums.get(k, torch.zeros_like(v)) + v

        if arc_evaluator is not None and _preds:
            arc_evaluator.update_batch(batch, _preds)

        if rank == 0 and n_batches % 100 == 0:
            elapsed = time.time() - t0
            print(f"  eval batch {n_batches}  steps={steps}  elapsed={elapsed:.1f}s")
    if world > 1:
        for k, v in metric_sums.items():
            dist.all_reduce(v)
    out = {}
    if rank == 0 and metric_sums:
        count = max(float(metric_sums.get("count", torch.tensor(1.0))), 1.0)
        for k, v in metric_sums.items():
            out[f"eval/{k}"] = float(v.item() / count if k != "count" else v.item())
        out["eval/n_batches"] = n_batches
        out["eval/avg_inference_steps"] = n_inference_steps_total / max(n_batches, 1)
        out["eval/elapsed_s"] = time.time() - t0

    if arc_evaluator is not None:
        arc_results = arc_evaluator.result(
            save_path=str(Path(cfg.out_dir)) if rank == 0 else None,
            rank=rank, world_size=world,
        )
        if arc_results is not None:
            out.update({f"eval/{k}": v for k, v in arc_results.items()})

    return out


def main(cfg: TrainTRMDEQConfig):
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = _rank_world()
    _seed_all(cfg.seed, rank)
    out_dir = Path(cfg.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "train_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        print(f"[{time.ctime()}] host={socket.gethostname()} world={world}")
        print(f"  data_dir={cfg.data_dir}  out_dir={cfg.out_dir}")

    train_loader = _build_dataloader(cfg, "train", rank, world)
    test_loader = _build_dataloader(cfg, "test", rank, world)
    metadata = train_loader.dataset.metadata
    if rank == 0:
        print(f"  metadata: {metadata.model_dump()}")

    model, cfg_dict = _make_model(cfg, metadata, rank, world)
    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"  TRM-DEQ params: {n/1e6:.3f}M")
        print(f"  L_layers={cfg.L_layers}  H_cycles={cfg.H_cycles}  "
              f"DEQ inner: max_iter={cfg.deq_max_iter} min_iter={cfg.deq_min_iter} tol={cfg.deq_tol}")
        print(f"  bptt_through={cfg.bptt_through}  jacobian_reg_lambda={cfg.jacobian_reg_lambda}")
        print(f"  halt_max_steps={cfg.halt_max_steps}")
        with open(out_dir / "model_config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

    arc_evaluator = None
    if cfg.evaluator == "arc":
        from experiments.eqlm_sudoku.trm.arc_evaluator import ARCEvaluator
        arc_evaluator = ARCEvaluator(cfg.data_dir, metadata)
        if rank == 0:
            print(f"  ARC evaluator enabled: {len(arc_evaluator.test_puzzles)} test puzzles")

    if world > 1:
        with torch.no_grad():
            for p in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(p, src=0)

    optimizers, optimizer_lrs = _build_optimizers(cfg, model, world)

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ["LOCAL_RANK"])],
            find_unused_parameters=True, broadcast_buffers=False,
        )

    ema = EMAHelper(mu=cfg.ema_rate) if cfg.ema else None
    if ema is not None:
        ema.register(model)

    total = int(
        cfg.epochs * metadata.total_groups * metadata.mean_puzzle_examples
        / cfg.global_batch_size
    )
    if rank == 0:
        print(f"  total optim steps target = {total:,}")

    train_t0 = time.time()
    last_log_t = time.time()
    step = 0
    train_epochs_per_iter = cfg.eval_interval
    total_iters = max(cfg.epochs // train_epochs_per_iter, 1)
    carry = None

    for outer in range(total_iters):
        if rank == 0:
            print(f"[{time.ctime()}] Outer iter {outer}/{total_iters} (epoch={outer * train_epochs_per_iter})")
        model.train()
        for _set, batch, _ in train_loader:
            carry, reduced = _train_step(cfg, model, batch, carry, optimizers,
                                         optimizer_lrs, step, total, world, rank)
            if ema is not None:
                ema.update(model)

            step += 1
            if rank == 0 and step % cfg.log_interval == 0:
                dt = time.time() - last_log_t
                last_log_t = time.time()
                elapsed = time.time() - train_t0
                spsec = step / max(elapsed, 1e-6)
                eta_s = max(total - step, 0) / max(spsec, 1e-6)
                eta_h, eta_m = divmod(int(eta_s), 3600)
                eta_m = eta_m // 60
                msg = f"  step={step:>7d}"
                if reduced:
                    for k in ("train/lm_loss", "train/q_halt_loss",
                              "train/exact_accuracy", "train/accuracy",
                              "train/jacobian_reg", "train/deq_iters",
                              "train/deq_resid", "train/lr"):
                        if k in reduced:
                            tag = k.split("/")[-1]
                            msg += f"  {tag}={reduced[k]:.4f}"
                msg += f"  dt/{cfg.log_interval}={dt:.2f}s eta={eta_h:d}h{eta_m:02d}m"
                print(msg)

        # End of outer iter -> eval (with EMA copy)
        if ema is not None:
            saved = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
            ema.ema(model)
        try:
            metrics = _eval(cfg, model, test_loader, world, rank,
                           max_batches=cfg.eval_max_batches,
                           arc_evaluator=arc_evaluator)
        finally:
            if ema is not None:
                for n, p in model.named_parameters():
                    if n in saved:
                        p.data.copy_(saved[n])

        if rank == 0:
            print(f"[{time.ctime()}] EVAL step={step}  metrics={metrics}")
            with open(out_dir / "eval_log.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "epoch": outer * train_epochs_per_iter, **metrics}) + "\n")
            inner = model.module if hasattr(model, "module") else model
            ckpt_path = out_dir / ("last.pt" if cfg.keep_last_only else f"step-{step:08d}.pt")
            payload = {
                "step": step,
                "model_state": inner.state_dict(),
                "ema_state": ema.state_dict() if ema is not None else None,
                "model_config": cfg_dict,
                "train_config": asdict(cfg),
                "metrics": metrics,
            }
            torch.save(payload, ckpt_path)
            print(f"  checkpoint -> {ckpt_path}")

    if _is_dist():
        dist.destroy_process_group()
    if rank == 0:
        print(f"[{time.ctime()}] training complete.")


if __name__ == "__main__":
    from jsonargparse import CLI
    cfg: TrainTRMDEQConfig = CLI(TrainTRMDEQConfig, as_positional=False)  # type: ignore[arg-type]
    main(cfg)
