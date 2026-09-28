"""Per-phase GPU memory profile of TRMRoot training steps, extrapolated to large batches.

Builds the model from config/arch/trm_root.yaml (via the cfg_pretrain Hydra config), runs a few training steps on
random Sudoku-shaped batches (seq_len 81, vocab 11) and reports, for the last step:

  P0  allocated before the forward (weights, optimizer state, carry)
  P1  peak during the Anderson solves
  P2  peak during the Jacobian block, and allocated right after it (graph it leaves alive)
  P3  peak over the whole forward, and allocated after it (graphs alive going into backward)
  P4  peak during backward
  reserved  max_memory_reserved over the step (allocator footprint, what nvidia-smi shows)

Each quantity is fitted as a + b*B over the batch sizes run and extrapolated to --extrapolate.

Usage (from the repo root):
  python scripts/profile_trm_root_memory.py --batches 8 16 24 32
  python scripts/profile_trm_root_memory.py --batches 8 16 --set jac_two_pass=true checkpoint_phantom=true
  python scripts/profile_trm_root_memory.py --batches 16 --compile --flops
"""
import argparse
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.losses.loss_heads import ACTLossHeadRoot  # noqa: E402
from models.recursive_reasoning.trm_root import TRMRoot  # noqa: E402
from utils.metrics import TotalFlopCounter  # noqa: E402

GB = 1024 ** 3
SEQ_LEN, VOCAB = 81, 11


def build(batch_size: int, jac_lambda: float, overrides: list[str]):
    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose("cfg_pretrain", overrides=["arch=trm_root", f"arch.loss.jacobian_reg_lambda={jac_lambda}"]
                      + [f"arch.{o}" for o in overrides])
    arch = OmegaConf.to_container(cfg.arch, resolve=True)
    loss_cfg = arch.pop("loss")
    for key in ("name", "short_name"):
        arch.pop(key, None)
    model_cfg = dict(arch, batch_size=batch_size, vocab_size=VOCAB, seq_len=SEQ_LEN, num_puzzle_identifiers=1, causal=False)
    with torch.device("cuda"):
        model = ACTLossHeadRoot(TRMRoot(model_cfg), loss_type=loss_cfg["loss_type"],
                                jacobian_reg_lambda=loss_cfg.get("jacobian_reg_lambda", 0.0))
    return model


class Probe:
    """Tracks the running peak across resets, so phase peaks and the overall peak can both be read."""

    def __init__(self):
        self.running_peak = 0
        self.phase = {}

    def fold(self):
        self.running_peak = max(self.running_peak, torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()

    def wrap(self, obj, name, key):
        original = getattr(obj, name)

        @torch.compiler.disable
        def wrapped(*args, **kwargs):
            self.fold()
            out = original(*args, **kwargs)
            peak = torch.cuda.max_memory_allocated()
            prev_peak, prev_after = self.phase.get(key, (0, 0))
            self.phase[key] = (max(prev_peak, peak), torch.cuda.memory_allocated())
            self.fold()
            return out

        setattr(obj, name, wrapped)


def run(batch_size: int, args) -> dict:
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    model = build(batch_size, args.jac_lambda, args.set)
    model.train()
    inner = model.model.inner
    dtypes = []
    inner.L_level.register_forward_hook(lambda m, inp, out: dtypes.append((str(inp[0].dtype), torch.is_grad_enabled())))
    probe = Probe()
    probe.wrap(inner, "_root_solve", "anderson")
    probe.wrap(inner, "_jacobian_regularizer", "jacobian")
    net = torch.compile(model, dynamic=False) if args.compile else model
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    g = torch.Generator(device="cuda").manual_seed(0)
    batch = {
        "inputs": torch.randint(1, VOCAB, (batch_size, SEQ_LEN), device="cuda", generator=g, dtype=torch.int32),
        "labels": torch.randint(1, VOCAB, (batch_size, SEQ_LEN), device="cuda", generator=g, dtype=torch.int32),
        "puzzle_identifiers": torch.zeros(batch_size, device="cuda", dtype=torch.int32),
    }
    carry = model.initial_carry(batch)
    for step in range(args.steps):
        last = step == args.steps - 1
        opt.zero_grad(set_to_none=True)
        if last:
            dtypes.clear()
            probe.phase.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            probe.running_peak = 0
            p0 = torch.cuda.memory_allocated()
        with TotalFlopCounter(display=False) if (args.flops and last) else nullcontext():
            carry, loss, _, _, _ = net(carry=carry, batch=batch, return_keys=[])
            if last:
                probe.fold()
                fwd_peak, after_fwd = probe.running_peak, torch.cuda.memory_allocated()
                probe.running_peak = 0
            (loss / batch_size).backward()
        if last:
            torch.cuda.synchronize()
            probe.fold()
            bwd_peak = probe.running_peak
        opt.step()
    result = {
        "B": batch_size,
        "P0 alloc": p0,
        "P1 anderson peak": probe.phase.get("anderson", (0, 0))[0],
        "P2 jac peak": probe.phase.get("jacobian", (0, 0))[0],
        "P2 after jac": probe.phase.get("jacobian", (0, 0))[1],
        "P3 fwd peak": fwd_peak,
        "P3 after fwd": after_fwd,
        "P4 bwd peak": bwd_peak,
        "reserved": torch.cuda.max_memory_reserved(),
    }
    grad_dtypes = sorted({d for d, grad in dtypes if grad})
    nograd_dtypes = sorted({d for d, grad in dtypes if not grad})
    result["_dtypes"] = f"L_level inputs: grad {grad_dtypes}, no-grad {nograd_dtypes}; carry z_H {carry.inner_carry.z_H.dtype}"
    del model, net, opt, carry, batch, loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--jac-lambda", type=float, default=1.0, help="arch.loss.jacobian_reg_lambda (0 = penalty off)")
    parser.add_argument("--set", nargs="*", default=[], help="extra arch overrides, e.g. jac_two_pass=true")
    parser.add_argument("--compile", action="store_true", help="torch.compile(dynamic=False) like pretrain.py")
    parser.add_argument("--flops", action="store_true", help="measure the step inside TotalFlopCounter like pretrain.py")
    parser.add_argument("--steps", type=int, default=3, help="training steps; the last one is measured")
    parser.add_argument("--extrapolate", type=int, nargs="+", default=[768, 1536])
    args = parser.parse_args()

    print(f"config: jac_lambda={args.jac_lambda} compile={args.compile} flops={args.flops} overrides={args.set}")
    rows = [run(b, args) for b in args.batches]
    print(rows[-1]["_dtypes"])
    keys = [k for k in rows[0] if k not in ("B", "_dtypes")]
    print(f"{'B':>6} " + " ".join(f"{k:>15}" for k in keys) + "   (GB)")
    for r in rows:
        print(f"{r['B']:>6} " + " ".join(f"{r[k] / GB:>15.3f}" for k in keys))
    if len(rows) >= 2:
        bs = np.array([r["B"] for r in rows], dtype=float)
        for target in args.extrapolate:
            fitted = []
            for k in keys:
                slope, intercept = np.polyfit(bs, [r[k] for r in rows], 1)
                fitted.append((intercept + slope * target) / GB)
            print(f"{target:>5}* " + " ".join(f"{v:>15.2f}" for v in fitted) + "   (linear fit)")


if __name__ == "__main__":
    main()
