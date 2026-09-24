"""Paper-style training-dynamics diagnostics for TRM (see utils/dynamics/plots.py for the figure <-> paper mapping)."""

import os
from typing import Any, Dict, List

import numpy as np
import torch

from utils.dynamics.probe import run_probe, unwrap_inner
from utils.dynamics.plots import ALL_PROBE_FIGURES, auroc, halt_prob


class DynamicsLogger:
    """Holds a fixed probe batch and produces wandb-ready scalars + figures at every eval."""

    def __init__(self, cfg: Any, eval_loader, device: torch.device, out_dir: str):
        self.cfg = cfg
        self.out_dir = out_dir
        rows: Dict[str, List[torch.Tensor]] = {}
        n = 0
        for _set_name, batch, _ in eval_loader:
            for k, v in batch.items():
                rows.setdefault(k, []).append(v)
            n += batch["inputs"].shape[0]
            if n >= cfg.num_puzzles:
                break
        self.batch = {k: torch.cat(v)[: cfg.num_puzzles].to(device) for k, v in rows.items()}

    def __call__(self, model: torch.nn.Module, step: int) -> Dict[str, Any]:
        import wandb
        import matplotlib.pyplot as plt

        cfg = self.cfg
        inner = unwrap_inner(model)
        T = int(inner.config.halt_max_steps)
        D = cfg.depth_mult * T
        r = run_probe(inner, self.batch, depth=D)

        final_T = r.exact[T - 1]
        modes = r.modes()
        out: Dict[str, Any] = {
            f"dynamics/exact_acc@{T}": float(final_T.mean()),
            f"dynamics/exact_acc@{D}": float(r.exact[-1].mean()),
            f"dynamics/token_acc@{T}": float(r.token_acc[T - 1].mean()),
            f"dynamics/residual@{T}": float(r.res[T - 1].mean()),
            f"dynamics/residual@{D}": float(r.res[-1].mean()),
            f"dynamics/rel_residual@{T}": float(r.rel_res[T - 1].mean()),
            "dynamics/settle_time_mean": float(r.settle_step.mean()),
            "dynamics/switch_freq_mean": float(r.switch_freq.mean()),
            "dynamics/q_auroc": auroc(r.q_logit[T - 1], final_T),
            # mean halt-head P(correct) at the trained depth, for puzzles that are / aren't actually correct
            "dynamics/halt_prob_correct": float(halt_prob(r.q_logit[T - 1, final_T]).mean()) if final_T.any() else float("nan"),
            "dynamics/halt_prob_wrong": float(halt_prob(r.q_logit[T - 1, ~final_T]).mean()) if (~final_T).any() else float("nan"),
        }
        for i, name in enumerate(("quick", "delayed", "failure")):
            out[f"dynamics/mode/{name}"] = float((modes == i).mean())

        figures = {name: fn(r) for name, fn in ALL_PROBE_FIGURES.items()}

        for name, fig in figures.items():
            out[f"dynamics_fig/{name}"] = wandb.Image(fig)
            if self.out_dir:
                os.makedirs(os.path.join(self.out_dir, "figures"), exist_ok=True)
                fig.savefig(os.path.join(self.out_dir, "figures", f"{name}_step{step}.png"), dpi=130)
            plt.close(fig)

        if self.out_dir:
            np.savez_compressed(
                os.path.join(self.out_dir, f"probe_step{step}.npz"),
                **{k: v for k, v in vars(r).items() if isinstance(v, np.ndarray)},
                scalars=np.array({k: v for k, v in out.items() if isinstance(v, float)}, dtype=object),
            )
        return out
