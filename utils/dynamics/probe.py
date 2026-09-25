"""Roll a fixed batch of puzzles through the TRM inner network step-by-step and record its dynamics.

One "step" is one outer supervision step (a full H_cycles x L_cycles recursion), i.e. one call of
`InnerNetwork.forward`. Halting and row resets are bypassed so the rollout can run past `halt_max_steps`.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from models.losses import IGNORE_LABEL_ID
from models.recursive_reasoning.merged_arch import LatentCarry


def unwrap_inner(model: torch.nn.Module) -> torch.nn.Module:
    """loss head (possibly torch.compile'd) -> TRMModel -> InnerNetwork"""
    model = getattr(model, "_orig_mod", model)
    return model.model.inner


def initial_latent(inner: torch.nn.Module, batch_size: int, device: torch.device) -> LatentCarry:
    carry = inner.empty_carry(batch_size, device=device)
    reset = torch.ones((batch_size,), dtype=torch.bool, device=device)
    return inner.reset_carry(reset, carry, None)


@dataclass
class ProbeResult:
    """All arrays are numpy. Leading axis t indexes steps 1..D (t=0 in the array is step 1)."""

    token_acc: np.ndarray  # [D, B]
    exact: np.ndarray  # [D, B] bool
    q_logit: np.ndarray  # [D, B] raw q_halt logit
    res_H: np.ndarray  # [D, B] ||z_H^t - z_H^{t-1}||
    res_L: np.ndarray  # [D, B]
    res: np.ndarray  # [D, B] joint residual over (z_H, z_L)
    rel_res: np.ndarray  # [D, B] res / ||(z_H, z_L)^t||
    dist_to_final: np.ndarray  # [D + 1, B] ||z_t - z_D|| / ||z_0 - z_D||, t = 0..D
    answer_changed: np.ndarray  # [D, B] bool, decoded answer differs from step t-1 (False at t=1)
    pooled_zH: np.ndarray  # [D + 1, B, H] token-mean of z_H, t = 0..D (for PCA)
    settle_step: np.ndarray  # [B] first step after which the decoded answer never changes
    switch_freq: np.ndarray  # [B] answer changes / settle_step
    train_depth: int  # halt_max_steps

    @property
    def depth(self) -> int:
        return self.exact.shape[0]

    def modes(self) -> np.ndarray:
        """PTRM trajectory modes at the trained depth: 0 = quick success, 1 = delayed success, 2 = failure.

        Quick: exactly correct from some step <= T/2 through step T. Delayed: correct at T but only from later on.
        """
        T = self.train_depth
        exact_T = self.exact[:T]
        correct_at_T = exact_T[-1]
        # first step from which the answer is correct through T (1-indexed)
        stays_correct = np.flip(np.cumprod(np.flip(exact_T, 0), 0), 0).astype(bool)
        first_stable = np.where(stays_correct.any(0), stays_correct.argmax(0) + 1, T + 1)
        return np.where(~correct_at_T, 2, np.where(first_stable <= T // 2, 0, 1))


def _flat(carry: LatentCarry) -> torch.Tensor:
    return torch.cat((carry.z_H.flatten(1), carry.z_L.flatten(1)), dim=1).float()


@torch.no_grad()
def _rollout(inner, batch, init: LatentCarry, depth: int, noise_sigma: float, seed: int, final: Optional[torch.Tensor]):
    """Single pass. If `final` (flattened z_D) is given, returns distances to it instead of recording everything else."""
    device = batch["inputs"].device
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        carry = LatentCarry(init.z_H.clone(), init.z_L.clone())
        if final is not None:
            dists = [(_flat(carry) - final).norm(dim=1)]
            for _ in range(depth):
                if noise_sigma > 0:
                    carry.z_L = carry.z_L + noise_sigma * torch.randn_like(carry.z_L)
                carry, _ = inner(carry, batch)
                dists.append((_flat(carry) - final).norm(dim=1))
            return torch.stack(dists)

        labels = batch["labels"]
        mask = labels != IGNORE_LABEL_ID
        n_valid = mask.sum(-1).clamp_min(1)
        rec: Dict[str, list] = {k: [] for k in ("token_acc", "exact", "q", "res_H", "res_L", "norm", "preds", "pooled")}
        rec["pooled"].append(carry.z_H.float().mean(1))
        for _ in range(depth):
            prev = carry
            if noise_sigma > 0:
                carry = LatentCarry(carry.z_H, carry.z_L + noise_sigma * torch.randn_like(carry.z_L))
            carry, info = inner(carry, batch)
            preds = info["logits"].argmax(-1)
            correct = (preds == labels) & mask
            rec["token_acc"].append(correct.sum(-1) / n_valid)
            rec["exact"].append(correct.sum(-1) == mask.sum(-1))
            rec["q"].append(info["q_halt_logits"].float())
            rec["res_H"].append((carry.z_H.float() - prev.z_H.float()).flatten(1).norm(dim=1))
            rec["res_L"].append((carry.z_L.float() - prev.z_L.float()).flatten(1).norm(dim=1))
            rec["norm"].append(_flat(carry).norm(dim=1))
            rec["preds"].append(torch.where(mask, preds, -1))
            rec["pooled"].append(carry.z_H.float().mean(1))
        return {k: torch.stack(v) for k, v in rec.items()}, _flat(carry)


def run_probe(
    inner: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    depth: int,
    noise_sigma: float = 0.0,
    init: Optional[LatentCarry] = None,
    seed: int = 0,
) -> ProbeResult:
    """Roll `batch` for `depth` outer steps from `init` (default: the model's own reset state).

    Runs the rollout twice with the same RNG seed so the distance to the final latent can be computed
    without storing every full latent state.
    """
    was_training = inner.training
    inner.eval()
    try:
        if init is None:
            init = initial_latent(inner, batch["inputs"].shape[0], batch["inputs"].device)
        rec, z_final = _rollout(inner, batch, init, depth, noise_sigma, seed, None)
        dists = _rollout(inner, batch, init, depth, noise_sigma, seed, z_final)
    finally:
        inner.train(was_training)

    res_H, res_L = rec["res_H"], rec["res_L"]
    res = (res_H.square() + res_L.square()).sqrt()
    preds = rec["preds"]
    changed = torch.zeros_like(res_H, dtype=torch.bool)
    changed[1:] = (preds[1:] != preds[:-1]).any(-1)

    # settle step: 1 + index of last change (1 if the answer never changes)
    D = preds.shape[0]
    idx = torch.arange(1, D + 1, device=changed.device).unsqueeze(1)
    settle = torch.where(changed, idx, torch.ones_like(idx)).max(0).values
    switch_freq = changed.sum(0) / settle.clamp_min(1)

    np_ = lambda t: t.detach().float().cpu().numpy()
    return ProbeResult(
        token_acc=np_(rec["token_acc"]),
        exact=rec["exact"].cpu().numpy(),
        q_logit=np_(rec["q"]),
        res_H=np_(res_H),
        res_L=np_(res_L),
        res=np_(res),
        rel_res=np_(res / rec["norm"].clamp_min(1e-8)),
        dist_to_final=np_(dists / dists[:1].clamp_min(1e-8)),
        answer_changed=changed.cpu().numpy(),
        pooled_zH=np_(rec["pooled"]),
        settle_step=settle.cpu().numpy(),
        switch_freq=np_(switch_freq),
        train_depth=int(inner.config.halt_max_steps),
    )
