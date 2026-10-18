from typing import Any, Dict, Tuple
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.losses import IGNORE_LABEL_ID
from models.losses.base import kl_balanced
from ..layers import (
    CastedEmbedding,
    CastedLinear,
    SwiGLU,
    rms_norm,
)
from .merged_arch import (
    LatentCarry,
    ModelCarry,
    InnerNetwork,
    _add_puzzle_emb_positions,
    _reset_rows,
    TRMModel,
)


class GaussianHead(nn.Module):
    def __init__(self, dim: int, expansion: float, norm_eps: float = 1e-5):
        super().__init__()

        self.norm_eps = norm_eps
        self.mlp = SwiGLU(dim, expansion)
        self.to_stats = CastedLinear(dim, 2 * dim, bias=True)

        # Start the guidance neutral (mu = 0, var = softplus(0)), as both reference implementations do
        with torch.no_grad():
            self.to_stats.weight.zero_()
            self.to_stats.bias.zero_()

    def forward(self, x: torch.Tensor):
        x = rms_norm(x, variance_epsilon=self.norm_eps)
        x = self.mlp(x)

        mu, raw_var = self.to_stats(x).chunk(2, dim=-1)

        # Must be positive because this is an actual variance, not log-variance.
        var = F.softplus(raw_var) + 1e-4
        return mu, var

class PosteriorHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, puzzle_emb_len: int, dtype: torch.dtype, expansion: float = 4):
        super().__init__()
        self.puzzle_emb_len = puzzle_emb_len
        embed_std = 1.0 / math.sqrt(hidden_size)
        self.embedding = CastedEmbedding(vocab_size, hidden_size, embed_std, dtype)

        self.combine = CastedLinear(2*hidden_size, hidden_size, bias=False)
        self.head = GaussianHead(dim=hidden_size, expansion=expansion)

    def forward(self, u_t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ignored label positions have no target token, so they get the pad token (0)
        y = torch.where(y == IGNORE_LABEL_ID, 0, y)
        y = self.embedding(y.to(torch.int32)) * math.sqrt(u_t.shape[-1])
        # The puzzle embedding prefix has no target, so its slots are zeros
        y = F.pad(y, (0, 0, self.puzzle_emb_len, 0))

        return self.head(self.combine(torch.cat([u_t, y], dim=-1)))


class GRAM_Inner(InnerNetwork):
    def __init__(self, config: dict):
        super().__init__(config)

        self.prior_head = GaussianHead(
            dim=self.config.hidden_size,
            expansion=self.config.expansion,
            norm_eps=self.config.rms_norm_eps,
        )
        self.posterior_head = PosteriorHead(
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
            puzzle_emb_len=self.config.puzzle_emb_len,
            dtype=self.forward_dtype,
            expansion=self.config.expansion,
        )
        # LPRM value head (paper App. A.2): predicts the prediction's accuracy from the first token of z_H
        self.v_head = CastedLinear(self.config.hidden_size, 1, bias=True)
        # No halt head: every puzzle runs all halt_max_steps supervision steps (see GRAM.forward)
        del self.q_head

        if not 1 <= self.config.grad_cycles <= self.config.H_cycles:
            raise ValueError(f"grad_cycles must be in [1, H_cycles={self.config.H_cycles}], got {self.config.grad_cycles}.")

    def _guidance(self, u_t: torch.Tensor, y: torch.Tensor, use_posterior: bool,
                  prior_rows: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | int]:
        """Samples the stochastic guidance eps for u_t and returns it with its beta-weighted balanced KL: from the
        posterior q(eps | u_t, y) if use_posterior (KL only when gradients are on), otherwise from the prior p(eps | u_t).
        prior_rows ([B] bool): puzzles that take the prior's guidance even when use_posterior (the KL is still paid)."""
        mu_prior, var_prior = self.prior_head(u_t)

        if use_posterior:
            mu_post, var_post = self.posterior_head(u_t, y)
            eps = mu_post + torch.randn_like(u_t) * var_post.sqrt()

            kl_loss = 0

            if prior_rows is not None:
                eps = torch.where(prior_rows.view(-1, 1, 1), mu_prior + torch.randn_like(u_t) * var_prior.sqrt(), eps)

            if torch.is_grad_enabled():
                # kl_balanced sums over the hidden dim -> [B, S]; average over tokens like the lm loss, then sum over the batch
                kl_loss = kl_balanced(
                    mu_post = mu_post,
                    var_post = var_post,
                    mu_prior = mu_prior,
                    var_prior = var_prior,
                    alpha=self.config.kl_grad_prior,
                ).mean(-1).sum() * self.config.kl_strength

            return eps, kl_loss

        if self.config.eval_prior_mean:
            return mu_prior, 0
        eps = mu_prior + torch.randn_like(u_t) * var_prior.sqrt()
        return eps, 0

    def _lprm_rollout_loss(self, u_t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Value bootstrapping for the LPRM head (TreeRM's, with the prior's draws as the rollouts): K draws of eps from
        the prior at u_t, each decoded without gradient, and the head trained to predict each draw's token accuracy
        from the draw's first token. The loss head only ever shows the head posterior-guided states, which is not
        what best-of-N or a search over prior draws asks it to rank. Batch-summed, averaged over the K draws, weighted
        by value_bootstrap_weight."""
        K = self.config.lprm_prior_rollouts
        with torch.no_grad():
            mu, var = self.prior_head(u_t)
            noise = torch.randn((u_t.shape[0], K) + u_t.shape[1:], device=u_t.device, dtype=u_t.dtype)
            z = rms_norm(u_t.unsqueeze(1) + mu.unsqueeze(1) + var.sqrt().unsqueeze(1) * noise, self.config.rms_norm_eps)  # [B, K, S, H]
            logits = self.lm_head(z[:, :, self.config.puzzle_emb_len :] if self.config.puzzle_emb_ndim != 0 else z)
            mask = y != IGNORE_LABEL_ID
            acc = ((logits.argmax(-1) == y.unsqueeze(1)) & mask.unsqueeze(1)).sum(-1) / mask.sum(-1, keepdim=True).clamp_min(1)  # [B, K]
        predicted = torch.sigmoid(self.v_head(z[:, :, 0]).to(torch.float32)).squeeze(-1)
        return F.mse_loss(predicted, acc.float(), reduction="sum") / K * self.config.value_bootstrap_weight

    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, dict]:
        z_H, z_L = carry.z_H, carry.z_L
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"], batch.get("object_ids"))
        y = batch["labels"]
        seq_dict = self._seq_dict()

        # Puzzles whose grad cycle takes the prior's guidance, so the lm loss also trains the trajectory inference runs
        # (prior_sample_frac): drawn per puzzle (GRAM.forward, kept for the puzzle's whole trajectory) or per step
        prior_rows = None
        if self.training and torch.is_grad_enabled() and self.config.prior_sample_frac > 0:
            if self.config.prior_sample_per_puzzle:
                prior_rows = batch["prior_traj"]
            else:
                prior_rows = torch.rand(y.shape[0], device=y.device) < self.config.prior_sample_frac

        # Run the recursion cycle for H_cycles - grad_cycles times without the gadient
        with torch.no_grad():
            for _ in range(self.config.H_cycles - self.config.grad_cycles):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + x, **seq_dict)

                u_t = self.L_level(z_H, z_L, **seq_dict)

                # Change with the GRAM network is that now noise from added to the z_H at each iteration
                # The posterior here pays no KL, so by default these cycles use the prior: otherwise y leaks into the trajectory for free
                eps, _ = self._guidance(u_t, y, use_posterior=self.training and self.config.posterior_nograd_cycles)
                z_H = u_t + eps


        # The last grad_cycles cycles carry the gradient; each samples the posterior and pays its KL
        kl_loss = 0
        for _ in range(self.config.grad_cycles):
            for __ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + x, **seq_dict)

            u_t = self.L_level(z_H, z_L, **seq_dict)
            eps, guidance_loss = self._guidance(u_t, y, use_posterior=self.training, prior_rows=prior_rows)
            kl_loss += guidance_loss
            if self.training and torch.is_grad_enabled() and self.config.lprm_prior_rollouts:
                kl_loss = kl_loss + self._lprm_rollout_loss(u_t, y)

            z_H = u_t + eps

        z_H = rms_norm(z_H, self.config.rms_norm_eps)
        
        # Clip out the puzzle embedding dims from the decoder
        if self.config.puzzle_emb_ndim != 0:
            logits = self.lm_head(z_H[:, self.config.puzzle_emb_len :]).contiguous()
        else:
            logits = self.lm_head(z_H).contiguous()

        info = {
            "lprm_value": torch.sigmoid(self.v_head(z_H[:, 0]).to(torch.float32)).squeeze(-1),
            "logits": logits,
            "kl_loss": kl_loss,
        }

        return LatentCarry(z_H, z_L), info


class GRAM(TRMModel):
    def __init__(self, config: dict):
        super().__init__(config)
        print("~"*80)
        print(config)
        print("GRAM architecture initialized")
        self.inner = GRAM_Inner(_add_puzzle_emb_positions(config))

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ModelCarry:
        """TRMModel.initial_carry, with room in current_data for each puzzle's prior_traj flag (see forward)."""
        return super().initial_carry(self._with_prior_traj(batch))

    def _with_prior_traj(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds prior_traj ([B] bool): whether a puzzle starting now runs its whole training trajectory on the prior's
        guidance (prior_sample_per_puzzle). It enters current_data on reset, so it stays fixed for the puzzle."""
        n, device = batch["inputs"].shape[0], batch["inputs"].device
        return {**batch, "prior_traj": torch.rand(n, device=device) < self.config.prior_sample_frac}

    def forward(self, carry: ModelCarry, batch: Dict[str, torch.Tensor], **kwargs: Any) -> Tuple[ModelCarry, Dict[str, torch.Tensor]]:
        """TRMModel.forward without ACT: there's no halt head, so a puzzle halts only after halt_max_steps
        supervision steps, in training and at eval."""
        del kwargs
        batch = self._with_prior_traj(batch)

        if carry is None:
            carry = self.initial_carry(batch)

        # Replace the halted puzzles in the batch with new ones
        device = batch["inputs"].device
        reset = carry.halted.to(device=device)
        inner_carry, current_data = _reset_rows(
            inner = self.inner,
            reset_flag = reset,
            inner_carry = LatentCarry(carry.inner_carry.z_H.to(device=device), carry.inner_carry.z_L.to(device=device)),
            current_data = carry.current_data,
            batch = batch,
            device = device,
        )

        steps = torch.where(reset, torch.zeros_like(carry.steps, device=device), carry.steps.to(device=device))
        prev_latent = inner_carry
        inner_carry, fpass_info = self.inner(inner_carry, current_data)

        with torch.no_grad():
            # NFE = L_level (the shared block f) evaluations
            prev_nfe = carry.nfe.to(device=device) if carry.nfe is not None else torch.zeros_like(steps)
            nfe = torch.where(reset, torch.zeros_like(prev_nfe), prev_nfe) + self.config.H_cycles * (self.config.L_cycles + 1)
            # normalised residuals over this supervision step: ||z^t - z^(t-1)|| / ||z^t||, for z_H and z_L
            for name in ("z_H", "z_L"):
                z, z_prev = getattr(inner_carry, name).float(), getattr(prev_latent, name).float()
                d_z = (z - z_prev).flatten(1).norm(dim=1)
                fpass_info[f"{name.replace('_', '')}_rel_residual"] = d_z / z.flatten(1).norm(dim=1).clamp_min(1e-8)

            steps = steps + 1
            halted = steps >= self.config.halt_max_steps

        return ModelCarry(inner_carry, steps, halted, current_data, halt_counter=torch.zeros_like(steps), nfe=nfe), fpass_info
