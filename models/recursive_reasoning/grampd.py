from typing import Any, Dict, Tuple
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.losses import IGNORE_LABEL_ID
from models.losses.base import kl_balanced_logits
from ..layers import (
    CastedEmbedding,
    CastedLinear,
    CosSin,
    SwiGLU,
    apply_rotary_pos_emb,
    rms_norm,
)
from .merged_arch import ModelCarry, _add_puzzle_emb_positions
from .gram import GRAM, GRAM_Inner


class CandidateHead(nn.Module):
    """Proposes num_candidates guidance vectors for every token of u_t: [B, S, H] -> [B, K, S, H]."""

    def __init__(self, hidden_size: int, num_candidates: int, expansion: float, norm_eps: float = 1e-5):
        super().__init__()

        self.num_candidates = num_candidates
        self.norm_eps = norm_eps
        self.mlp = SwiGLU(hidden_size, expansion)
        # Not zero-initialised: identical candidates would all get the same gradient direction and never separate
        self.to_candidates = CastedLinear(hidden_size, num_candidates * hidden_size, bias=True)

    def forward(self, u_t: torch.Tensor) -> torch.Tensor:
        x = self.mlp(rms_norm(u_t, variance_epsilon=self.norm_eps))
        cand = self.to_candidates(x).unflatten(-1, (self.num_candidates, -1))
        return cand.transpose(1, 2)


class CandidateScorer(nn.Module):
    """Logits over the candidates by cross-attention: for each candidate, the context tokens (queries) attend over that
    candidate's tokens (keys / values), the attended tokens are averaged and a linear layer gives the logit.
    [B, S, H] x [B, K, S, H] -> [B, K]."""

    def __init__(self, hidden_size: int, num_heads: int, norm_eps: float = 1e-5):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm_eps = norm_eps
        self.q_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.kv_proj = CastedLinear(hidden_size, 2 * hidden_size, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.to_logit = CastedLinear(hidden_size, 1, bias=True)

    def forward(self, ctx: torch.Tensor, cand: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        batch_size, num_candidates, seq_len, hidden_size = cand.shape

        # One attention problem per (puzzle, candidate): the same context queries against each candidate's tokens
        query = self.q_proj(rms_norm(ctx, variance_epsilon=self.norm_eps)).view(batch_size, 1, seq_len, self.num_heads, self.head_dim)
        query = query.expand(-1, num_candidates, -1, -1, -1).reshape(batch_size * num_candidates, seq_len, self.num_heads, self.head_dim)
        kv = self.kv_proj(rms_norm(cand, variance_epsilon=self.norm_eps)).view(batch_size * num_candidates, seq_len, 2, self.num_heads, self.head_dim)
        key, value = kv.unbind(2)

        # Context token s and candidate token s are the same cell, so both get that cell's RoPE position
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # [B*K, S, heads, head_dim] -> [B*K, heads, S, head_dim] for SDPA, then back
        attended = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        attended = self.o_proj(attended.transpose(1, 2).reshape(batch_size * num_candidates, seq_len, hidden_size))

        return self.to_logit(attended.mean(dim=1)).view(batch_size, num_candidates).float()


class LabelEncoder(nn.Module):
    """Mixes the labels y into u_t for the posterior, embedded as in gram.PosteriorHead.
    [B, S, H] x [B, S - puzzle_emb_len] -> [B, S, H]."""

    def __init__(self, hidden_size: int, vocab_size: int, puzzle_emb_len: int, dtype: torch.dtype):
        super().__init__()
        self.puzzle_emb_len = puzzle_emb_len
        self.embedding = CastedEmbedding(vocab_size, hidden_size, 1.0 / math.sqrt(hidden_size), dtype)
        self.combine = CastedLinear(2 * hidden_size, hidden_size, bias=False)

    def forward(self, u_t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ignored label positions have no target token, so they get the pad token (0)
        y = torch.where(y == IGNORE_LABEL_ID, 0, y)
        y = self.embedding(y.to(torch.int32)) * math.sqrt(u_t.shape[-1])
        # The puzzle embedding prefix has no target, so its slots are zeros
        y = F.pad(y, (0, 0, self.puzzle_emb_len, 0))
        return self.combine(torch.cat([u_t, y], dim=-1))


class PosteriorCandidateScorer(nn.Module):
    """Posterior logits over the candidates: its own CandidateScorer, with the labels mixed into the context first."""

    def __init__(self, hidden_size: int, num_heads: int, vocab_size: int, puzzle_emb_len: int, dtype: torch.dtype, norm_eps: float = 1e-5):
        super().__init__()
        self.label_encoder = LabelEncoder(hidden_size, vocab_size, puzzle_emb_len, dtype)
        self.scorer = CandidateScorer(hidden_size, num_heads, norm_eps)

    def forward(self, u_t: torch.Tensor, y: torch.Tensor, cand: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        return self.scorer(self.label_encoder(u_t, y), cand, cos_sin)


class GRAMDP_Inner(GRAM_Inner):
    """GRAM_Inner with discrete guidance: a prior p(k | u_t) / posterior q(k | u_t, y) over noise_options candidate
    vectors picks eps as exactly one candidate (Gumbel-max), trained with straight-through Gumbel-softmax gradients."""

    def __init__(self, config: dict):
        super().__init__(config)

        self.candidate_head = CandidateHead(
            hidden_size=self.config.hidden_size,
            num_candidates=self.config.noise_options,
            expansion=self.config.expansion,
            norm_eps=self.config.rms_norm_eps,
        )
        # Replace GRAM's Gaussian prior / posterior heads
        self.prior_head = CandidateScorer(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            norm_eps=self.config.rms_norm_eps,
        )
        self.posterior_head = PosteriorCandidateScorer(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            vocab_size=self.config.vocab_size,
            puzzle_emb_len=self.config.puzzle_emb_len,
            dtype=self.forward_dtype,
            norm_eps=self.config.rms_norm_eps,
        )
        # Straight-through gradient temperature; a buffer so GRAMDP can anneal it each step without recompiling
        self.register_buffer("gumbel_tau", torch.tensor(float(self.config.gumbel_temp)), persistent=False)

    def _guidance(self, u_t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | int]:
        """Samples the guidance eps for u_t as one of the candidates, with its beta-weighted balanced KL: drawn from the
        posterior q(k | u_t, y) in training, otherwise from the prior p(k | u_t) with no KL."""
        cand = self.candidate_head(u_t)
        cos_sin = self._cos_sin()
        prior_logits = self.prior_head(u_t, cand, cos_sin)

        kl_loss = 0
        logits = prior_logits
        if self.training:
            post_logits = self.posterior_head(u_t, y, cand, cos_sin)

            # Only the grad cycles pay the KL
            if torch.is_grad_enabled():
                kl_loss = kl_balanced_logits(
                    prior_logits=prior_logits,
                    post_logits=post_logits,
                    alpha=self.config.kl_grad_prior,
                ).sum() * self.config.kl_strength
            logits = post_logits

        # Straight-through Gumbel-softmax: the forward pass is the one-hot Gumbel-max sample argmax(logits + g), an exact
        # draw from Cat(softmax(logits)); the backward pass uses the gradient of softmax((logits + g) / tau)
        weights = F.gumbel_softmax(logits.float(), tau=self.gumbel_tau, hard=True, dim=-1)

        eps = torch.einsum("bk,bksh->bsh", weights, cand.float()).to(u_t.dtype)
        return eps, kl_loss


class GRAMDP(GRAM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.inner = GRAMDP_Inner(_add_puzzle_emb_positions(config))

    @torch.compiler.disable
    def _anneal_gumbel_tau(self, carry: ModelCarry | None) -> None:
        """Linearly anneals the Gumbel-softmax temperature from gumbel_temp to gumbel_temp_min over gumbel_anneal_steps of
        carry.global_step (the training step pretrain.py sets; without one, e.g. outside pretrain.py, it stays at gumbel_temp)."""
        step = getattr(carry, "global_step", None)
        if step is None:
            return
        frac = min(1.0, step / self.config.gumbel_anneal_steps) if self.config.gumbel_anneal_steps > 0 else 1.0
        self.inner.gumbel_tau.fill_(self.config.gumbel_temp + frac * (self.config.gumbel_temp_min - self.config.gumbel_temp))

    def forward(self, carry: ModelCarry, batch: Dict[str, torch.Tensor], **kwargs: Any) -> Tuple[ModelCarry, Dict[str, torch.Tensor]]:
        self._anneal_gumbel_tau(carry)
        return super().forward(carry, batch, **kwargs)
