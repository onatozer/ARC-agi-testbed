from typing import Dict, Tuple
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.losses import IGNORE_LABEL_ID
from models.losses.base import kl_balanced_logits
from ..layers import (
    CastedEmbedding,
    CastedLinear,
    SwiGLU,
    rms_norm,
)
from .merged_arch import _add_puzzle_emb_positions
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
    """Logits over the candidates: a per-token query from the context, dotted with each candidate and averaged
    over the sequence, so each logit sees the whole latent and the whole candidate. [B, S, H] x [B, K, S, H] -> [B, K]."""

    def __init__(self, hidden_size: int, expansion: float, norm_eps: float = 1e-5):
        super().__init__()

        self.norm_eps = norm_eps
        self.mlp = SwiGLU(hidden_size, expansion)
        self.to_query = CastedLinear(hidden_size, hidden_size, bias=False)

    def forward(self, ctx: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        query = self.to_query(self.mlp(rms_norm(ctx, variance_epsilon=self.norm_eps)))
        seq_len, hidden_size = ctx.shape[-2:]
        return torch.einsum("bsh,bksh->bk", query.float(), cand.float()) / (seq_len * math.sqrt(hidden_size))


class PosteriorCandidateScorer(CandidateScorer):
    """CandidateScorer whose context also sees the labels y, embedded as in gram.PosteriorHead."""

    def __init__(self, hidden_size: int, vocab_size: int, puzzle_emb_len: int, dtype: torch.dtype, expansion: float, norm_eps: float = 1e-5):
        super().__init__(hidden_size, expansion, norm_eps)
        self.puzzle_emb_len = puzzle_emb_len
        self.embedding = CastedEmbedding(vocab_size, hidden_size, 1.0 / math.sqrt(hidden_size), dtype)
        self.combine = CastedLinear(2 * hidden_size, hidden_size, bias=False)

    def forward(self, u_t: torch.Tensor, y: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        # Ignored label positions have no target token, so they get the pad token (0)
        y = torch.where(y == IGNORE_LABEL_ID, 0, y)
        y = self.embedding(y.to(torch.int32)) * math.sqrt(u_t.shape[-1])
        # The puzzle embedding prefix has no target, so its slots are zeros
        y = F.pad(y, (0, 0, self.puzzle_emb_len, 0))

        return super().forward(self.combine(torch.cat([u_t, y], dim=-1)), cand)


class GRAMDP_Inner(GRAM_Inner):
    """GRAM_Inner with discrete guidance: eps is a Gumbel-softmax mixture of noise_options candidate vectors,
    chosen by a prior p(k | u_t) / posterior q(k | u_t, y) over the candidates."""

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
            expansion=self.config.expansion,
            norm_eps=self.config.rms_norm_eps,
        )
        self.posterior_head = PosteriorCandidateScorer(
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
            puzzle_emb_len=self.config.puzzle_emb_len,
            dtype=self.forward_dtype,
            expansion=self.config.expansion,
            norm_eps=self.config.rms_norm_eps,
        )

    def _guidance(self, u_t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | int]:
        """Samples the guidance eps for u_t as a Gumbel-softmax mixture of the candidates, with its beta-weighted
        balanced KL: from the posterior q(k | u_t, y) in training, otherwise from the prior p(k | u_t) with no KL."""
        cand = self.candidate_head(u_t)
        prior_logits = self.prior_head(u_t, cand)

        kl_loss = 0
        if self.training:
            post_logits = self.posterior_head(u_t, y, cand)
            logits = post_logits

            # Only the grad cycles pay the KL
            if torch.is_grad_enabled():
                kl_loss = kl_balanced_logits(
                    prior_logits=prior_logits,
                    post_logits=post_logits,
                    alpha=self.config.kl_grad_prior,
                ).sum() * self.config.kl_strength
        else:
            logits = prior_logits

        weights = F.gumbel_softmax(logits, tau=self.config.gumbel_temp, dim=-1)
        eps = torch.einsum("bk,bksh->bsh", weights, cand.float()).to(u_t.dtype)
        return eps, kl_loss


class GRAMDP(GRAM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.inner = GRAMDP_Inner(_add_puzzle_emb_positions(config))
