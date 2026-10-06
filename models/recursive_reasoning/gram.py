from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import math

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.func import functional_call, jacrev
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from torchdeq.solver.anderson import anderson_solver

from ..common import trunc_normal_init_
from ..layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    LinearSwish,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)
from ..sparse_embedding import CastedSparseEmbedding
from .merged_arch import (
    Block,
    Blocks,
    TRMConfig,
    InnerNetwork,
    ModelCarry,
    LatentCarry,
    _reset_rows,
    _add_puzzle_emb_positions,
    TRMModel,
)

@dataclass
class TransitionInfo:
    """Diagnostics returned per transition; used by the loss + by ablation logging."""

    u_t: torch.Tensor              # (B, L, D) — deterministic high-level update
    eps: torch.Tensor              # (B, L, D) — sampled stochastic guidance
    mu_p: torch.Tensor             # (B, L, D) — prior mean
    var_p: torch.Tensor            # (B, L, D) — prior σ²
    mu_q: torch.Tensor | None      # (B, L, D) — posterior mean (None at inference)
    var_q: torch.Tensor | None     # (B, L, D) — posterior σ² (None at inference)

@dataclass
class GramCarry(ModelCarry):
    transition: TransitionInfo


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

class GRAMConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    
    kl_grad_prior: float = .8
    kl_strength: float = .07

class GaussianHead(nn.Module):
    def __init__(self, dim: int, expansion):
        super().__init__()

        self.norm = rms_norm(dim)
        self.mlp = SwiGLU(dim, expansion)
        self.to_stats = CastedLinear(dim, 2 * dim, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.mlp(x)

        mu, raw_var = self.to_stats(x).chunk(2, dim=-1)

        # Must be positive because this is an actual variance, not log-variance.
        var = F.softplus(raw_var) + 1e-4
        return mu, var

class PosteriorHead(nn.module):
    def __init__(self, hidden_size: int, vocab_size: int, dtype: str, expansion: int = 4):
        embed_std = 1.0 / math.sqrt(hidden_size)
        self.embedding = CastedEmbedding(vocab_size, hidden_size, embed_std, dtype)

        self.head = nn.Sequential(
            CastedLinear(2*hidden_size, hidden_size),
            GaussianHead(dim=hidden_size, expansion=expansion),
        )

    def forward(self, u_t: torch.tensor, y: torch.tensor) -> torch.tensor:
        y = self.emebdding(y) * math.sqrt(u_t.shape[-1])

        return self.head(torch.cat([u_t, y]))

        

class GRAM_Inner(InnerNetwork):
    def __init__(self, config: GRAMConfig):
        super().__init__(config)
        # super().__init__(config.TRM_cfg)
        config = GRAMConfig(**config) if isinstance(config, dict) else config
        self.gram_config = config

        self.prior_head = GaussianHead(
            dim=config.hidden_size,
            expansion=config.expansion,
        )

        self.prior_head = nn.Sequential(
            rms_norm(config.hidden_size),
            SwiGLU(2, config.expansion),
            CastedLinear(config.hidden_size, 2 * config.hidden_size),
        )


    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, dict]:
        z_H, z_L = carry.z_H, carry.z_L
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"], batch.get("object_ids"))
        y = batch["labels"]
        seq_dict = self._seq_dict()

        # Run the recursion cycle for H_cycle -1 times without the gadient
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + x, **seq_dict)
                    
                u_t = self.L_level(z_H, z_L, **seq_dict)

                mu_post, sigma_post = self.posterior_head(u_t, y)
                mu_prior, sigma_prior = self.prior_head(u_t)

                # Change with the GRAM network is that now noise from added to the z_H at each iteration
                if self.is_training: # Noise added from posterior network if training, otherwise from prior network
                    eps = torch.randn_like(z_H)*sigma_post + mu_post
                else:
                    eps = torch.randn_like(z_H)*sigma_prior + mu_prior

                z_H = u_t + eps
             

        # Finally, last recursion cycle is ran with the gradient attached
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + x, **seq_dict)

        u_t = self.L_level(z_H, z_L, **seq_dict)
        mu_post, sigma_post = self.posterior_head(u_t, x, y)
        mu_prior, sigma_prior = self.prior_head(u_t, x)

        # Change with the GRAM network is that now noise from added to the u_t at each iteration
        if self.is_training: # Noise added from posterior network if training, otherwise from prior network
            eps = torch.randn_like(u_t)*sigma_post + mu_post
        else:
            eps = torch.randn_like(u_t)*sigma_prior + mu_prior
            
        z_H = z_H + eps


        # Clip out the puzzle embedding dims from the decoder
        if self.config.puzzle_emb_ndim != 0:
            logits = self.lm_head(z_H[:, self.config.puzzle_emb_len :]).contiguous()
        else:
            logits = self.lm_head(z_H).contiguous()

        q = self.q_head(z_H[:, 0]).to(torch.float32) 

        info = {
            "q_halt_logits": q[..., 0],
            "logits": logits,
            "u_t": u_t,
            "mu_prior": mu_prior,
            "sigma_prior": sigma_prior,
            "mu_post": mu_post,
            "sigma_post": sigma_post,
        }
        
        return LatentCarry(z_H, z_L), info



class GRAMModel(TRMModel):
    def __init__(self, config: GRAMConfig):
        super().__init__(config)
        print("~"*80)
        print(config)
        print("GRAM architecture initialized")


class GRAM(TRMModel):
    def __init__(self, config: GRAMConfig):
        # config = GRAMConfig.model_validate(config)
        # super().__init__(config.TRM_cfg)
        super().__init__(config)
        print("~"*80)
        print(config)
        print("DEQ architecture initialized")
        self.root_config = GRAMConfig(**config)
        self.inner = GRAM_Inner(_add_puzzle_emb_positions(config))