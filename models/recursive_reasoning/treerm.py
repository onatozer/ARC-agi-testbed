import torch
import torch.nn.functional as F
from torch import nn
from .merged_arch import TRMModel, ModelCarry, InnerNetwork, LatentCarry, _add_puzzle_emb_positions, _reset_rows
from einops.layers.torch import Rearrange 
from typing import Any, Dict, Tuple
import math


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
from .gram import GRAM, GRAM_Inner

from layers import SwiGLU, CastedLinear, rms_norm

class CandidateGenerator(nn.module):
    def __init__(self, hidden_dim: int, num_candidates: int):

        self.hidden_dim = hidden_dim
        self.num_candidates = num_candidates

        self.head = nn.Sequential(
            CastedLinear(hidden_dim*2, hidden_dim, bias=True),
            SwiGLU(hidden_dim),
            CastedLinear(hidden_dim, hidden_dim*num_candidates, bias=True),
        )

    def forward(self, z_L: torch.Tensor, z_H: torch.Tensor):
        x = torch.cat([z_L, z_H], dim=-1)
        x = self.head(x) # K, B, L, D
        return Rearrange('K B L D -> B (K L D)')


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


class TreeRMInner(InnerNetwork):
    def __int__(self, config):
        self.candidate_generator = CandidateGenerator(
            hidden_dim=config.hidden_dim, 
            num_candidates=config.num_candidates)

        self.candidate_scorer = CandidateScorer(
            hidden_size=config.hidden_size,
            num_heads=config.num_scorer_heads
        )
        
        self.v_head = CastedLinear(self.config.hidden_size, 1, bias=True)

    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, dict]:
        z_H, z_L = carry.z_H, carry.z_L
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"], batch.get("object_ids"))
        y = batch["labels"]
        seq_dict = self._seq_dict()

        # Run the recursion cycle for H_cycles - grad_cycles times without the gadient
        with torch.no_grad():
            for _ in range(self.config.H_cycles - self.config.grad_cycles):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + x, **seq_dict)

                z_H = self.L_level(z_H, z_L, **seq_dict)

        # The last grad_cycles cycles carry the gradient; each samples the posterior and pays its KL
        kl_loss = 0
        for _ in range(self.config.grad_cycles):
            for __ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + x, **seq_dict)

            u_t = self.L_level(z_H, z_L, **seq_dict)
            eps, guidance_loss = self._guidance(u_t, y)
            kl_loss += guidance_loss

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




class TreeRM(TRMModel):

    def __init__(self, config: dict):
        self.inner = TreeRMInner(config=config)

    def forward(self, carry: ModelCarry, batch: Dict[str, torch.Tensor], **kwargs: Any) -> Tuple[ModelCarry, Dict[str, torch.Tensor]]:
        del kwargs
        
        if carry is None:
            carry = self.initial_carry(batch)
        
        # Bring everything from the carry onto the same device used by the batch
        device = batch["inputs"].device
        reset = carry.halted.to(device=device)

        # Call the helper function that replaces all the problems in the batch that have been halted on and replaced them with new ones
        inner_carry, current_data = _reset_rows(
            inner = self.inner,
            reset_flag = reset,
            inner_carry = LatentCarry(carry.inner_carry.z_H.to(device=device), carry.inner_carry.z_L.to(device=device)),
            current_data = carry.current_data,
            batch = batch,
            device = device,
        )
        
        steps = torch.where(reset, torch.zeros_like(carry.steps, device=device), carry.steps.to(device=device))
        counter = torch.zeros_like(steps) if carry.halt_counter is None else torch.where(reset, torch.zeros_like(steps), carry.halt_counter.to(device=device))
        inner_carry, fpass_info = self.inner(inner_carry, current_data)


        return ModelCarry(inner_carry, steps, halted, current_data, halt_counter=counter, nfe=nfe), fpass_info