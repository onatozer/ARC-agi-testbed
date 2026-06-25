"""TRM-DEQ: TRM with the L-cycles inner recurrence replaced by an Anderson
fixed-point solve, plus a Jacobian-norm regulariser (Bai et al. 2021 style).

Why this exists
---------------
TRM's paper reports (in the "Ideas that failed" section) that a DEQ-style
forward (TorchDEQ) hurt generalisation on Sudoku. The most plausible cause is
*loss of gradient richness*: DEQ + IFT 1-step backward gives k=1 BPTT signal,
and Table 1 of the paper shows k=1 vs full BPTT drops Sudoku-Extreme accuracy
from 87 % to 56 %. Two known mitigations:

  1. **Phantom-gradient backward**: solve to (approximate) FP via Anderson, then
     unroll the last K iterates with grad and BPTT through those. We expose
     this via ``--bptt_through K`` (K=1 = pure IFT, K = L_cycles = full BPTT).
  2. **Jacobian regularisation**: penalise ``||J·v||^2`` at z*, encouraging the
     FP map to be well-conditioned (Bai, Koltun & Kolter 2021). Helps the
     solver converge faster and stabilises training.

We DEQ-ify only the **inner L-cycles** loop (z_L update). The outer H-cycles
loop and z_H update stay explicit. This preserves TRM's two-latent design and
deep supervision, only changing how z_L is refined.

Everything else (puzzle embeddings + sparse SignSGD, ACT halting / Q-head,
deep supervision via halted-carry, EMA, stablemax CE) is identical to vanilla
TRM and reused via composition with the existing TRM modules in this folder.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from .common import trunc_normal_init_
from .layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    LinearSwish,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)
from .sparse_embedding import CastedSparseEmbedding
from .trm import (
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

class TRMDEQConfig(BaseModel):
    """Mirrors TRM's config plus DEQ-specific knobs."""
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    # Outer (H_cycles) is preserved from TRM; inner L_cycles becomes the
    # solver max_iter when ``deq_inner=True``.
    H_cycles: int
    L_cycles: int            # -> deq_max_iter when DEQ replaces inner loop

    H_layers: int = 0
    L_layers: int

    # Transformer
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # ACT halting
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

    # ---- DEQ-specific knobs --------------------------------------------- #
    deq_inner: bool = True            # if False, this acts as plain TRM
    deq_max_iter: int = 8             # forward Anderson iterations
    deq_min_iter: int = 4
    deq_tol: float = 1e-3
    deq_anderson_m: int = 5           # history length
    deq_anderson_beta: float = 1.0
    # K last solver iterates to backprop through; K=1 -> pure IFT 1-step JVP,
    # K=L_cycles -> equivalent to full BPTT through the inner loop.
    bptt_through: int = 1
    jacobian_reg_lambda: float = 0.0  # 0 disables; 1e-3 ≈ Bai et al. 2021 default
    jacobian_reg_n_samples: int = 1


# --------------------------------------------------------------------------- #
# Anderson acceleration -- thin local copy so this file stays self-contained.
# Operates in fp32 for numerical stability; the caller can pass any dtype.
# --------------------------------------------------------------------------- #

def _anderson_solve(f, y0: torch.Tensor, *, max_iter: int, tol: float,
                    min_iter: int, m: int, beta: float):
    """Anderson-accelerated FP iteration. Returns (y_star, info_dict)."""
    B = y0.size(0)
    n = y0.numel() // B
    dtype = torch.float32  # solver in fp32 for stability
    device = y0.device

    Y = torch.zeros(B, m, n, dtype=dtype, device=device)
    F_ = torch.zeros(B, m, n, dtype=dtype, device=device)
    H = torch.zeros(B, m + 1, m + 1, dtype=dtype, device=device)
    H[:, 0, 1:] = 1.0
    H[:, 1:, 0] = 1.0
    rhs = torch.zeros(B, m + 1, 1, dtype=dtype, device=device)
    rhs[:, 0] = 1.0

    Y[:, 0] = y0.detach().reshape(B, n).to(dtype)
    F_[:, 0] = f(y0).detach().reshape(B, n).to(dtype)
    Y[:, 1] = F_[:, 0]
    F_[:, 1] = f(F_[:, 0].view_as(y0)).detach().reshape(B, n).to(dtype)

    iters = 2
    rel = float("inf")
    converged = False
    for k in range(2, max_iter):
        nn_ = min(k, m)
        G = F_[:, :nn_] - Y[:, :nn_]
        H[:, 1:nn_+1, 1:nn_+1] = (
            torch.bmm(G, G.transpose(1, 2))
            + 1e-4 * torch.eye(nn_, dtype=dtype, device=device).unsqueeze(0)
        )
        try:
            alpha = torch.linalg.solve(H[:, :nn_+1, :nn_+1], rhs[:, :nn_+1])[:, 1:nn_+1, 0]
        except RuntimeError:
            break
        y_new = (
            beta * (alpha[..., None] * F_[:, :nn_]).sum(dim=1)
            + (1 - beta) * (alpha[..., None] * Y[:, :nn_]).sum(dim=1)
        )
        F_new = f(y_new.view_as(y0)).detach().reshape(B, n).to(dtype)

        Y = torch.roll(Y, shifts=-1, dims=1)
        F_ = torch.roll(F_, shifts=-1, dims=1)
        Y[:, -1] = y_new
        F_[:, -1] = F_new

        diff = (F_new - y_new).norm(dim=1)
        ref = F_new.norm(dim=1).clamp_min(1e-9)
        rel = float((diff / ref).max())
        iters = k + 1
        if iters >= min_iter and rel < tol:
            converged = True
            break

    y_star = Y[:, -1].view_as(y0).to(y0.dtype)
    return y_star, {"iters": iters, "rel_residual": rel, "converged": converged}


# --------------------------------------------------------------------------- #
# DEQ inner module
# --------------------------------------------------------------------------- #

class _InnerDEQ(nn.Module):
    """The ``z_L`` inner refinement, but as a fixed-point solve over::

        f(z_L) = L_level(z_L, z_H + input_embeddings)

    Forward: anderson-solve until tol/min_iter/max_iter; the last
    ``bptt_through`` iterates carry grad (phantom-gradient style).

    Returns (z_star, jac_reg) where jac_reg is a scalar (0 if disabled).
    """

    def __init__(self, config: TRMDEQConfig):
        super().__init__()
        self.cfg = config

    def forward(self, *, L_level, z_L, ctx, seq_info):
        """ctx = z_H + input_embeddings (fixed during the solve)."""
        cfg = self.cfg

        def fmap(z):
            return L_level(z, ctx, **seq_info)

        # Forward solve (no grad)
        with torch.no_grad():
            z_star, info = _anderson_solve(
                fmap, z_L,
                max_iter=int(cfg.deq_max_iter),
                tol=float(cfg.deq_tol),
                min_iter=int(cfg.deq_min_iter),
                m=int(cfg.deq_anderson_m),
                beta=float(cfg.deq_anderson_beta),
            )

        # Phantom-gradient: re-run the last K iterates WITH grad starting from
        # z_star, so the loss can backprop through K layer applications.
        K = max(1, int(cfg.bptt_through))
        z = z_star.detach()
        for _ in range(K):
            z = fmap(z)

        # Jacobian regulariser ||J^T·v||^2 (Bai et al. 2021).
        # Force math SDPA -- flash attn lacks double-backward.
        jac_reg = z.new_zeros(())
        if cfg.jacobian_reg_lambda > 0 and self.training:
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                _sdpa_ctx = sdpa_kernel(SDPBackend.MATH)
            except ImportError:
                _sdpa_ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False,
                )
            d = z.shape[-1]
            for _ in range(int(cfg.jacobian_reg_n_samples)):
                v = (torch.randn_like(z) / math.sqrt(d)).detach()
                with torch.enable_grad(), _sdpa_ctx:
                    z_in = z.detach().requires_grad_(True)
                    f_z = fmap(z_in)
                    s = (v * f_z).sum()
                    # create_graph=True so the loss can backprop through Jt_v;
                    # retain_graph defaults to create_graph here (we need it).
                    (Jt_v,) = torch.autograd.grad(s, z_in, create_graph=True)
                jac_reg = jac_reg + Jt_v.pow(2).mean()
            jac_reg = jac_reg / int(cfg.jacobian_reg_n_samples)

        # Convergence diagnostics for monitoring
        self._last_info = info
        return z, jac_reg


# --------------------------------------------------------------------------- #
# Inner module mirroring TRM's _Inner.forward but DEQ-ifying the L-cycles loop
# --------------------------------------------------------------------------- #

class TRMDEQ_Inner(nn.Module):
    def __init__(self, config: TRMDEQConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # I/O (copied verbatim from TRM)
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(
            config.vocab_size, config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = (
            -(config.puzzle_emb_ndim // -config.hidden_size)
            if config.puzzle_emb_len == 0 else config.puzzle_emb_len
        )
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                batch_size=config.batch_size, init_std=0,
                cast_to=self.forward_dtype,
            )

        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len, config.hidden_size,
                init_std=embed_init_std, cast_to=self.forward_dtype,
            )

        # Reasoning layer (one shared L_level used for both z_L update and
        # the explicit z_H update -- this matches TRM's "single network" choice)
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[
            TinyRecursiveReasoningModel_ACTV1Block(config) for _ in range(config.L_layers)
        ])

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

        # The DEQ inner module
        self.inner_deq = _InnerDEQ(config)

        # Track the most recent jacobian-reg value for the loss head to grab
        self._last_jac_reg: torch.Tensor = torch.zeros(())

    # --- helpers (verbatim from TRM) ------------------------------------- #
    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - pe.shape[-1]
            if pad_count > 0:
                pe = F.pad(pe, (0, pad_count))
            embedding = torch.cat(
                (pe.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len,
                            self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len,
                            self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    # --- main forward: DEQ on inner L-cycles ----------------------------- #
    def forward(self, carry, batch: Dict[str, torch.Tensor]):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        z_H, z_L = carry.z_H, carry.z_L

        cfg = self.config
        jac_reg_total = z_H.new_zeros(())

        # --- Outer H_cycles: T-1 no_grad warm-ups --------------------------- #
        with torch.no_grad():
            for _ in range(cfg.H_cycles - 1):
                z_L_star, _ = self.inner_deq(
                    L_level=self.L_level, z_L=z_L,
                    ctx=z_H + x, seq_info=seq_info,
                )
                z_L = z_L_star
                z_H = self.L_level(z_H, z_L, **seq_info)

        # --- Last outer iter WITH grad ------------------------------------ #
        z_L_star, jac_reg = self.inner_deq(
            L_level=self.L_level, z_L=z_L,
            ctx=z_H + x, seq_info=seq_info,
        )
        jac_reg_total = jac_reg_total + jac_reg
        z_L = z_L_star
        z_H = self.L_level(z_H, z_L, **seq_info)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(), z_L=z_L.detach(),
        )
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        # Stash for the outer loss head to read.
        self._last_jac_reg = jac_reg_total

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


# --------------------------------------------------------------------------- #
# ACT wrapper (mirrors TRM's outer wrapper)
# --------------------------------------------------------------------------- #

class TRMDEQ(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRMDEQConfig(**config_dict)
        self.inner = TRMDEQ_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        bsz = batch["inputs"].shape[0]
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(bsz),
            steps=torch.zeros((bsz,), dtype=torch.int32),
            halted=torch.ones((bsz,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry, batch: Dict[str, torch.Tensor]):
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            # Surface the Jacobian regulariser so the loss head can include it.
            "jacobian_reg": self.inner._last_jac_reg,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step
            if self.training and self.config.halt_max_steps > 1:
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)
                min_halt = (
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt)
                if not self.config.no_ACT_continue:
                    _, _, (next_qh, next_qc) = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(is_last_step, next_qh, torch.maximum(next_qh, next_qc))
                    )

        return TinyRecursiveReasoningModel_ACTV1Carry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs
