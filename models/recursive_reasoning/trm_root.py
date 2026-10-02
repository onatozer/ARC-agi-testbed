from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

class TRMRootConfig(BaseModel):
    # TRM_cfg: TRMConfig = Field(default_factory=TRMConfig)
    model_config = ConfigDict(extra='allow')

    deq_max_iter: int = 8             # forward Anderson iterations
    deq_tol: float = 1e-3
    deq_stop_mode: str = "rel"        # "rel": ||f(z)-z|| / ||f(z)||, "abs": ||f(z)-z|| (scales with latent size)
    deq_anderson_m: int = 5           # history length of Anderson solver
    deq_anderson_tau: float = 1.0     # Anderson damping (torchdeq's tau)

    bptt_through: int = 6 
    phantom_gradient_lambda: float = .5
    jacobian_reg_n_samples: int = 1
    log_jacobian_norm: bool = False  # estimate + log ||J||_F^2 / d every training step, even when the penalty is off

    # Memory savers; all give the same gradients as with them off
    checkpoint_phantom: bool = False  # recompute the phantom-gradient L_level calls in backward instead of storing them
    jac_two_pass: bool = False        # take the penalty's parameter gradients during the forward, freeing its graph early
    jac_chunk_size: int = 0           # with jac_two_pass: compute the penalty over this many batch rows at a time (0 = all)

def jac_loss_estimate(f0, z0, vecs=2, create_graph=True):
    """Estimating tr(J^TJ)=tr(JJ^T) via Hutchinson estimator

    Args:
        f0 (torch.Tensor): Output of the function f (whose J is to be analyzed)
        z0 (torch.Tensor): Input to the function f
        vecs (int, optional): Number of random Gaussian vectors to use. Defaults to 2.
        create_graph (bool, optional): Whether to create backward graph (e.g., to train on this loss). 
                                       Defaults to True.

    Returns:
        torch.Tensor: A 1x1 torch tensor that encodes the (shape-normalized) jacobian loss
    """
    vecs = vecs
    result = 0
    for i in range(vecs):
        v = torch.randn(*f0.shape).to(z0)
        vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
        result += vJ.square().sum()

    return result / (vecs * z0.numel())


class _InjectParamGrads(torch.autograd.Function):
    """Returns `value` unchanged; in backward, adds grad_out * precomputed gradient to each parameter.

    Lets the Jacobian penalty take its parameter gradients during the forward (and free its double-backward graph)
    while the single loss.backward() still applies the loss weight and batch scaling, so .grad ends up the same.
    """

    @staticmethod
    def forward(ctx, value, *params_and_grads):
        n = len(params_and_grads) // 2
        ctx.n = n
        ctx.save_for_backward(*params_and_grads[n:])
        return value.clone()

    @staticmethod
    def backward(ctx, grad_out):
        grads = [(grad_out * g).to(g.dtype) for g in ctx.saved_tensors]
        return (None, *grads, *([None] * ctx.n))


class TRMRoot_Inner(InnerNetwork):
    def __init__(self, config: TRMRootConfig):
        super().__init__(config)
        # super().__init__(config.TRM_cfg)
        config = TRMRootConfig(**config) if isinstance(config, dict) else config
        self.root_config = config
        # Set to False by ACTLossHeadRoot when jacobian_reg_lambda == 0
        self.compute_jacobian_reg = True

    @torch.compiler.disable(recursive=True)
    def _jacobian_regularizer(self, z_in, input_injection, seq_dict, create_graph=True):
        """Hutchinson estimate of ||J||_F^2 / d for one L_level step (the loss head weights it by jacobian_reg_lambda).

        create_graph=True keeps it differentiable so it can be used as a penalty; logging only needs False.
        """
        cfg = self.root_config

        jac_input = input_injection.detach()

        jac_seq_dict = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in seq_dict.items()
        }

        if create_graph and cfg.jac_two_pass:
            return self._jacobian_two_pass(z_in, jac_input, jac_seq_dict)

        # the penalty's gradient needs a double backward; fused attention kernels have none
        with sdpa_kernel(SDPBackend.MATH) if create_graph else nullcontext():
            z_out = self.L_level(
                z_in,
                jac_input,
                **jac_seq_dict,
            )

        return jac_loss_estimate(
            z_out,
            z_in,
            vecs=cfg.jacobian_reg_n_samples,
            create_graph=create_graph,
        )

    @torch.compiler.disable(recursive=True)
    def _jacobian_two_pass(self, z_in, jac_input, jac_seq_dict):
        """Same penalty value and parameter gradients as the single-pass path, but the double-backward graph is freed here,
        before the phantom-gradient graph is built, and only jac_chunk_size rows of it exist at a time.

        Chunking is exact because L_level acts on each batch row independently (no cross-row ops), so J is block-diagonal
        over rows and ||vJ||^2 is a sum over rows.
        """
        cfg = self.root_config
        params = [p for p in self.L_level.parameters() if p.requires_grad]
        batch = z_in.shape[0]
        chunk = cfg.jac_chunk_size if cfg.jac_chunk_size > 0 else batch
        vecs = cfg.jacobian_reg_n_samples

        # Consume the RNG exactly as the single-pass path does: L_level's noise draw (device RNG), then the Hutchinson
        # vectors (CPU RNG, as in jac_loss_estimate). The chunks' own noise draws are then discarded via fork_rng; the
        # noise is additive, so it never affects J.
        torch.randn(z_in.shape, dtype=torch.promote_types(z_in.dtype, jac_input.dtype), device=z_in.device)
        vs = [torch.randn(*z_in.shape).to(z_in) for _ in range(vecs)]

        total = torch.zeros((), dtype=torch.float32, device=z_in.device)
        grads = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        with torch.random.fork_rng(devices=[z_in.device] if z_in.is_cuda else []):
            for start in range(0, batch, chunk):
                rows = slice(start, start + chunk)
                z_chunk = z_in[rows].detach().requires_grad_(True)
                with sdpa_kernel(SDPBackend.MATH):
                    z_out = self.L_level(z_chunk, jac_input[rows], **jac_seq_dict)
                penalty = 0
                for v in vs:
                    vJ = torch.autograd.grad(z_out, z_chunk, v[rows], retain_graph=True, create_graph=True)[0]
                    penalty = penalty + vJ.square().sum()
                penalty = penalty / (vecs * z_in.numel())
                chunk_grads = torch.autograd.grad(penalty, params, allow_unused=True)
                for acc, g in zip(grads, chunk_grads):
                    if g is not None:
                        acc.add_(g)
                total += penalty.detach().float()
                del z_out, penalty, chunk_grads

        grads = [g.to(p.dtype) for g, p in zip(grads, params)]
        return _InjectParamGrads.apply(total, *params, *grads)

    def _root_solve(self, z_L: torch.tensor, input_injection, seq_dict):
        """Uses the torch DEQ library to compute the fixed point of the inner network

        Returns:
            z: The output tensor of the equilibrium finding, attached to the computational graph
            jac_reg: The jacobian regularization penalty
        """
        cfg = self.root_config

        # Anderson solver only works in f32 precision type, but we may not want that for our network
        def fmap(z):
            self._nfe += 1
            z = z.to(getattr(torch, cfg.forward_dtype))
            z_out = self.L_level(z, input_injection, **seq_dict).float() #NOTE: Potentially reshape to be friendlier to anderson solver?? 
            return z_out

        # Find equilibrium point, as estimated by the anderson solver
        with torch.no_grad():
            z_star, _, info = anderson_solver(
                func = fmap,
                x0 = z_L.detach().float(),
                max_iter=int(cfg.deq_max_iter),
                tol=float(cfg.deq_tol),
                stop_mode=cfg.deq_stop_mode,
                m=int(cfg.deq_anderson_m),
                tau=float(cfg.deq_anderson_tau),
            )

        # Convergence diagnostics for monitoring
        self._last_info = info

        # The solver works in fp32; hand the rest of the network forward_dtype, as the other recursive models use
        return z_star.to(getattr(torch, cfg.forward_dtype))

    def _L_grad(self, hidden_states, input_injection, seq_dict):
        """L_level on the gradient path; with checkpoint_phantom its activations are recomputed in backward."""
        if self.root_config.checkpoint_phantom and self.training and torch.is_grad_enabled():
            return checkpoint(self.L_level, hidden_states, input_injection, use_reentrant=False, **seq_dict)
        return self.L_level(hidden_states, input_injection, **seq_dict)

    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the network using a fixed-root solver to compute z_L,
        then doing extra phantom-gradient iterations so that the inner network graph can be backpropagated through.
        """
        z_H, z_L = carry.z_H, carry.z_L
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"], batch.get("object_ids"))
        seq_dict = self._seq_dict()
        self._nfe = 0  # L_level evaluations in this forward (solver iterations are data-dependent)
        self._last_info = None

        # Replace the L_cycle recursions with a root-finding operation
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                z_L = self._root_solve(
                    z_L = z_L,
                    input_injection = z_H + x,
                    seq_dict = seq_dict
                )
                z_H = self.L_level(z_H, z_L, **seq_dict)
                self._nfe += 1


        z_jac = z_L.detach()
        injection = z_H + x

        # Only pay for the extra L_level pass + VJPs when the penalty enters the loss or the norm is logged
        use_penalty = self.compute_jacobian_reg
        jac_reg = torch.zeros((), device=z_L.device)
        jac_norm = None
        if self.training and torch.is_grad_enabled() and (use_penalty or self.root_config.log_jacobian_norm):
            z_phantom = z_L.detach().requires_grad_(True)
            jac_norm = self._jacobian_regularizer(z_phantom, injection, seq_dict, create_graph=use_penalty)
            if use_penalty:
                jac_reg = jac_norm
                self._nfe += 1  # a logging-only estimate is a diagnostic, not part of the model's computation

        # Phantom Graidient iterations
        #TODO: Von-neuman convergance could also be added
        damp = self.root_config.phantom_gradient_lambda
        for _ in range(self.root_config.bptt_through):
            z_jac = z_jac*damp + (1-damp) * self._L_grad(z_jac, injection, seq_dict)

        z_H = self._L_grad(z_H, z_jac, seq_dict)
        self._nfe += self.root_config.bptt_through + 1

        
        # Clip out the puzzle embedding dims from the decoder
        if self.config.puzzle_emb_ndim != 0:
            logits = self.lm_head(z_H[:, self.config.puzzle_emb_len :]).contiguous()
        else:
            logits = self.lm_head(z_H).contiguous()

        q = self.q_head(z_H[:, 0]).to(torch.float32) 
        
        info = {
            "q_halt_logits": q[..., 0],
            "logits": logits,
            "jacobian_reg": jac_reg,
            "nfe": self._nfe,
        }
        if jac_norm is not None:
            info["jacobian_norm"] = jac_norm.detach()
        if self._last_info is not None:
            # per-row fixed-point error of the last solve (the z_L that is carried): ||f(z)-z|| and ||f(z)-z|| / ||f(z)||,
            # each the lowest the solver reached
            info["solver_abs_residual"] = self._last_info["abs_lowest"]
            info["solver_rel_residual"] = self._last_info["rel_lowest"]
        return LatentCarry(z_H.detach(), z_L.detach()), info


class TRMRoot(TRMModel):
    def __init__(self, config: TRMRootConfig):
        # config = TRMRootConfig.model_validate(config)
        # super().__init__(config.TRM_cfg)
        super().__init__(config)
        print("~"*80)
        print(config)
        print("DEQ architecture initialized")
        self.root_config = TRMRootConfig(**config)
        self.inner = TRMRoot_Inner(_add_puzzle_emb_positions(config))