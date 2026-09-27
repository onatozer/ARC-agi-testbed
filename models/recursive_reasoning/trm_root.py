from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.func import functional_call, jacrev
from torch.nn.attention import SDPBackend, sdpa_kernel
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

        return z_star 

    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the network using a fixed-root solver to compute z_L,
        then doing extra phantom-gradient iterations so that the inner network graph can be backpropagated through.
        """
        z_H, z_L = carry.z_H, carry.z_L
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        seq_dict = {"cos_sin": self._cos_sin()}
        self._nfe = 0  # L_level evaluations in this forward (solver iterations are data-dependent)

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

        # Only pay for the extra L_level pass + VJPs when the penalty enters the loss or the norm is logged
        use_penalty = self.compute_jacobian_reg
        jac_reg = torch.zeros((), device=z_L.device)
        jac_norm = None
        if self.training and torch.is_grad_enabled() and (use_penalty or self.root_config.log_jacobian_norm):
            z_phantom = z_L.detach().requires_grad_(True)
            jac_norm = self._jacobian_regularizer(z_phantom, z_H + x, seq_dict, create_graph=use_penalty)
            if use_penalty:
                jac_reg = jac_norm
                self._nfe += 1  # a logging-only estimate is a diagnostic, not part of the model's computation
        
        # Phantom Graidient iterations 
        #TODO: Von-neuman convergance could also be added
        damp = self.root_config.phantom_gradient_lambda
        for _ in range(self.root_config.bptt_through):
            z_jac = z_jac*damp + (1-damp) * self.L_level(z_jac, z_H + x, **seq_dict)
        
        z_H = self.L_level(z_H, z_jac, **seq_dict)
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
        return LatentCarry(z_H, z_L), info


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