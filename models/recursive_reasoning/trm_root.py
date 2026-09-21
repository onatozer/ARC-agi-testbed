from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.func import functional_call, jacrev
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
    TRMModel,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

class TRMRootConfig(BaseModel):
    # TRM_cfg: TRMConfig = Field(default_factory=TRMConfig)
    model_config = ConfigDict(extra='allow')

    deq_max_iter: int = 8             # forward Anderson iterations
    deq_min_iter: int = 4
    deq_tol: float = 1e-3
    deq_anderson_m: int = 5           # history length of Anderson solver
    deq_anderson_beta: float = 1.0

    bptt_through: int = 6 
    phantom_gradient_lambda: float = .5
    jacobian_reg_gamma: float = 0.6  
    jacobian_reg_n_samples: int = 1

class TRMRoot_Inner(InnerNetwork):
    def __init__(self, config: TRMRootConfig):
        super().__init__(config)
        # super().__init__(config.TRM_cfg)
        config = TRMRootConfig(**config) if isinstance(config, dict) else config
        self.root_cfg = config
        
    def _root_solve(self, z_L: torch.tensor, input_injection, seq_dict):
        """Uses the torch DEQ library to compute the fixed point of the inner network

        Returns:
            z: The output tensor of the equilibrium finding, attached to the computational graph
            jac_reg: The jacobian regularization penalty
        """
        cfg = self.root_cfg

        # Anderson solver only works in f32 precision type, but we may not want that for our network
        def fmap(z):
            z = z.to(getattr(torch, cfg.forward_dtype))
            z_out = self.L_level(z, input_injection, **seq_dict).float() #NOTE: Potentially reshape to be friendlier to anderson solver?? 
            return z_out

        # Find equilibrium point, as estimated by the anderson solver
        with torch.no_grad():
            z_star, _, info = anderson_solver(
                func = fmap,
                x0 = z_L.float(),
                max_iter=int(cfg.deq_max_iter),
                tol=float(cfg.deq_tol),
                min_iter=int(cfg.deq_min_iter),
                m=int(cfg.deq_anderson_m),
                beta=float(cfg.deq_anderson_beta),
            )

        # Phantom-gradient: re-run the last K iterates WITH grad starting from
        # z_star, so the loss can backprop through K layer applications.
        K = max(1, int(cfg.bptt_through))
        damp = cfg.phantom_gradient_lambda
        z = z_star.detach()
        
        #TODO: Von-neuman convergance could also be added
        for _ in range(K):
            z = z*damp + (1-damp) * self.L_level(z, input_injection, **seq_dict)

        # Compute the Hutchinson estimation of the forbenius norm of the Jacobain

        jacobian = jacrev(self.L_level, argnums=0)(z_star, input_injection)
        print("*"*20, "Jacobain", "*"*20)
        print(jacobian.shape)

        n = cfg.jacobian_reg_n_samples
        noise = torch.randn(n, jacobian.shape[0])
        jacobian_norm = noise @ jacobian
        jac_reg = torch.linalg.vector_norm(jacobian_norm, ord=2) if n == 1 else torch.linalg.matrix_norm(jacobian_norm, ord=2)/n

        # Convergence diagnostics for monitoring
        self._last_info = info
        return z, jac_reg*cfg.jacobian_rec_gamma

    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the network using a fixed-root solver to compute z_L,
        then doing extra phantom-gradient iterations so that the inner network graph can be backpropagated through.
        """
        z_H, z_L = carry.z_H, carry.z_L
        x = self._input_embeddings(batch["inputs"])
        seq_dict = {"cos_sin": self._cos_sin()}
        
        z_L, jac_reg  = self._root_solve(
            z_L = carry.z_L,
            input_injection = z_H + x,
            seq_dict = seq_dict
        )
        z_H = self.L_level(z_H, z_L, **seq_dict)
        
        # Clip out the puzzle embedding dims from the decoder
        if self.config.puzzle_emb_ndim != 0:
            logits = self.lm_head(z_H[:, self.config.puzzle_emb_len :]).contiguous()
        else:
            logits = self.lm_head(z_H).contiguous()

        q = self.q_head(z_H[:, 0]).to(torch.float32) 
        
        info = {
            "q_halt": q,
            "logits": logits,
            "jacobain_reg": jac_reg,
        }
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
        self.inner = TRMRoot_Inner(config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb