from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math

import torch
from pydantic import BaseModel
from torch import nn

from models.common import trunc_normal_init_
from models.layers import Attention, CastedEmbedding, CastedLinear, CosSin, RotaryEmbedding, RotaryEmbedding2D, SwiGLU, rms_norm

@dataclass
class LatentCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class ModelCarry:
    inner_carry: LatentCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]
    global_step: Optional[int] = None
    halt_counter: Optional[torch.Tensor] = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class TRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    board_height: Optional[int] = None
    board_width: Optional[int] = None
    halt_max_steps: int
    halt_exploration_prob: float
    disable_act_halt: bool = False
    act_inference: bool = False
    halt_threshold: float = 0.0
    halt_confirm_steps: int = 0
    halt_confirm_mode: str = "consecutive"
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False
    no_ACT_continue: bool = True

    puzzle_emb_len: int = 16
    puzzle_emb_ndim: int

    # Noise injection params
    lambda_: float = 0.95
    noise_scale: float = 0.01
    H_init_std: float = 1.0
    L_init_std: float = 1.0

    random_initialization: bool = True



class Block(nn.Module):
    def __init__(self, config: TRMConfig) -> None:
        super().__init__()
        self.config = config
        if config.mlp_t:
            self.mlp_t = SwiGLU(config.seq_len, config.expansion)
        else:
            self.self_attn = Attention(config.hidden_size, 
                                       config.hidden_size // config.num_heads, 
                                       config.num_heads, 
                                       config.num_heads, 
                                       causal=False)
            
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = rms_norm(hidden_states + self.mlp_t(hidden_states), variance_epsilon = self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )
        return rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)


class Blocks(nn.Module):
    def __init__(self, layers: list[Block], config: TRMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.lambda_ = config.lambda_
        self.noise_scale = config.noise_scale

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        updated_states = hidden_states + input_injection

        for layer in self.layers:
            updated_states = layer(hidden_states=updated_states, **kwargs)
        noise = torch.randn_like(updated_states) * self.noise_scale
    
        return (1 - self.lambda_) * hidden_states + self.lambda_*(updated_states) + noise


class InnerNetwork(nn.Module):
    def __init__(self, config: TRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, embed_std, self.forward_dtype)
        self._init_pos(embed_std)

        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True) 
        #TODO: The way the Q-head is being implemented has to be wrong, 
        # like no way it's optimal to use only the puzzle embedding and only the first of the 2 output dims

        self.L_level = Blocks(layers = [Block(config) for _ in range(config.L_layers)], config = self.config)
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        
        self._random_reset_field_stds = {"z_H": float(config.H_init_std), "z_L": float(config.L_init_std)}
        self._random_reset_z_H_buffer: Optional[torch.Tensor] = None
        self._random_reset_z_L_buffer: Optional[torch.Tensor] = None
        self.random_initialization = config.random_initialization
        
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _board_dims(self) -> Tuple[int, int]:
        if self.config.board_height is not None and self.config.board_width is not None:
            return self.config.board_height, self.config.board_width
        board_size = int(self.config.seq_len**0.5)
        if board_size * board_size != self.config.seq_len:
            raise ValueError(f"seq_len {self.config.seq_len} is not a perfect square. Specify board_height and board_width explicitly.")
        return board_size, board_size

    def _init_pos(self, embed_std: float) -> None:
        def _sinusoidal(length: int, dim: int) -> torch.Tensor:
            pe = torch.zeros(length, dim, dtype=torch.float32)
            if dim == 0:
                return pe
            pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
            angles = pos * div.unsqueeze(0)
            pe[:, 0::2] = torch.sin(angles)
            pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])
            return pe
        
        head_dim = self.config.hidden_size // self.config.num_heads
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(head_dim, self.config.seq_len, self.config.rope_theta)

        elif self.config.pos_encodings == "rope2d":
            h, w = self._board_dims()
            self.rotary_emb = RotaryEmbedding2D(head_dim, h, w, self.config.rope_theta)

        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len, self.config.hidden_size, embed_std, self.forward_dtype)
            
        elif self.config.pos_encodings == "absolute":
            self.register_buffer("absolute_pos_embedding", _sinusoidal(self.config.seq_len, self.config.hidden_size).to(self.forward_dtype), persistent=True)

    def _cos_sin(self) -> CosSin:
        return self.rotary_emb() if hasattr(self, "rotary_emb") else None

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids.to(torch.int32))
        if self.config.pos_encodings == "learned":
            return self.embed_scale* 0.707106781 * (x + self.embed_pos.embedding_weight.to(self.forward_dtype))
        if self.config.pos_encodings == "absolute":
            return self.embed_scale* 0.707106781 * (x + self.absolute_pos_embedding.to(device=x.device, dtype=self.forward_dtype))

        return self.embed_scale*x

    def empty_carry(self, batch_size: int, device: str = "cpu") -> LatentCarry:
        """Creates a LatentCarry containing no values"""
        shape = (batch_size, self.config.seq_len, self.config.hidden_size)
        return LatentCarry(torch.empty(shape, dtype=self.forward_dtype, device=device), torch.empty(shape, dtype=self.forward_dtype, device=device))

    def reset_carry(self, reset_flag: torch.Tensor, carry: LatentCarry, reset_indices: Optional[torch.tensor]) -> LatentCarry:
        """
        Resets the initial carry before the recursion process begins. If random initilization is on, samples from guassian distribution.
        Otherwise, loads identical distribution from initilazation.
        """
        # Edge case to check that the carry is actually a populated tensor, broadcast breaks if there's a zero dim
        if 0 in carry.z_L.shape or 0 in carry.z_H.shape:
            carry = self.empty_carry(batch_size = self.config.batch_size, device = carry.z_L.device)
        
        if not self.random_initialization:

            h = self.H_init.to(device=carry.z_H.device, dtype=carry.z_H.dtype).view(1, 1, -1)
            l = self.L_init.to(device=carry.z_L.device, dtype=carry.z_L.dtype).view(1, 1, -1)
            mask = reset_flag.to(device=carry.z_H.device).view(-1, 1, 1)
            
            return LatentCarry(torch.where(mask, h, carry.z_H), torch.where(mask, l, carry.z_L))
        
        else:
            reset_indices = reset_flag.nonzero(as_tuple=False).flatten() if reset_indices is None else reset_indices
            n = int(reset_indices.numel())
            
            if n == 0: # Nothing to reset
                return carry
            
            for name in ("z_H", "z_L"):
                state = getattr(carry, name)
                shape = state.shape[1:]
                buf_name = f"_random_reset_{name}_buffer"
                buf = getattr(self, buf_name)

                # Deciding whether or not to reinitialize the buffer
                if buf is None or buf.device != state.device or buf.dtype != state.dtype or buf.shape[1:] != shape or buf.shape[0] < n:
                    buf = torch.empty((n if buf is None else max(n, buf.shape[0] * 2),) + shape, dtype=state.dtype, device=state.device)
                    setattr(self, buf_name, buf)
                
                init = trunc_normal_init_(buf[:n], std=float(self._random_reset_field_stds[name]))
                state[reset_indices.to(device=state.device)] = init
                
            return carry

    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z_H, z_L = carry.z_H, carry.z_L
        x = self._input_embeddings(batch["inputs"])
        seq_dict = {"cos_sin": self._cos_sin()}

        # Run the recursion cycle for H_cycle -1 times without the gadient
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + x, **seq_dict)
                z_H = self.L_level(z_L, z_H, **seq_dict) 

        # Finally, last recursion cycle is ran with the gradient attached
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + x, **seq_dict)
        z_H = self.L_level(z_L, z_H, **seq_dict) 

        # Clip out the puzzle embedding dims from the decoder
        if self.config.puzzle_emb_ndim != 0:
            # The slice [:, puzzle_emb_len:] strips the prepended summary slot(s) (the [CLS]-like position 0 that the q_head reads),
            # so the logits line up 1:1 with actual puzzle cells.
            logits = self.lm_head(z_H[:, self.config.puzzle_emb_len :]).contiguous()
        else:
            logits = self.lm_head(z_H).contiguous()

        # Struggling to find a super good reason for why z_H[:,0] is being passed in and not z_H. When there's a puzzle embedding, its added to the front of z_H, 
        # so z_H[:,0] is just running on the puzzle embedding of each batch, but otherwise it's hard to interpert exactly what z_H[0] is supposed to be
        q = self.q_head(z_H[:, 0]).to(torch.float32) 

        info = {
            "q_halt_logits": q[..., 0],
            "logits": logits,
        }
        return LatentCarry(z_H, z_L), info


def _disable_torch_compile(fn: Any) -> Any:
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    if disable is None:
        disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    return disable(fn) if disable is not None else fn


@_disable_torch_compile
def _reset_rows(
    inner: Any,
    reset_flag: torch.Tensor,
    inner_carry: LatentCarry,
    current_data: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[LatentCarry, Dict[str, torch.Tensor]]:
    # print("Reset rows called with", inner, reset_flag.shape, batch)
    idx = reset_flag.nonzero(as_tuple=False).flatten()
    inner_carry = inner.reset_carry(reset_flag, inner_carry, reset_indices=idx)
    current_data = {k: v.to(device=batch[k].device) for k, v in current_data.items()}
    
    if idx.numel() > 0:
        idx = idx.to(device=device)
        for k, v in current_data.items():
            v[idx] = batch[k][idx]
            
    return inner_carry, current_data


class TRMModel(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.config = TRMConfig(**config_dict)
        print(config_dict)
        self.inner = InnerNetwork(self.config)
        print("*"*80)
        print("Merged arch live!!")
        print("*"*80)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ModelCarry:
        """Creates the an empty 'ModelCarry' object, which contains all the information the class needs throughout the recursion process"""
        batch_size, device = batch["inputs"].shape[0], batch["inputs"].device
        c = self.inner.empty_carry(batch_size)

        c.z_H = c.z_H.to(device)
        c.z_L = c.z_L.to(device)

        return ModelCarry(
            inner_carry=c,
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()},
            halt_counter=torch.zeros((batch_size,), dtype=torch.int32, device=device),
        )

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
        inner_carry, fpass_info = self.inner(inner_carry, current_data)
        q_halt = fpass_info["q_halt_logits"]
        
        # ACT decision with ranomdizer that continues according to some exploration probability
        with torch.no_grad():
            steps = steps + 1
            halted = steps >= self.config.halt_max_steps
            if self.training and self.config.halt_max_steps > 1:
                halted = (halted | (q_halt > 0)) & (
                    steps
                    >= (
                        (torch.rand_like(q_halt) < self.config.halt_exploration_prob)
                        * torch.randint_like(steps, low=2, high=self.config.halt_max_steps + 1)
                    )
                )
        return ModelCarry(inner_carry, steps, halted, current_data), fpass_info

