from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
import torch.nn.functional as F

from models.common import trunc_normal_init_
from models.layers import ALiBi2D, Attention, CastedEmbedding, CastedLinear, CosSin, RotaryEmbedding, RotaryEmbedding2D, ConvSwiGLU, SwiGLU, rms_norm
from models.sparse_embedding import CastedSparseEmbedding
from einops.layers.torch import Rearrange

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
    nfe: Optional[torch.Tensor] = None  # per-row L_level evaluations since the puzzle was (re)started

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class TRMConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    
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
    activation_function: str = "swiglu"  # "swiglu" or "conv_swiglu" (URM's ConvSwiGLU)
    conv_kernel: int = 2  # ConvSwiGLU depthwise conv kernel size

    # seq_len includes the puzzle_emb_len prefix slots; see _add_puzzle_emb_positions
    puzzle_emb_len: int = 16
    puzzle_emb_ndim: int
    num_puzzle_identifiers: int = 1

    # Noise injection params
    lambda_: float = 0.95
    noise_scale: float = 0.01
    H_init_std: float = 1.0
    L_init_std: float = 1.0

    random_initialization: bool = True
    visual_attention: bool = False
    pe_mixer: bool = False  # ViTARC PEmixer: learned per-dim weights on the token and positional embeddings
    alibi_2d: bool = False  # ViTARC 2D-RPE: additive 2D ALiBi attention bias


def _add_puzzle_emb_positions(config_dict: dict) -> dict:
    """Extends seq_len by the puzzle-embedding slots prepended to the input (none when puzzle_emb_ndim == 0).
    A puzzle_emb_len of 0 means ceil(puzzle_emb_ndim / hidden_size) slots."""
    config_dict = dict(config_dict)
    puzzle_emb_ndim = int(config_dict.get("puzzle_emb_ndim", 0) or 0)
    puzzle_emb_len = 0
    if puzzle_emb_ndim > 0:
        puzzle_emb_len = int(config_dict.get("puzzle_emb_len", 0) or 0) or -(puzzle_emb_ndim // -int(config_dict["hidden_size"]))
    config_dict["puzzle_emb_len"] = puzzle_emb_len
    config_dict["seq_len"] = int(config_dict["seq_len"]) + puzzle_emb_len
    return config_dict


def _sinusoid(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Fixed sinusoidal encoding of the given positions, interleaved sin/cos over dim channels."""
    pe = torch.zeros(positions.shape[0], dim, dtype=torch.float32, device=positions.device)
    if dim == 0:
        return pe
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) * (-math.log(10000.0) / dim))
    angles = positions.to(torch.float32).unsqueeze(1) * div.unsqueeze(0)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])
    return pe


def _sinusoid_2d(height: int, width: int, dim: int, prefix_len: int) -> torch.Tensor:
    """ViTARC 2D APE (arXiv 2410.06405, eq. 9): concat(sinusoid(row), sinusoid(col)) for a row-major grid.
    The prefix_len leading tokens (puzzle embedding slots) get zeros, i.e. no position."""
    rows = torch.arange(height).repeat_interleave(width)
    cols = torch.arange(width).repeat(height)
    pe = torch.cat((_sinusoid(rows, dim // 2), _sinusoid(cols, dim - dim // 2)), dim=-1)
    return torch.cat((pe.new_zeros(prefix_len, dim), pe), dim=0)



class Block(nn.Module):
    def __init__(self, config: TRMConfig, seq_len: int) -> None:
        super().__init__()
        self.config = config

        if config.activation_function == "swiglu":
            self.activation_function = SwiGLU
        elif config.activation_function == "conv_swiglu":
            self.activation_function = ConvSwiGLU
        else:
            raise ValueError(f"Unknown activation function {config.activation_function!r}; expected 'swiglu' or 'conv_swiglu'.")
        
        if config.mlp_t:
            self.mlp_t = self.activation_function(seq_len, config.expansion)
        else:
            self.self_attn = Attention(config.hidden_size, 
                                       config.hidden_size // config.num_heads, 
                                       config.num_heads, 
                                       config.num_heads, 
                                       causal=False)
            
        self.mlp = self.activation_function(hidden_size=config.hidden_size, expansion=config.expansion)
        
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = rms_norm(hidden_states + self.mlp_t(hidden_states), variance_epsilon = self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states, attn_bias=attn_bias),
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
    def __init__(self, config: TRMConfig, seq_len: Optional[int] = None) -> None:
        super().__init__()
        config = TRMConfig(**config) if isinstance(config, dict) else config 
        self.config = config
        # Number of tokens the blocks see; differs from config.seq_len when the input is patched
        self.seq_len = config.seq_len if seq_len is None else seq_len
        self.forward_dtype = getattr(torch, config.forward_dtype)
        
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, embed_std, self.forward_dtype)
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                                                    batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype)
        self._init_pos(embed_std)
        if config.alibi_2d:
            self.alibi_2d = ALiBi2D(config.num_heads, *self._grid_dims(), prefix_len=config.puzzle_emb_len, dtype=self.forward_dtype)

        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True) 
        #TODO: The way the Q-head is being implemented has to be wrong, 
        # like no way it's optimal to use only the puzzle embedding and only the first of the 2 output dims

        self.L_level = Blocks(layers = [Block(config, self.seq_len) for _ in range(config.L_layers)], config = self.config)
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
        head_dim = self.config.hidden_size // self.config.num_heads
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(head_dim, self.seq_len, self.config.rope_theta)

        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.seq_len, self.config.hidden_size, embed_std, self.forward_dtype)

        elif self.config.pos_encodings == "absolute":
            self.register_buffer("absolute_pos_embedding", _sinusoid(torch.arange(self.seq_len), self.config.hidden_size).to(self.forward_dtype), persistent=True)

        elif self.config.pos_encodings == "absolute_2d":
            self.register_buffer("absolute_pos_embedding", _sinusoid_2d(*self._grid_dims(), self.config.hidden_size, self.config.puzzle_emb_len).to(self.forward_dtype), persistent=True)

        elif self.config.pos_encodings == "absolute_2d_ope":
            # ViTARC OPE (arXiv 2410.06405, eq. 13): concat(sinusoid(object), sinusoid(row), sinusoid(col)), about a third each.
            # The (row, col) part is fixed; the object part is computed per batch from batch["object_ids"]
            self.ope_dim = 2 * (self.config.hidden_size // 6)
            self.register_buffer("absolute_pos_embedding", _sinusoid_2d(*self._grid_dims(), self.config.hidden_size - self.ope_dim, self.config.puzzle_emb_len).to(self.forward_dtype), persistent=True)

        self._init_pe_mixer()

    def _grid_dims(self) -> Tuple[int, int]:
        """(height, width) of the row-major token grid the blocks see after the puzzle prefix."""
        h, w = self._board_dims()
        if h * w != self.seq_len - self.config.puzzle_emb_len:
            raise ValueError(f"board {h}x{w} does not match the {self.seq_len - self.config.puzzle_emb_len} grid tokens; set board_height and board_width.")
        return h, w

    def _init_pe_mixer(self) -> None:
        if not self.config.pe_mixer:
            return
        if self.config.pos_encodings not in ("learned", "absolute", "absolute_2d", "absolute_2d_ope"):
            raise ValueError(f"pe_mixer needs an additive pos_encodings (learned, absolute, absolute_2d or absolute_2d_ope), got {self.config.pos_encodings!r}.")
        # Initialised to the fixed 1/sqrt(2) weights of the unmixed sum, so step 0 matches pe_mixer=False
        self.pe_alpha = nn.Parameter(torch.full((self.config.hidden_size,), 0.707106781))
        self.pe_beta = nn.Parameter(torch.full((self.config.hidden_size,), 0.707106781))

    def _add_pos(self, x: torch.Tensor, object_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Adds the positional encoding to the (prefix + token) embeddings and applies the embedding scale."""
        if self.config.pos_encodings == "learned":
            pos = self.embed_pos.embedding_weight.to(self.forward_dtype)
        elif self.config.pos_encodings in ("absolute", "absolute_2d"):
            pos = self.absolute_pos_embedding.to(device=x.device, dtype=self.forward_dtype)
        elif self.config.pos_encodings == "absolute_2d_ope":
            if object_ids is None:
                raise ValueError("pos_encodings=absolute_2d_ope needs batch['object_ids']; build the dataset with object_ids=true.")
            obj = _sinusoid(object_ids.flatten(), self.ope_dim).view(*object_ids.shape, self.ope_dim)
            obj = F.pad(obj, (0, 0, self.config.puzzle_emb_len, 0)).to(self.forward_dtype)  # prefix slots have no object
            xy = self.absolute_pos_embedding.to(device=x.device, dtype=self.forward_dtype)
            pos = torch.cat((obj, xy.expand(obj.shape[0], -1, -1)), dim=-1)
        else:
            return self.embed_scale*x

        if self.config.pe_mixer:
            # ViTARC PEmixer (arXiv 2410.06405, eq. 10 / App. C weighted_sum_no_norm_vec): alpha * E_p + beta * E_pos
            return self.embed_scale * (self.pe_alpha.to(self.forward_dtype) * x + self.pe_beta.to(self.forward_dtype) * pos)
        return self.embed_scale* 0.707106781 * (x + pos)

    def _grid_dims(self) -> Tuple[int, int]:
        """(height, width) of the row-major token grid the blocks see after the puzzle prefix."""
        h, w = self._board_dims()
        if h * w != self.seq_len - self.config.puzzle_emb_len:
            raise ValueError(f"board {h}x{w} does not match the {self.seq_len - self.config.puzzle_emb_len} grid tokens; set board_height and board_width.")
        return h, w

    def _cos_sin(self) -> CosSin:
        return self.rotary_emb() if hasattr(self, "rotary_emb") else None

    def _seq_dict(self) -> Dict[str, Any]:
        """Per-sequence tensors every L_level call takes: RoPE tables and the attention bias (None when unused)."""
        return {"cos_sin": self._cos_sin(), "attn_bias": self.alibi_2d() if hasattr(self, "alibi_2d") else None}

    def _input_embeddings(self, input_ids: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            # Prepend the per-puzzle embedding, zero-padded to puzzle_emb_len full hidden-size slots
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers.to(torch.int32))
            pad = self.config.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad > 0:
                puzzle_embedding = torch.nn.functional.pad(puzzle_embedding, (0, pad))
            x = torch.cat((puzzle_embedding.view(-1, self.config.puzzle_emb_len, self.config.hidden_size), x), dim=1)
        return self._add_pos(x, object_ids)

    def empty_carry(self, batch_size: int, device: str = "cpu") -> LatentCarry:
        """Creates a LatentCarry containing no values"""
        shape = (batch_size, self.seq_len, self.config.hidden_size)
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
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        seq_dict = self._seq_dict()

        # Run the recursion cycle for H_cycle -1 times without the gadient
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + x, **seq_dict)
                z_H = self.L_level(z_H, z_L, **seq_dict)

        # Finally, last recursion cycle is ran with the gradient attached
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + x, **seq_dict)
        z_H = self.L_level(z_H, z_L, **seq_dict)

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


class InnerNetworkVIT(InnerNetwork):
    def __init__(self, config: TRMConfig):
        # One token per patch, plus the puzzle embedding prefix slots
        num_patches = (config.seq_len - config.puzzle_emb_len) // (config.patch_height * config.patch_width)
        super().__init__(config, seq_len=num_patches + config.puzzle_emb_len)

        
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_std = 1.0 / self.embed_scale

        #Initialize the VIT specific args
        patch_height = config.patch_height
        patch_width = config.patch_width
        
        # ARC cells are categorical, so each cell is one-hot encoded and the vocab plays the role of ViT's channel dim
        _, board_width = self._board_dims()
        patch_dim  = patch_height*patch_width*config.vocab_size
        dim = config.hidden_size
        self.patch_embedding = nn.Sequential(
            Rearrange('b (h p1 w p2) c -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width, w = board_width // patch_width),
            CastedLinear(patch_dim, dim, bias=False),
        )
        # Predict every cell of a patch from its token, then un-patchify back to one logit row per cell
        self.lm_head = nn.Sequential(
            CastedLinear(dim, patch_dim, bias=False),
            Rearrange('b (h w) (p1 p2 c) -> b (h p1 w p2) c', p1 = patch_height, p2 = patch_width, w = board_width // patch_width),
        )

    def _grid_dims(self) -> Tuple[int, int]:
        """(height, width) of the patch grid."""
        h, w = self._board_dims()
        return h // self.config.patch_height, w // self.config.patch_width

    def _init_pos(self, embed_std: float) -> None:
        head_dim = self.config.hidden_size // self.config.num_heads

        # When it's the vision transformer rope is just automatically gonna imply 2d rope, IMO makes more sense, and 2d rope doesn't even work for the regular trans anyways
        if self.config.pos_encodings == "rope":
            h, w = self._board_dims()
            self.rotary_emb = RotaryEmbedding2D(head_dim, h // self.config.patch_height, w // self.config.patch_width, self.config.rope_theta, prefix_len=self.config.puzzle_emb_len)

        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.seq_len, self.config.hidden_size, embed_std, self.forward_dtype)

        elif self.config.pos_encodings == "absolute":
            self.register_buffer("absolute_pos_embedding", _sinusoid(torch.arange(self.seq_len), self.config.hidden_size).to(self.forward_dtype), persistent=True)

        elif self.config.pos_encodings == "absolute_2d":
            self.register_buffer("absolute_pos_embedding", _sinusoid_2d(*self._grid_dims(), self.config.hidden_size, self.config.puzzle_emb_len).to(self.forward_dtype), persistent=True)

        elif self.config.pos_encodings == "absolute_2d_ope":
            raise ValueError("absolute_2d_ope gives each cell an object index, which is undefined for a multi-cell patch; use visual_attention=false.")

        self._init_pe_mixer()


    def _grid_dims(self) -> Tuple[int, int]:
        """(height, width) of the patch grid."""
        h, w = self._board_dims()
        return h // self.config.patch_height, w // self.config.patch_width

    def _input_embeddings(self, input_ids: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(F.one_hot(input_ids.long(), self.config.vocab_size).to(self.forward_dtype))
        
        if self.config.puzzle_emb_ndim > 0:
            # Prepend the per-puzzle embedding, zero-padded to puzzle_emb_len full hidden-size slots
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers.to(torch.int32))
            pad = self.config.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad > 0:
                puzzle_embedding = torch.nn.functional.pad(puzzle_embedding, (0, pad))
            x = torch.cat((puzzle_embedding.view(-1, self.config.puzzle_emb_len, self.config.hidden_size), x), dim=1)

        return self._add_pos(x, object_ids)


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
        self.config = TRMConfig(**_add_puzzle_emb_positions(config_dict))
        print(config_dict)
        if self.config.visual_attention:
            self.inner = InnerNetworkVIT(self.config)
        else:
            self.inner = InnerNetwork(self.config)

        print("*"*80)
        print("Merged arch live!!")
        print("*"*80)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

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
            nfe=torch.zeros((batch_size,), dtype=torch.int32, device=device),
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
        counter = torch.zeros_like(steps) if carry.halt_counter is None else torch.where(reset, torch.zeros_like(steps), carry.halt_counter.to(device=device))
        prev_latent = inner_carry
        inner_carry, fpass_info = self.inner(inner_carry, current_data)
        q_halt = fpass_info["q_halt_logits"]

        with torch.no_grad():
            # NFE = L_level (the shared block f) evaluations; inners with a variable count (e.g. DEQ solvers) report it
            step_nfe = fpass_info.pop("nfe", self.config.H_cycles * (self.config.L_cycles + 1))
            prev_nfe = carry.nfe.to(device=device) if carry.nfe is not None else torch.zeros_like(steps)
            nfe = torch.where(reset, torch.zeros_like(prev_nfe), prev_nfe) + step_nfe
            # normalised residuals over this supervision step: ||z^t - z^(t-1)|| / ||z^t||, for z_H and z_L
            for name in ("z_H", "z_L"):
                z, z_prev = getattr(inner_carry, name).float(), getattr(prev_latent, name).float()
                d_z = (z - z_prev).flatten(1).norm(dim=1)
                fpass_info[f"{name.replace('_', '')}_rel_residual"] = d_z / z.flatten(1).norm(dim=1).clamp_min(1e-8)
        
        # This section determines which puzzles in the batch to halt on based on if it's either
        # 1. Reached the max number of steps
        # 2. If the q-values from the ACT head say it's better to halt than to continue
        with torch.no_grad():
            steps = steps + 1
            is_last = steps >= self.config.halt_max_steps # Condition 1
            halted = is_last

            if self.config.halt_max_steps > 1 and (self.training or self.config.act_inference) and not self.config.disable_act_halt:
                # TRM-style ACT: halt once the predicted probability of being correct is high enough (no continue head)
                signal = q_halt > float(self.config.halt_threshold)

                if self.config.halt_confirm_steps > 0:
                    mode = str(self.config.halt_confirm_mode or "consecutive").lower()
                    counter = torch.where(signal, counter + 1, counter if mode in {"cumulative", "accumulate", "total"} else torch.zeros_like(counter))
                    halted = halted | (counter >= 1 + int(self.config.halt_confirm_steps))
                else:
                    counter = torch.where(signal, counter + 1, torch.zeros_like(counter))
                    halted = halted | signal

                # Exploration (training only): with some probability, force a random minimum number of steps before halting
                if self.training:
                    min_steps = (torch.rand_like(q_halt) < self.config.halt_exploration_prob) * torch.randint_like(steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (steps >= min_steps)

        return ModelCarry(inner_carry, steps, halted, current_data, halt_counter=counter, nfe=nfe), fpass_info

