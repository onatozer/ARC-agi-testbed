"""VARC baseline: "ARC Is a Vision Problem!" (Hu et al., arXiv 2511.14761).

Each (input, output) pair is image-to-image translation: the raw grid is scaled by a random integer and
translated onto a fixed canvas, a ViT with 2x2 patches and one learned task token per (augmented) puzzle
classifies every canvas pixel, and the per-pixel logits are averaged back onto the raw grid.
Details the paper leaves open follow the authors' code (github.com/lillian039/VARC, src/ARC_ViT.py and
src/ARC_loader.py): post-LN blocks with a GELU MLP, learned absolute position embedding plus 2D RoPE,
a key-padding mask over background patches, and one shared scale/offset for input and output.

Dihedral and color augmentation stays in the dataset, where every augmented puzzle already has its own
puzzle_identifier, i.e. its own task token. Scale and translation augmentation happen here, per row.

Token ids follow dataset/build_arc_dataset.py: PAD=0, EOS=1, colors 0..9 -> 2..11. The canvas uses the same
ids as VARC's classes: PAD plays background [BG] and EOS (the dataset's row/column just past the grid)
plays the border [BD], so the head predicts dataset tokens directly.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
import torch.nn.functional as F

from models.layers import RotaryEmbedding2D, apply_rotary_pos_emb
from models.sparse_embedding import CastedSparseEmbedding

PAD_ID = 0
EOS_ID = 1
FIRST_COLOR_ID = 2
ARC_VOCAB_SIZE = 12


@dataclass
class VARCCarry:
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]
    global_step: Optional[int] = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class VARCConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int = 1
    puzzle_emb_ndim: int = 0  # 0 = no task token, hidden_size = one task token
    board_height: Optional[int] = None
    board_width: Optional[int] = None

    image_size: int = 64
    patch_size: int = 2
    hidden_size: int = 512
    num_layers: int = 10
    num_heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0
    task_emb_init_std: float = 0.02
    forward_dtype: str = "bfloat16"

    halt_max_steps: int = 1  # read by the loss heads and pretrain.py; VARC always halts after one pass


class VARCAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor], attn_mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.num_heads, self.head_dim).unbind(2)
        q, k = apply_rotary_pos_emb(q, k, *cos_sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        return self.proj_dropout(self.proj(out.transpose(1, 2).reshape(B, S, D)))


class VARCBlock(nn.Module):
    """Post-LN encoder layer, as in the reference ARCTransformerEncoderLayer."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = VARCAttention(hidden_size, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, hidden_size), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor], attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.attn(x, cos_sin, attn_mask)))
        return self.norm2(x + self.mlp(x))


class VARCInner(nn.Module):
    def __init__(self, config: VARCConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.forward_dtype = getattr(torch, c.forward_dtype)
        self.grid = c.image_size // c.patch_size
        self.task_token_len = 1 if c.puzzle_emb_ndim > 0 else 0

        self.color_embed = nn.Embedding(c.vocab_size, c.hidden_size)
        self.patch_embed = nn.Conv2d(c.hidden_size, c.hidden_size, kernel_size=c.patch_size, stride=c.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid, c.hidden_size))
        if self.task_token_len:
            self.puzzle_emb = CastedSparseEmbedding(
                c.num_puzzle_identifiers, c.hidden_size, batch_size=c.batch_size, init_std=c.task_emb_init_std, cast_to=torch.float32
            )
        self.rotary = RotaryEmbedding2D(c.hidden_size // c.num_heads, self.grid, self.grid, base=c.rope_theta, prefix_len=self.task_token_len)
        self.dropout = nn.Dropout(c.dropout)
        self.layers = nn.ModuleList([VARCBlock(c.hidden_size, c.num_heads, c.mlp_dim, c.dropout) for _ in range(c.num_layers)])
        self.norm = nn.LayerNorm(c.hidden_size)
        self.head = nn.Linear(c.hidden_size, c.patch_size * c.patch_size * c.vocab_size)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def place_on_canvas(self, inputs: torch.Tensor, labels: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick a random integer scale and offset per row and map every canvas pixel to its raw-frame cell.

        Returns (canvas_inputs [B, S, S] token ids with everything but colors as PAD/[BG],
                 raw_index [B, S*S] flat raw cell per canvas pixel, H*W where the pixel maps to no cell).
        The extent is taken over the non-PAD cells of the input, plus the label when given (training),
        as the reference does with max(input, output) size; at eval only the input is known.
        """
        c = self.config
        B, S = inputs.shape[0], c.image_size
        H, W = c.board_height, c.board_width
        device = inputs.device

        occupied = (inputs != PAD_ID).view(B, H, W)
        if labels is not None:
            occupied = occupied | ((labels != PAD_ID) & (labels >= 0)).view(B, H, W)
        rows, cols = occupied.any(2), occupied.any(1)
        r_idx = torch.arange(H, device=device)
        c_idx = torch.arange(W, device=device)
        r0 = torch.where(rows, r_idx, H).amin(1).clamp_max(H - 1)
        c0 = torch.where(cols, c_idx, W).amin(1).clamp_max(W - 1)
        ext_h = (torch.where(rows, r_idx, -1).amax(1) - r0 + 1).clamp_min(1)
        ext_w = (torch.where(cols, c_idx, -1).amax(1) - c0 + 1).clamp_min(1)

        # Random integer scale so the scaled grid fits, then a random offset keeping it fully visible
        max_scale = (S // torch.maximum(ext_h, ext_w)).clamp_min(1)
        scale = 1 + (torch.rand(B, device=device) * max_scale).long().clamp_max(max_scale - 1)
        off_y = (torch.rand(B, device=device) * (S - ext_h * scale + 1).clamp_min(1)).long()
        off_x = (torch.rand(B, device=device) * (S - ext_w * scale + 1).clamp_min(1)).long()

        pix = torch.arange(S, device=device)
        dy = pix.unsqueeze(0) - off_y.unsqueeze(1)
        dx = pix.unsqueeze(0) - off_x.unsqueeze(1)
        ry = r0.unsqueeze(1) + torch.div(dy, scale.unsqueeze(1), rounding_mode="floor")
        rx = c0.unsqueeze(1) + torch.div(dx, scale.unsqueeze(1), rounding_mode="floor")
        valid = ((dy >= 0) & (ry < H)).unsqueeze(2) & ((dx >= 0) & (rx < W)).unsqueeze(1)
        raw_index = torch.where(valid, ry.unsqueeze(2) * W + rx.unsqueeze(1), H * W).view(B, S * S)

        padded_inputs = F.pad(inputs, (0, 1), value=PAD_ID)
        canvas = torch.gather(padded_inputs, 1, raw_index)
        canvas = torch.where(canvas >= FIRST_COLOR_ID, canvas, PAD_ID)  # the input's EOS markers are background too
        return canvas.view(B, S, S), raw_index

    def canvas_to_raw(self, canvas_logits: torch.Tensor, raw_index: torch.Tensor) -> torch.Tensor:
        """Average per-pixel logits over each raw cell's s x s block. Cells no pixel maps to predict PAD."""
        B, _, V = canvas_logits.shape
        HW = self.config.board_height * self.config.board_width
        sums = canvas_logits.new_zeros(B, HW + 1, V).scatter_add_(1, raw_index.unsqueeze(-1).expand(-1, -1, V), canvas_logits)
        counts = canvas_logits.new_zeros(B, HW + 1).scatter_add_(1, raw_index, torch.ones_like(raw_index, dtype=canvas_logits.dtype))
        sums, counts = sums[:, :HW], counts[:, :HW, None]
        pad_logits = F.one_hot(torch.tensor(PAD_ID, device=sums.device), V).to(sums.dtype) * 100.0
        return torch.where(counts > 0, sums / counts.clamp_min(1), pad_logits)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        c = self.config
        B, S, P, G = batch["inputs"].shape[0], c.image_size, c.patch_size, self.grid
        canvas, raw_index = self.place_on_canvas(batch["inputs"], batch["labels"] if self.training else None)

        with torch.autocast(device_type=canvas.device.type, dtype=self.forward_dtype, enabled=self.forward_dtype != torch.float32):
            x = self.patch_embed(self.color_embed(canvas).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
            x = x + self.pos_embed

            # Key-padding mask: a patch is attendable if any of its pixels belongs to the input grid
            attend = (canvas != PAD_ID).view(B, G, P, G, P).any(4).any(2).view(B, G * G)
            if self.task_token_len:
                task = self.puzzle_emb(batch["puzzle_identifiers"]).view(B, 1, c.hidden_size)
                x = torch.cat((task.to(x.dtype), x), dim=1)
                attend = torch.cat((attend.new_ones(B, 1), attend), dim=1)
            x = self.dropout(x)

            cos_sin = self.rotary()
            attn_mask = attend[:, None, None, :]
            for layer in self.layers:
                x = layer(x, cos_sin, attn_mask)
            x = self.norm(x)[:, self.task_token_len:]
            logits = self.head(x)

        logits = logits.float().view(B, G, G, P, P, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, S * S, -1)
        return self.canvas_to_raw(logits, raw_index)


class VARC(nn.Module):
    """Single-pass model with the carry/halting interface ACTLossHead, InferenceLossHead and pretrain.py expect."""

    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        config = VARCConfig(**config_dict)
        if config.board_height is None or config.board_width is None:
            side = math.isqrt(config.seq_len)
            config.board_height = config.board_height or side
            config.board_width = config.board_width or config.seq_len // config.board_height
        if config.board_height * config.board_width != config.seq_len:
            raise ValueError(f"VARC needs board_height * board_width == seq_len, got {config.board_height}x{config.board_width} vs {config.seq_len}.")
        if config.vocab_size != ARC_VOCAB_SIZE:
            raise ValueError(f"VARC expects ARC token ids (PAD, EOS, 10 colors; vocab_size {ARC_VOCAB_SIZE}), got vocab_size {config.vocab_size}. Border-token datasets are not supported.")
        if config.puzzle_emb_ndim not in (0, config.hidden_size):
            raise ValueError(f"VARC uses one task token: puzzle_emb_ndim must be 0 or hidden_size ({config.hidden_size}), got {config.puzzle_emb_ndim}.")
        if config.image_size % config.patch_size or max(config.board_height, config.board_width) > config.image_size:
            raise ValueError(f"image_size {config.image_size} must be divisible by patch_size {config.patch_size} and hold the {config.board_height}x{config.board_width} frame.")
        if config.hidden_size % config.num_heads or (config.hidden_size // config.num_heads) % 4:
            raise ValueError("hidden_size / num_heads must be a multiple of 4 for 2D RoPE.")
        self.config = config
        self.inner = VARCInner(config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> VARCCarry:
        batch_size, device = batch["inputs"].shape[0], batch["inputs"].device
        return VARCCarry(
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()},
        )

    def forward(self, carry: Optional[VARCCarry], batch: Dict[str, torch.Tensor], **kwargs: Any) -> Tuple[VARCCarry, Dict[str, torch.Tensor]]:
        del carry, kwargs  # every row finishes in one pass, so each call starts from the new batch
        logits = self.inner(batch)
        batch_size, device = logits.shape[0], logits.device
        new_carry = VARCCarry(
            steps=torch.ones((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data=batch,
        )
        # No halting head in VARC; the loss heads and the ARC evaluator still read q_halt_logits
        return new_carry, {"logits": logits, "q_halt_logits": torch.zeros((batch_size,), dtype=torch.float32, device=device)}
