from typing import Optional, Tuple, Union
import einops
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.functional import scaled_dot_product_attention

from models.common import trunc_normal_init_



CosSin = Union[Tuple[torch.Tensor, torch.Tensor], None]


def _find_multiple(value: int, divisor: int) -> int:
    return (-(value // -divisor)) * divisor


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

class RotaryEmbedding2D(nn.Module):
    """Axial 2D RoPE for a row-major (height, width) grid of tokens, returned as per-token (cos, sin) tables
    so Attention can apply it like 1D RoPE: a quarter of the rotated pairs turn with the row, a quarter with the column.
    The prefix_len leading tokens (e.g. puzzle embedding slots) get angle 0, i.e. no rotation, like ViT's [class] token
    carrying no spatial position."""
    def __init__(self, dim: int, max_height: int, max_width: int, base: float, prefix_len: int = 0, device=None) -> None:
        super().__init__()
        dim_half = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_half, 2, dtype=torch.float32, device=device) / dim_half))
        rows = torch.arange(max_height, dtype=torch.float32, device=device).repeat_interleave(max_width)
        cols = torch.arange(max_width, dtype=torch.float32, device=device).repeat(max_height)
        freqs = torch.cat((torch.outer(rows, inv_freq), torch.outer(cols, inv_freq)), dim=-1)
        freqs = torch.cat((freqs.new_zeros(prefix_len, dim_half), freqs), dim=0)

        # Same (freqs, freqs) layout as RotaryEmbedding, so rotate_half pairs dim i with i + dim/2
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class ALiBi2D(nn.Module):
    """ViTARC 2D-RPE (arXiv 2410.06405, eq. 12): a per-head additive attention bias of -r * Manhattan distance between
    grid tokens on a row-major (height, width) grid, with slope r_before for keys at or before the query in raster
    order and r_after for keys after it. Slopes follow ALiBi's geometric 2^(-8/H) ratio, starting at 2^-1 (before)
    and 2^-0.5 (after). The prefix_len leading tokens (e.g. puzzle embedding slots) get no bias to or from anything."""
    def __init__(self, num_heads: int, height: int, width: int, prefix_len: int = 0, dtype: torch.dtype = torch.float32, device=None) -> None:
        super().__init__()
        rows = torch.arange(height, device=device).repeat_interleave(width)
        cols = torch.arange(width, device=device).repeat(height)
        dist = ((rows[:, None] - rows[None, :]).abs() + (cols[:, None] - cols[None, :]).abs()).float()

        ratio = 2.0 ** (-8.0 / num_heads)
        powers = ratio ** torch.arange(num_heads, dtype=torch.float32, device=device)
        r_before, r_after = 2.0 ** -1 * powers, 2.0 ** -0.5 * powers

        n = height * width
        before = torch.ones(n, n, dtype=torch.bool, device=device).tril()  # key j <= query i
        slope = torch.where(before, r_before[:, None, None], r_after[:, None, None])

        bias = torch.zeros(num_heads, prefix_len + n, prefix_len + n, device=device)
        bias[:, prefix_len:, prefix_len:] = -slope * dist
        self.bias = nn.Buffer(bias.to(dtype), persistent=False)

    def forward(self) -> torch.Tensor:
        return self.bias


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value)) # needed for scaled_dot_product_attention but not flash_attn_func
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=attn_bias, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore  # not .view: the math SDPA backend returns a non-contiguous tensor
        return self.o_proj(attn_output)

class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float = 2.67):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class ConvSwiGLU(nn.Module):
    """SwiGLU with a depthwise short convolution over the token axis after the gating (URM, arXiv 2512.14693):
    Y = SiLU(DWConv_k(SiLU(G) * U)) W_down. Padding follows the official code (k//2 on both sides, keep the first T),
    so with k=2 each token mixes itself and the token before it."""
    def __init__(self, hidden_size: int, expansion: float, conv_kernel: int = 2):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.dwconv       = nn.Conv1d(inter, inter, conv_kernel, padding=conv_kernel // 2, groups=inter, bias=True)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        h = (F.silu(gate) * up).transpose(1, 2)
        # Cast to the input dtype, as CastedLinear does
        h = F.conv1d(h, self.dwconv.weight.to(h.dtype), self.dwconv.bias.to(h.dtype),
                     padding=self.dwconv.padding, groups=self.dwconv.groups)[..., : x.shape[1]]
        return self.down_proj(F.silu(h).transpose(1, 2))

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
