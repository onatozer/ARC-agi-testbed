"""ViTARC (arXiv 2410.06405) positional options: 2D sinusoidal APE and the PEmixer."""

import pytest
import torch

from models.recursive_reasoning.merged_arch import _sinusoid, _sinusoid_2d
from tests.fixtures import tiny_model_and_batch


def test_sinusoid_2d_layout():
    h, w, dim, prefix = 3, 4, 16, 2
    pe = _sinusoid_2d(h, w, dim, prefix)
    assert pe.shape == (prefix + h * w, dim)
    assert torch.all(pe[:prefix] == 0)

    grid = pe[prefix:].view(h, w, dim)
    row_half, col_half = grid[..., : dim // 2], grid[..., dim // 2 :]
    # The first half only depends on the row, the second only on the column
    assert torch.equal(row_half, row_half[:, :1].expand_as(row_half))
    assert torch.equal(col_half, col_half[:1].expand_as(col_half))
    assert torch.equal(row_half[:, 0], _sinusoid(torch.arange(h), dim // 2))
    assert torch.equal(col_half[0], _sinusoid(torch.arange(w), dim // 2))


def test_absolute_2d_buffer_skips_puzzle_prefix():
    model, _ = tiny_model_and_batch(pos_encodings="absolute_2d", puzzle_emb_ndim=32, puzzle_emb_len=2, board_height=4, board_width=4)
    pe = model.model.inner.absolute_pos_embedding
    assert pe.shape == (2 + 16, 32)
    assert torch.equal(pe, _sinusoid_2d(4, 4, 32, 2).to(pe.dtype).to(pe.device))


def test_absolute_2d_rejects_mismatched_board():
    with pytest.raises(ValueError, match="does not match"):
        tiny_model_and_batch(pos_encodings="absolute_2d", board_height=4, board_width=5)


@pytest.mark.parametrize("pos_encodings", ["learned", "absolute", "absolute_2d"])
def test_pe_mixer_matches_unmixed_sum_at_init(pos_encodings):
    plain, batch = tiny_model_and_batch(pos_encodings=pos_encodings)
    mixed, _ = tiny_model_and_batch(pos_encodings=pos_encodings, pe_mixer=True)
    mixed.load_state_dict(plain.state_dict(), strict=False)  # same token/pos weights; mixer keeps its init

    args = (batch["inputs"], batch["puzzle_identifiers"])
    # Equal up to bf16 rounding: alpha*x + beta*pos vs 0.707*(x + pos)
    torch.testing.assert_close(mixed.model.inner._input_embeddings(*args), plain.model.inner._input_embeddings(*args), atol=1e-2, rtol=1.6e-2)


def test_pe_mixer_rejects_rope():
    with pytest.raises(ValueError, match="pe_mixer"):
        tiny_model_and_batch(pe_mixer=True)


@pytest.mark.parametrize("visual_attention", [False, True])
def test_absolute_2d_with_pe_mixer_trains(visual_attention):
    extra = dict(visual_attention=True, patch_height=2, patch_width=2) if visual_attention else {}
    model, batch = tiny_model_and_batch(pos_encodings="absolute_2d", pe_mixer=True, **extra)
    carry = model.initial_carry(batch)
    _, loss, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
    loss.backward()

    inner = model.model.inner
    assert torch.isfinite(loss)
    assert inner.pe_alpha.grad is not None and inner.pe_alpha.grad.abs().sum() > 0
    assert inner.pe_beta.grad is not None and inner.pe_beta.grad.abs().sum() > 0
