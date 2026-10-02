"""ViTARC (arXiv 2410.06405) object-based positional encoding (OPE)."""

import numpy as np
import pytest
import torch

from dataset.build_arc_dataset import object_index_canvas
from models.recursive_reasoning.merged_arch import _sinusoid, _sinusoid_2d
from tests.fixtures import tiny_model_and_batch


def _tokens(grid):
    return np.array(grid, dtype=np.uint8) + 2  # colors -> token ids


def test_two_objects_and_background():
    canvas = _tokens([
        [1, 1, 0, 0],
        [1, 0, 0, 2],
        [0, 0, 0, 2],
    ])
    np.testing.assert_array_equal(object_index_canvas(canvas), [
        [1, 1, 0, 0],  # each object's bounding box; black outside any box stays 0
        [1, 1, 0, 2],
        [0, 0, 0, 2],
    ])


def test_diagonal_cells_are_one_object():
    canvas = _tokens([
        [3, 0],
        [0, 3],
    ])
    np.testing.assert_array_equal(object_index_canvas(canvas), [[1, 1], [1, 1]])


def test_nested_object_keeps_its_own_index():
    canvas = _tokens([
        [4, 4, 4, 4],
        [4, 0, 5, 4],
        [4, 4, 4, 4],
    ])
    ids = object_index_canvas(canvas)
    assert ids[1, 2] != ids[0, 0]  # the inner object is painted over the ring's box
    assert set(np.unique(ids)) == {1, 2}


def test_multicolor_object_splits_by_color():
    canvas = _tokens([[6, 7]])
    np.testing.assert_array_equal(object_index_canvas(canvas), [[1, 2]])


def test_non_color_tokens_are_not_objects():
    canvas = np.array([[3, 1, 0], [1, 1, 0]], dtype=np.uint8)  # color 1, EOS, PAD
    np.testing.assert_array_equal(object_index_canvas(canvas), [[1, 0, 0], [0, 0, 0]])


def _ope_model(**overrides):
    return tiny_model_and_batch(pos_encodings="absolute_2d_ope", puzzle_emb_ndim=32, puzzle_emb_len=2, board_height=4, board_width=4, **overrides)


def test_ope_encoding_layout():
    model, batch = _ope_model()
    inner = model.model.inner
    object_ids = torch.randint(0, 5, batch["inputs"].shape, device="cuda")
    pos = inner._add_pos(torch.zeros(object_ids.shape[0], 2 + 16, 32, device="cuda", dtype=torch.bfloat16), object_ids)
    pos = pos / (inner.embed_scale * 0.707106781)

    d_o = inner.ope_dim
    assert d_o == 10
    assert torch.all(pos[:, :2] == 0)  # puzzle prefix: no object, no position
    expected_obj = _sinusoid(object_ids.flatten().cpu(), d_o).view(*object_ids.shape, d_o)
    torch.testing.assert_close(pos[:, 2:, :d_o].float().cpu(), expected_obj, atol=2e-2, rtol=0)
    expected_xy = _sinusoid_2d(4, 4, 32 - d_o, 0)
    torch.testing.assert_close(pos[0, 2:, d_o:].float().cpu(), expected_xy, atol=2e-2, rtol=0)


def test_ope_needs_object_ids():
    model, batch = _ope_model()
    with pytest.raises(ValueError, match="object_ids"):
        model(carry=model.initial_carry(batch), batch=batch, return_keys=[])


def test_ope_rejects_patches():
    with pytest.raises(ValueError, match="patch"):
        _ope_model(visual_attention=True, patch_height=2, patch_width=2)


def test_ope_trains_and_uses_object_ids():
    model, batch = _ope_model(pe_mixer=True)
    batch["object_ids"] = torch.randint(0, 5, batch["inputs"].shape, dtype=torch.int32, device="cuda")
    _, loss, _, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])
    loss.backward()
    assert torch.isfinite(loss)

    inner = model.model.inner
    args = (batch["inputs"], batch["puzzle_identifiers"])
    assert not torch.equal(inner._input_embeddings(*args, batch["object_ids"]), inner._input_embeddings(*args, torch.zeros_like(batch["object_ids"])))
