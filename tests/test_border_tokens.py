"""ViTARC (arXiv 2410.06405) border tokens for the ARC dataset builder."""

import numpy as np

from dataset.build_arc_dataset import (EndXToken, EndXYToken, EndYToken, NewlineToken, Pad2DToken,
                                       np_grid_to_seq_translational_augment)
from evaluators.arc import _crop

C = 5  # canvas side
INP = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
OUT = np.array([[9]], dtype=np.uint8)


def _canvas(seq):
    return seq.reshape(C, C + 1)


def test_untranslated_layout():
    inp, out = np_grid_to_seq_translational_augment(INP, OUT, do_translation=False, max_grid_size=C, border_tokens=True)
    P, X, Y, XY, NL = Pad2DToken, EndXToken, EndYToken, EndXYToken, NewlineToken
    expected = np.array([
        [2, 3, 4, X, P, NL],
        [5, 6, 7, X, P, NL],
        [Y, Y, Y, XY, P, NL],
        [P, P, P, P, P, NL],
        [P, P, P, P, P, NL],
    ])
    np.testing.assert_array_equal(_canvas(inp), expected)
    np.testing.assert_array_equal(_canvas(out)[:2, :2], [[11, X], [Y, XY]])
    assert (inp == 0).sum() == 0 and (out == 0).sum() == 0  # every canvas cell is supervised


def test_grid_touching_canvas_edge_has_no_end_tokens_past_it():
    full = np.zeros((C, C), dtype=np.uint8)
    inp, _ = np_grid_to_seq_translational_augment(full, OUT, do_translation=False, max_grid_size=C, border_tokens=True)
    canvas = _canvas(inp)
    np.testing.assert_array_equal(canvas[:, :C], 2)
    np.testing.assert_array_equal(canvas[:, C], NewlineToken)


def test_translated_grid_is_framed_by_pad():
    np.random.seed(3)
    inp, _ = np_grid_to_seq_translational_augment(INP, OUT, do_translation=True, max_grid_size=C, border_tokens=True)
    canvas = _canvas(inp)
    rows, cols = np.nonzero((canvas >= 2) & (canvas <= 11))
    r, c = rows.min(), cols.min()
    np.testing.assert_array_equal(canvas[r:r + 2, c:c + 3], INP + 2)
    assert np.all(canvas[:r, :C] == Pad2DToken) and np.all(canvas[:, :c] == Pad2DToken)


def test_crop_recovers_grid():
    inp, out = np_grid_to_seq_translational_augment(INP, OUT, do_translation=False, max_grid_size=C, border_tokens=True)
    np.testing.assert_array_equal(_crop(inp, C, C + 1), INP)
    np.testing.assert_array_equal(_crop(out, C, C + 1), OUT)


def test_off_keeps_original_format():
    inp, _ = np_grid_to_seq_translational_augment(INP, OUT, do_translation=False, max_grid_size=C, border_tokens=False)
    expected = np.array([
        [2, 3, 4, 1, 0],
        [5, 6, 7, 1, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    np.testing.assert_array_equal(inp.reshape(C, C), expected)
    np.testing.assert_array_equal(_crop(inp, C, C), INP)
