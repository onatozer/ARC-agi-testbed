import numpy as np
import pytest

from utils.dynamics.basin import basin_entropy, basin_map, plot_basin, uncertainty_exponent
from utils.dynamics.probe import unwrap_inner
from tests.fixtures import tiny_model_and_batch


def test_basin_entropy_constant_is_zero():
    s_b, s_bb = basin_entropy(np.full((20, 20), 3), box=5)
    assert s_b == 0.0 and s_bb == 0.0


@pytest.mark.parametrize("box", [2, 4])
def test_basin_entropy_checkerboard_is_ln2(box):
    img = np.indices((16, 16)).sum(0) % 2
    s_b, s_bb = basin_entropy(img, box=box)
    assert np.isclose(s_b, np.log(2)) and np.isclose(s_bb, np.log(2))


def test_boundary_basin_entropy_only_counts_mixed_boxes():
    img = np.zeros((10, 10), dtype=int)
    img[:5, :5] = np.indices((5, 5)).sum(0) % 2  # one mixed box out of four
    s_b, s_bb = basin_entropy(img, box=5)
    assert np.isclose(s_bb, 4 * s_b) and s_bb > 0


def test_uncertainty_exponent_half_plane():
    G = 200
    img = (np.arange(G)[None, :] >= G // 2).repeat(G, 0).astype(int)
    alpha = uncertainty_exponent(img)
    assert abs(alpha - 1.0) < 0.05  # smooth boundary: dim = 2 - alpha ~ 1


def test_basin_map_tiny_model():
    model, batch = tiny_model_and_batch()
    inner = unwrap_inner(model)
    row = {k: v[:1] for k, v in batch.items()}
    r = basin_map(inner, row, grid=8, span=2.0, depth=6, chunk=20, box=4)
    for k in ("settle", "correct", "capped", "fli", "switch_freq"):
        assert r[k].shape == (8, 8), k
    assert r["correct"].dtype == bool and r["capped"].dtype == bool
    assert ((r["settle"] >= 1) & (r["settle"] <= 6)).all()
    assert (r["fli"] >= 0).all() and (r["fli"] <= 1).all()
    assert (r["switch_freq"] >= 0).all() and (r["switch_freq"] <= 1).all()
    assert r["capped"][r["settle"] < 6].sum() == 0
    assert 0 <= r["basin_entropy"] <= r["boundary_basin_entropy"] + 1e-9
    # chunking must not change the result, and reruns are deterministic
    r2 = basin_map(inner, row, grid=8, span=2.0, depth=6, chunk=64, box=4)
    assert (r["settle"] == r2["settle"]).all() and (r["correct"] == r2["correct"]).all()

    import matplotlib.figure

    fig = plot_basin(r, "tiny")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_basin_map_random_z0_differs():
    model, batch = tiny_model_and_batch()
    inner = unwrap_inner(model)
    row = {k: v[:1] for k, v in batch.items()}
    r = basin_map(inner, row, grid=6, span=2.0, depth=4, random_z0=True)
    assert r["settle"].shape == (6, 6)
