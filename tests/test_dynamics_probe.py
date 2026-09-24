import numpy as np

from utils.dynamics.probe import run_probe, unwrap_inner
from tests.fixtures import tiny_model_and_batch


def test_probe_shapes_and_invariants():
    model, batch = tiny_model_and_batch()
    inner = unwrap_inner(model)
    r = run_probe(inner, batch, depth=8)
    B = batch["inputs"].shape[0]
    assert r.exact.shape == (8, B) and r.pooled_zH.shape[:2] == (9, B)
    assert np.allclose(r.dist_to_final[0], 1.0) and np.allclose(r.dist_to_final[-1], 0.0, atol=1e-4)
    assert ((r.settle_step >= 1) & (r.settle_step <= 8)).all()
    assert set(np.unique(r.modes())) <= {0, 1, 2}
    # deterministic model: same seed -> same rollout
    r2 = run_probe(inner, batch, depth=8)
    assert np.allclose(r.res, r2.res)


def test_probe_noise_changes_rollout():
    model, batch = tiny_model_and_batch()
    inner = unwrap_inner(model)
    a = run_probe(inner, batch, depth=4, noise_sigma=0.5, seed=1)
    b = run_probe(inner, batch, depth=4, noise_sigma=0.5, seed=2)
    assert not np.allclose(a.res, b.res)
    assert np.allclose(a.dist_to_final[-1], 0.0, atol=1e-4)
