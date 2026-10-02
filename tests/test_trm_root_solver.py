import inspect

import pytest
import torch
from torchdeq.solver.anderson import anderson_solver

import models.recursive_reasoning.trm_root as trm_root
from models.losses.loss_heads import ACTLossHeadRoot
from models.recursive_reasoning.trm_root import TRMRoot
from tests.fixtures import tiny_model_and_batch


def _tiny_root(**overrides):
    _, batch = tiny_model_and_batch(batch_size=4)
    cfg = dict(
        batch_size=4, seq_len=16, vocab_size=6, H_cycles=3, L_cycles=2, H_layers=0, L_layers=1, hidden_size=32,
        expansion=2, num_heads=2, pos_encodings="rope", halt_max_steps=2, halt_exploration_prob=0.1, puzzle_emb_ndim=0,
        noise_scale=0.0, random_initialization=False, num_puzzle_identifiers=1, causal=False, forward_dtype="bfloat16",
        deq_max_iter=3, bptt_through=1, jacobian_reg_n_samples=1,
    )
    cfg.update(overrides)
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = ACTLossHeadRoot(TRMRoot(cfg), loss_type="stablemax_cross_entropy", jacobian_reg_lambda=0.0)
    return model, batch


def _record_solver_calls(monkeypatch):
    calls = []
    real = trm_root.anderson_solver

    def recording_solver(func, x0, **kwargs):
        out = real(func, x0, **kwargs)
        calls.append({"x0": x0.clone(), "z_star": out[0].clone(), "kwargs": kwargs})
        return out

    monkeypatch.setattr(trm_root, "anderson_solver", recording_solver)
    return calls


def _forward(model, batch):
    model.eval()
    with torch.no_grad():
        model(carry=model.initial_carry(batch), batch=batch, return_keys=[])


def test_solver_kwargs_are_ones_torchdeq_uses(monkeypatch):
    calls = _record_solver_calls(monkeypatch)
    model, batch = _tiny_root(deq_tol=0.25, deq_stop_mode="abs", deq_anderson_m=4, deq_anderson_tau=0.5)
    _forward(model, batch)

    params = inspect.signature(anderson_solver).parameters
    accepted = {n for n, p in params.items() if p.kind is not inspect.Parameter.VAR_KEYWORD}
    kwargs = calls[0]["kwargs"]
    assert set(kwargs) <= accepted  # torchdeq silently drops anything else via **kwargs
    assert (kwargs["tol"], kwargs["stop_mode"], kwargs["m"], kwargs["tau"]) == (0.25, "abs", 4, 0.5)


def test_stop_mode_defaults_to_relative(monkeypatch):
    calls = _record_solver_calls(monkeypatch)
    model, batch = _tiny_root()
    _forward(model, batch)
    assert calls[0]["kwargs"]["stop_mode"] == "rel"


def test_each_solve_warm_starts_from_the_previous_solution(monkeypatch):
    calls = _record_solver_calls(monkeypatch)
    model, batch = _tiny_root(H_cycles=3)  # two solves per forward
    _forward(model, batch)
    assert len(calls) == 2
    # the solve's fp32 output is handed back to the network in forward_dtype, then re-widened by the solver
    assert torch.equal(calls[1]["x0"], calls[0]["z_star"].to(torch.bfloat16).float())


def test_solver_residuals_are_the_last_solves_per_row_errors(monkeypatch):
    infos = []
    real = trm_root.anderson_solver

    def recording_solver(func, x0, **kwargs):
        out = real(func, x0, **kwargs)
        infos.append(out[2])
        return out

    monkeypatch.setattr(trm_root, "anderson_solver", recording_solver)
    model, batch = _tiny_root(H_cycles=3, halt_max_steps=1)  # two solves per forward; every row halts after one step
    model.eval()
    with torch.no_grad():
        _, _, metrics, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])

    count = metrics["count"].item()
    assert count == batch["inputs"].shape[0]
    last = infos[-1]
    for key, info_key in (("solver_abs_residual", "abs_lowest"), ("solver_rel_residual", "rel_lowest")):
        assert metrics[key].item() / count == pytest.approx(last[info_key].mean().item(), rel=1e-5)
    assert 0 < metrics["solver_rel_residual"].item() / count < metrics["solver_abs_residual"].item() / count


@pytest.mark.parametrize("deq_tol", [1e-12, 1e3])  # never stops early / stops at the first Anderson iteration
def test_solver_residual_trace_is_logged_at_fixed_iterations(monkeypatch, deq_tol):
    infos = []
    real = trm_root.anderson_solver

    def recording_solver(func, x0, **kwargs):
        out = real(func, x0, **kwargs)
        infos.append(out[2])
        return out

    monkeypatch.setattr(trm_root, "anderson_solver", recording_solver)
    model, batch = _tiny_root(H_cycles=3, halt_max_steps=1, deq_max_iter=8, deq_tol=deq_tol)
    model.eval()
    with torch.no_grad():
        _, _, metrics, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])

    count = metrics["count"].item()
    trace = infos[-1]["rel_trace"]  # column j is iteration j + 2
    logged = sorted(int(k.split("@")[1]) for k in metrics if k.startswith("solver_rel_residual@"))
    assert logged == [3, 4, 8]  # the same keys whether or not the solve stops early, so eval can aggregate them
    for n in logged:
        assert metrics[f"solver_rel_residual@{n}"].item() / count == pytest.approx(trace[:, n - 2].mean().item(), rel=1e-5)
    # the last traced iteration is never better than the lowest residual the solve reached
    assert metrics["solver_rel_residual@8"].item() >= metrics["solver_rel_residual"].item() - 1e-6
