import math

import pytest
import torch

from models.losses.base import stablemax_cross_entropy
from models.losses.loss_heads import LossHeadTreeRM, MaxStepsInferenceLossHead, MaxStepsInferenceLossHeadTreeRM
from models.mcts import MCTS, Node
from models.recursive_reasoning.treerm import TreeRM

from tests.test_gram import _batch, _cfg

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _treerm_head(**overrides):
    torch.manual_seed(0)
    cfg = _cfg(**{
        "kl_grad_prior": 0.8, "kl_strength": 0.07, "grad_cycles": 1, "posterior_nograd_cycles": False, "prior_sample_frac": 0.0, "prior_sample_per_puzzle": False, "num_candidates": 3, "mcts_simulations": 6,
        "mcts_c_puct": 1.25, "mcts_impl": "python", "search_compile": False, "candidate_loss": "none", "candidate_loss_eps": 0.05,
        "value_bootstrap_weight": 0.0, "eval_mcts_simulations": None, "eval_pick": "argmax", "mcts_horizon": "step", "lprm_pool": "first", "lprm_target": "step", "lprm_prior_rollouts": 0, "posterior_y_pool": "position", "lprm_rollout_cycles": "grad", **overrides,
    })
    with torch.device("cuda"):
        model = LossHeadTreeRM(TreeRM(cfg), loss_type="stablemax_cross_entropy")
    return model, _batch(cfg)


# MCTS on a toy problem: a state is the tuple of child indices taken from the root

def _toy_search(values, depth, num_simulations):
    root = Node(latent=(), allowed_depth=depth)
    root.expand([(k,) for k in range(len(values))], [1 / len(values)] * len(values))
    expanded_depths = []

    def expand_fn(states):
        expanded_depths.extend(len(state) for state in states)
        return [[state + (k,) for k in range(len(values))] for state in states], [[1 / len(values)] * len(values) for _ in states]

    [counts] = MCTS(num_simulations, c_puct=1.0).run([root], expand_fn, lambda states, _: [values[state[0]] for state in states])
    return root, counts, expanded_depths


def test_mcts_visits_sum_to_simulations_and_favour_the_best_child():
    root, counts, _ = _toy_search(values=[0.0, 1.0, 0.2], depth=1, num_simulations=30)
    assert sum(counts) == 30 == root.num_visits
    assert counts.index(max(counts)) == 1


def test_mcts_never_expands_past_the_allowed_depth():
    _, _, expanded_depths = _toy_search(values=[0.0, 1.0], depth=2, num_simulations=40)
    # Root children (depth 1 left) expand once each; their children have no depth left
    assert expanded_depths and max(expanded_depths) == 1


@needs_cuda
def test_treerm_replaces_gaussian_heads_and_has_no_halt_head():
    model, _ = _treerm_head()
    inner = model.model.inner
    assert type(inner).__name__ == "TreeRMInner" and not hasattr(inner, "q_head")
    assert not hasattr(inner, "prior_head") and not hasattr(inner, "posterior_head")
    assert inner.rollout_loss_fn is stablemax_cross_entropy


@needs_cuda
def test_treerm_train_step_trains_generator_and_scorer():
    model, batch = _treerm_head()
    model.train()
    _, loss, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics["kl_loss"].ndim == 0 and metrics["kl_loss"] > 0

    loss.backward()
    inner = model.model.inner
    for head in (inner.candidate_generator, inner.candidate_scorer):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


@needs_cuda
def test_treerm_straight_through_reaches_scorer_without_kl():
    model, batch = _treerm_head(kl_strength=0.0)
    model.train()
    _, loss, _, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.inner.candidate_scorer.parameters())


@needs_cuda
def test_treerm_eval_search_ignores_labels():
    model, batch = _treerm_head()
    model.eval()
    inner = model.model.inner
    inner.rollout_loss_fn = None  # the eval search must value rollouts by the LPRM head only
    shape = (batch["inputs"].shape[0], inner.config.seq_len, inner.config.hidden_size)
    u_t, z_L, x = (torch.randn(shape, device="cuda", dtype=inner.forward_dtype) for _ in range(3))
    with torch.inference_mode():
        cand = inner._candidates(z_L, u_t)
        prior = inner.candidate_scorer(u_t, cand, inner._cos_sin())
        post_a = inner._search(cand, prior, z_L, x, batch["labels"], [2] * shape[0])
        post_b = inner._search(cand, prior, z_L, x, torch.zeros_like(batch["labels"]), [2] * shape[0])
        _, outputs = model.model(carry=model.initial_carry(batch), batch=batch)
    torch.testing.assert_close(post_a, post_b)
    assert outputs["kl_loss"] == 0 and torch.isfinite(outputs["logits"]).all()


@needs_cuda
@pytest.mark.parametrize("training", [True, False])
def test_treerm_next_z_H_is_exactly_one_candidate(training):
    model, batch = _treerm_head()
    model.train(training)
    inner = model.model.inner
    shape = (batch["inputs"].shape[0], inner.config.seq_len, inner.config.hidden_size)
    u_t, z_L, x = (torch.randn(shape, device="cuda", dtype=inner.forward_dtype) for _ in range(3))
    with torch.no_grad():
        cand = inner._candidates(z_L, u_t)
        z_H, _, _ = inner._mcts_guidance(u_t, z_L, x, batch["labels"], [2] * shape[0])
    dist = (z_H.float().unsqueeze(1) - cand.float()).flatten(2).abs().amax(-1)
    assert (dist.amin(-1) < 1e-2).all()


@needs_cuda
def test_treerm_runs_every_supervision_step_and_infers():
    model, batch = _treerm_head()
    model.train()
    carry = model.initial_carry(batch)
    for step in range(1, 5):  # halt_max_steps = 4, H_cycles = 2
        carry, loss, _, _, _ = model(return_keys=[], carry=carry, batch=batch)
        assert bool(carry.halted.all()) == (step == 4) and torch.isfinite(loss)

    head = MaxStepsInferenceLossHead(model.model, loss_type="stablemax_cross_entropy")
    head.eval()
    carry = head.initial_carry(batch)
    with torch.inference_mode():
        for _ in range(4):
            carry, loss, _, _, finished = head(return_keys=None, carry=carry, batch=batch)
    assert bool(finished) and torch.isfinite(loss)


@needs_cuda
def test_treerm_reports_the_lprm_value_as_its_halt_confidence():
    # The ARC evaluator breaks vote ties by sigmoid(q_halt_logits); TreeRM has no halt head, so it reports the LPRM value
    model, batch = _treerm_head()
    model.eval()
    with torch.inference_mode():
        _, outputs = model.model(carry=model.initial_carry(batch), batch=batch)
    torch.testing.assert_close(outputs["q_halt_logits"].sigmoid(), outputs["lprm_value"], atol=1e-5, rtol=1e-4)


@needs_cuda
def test_treerm_logs_search_diagnostics_in_train_and_eval():
    model, batch = _treerm_head()
    model.train()
    _, _, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    for key in ("search/cand_div", "search/post_entropy", "search/kl_nats", "search/argmax_agree"):
        assert key in metrics and torch.isfinite(metrics[key])

    head = MaxStepsInferenceLossHeadTreeRM(model.model, loss_type="stablemax_cross_entropy")
    head.eval()
    carry = head.initial_carry(batch)
    with torch.inference_mode():
        for _ in range(4):
            carry, _, metrics, _, _ = head(return_keys=None, carry=carry, batch=batch)
    assert "search/post_entropy" in metrics and 0 <= metrics["search/argmax_agree"] <= metrics["count"]


@needs_cuda
def test_treerm_value_bootstrap_trains_the_value_head_from_the_search():
    model, batch = _treerm_head(value_bootstrap_weight=1.0, kl_strength=0.0)
    model.train()
    inner = model.model.inner
    carry = model.initial_carry(batch)
    _, info = model.model(carry=carry, batch=batch)
    info["kl_loss"].backward()  # the bootstrap term rides in the guidance loss
    assert inner.v_head.weight.grad is not None and inner.v_head.weight.grad.abs().sum() > 0
    assert len(inner._bootstrap) == inner.config.mcts_simulations


@needs_cuda
def test_treerm_searches_only_on_grad_cycles():
    model, batch = _treerm_head()  # H_cycles=2, grad_cycles=1
    model.train()
    inner = model.model.inner
    calls = []
    original = inner._search
    inner._search = lambda *args, **kwargs: calls.append(torch.is_grad_enabled()) or original(*args, **kwargs)
    model.model(carry=model.initial_carry(batch), batch=batch)
    assert calls == [True]  # one search, on the grad cycle


@needs_cuda
def test_flag_candidates_are_u_t_plus_orthonormal_directions():
    model, batch = _treerm_head()
    inner = model.model.inner
    directions = inner.candidate_generator.directions.float()
    # Slot 0 is the identity; the other slots' directions are orthonormal
    assert not directions[0].any()
    flags = directions[1:]
    torch.testing.assert_close(flags @ flags.T, torch.eye(flags.shape[0], device=flags.device), atol=1e-5, rtol=0)
    shape = (batch["inputs"].shape[0], inner.config.seq_len, inner.config.hidden_size)
    z_L, u_t = (torch.randn(shape, device="cuda", dtype=inner.forward_dtype) for _ in range(2))
    with torch.no_grad():
        cand = inner._candidates(z_L, u_t)
    torch.testing.assert_close(cand[:, 0], u_t, atol=0, rtol=0)  # bit-exact: K=1 is plain TRM
    delta = (cand.float() - u_t.float().unsqueeze(1))  # [B, K, S, H]
    # At init the gate is 1, so each slot's offset is sqrt(H) * e_k at every token
    expected = (math.sqrt(inner.config.hidden_size) * directions)[None, :, None, :].expand_as(delta)
    torch.testing.assert_close(delta, expected, atol=0.1, rtol=0.02)


@needs_cuda
def test_treerm_eval_can_search_with_more_simulations():
    model, batch = _treerm_head(eval_mcts_simulations=16)  # training: 6
    inner = model.model.inner
    seen = []
    original = inner.mcts.run
    inner.mcts.run = lambda *args, **kwargs: seen.append(inner.mcts.num_simulations) or original(*args, **kwargs)
    model.train()
    model.model(carry=model.initial_carry(batch), batch=batch)
    model.eval()
    with torch.inference_mode():
        model.model(carry=model.initial_carry(batch), batch=batch)
    assert seen[0] == 6 and seen[-1] == 16


@needs_cuda
@pytest.mark.parametrize("pick", ["argmax", "sample"])
def test_eval_pick_is_one_hot_and_argmax_takes_the_most_visited(pick):
    model, batch = _treerm_head(eval_pick=pick)
    inner = model.model.inner
    seen = []
    original = inner._search
    inner._search = lambda *args, **kwargs: seen.append(original(*args, **kwargs)) or seen[-1]
    picks = []
    einsum = torch.einsum
    torch.einsum = lambda eq, w, c: picks.append(w) or einsum(eq, w, c)
    try:
        model.eval()
        with torch.inference_mode():
            _, info = model.model(carry=model.initial_carry(batch), batch=batch)
    finally:
        torch.einsum = einsum
    assert picks and all(((w == 0) | (w == 1)).all() and (w.sum(-1) == 1).all() for w in picks)
    if pick == "argmax":
        assert all((w.argmax(-1) == post.argmax(-1)).all() for w, post in zip(picks, seen))
    for key in ("search/pick_ce_regret", "search/pick_is_best", "search/cand_ce_gap", "search/value_rank_corr"):
        assert key in info and torch.isfinite(info[key]).all()
    assert (info["search/pick_ce_regret"] >= 0).all() and (info["search/value_rank_corr"].abs() <= 1 + 1e-5).all()


@needs_cuda
@pytest.mark.parametrize("horizon", ["step", "full"])
def test_full_horizon_search_rolls_through_the_steps_left(horizon):
    """With mcts_horizon=full the eval search's allowed depth grows by H_cycles per supervision step left (the outer
    model passes steps_left); training and the step horizon keep the depth within the step."""
    model, batch = _treerm_head(mcts_horizon=horizon)
    inner = model.model.inner
    seen = []
    original = inner._search
    inner._search = lambda cand, prior, z_L, x, y, remaining: seen.append(remaining[0]) or original(cand, prior, z_L, x, y, remaining)
    model.eval()
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        for _ in range(2):  # steps 0 and 1 of halt_max_steps 4: 3 and 2 steps left
            carry, _ = model.model(carry=carry, batch=batch)
    H = inner.config.H_cycles
    extra = [3 * H, 3 * H, 2 * H, 2 * H] if horizon == "full" else [0, 0, 0, 0]  # H_cycles 2: two searches per step
    assert seen == [H + extra[0], H - 1 + extra[1], H + extra[2], H - 1 + extra[3]]
    model.train()
    seen.clear()
    model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert seen == [1]  # training searches on the grad cycle only (the last one) and never rolls past the step


# Candidate training (REPORTS_CANDIDATES.md): without a candidate loss only the picked candidate gets a gradient

def _slot_grad_norms(model):
    """Gradient norm of each flag slot's gate (the gate is a [K, 2H] linear: one row per candidate; row 0 is the
    identity slot, whose flag is zero, so it never gets a gradient and is left out)."""
    return model.model.inner.candidate_generator.gate.weight.grad.norm(dim=-1)[1:]


def _train_backward(**overrides):
    """One training step's backward with the scorer's candidate input detached: the KL and the straight-through pick
    reach every candidate through the scorer (to be scored right), so what is left is the gradient that improves a
    candidate as a z_H, which is what the candidate loss is about."""
    model, batch = _treerm_head(**overrides)
    model.train()
    scorer = model.model.inner.candidate_scorer
    scorer_forward = scorer.forward
    scorer.forward = lambda u_t, cand, cos_sin: scorer_forward(u_t, cand.detach(), cos_sin)
    _, loss, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert torch.isfinite(loss)
    loss.backward()
    return model, metrics


@needs_cuda
def test_without_candidate_loss_only_the_picked_candidate_gets_a_gradient():
    model, metrics = _train_backward(batch_size=1)
    assert (_slot_grad_norms(model) > 0).sum() <= 1  # 0 when the pick was the identity slot
    assert "search/cand_loss" not in metrics


@needs_cuda
@pytest.mark.parametrize("loss", ["wta", "posterior"])
def test_candidate_loss_gives_every_candidate_a_gradient(loss):
    model, metrics = _train_backward(batch_size=1, candidate_loss=loss)
    assert (_slot_grad_norms(model) > 0).all()
    assert all(torch.isfinite(metrics[k]) for k in ("search/cand_loss", "search/cand_ce_gap"))
    assert 0 <= metrics["search/cand_best_agree"] <= 1


@needs_cuda
def test_candidate_loss_rolls_out_the_updates_left_in_the_step():
    # grad_cycles=2 of H_cycles=2: the first grad cycle's candidates are one TRM update away from the decode
    model, _ = _train_backward(batch_size=1, candidate_loss="wta", grad_cycles=2)
    assert (_slot_grad_norms(model) > 0).all()


@needs_cuda
def test_candidate_loss_rejects_unknown_modes():
    with pytest.raises(ValueError):
        _treerm_head(candidate_loss="all")


@needs_cuda
def test_search_refuses_non_finite_prior_logits():
    model, batch = _treerm_head()
    inner = model.model.inner
    shape = (batch["inputs"].shape[0], inner.config.seq_len, inner.config.hidden_size)
    u_t, z_L, x = (torch.randn(shape, device="cuda", dtype=inner.forward_dtype) for _ in range(3))
    with torch.no_grad():
        cand = inner._candidates(z_L, u_t)
        prior = torch.full((shape[0], cand.shape[1]), float("nan"), device="cuda")
        with pytest.raises(FloatingPointError):
            inner._search(cand, prior, z_L, x, batch["labels"], [1] * shape[0])


@needs_cuda
def test_grad_cycle_diagnostics_are_averaged_over_the_grad_cycles_only():
    # H_cycles=2, grad_cycles=1: the candidate loss reports once; its diagnostics must not be divided by H_cycles
    model, batch = _treerm_head(candidate_loss="wta")
    model.train()
    inner = model.model.inner
    reported = []
    original = inner._candidate_loss
    inner._candidate_loss = lambda *args: (lambda out: reported.append(out[1]) or out)(original(*args))
    _, outputs = model.model(carry=model.initial_carry(batch), batch=batch)
    [diag] = reported
    torch.testing.assert_close(outputs["search/cand_ce_gap"], diag["search/cand_ce_gap"])
    assert outputs["search/post_entropy"].shape == outputs["search/cand_ce_gap"].shape
