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
        "kl_grad_prior": 0.8, "kl_strength": 0.07, "grad_cycles": 1, "num_candidates": 3, "mcts_simulations": 6,
        "mcts_c_puct": 1.25, "mcts_horizon": "step", "gumbel_temp": 1.0, "candidate_residual": False, "eval_search": True, "posterior_smoothing": 0.0, "candidate_diversity_weight": 0.0, **overrides,
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
@pytest.mark.parametrize("horizon", ["step", "full"])
def test_treerm_runs_every_supervision_step_and_infers(horizon):
    model, batch = _treerm_head(mcts_horizon=horizon)
    model.train()
    inner = model.model.inner
    carry = model.initial_carry(batch)
    for step in range(1, 5):  # halt_max_steps = 4, H_cycles = 2
        carry, loss, _, _, _ = model(return_keys=[], carry=carry, batch=batch)
        assert bool(carry.halted.all()) == (step == 4) and torch.isfinite(loss)
        # Updates left from the first cycle of the step just run
        expected = 2 if horizon == "step" else 2 + 2 * (4 - step)
        assert inner._remaining_updates(0, batch["inputs"].shape[0]) == [expected] * batch["inputs"].shape[0]

    head = MaxStepsInferenceLossHead(model.model, loss_type="stablemax_cross_entropy")
    head.eval()
    carry = head.initial_carry(batch)
    with torch.inference_mode():
        for _ in range(4):
            carry, loss, _, _, finished = head(return_keys=None, carry=carry, batch=batch)
    assert bool(finished) and torch.isfinite(loss)


@needs_cuda
def test_treerm_eval_without_search_never_searches():
    model, batch = _treerm_head(eval_search=False)
    model.eval()
    inner = model.model.inner
    inner._search = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("searched at eval"))
    with torch.inference_mode():
        _, outputs = model.model(carry=model.initial_carry(batch), batch=batch)
    assert torch.isfinite(outputs["logits"]).all()


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
