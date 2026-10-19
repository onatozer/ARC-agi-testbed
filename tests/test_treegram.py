import pytest
import torch

from models.losses.loss_heads import LossHeadGRAM, LossHeadTreeRM
from models.recursive_reasoning.gram import GRAM
from models.recursive_reasoning.treegram import TreeGRAM

from tests.test_gram import _batch, _cfg
from tests.test_treerm import _treerm_head

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _treegram_head(**overrides):
    torch.manual_seed(0)
    cfg = _cfg(**{
        "kl_grad_prior": 0.8, "kl_strength": 0.07, "grad_cycles": 1, "posterior_nograd_cycles": False, "prior_sample_frac": 0.0, "prior_sample_per_puzzle": False, "num_candidates": 3, "mcts_simulations": 6,
        "mcts_c_puct": 1.25, "mcts_impl": "python", "search_compile": False, "candidate_loss": "wta", "candidate_loss_eps": 0.05,
        "value_bootstrap_weight": 0.0, "eval_mcts_simulations": None, "eval_pick": "argmax", "mcts_horizon": "step", "lprm_pool": "first", "lprm_target": "step", "lprm_prior_rollouts": 0, "posterior_y_pool": "position", "lprm_rollout_cycles": "grad", "prior_target": "visits", **overrides,
    })
    with torch.device("cuda"):
        model = LossHeadTreeRM(TreeGRAM(cfg), loss_type="stablemax_cross_entropy")
    return model, _batch(cfg), cfg


@needs_cuda
@pytest.mark.parametrize("target", ["visits", "none"])
def test_treegram_prior_target_trains_the_variance(target):
    model, batch, _ = _treegram_head(prior_target=target, candidate_loss="none")
    model.train()
    _, loss, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert torch.isfinite(loss)
    loss.backward()
    # to_stats' second half of rows is the raw variance; only the visits target reaches it through the NLL (pathwise,
    # var only trains through a picked noise sample, which a 3-candidate search with the mean first may never pick)
    var_rows = model.model.inner.prior_head.to_stats.weight.grad[model.model.inner.config.hidden_size:]
    if target == "visits":
        assert metrics["kl_loss"] != 0 and var_rows.abs().sum() > 0
    else:
        assert metrics["kl_loss"] == 0


@needs_cuda
def test_treegram_has_the_prior_head_and_no_candidate_heads():
    model, _, _ = _treegram_head()
    inner = model.model.inner
    assert hasattr(inner, "prior_head") and not hasattr(inner, "posterior_head")
    assert not hasattr(inner, "candidate_generator") and not hasattr(inner, "candidate_scorer")


@needs_cuda
def test_treegram_slot_0_is_the_prior_mean_and_the_others_are_draws():
    model, batch, _ = _treegram_head()
    inner = model.model.inner
    shape = (batch["inputs"].shape[0], inner.config.seq_len, inner.config.hidden_size)
    z_L, u_t = (torch.randn(shape, device="cuda", dtype=inner.forward_dtype) for _ in range(2))
    with torch.no_grad():
        mu, var = inner.prior_head(u_t)
        cand = inner._candidates(z_L, u_t)
    torch.testing.assert_close(cand[:, 0], u_t + mu, atol=0, rtol=0)
    assert (cand[:, 1] != cand[:, 2]).any() and (cand[:, 1] != cand[:, 0]).any()
    # The uniform search prior puts the prior's argmax on slot 0
    assert (inner._prior_logits(u_t, cand) == 0).all()


@needs_cuda
def test_treegram_train_step_trains_the_prior_head():
    model, batch, _ = _treegram_head()
    model.train()
    _, loss, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.model.inner.prior_head.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


@needs_cuda
def test_treegram_k1_is_gram_prior_mean_inference():
    """With one candidate the search can only pick the prior mean, so eval equals GRAM with eval_prior_mean."""
    model, batch, cfg = _treegram_head(num_candidates=1, mcts_simulations=1)
    torch.manual_seed(0)
    with torch.device("cuda"):
        gram = LossHeadGRAM(GRAM({**cfg, "eval_prior_mean": True}), loss_type="stablemax_cross_entropy")
    missing, unexpected = gram.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("model.inner.posterior_head.") for k in missing)
    model.eval(), gram.eval()
    with torch.inference_mode():  # one supervision step of each bare model from the same carry
        _, out_tree = model.model(carry=model.initial_carry(batch), batch=batch)
        _, out_gram = gram.model(carry=gram.initial_carry(batch), batch=batch)
    torch.testing.assert_close(out_tree["logits"], out_gram["logits"], atol=0, rtol=0)


@needs_cuda
def test_treerm_prior_logits_hook_is_the_scorer():
    model, batch = _treerm_head()
    inner = model.model.inner
    shape = (batch["inputs"].shape[0], inner.config.seq_len, inner.config.hidden_size)
    z_L, u_t = (torch.randn(shape, device="cuda", dtype=inner.forward_dtype) for _ in range(2))
    with torch.no_grad():
        cand = inner._candidates(z_L, u_t)
        torch.testing.assert_close(inner._prior_logits(u_t, cand), inner.candidate_scorer(u_t, cand, inner._cos_sin()))
