import pytest
import torch
import torch.nn.functional as F

from models.layers import apply_rotary_pos_emb, rms_norm
from models.losses.base import kl_balanced_logits
from models.losses.loss_heads import LossHeadGRAM, MaxStepsInferenceLossHead
from models.recursive_reasoning.grampd import GRAMDP

from tests.test_gram import _batch, _cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _grampd_head(**overrides):
    torch.manual_seed(0)
    cfg = _cfg(**{"kl_grad_prior": 0.8, "kl_strength": 0.07, "grad_cycles": 1, "posterior_nograd_cycles": False, "prior_sample_frac": 0.0, "prior_sample_per_puzzle": False, "lprm_prior_rollouts": 0, "posterior_y_pool": "position", "value_bootstrap_weight": 0.5, "lprm_rollout_cycles": "grad", "lprm_pool": "first", "lprm_target": "step", "noise_options": 4, "gumbel_temp": 1.0, "gumbel_temp_min": 0.5, "gumbel_anneal_steps": 100, "usage_entropy_weight": 0.0, **overrides})
    with torch.device("cuda"):
        model = LossHeadGRAM(GRAMDP(cfg), loss_type="stablemax_cross_entropy")
    return model, _batch(cfg)


def test_kl_balanced_logits_is_categorical_kl_of_posterior_from_prior():
    torch.manual_seed(0)
    prior, post = torch.randn(3, 5), torch.randn(3, 5)
    expected = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=post), torch.distributions.Categorical(logits=prior))
    torch.testing.assert_close(kl_balanced_logits(prior, post, alpha=0.8), expected)
    assert torch.allclose(kl_balanced_logits(prior, prior, alpha=0.8), torch.zeros(3), atol=1e-6)


def test_grampd_replaces_gaussian_heads_and_has_no_halt_head():
    model, _ = _grampd_head()
    inner = model.model.inner
    assert type(inner).__name__ == "GRAMDP_Inner" and not hasattr(inner, "q_head")
    assert not any(name.endswith("to_stats.weight") for name, _ in inner.named_parameters())


def test_grampd_train_step_trains_every_guidance_head():
    model, batch = _grampd_head()
    model.train()
    _, loss, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics["kl_loss"].ndim == 0 and metrics["kl_loss"] > 0

    loss.backward()
    inner = model.model.inner
    for head in (inner.candidate_head, inner.prior_head, inner.posterior_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


def test_grampd_eval_samples_prior_only():
    model, batch = _grampd_head()
    model.eval()
    calls = []
    head = model.model.inner.posterior_head
    original = head.forward
    head.forward = lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs)
    with torch.inference_mode():
        _, outputs = model.model(carry=model.initial_carry(batch), batch=batch)
    assert not calls and outputs["kl_loss"] == 0
    assert outputs["logits"].shape == (*batch["inputs"].shape, 6) and torch.isfinite(outputs["logits"]).all()


def test_grampd_runs_every_supervision_step_and_infers():
    model, batch = _grampd_head()
    model.train()
    carry = model.initial_carry(batch)
    for step in range(1, 5):  # halt_max_steps = 4
        carry, loss, _, _, _ = model(return_keys=[], carry=carry, batch=batch)
        assert bool(carry.halted.all()) == (step == 4) and torch.isfinite(loss)

    head = MaxStepsInferenceLossHead(model.model, loss_type="stablemax_cross_entropy")
    head.eval()
    carry = head.initial_carry(batch)
    with torch.inference_mode():
        for _ in range(4):
            carry, loss, _, _, finished = head(return_keys=None, carry=carry, batch=batch)
    assert bool(finished) and torch.isfinite(loss)


@pytest.mark.parametrize("training", [True, False])
def test_grampd_eps_is_exactly_one_candidate(training):
    model, batch = _grampd_head()
    model.train(training)
    inner = model.model.inner
    u_t = torch.randn(batch["inputs"].shape[0], inner.config.seq_len, inner.config.hidden_size, device="cuda", dtype=inner.forward_dtype)
    with torch.no_grad():
        cand = inner.candidate_head(u_t)
        eps, _ = inner._guidance(u_t, batch["labels"], use_posterior=training)
    # Gumbel-max forward pass: eps is one of the candidates, not a blend of several
    dist = (eps.float().unsqueeze(1) - cand.float()).flatten(2).abs().amax(-1)
    assert (dist.amin(-1) < 1e-2).all()


def test_candidate_scorer_ranking_depends_on_context_and_cell_positions():
    model, batch = _grampd_head()
    inner = model.model.inner
    torch.manual_seed(1)
    shape = (batch["inputs"].shape[0], inner.config.seq_len, inner.config.hidden_size)
    u_a, u_b = (torch.randn(shape, device="cuda", dtype=inner.forward_dtype) for _ in range(2))
    with torch.no_grad():
        inner.prior_head.to_logit.weight.normal_()  # the zero-mean init makes the logits tiny; spread them out
        cand = inner.candidate_head(u_a)
        cos_sin = inner._cos_sin()
        logits_a = inner.prior_head(u_a, cand, cos_sin)
        logits_b = inner.prior_head(u_b, cand, cos_sin)
        logits_rolled = inner.prior_head(u_a, cand.roll(1, dims=2), cos_sin)

    def centred(x):  # softmax ignores a shift shared by every candidate, so compare the logits relative to their mean
        return x - x.mean(-1, keepdim=True)

    # Same candidates, different context -> a different preference among them (not only a shared shift)
    assert (centred(logits_a) - centred(logits_b)).abs().max() > 1e-3
    # Same context, candidate content moved to different cells -> different preference (RoPE aligns cells)
    assert (centred(logits_a) - centred(logits_rolled)).abs().max() > 1e-3


def test_candidate_scorer_rotating_the_query_before_repeating_it_changes_nothing():
    model, batch = _grampd_head()
    inner = model.model.inner
    scorer = inner.prior_head
    torch.manual_seed(2)
    B, K, S, H = batch["inputs"].shape[0], inner.config.noise_options, inner.config.seq_len, inner.config.hidden_size
    ctx = torch.randn(B, S, H, device="cuda", dtype=inner.forward_dtype)
    cand = torch.randn(B, K, S, H, device="cuda", dtype=inner.forward_dtype)
    cos, sin = inner._cos_sin()
    with torch.no_grad():
        logits = scorer(ctx, cand, (cos, sin))

        # Reference: repeat the query per candidate first, then rotate (the order before this change)
        query = scorer.q_proj(rms_norm(ctx, variance_epsilon=scorer.norm_eps)).view(B, 1, S, scorer.num_heads, scorer.head_dim)
        query = query.expand(-1, K, -1, -1, -1).reshape(B * K, S, scorer.num_heads, scorer.head_dim)
        key, value = scorer.kv_proj(rms_norm(cand, variance_epsilon=scorer.norm_eps)).view(B * K, S, 2, scorer.num_heads, scorer.head_dim).unbind(2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        attended = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        attended = scorer.o_proj(attended.transpose(1, 2).reshape(B * K, S, H))
        reference = scorer.to_logit(attended.mean(dim=1)).view(B, K).float()

    assert torch.equal(logits, reference)


def test_grampd_anneals_gumbel_temperature_with_global_step():
    model, batch = _grampd_head(gumbel_temp=1.0, gumbel_temp_min=0.5, gumbel_anneal_steps=100)
    model.train()
    for step, tau in [(0, 1.0), (50, 0.75), (100, 0.5), (500, 0.5)]:
        carry = model.initial_carry(batch)
        carry.global_step = step
        model(return_keys=[], carry=carry, batch=batch)
        assert model.model.inner.gumbel_tau.item() == pytest.approx(tau)
