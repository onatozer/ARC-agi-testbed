import pytest
import torch

from models.losses import IGNORE_LABEL_ID
from models.losses.loss_heads import ACTLossHead, LossHeadGRAM, MaxStepsInferenceLossHead
from models.recursive_reasoning.gram import GRAM
from models.recursive_reasoning.merged_arch import TRMModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cfg(**overrides):
    cfg = dict(
        batch_size=4, seq_len=16, vocab_size=6, H_cycles=2, L_cycles=2, H_layers=0, L_layers=1,
        hidden_size=32, expansion=2, num_heads=2, pos_encodings="rope", halt_max_steps=4,
        halt_exploration_prob=0.1, puzzle_emb_ndim=32, puzzle_emb_len=2, num_puzzle_identifiers=3,
        noise_scale=0.0, random_initialization=False, causal=False, eval_prior_mean=False,
    )
    cfg.update(overrides)
    return cfg


def _batch(cfg):
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(2, cfg["vocab_size"], (cfg["batch_size"], cfg["seq_len"]), generator=g)
    labels[:, -3:] = IGNORE_LABEL_ID
    return {
        "inputs": torch.randint(1, cfg["vocab_size"], (cfg["batch_size"], cfg["seq_len"]), generator=g).cuda(),
        "labels": labels.cuda(),
        "puzzle_identifiers": torch.zeros(cfg["batch_size"], dtype=torch.int32).cuda(),
    }


def _gram_head(**overrides):
    torch.manual_seed(0)
    cfg = _cfg(**{"kl_grad_prior": 0.8, "kl_strength": 0.07, "grad_cycles": 1, "posterior_nograd_cycles": False, "prior_sample_frac": 0.0, "prior_sample_per_puzzle": False, "lprm_prior_rollouts": 0, "value_bootstrap_weight": 0.5, "lprm_rollout_cycles": "grad", "lprm_pool": "first", "lprm_target": "step", **overrides})
    with torch.device("cuda"):
        model = LossHeadGRAM(GRAM(cfg), loss_type="stablemax_cross_entropy")
    return model, _batch(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_lprm_prior_rollouts_train_only_the_value_head_on_prior_draws():
    model, batch = _gram_head(lprm_prior_rollouts=3)
    model.train()
    _, loss, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert torch.isfinite(loss)
    inner = model.model.inner
    # The bootstrap's contribution reaches the value head only: it is the extra kl_loss, and the backbone's gradient
    # from it is zero (the draws are decoded without gradient), so the lm/KL gradients alone are left when it is off
    loss.backward()
    v_grad = inner.v_head.weight.grad.clone()
    model_off, batch = _gram_head(lprm_prior_rollouts=0)
    model_off.train()
    _, loss_off, metrics_off, _, _ = model_off(return_keys=[], carry=model_off.initial_carry(batch), batch=batch)
    loss_off.backward()
    assert metrics["kl_loss"] > metrics_off["kl_loss"]
    assert not torch.equal(v_grad, model_off.model.inner.v_head.weight.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_lprm_pool_mean_reads_every_token():
    model, batch = _gram_head(lprm_pool="mean")
    inner = model.model.inner
    z = torch.randn(2, inner.config.seq_len, inner.config.hidden_size, device="cuda", dtype=inner.forward_dtype)
    with torch.no_grad():
        expected = torch.sigmoid(inner.v_head(z.mean(dim=1)).to(torch.float32)).squeeze(-1)
        torch.testing.assert_close(inner._lprm_value(z), expected)
        # Perturbing a token other than the first changes the pooled value, never the first-token one
        z2 = z.clone()
        z2[:, 1] += 1.0
        assert not torch.equal(inner._lprm_value(z2), inner._lprm_value(z))
    model_first, _ = _gram_head()
    with torch.no_grad():
        assert torch.equal(model_first.model.inner._lprm_value(z2), model_first.model.inner._lprm_value(z))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_lprm_rollout_cycles_all_scores_draws_from_every_cycle():
    seen = []
    model, batch = _gram_head(lprm_prior_rollouts=2, lprm_rollout_cycles="all")
    inner = model.model.inner
    original = inner._lprm_rollout_pairs
    inner._lprm_rollout_pairs = lambda u_t, z_L, x, y, remaining: seen.append(remaining) or original(u_t, z_L, x, y, remaining)
    model.train()
    _, loss, _, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert torch.isfinite(loss)
    assert seen == [1, 0]  # H_cycles 2: the no-grad cycle with one cycle left, then the grad cycle with none


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_lprm_target_final_is_the_trajectory_end_and_leaves_eval_alone():
    model, batch = _gram_head(lprm_target="final")
    model.train()
    carry = model.initial_carry(batch)
    _, loss, metrics, outputs, _ = model(return_keys=["lprm_target"], carry=carry, batch=batch)
    assert torch.isfinite(loss) and "lprm_loss" in metrics
    target = outputs["lprm_target"]
    assert target.shape == (batch["labels"].shape[0],) and set(target.unique().tolist()) <= {0.0, 1.0}
    assert model.model.inner.training  # the rollout restores the inner's mode
    model.eval()
    with torch.inference_mode():
        _, info = model.model(carry=model.initial_carry(batch), batch=batch)
    assert "lprm_target" not in info


def _count_posterior_calls(model):
    calls = []
    head = model.model.inner.posterior_head
    original = head.forward
    head.forward = lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs)
    return calls


def test_gram_train_step_trains_prior_and_posterior():
    model, batch = _gram_head()
    model.train()
    # The heads are zero-initialised, so prior == posterior and the KL (and its gradient) is 0; separate them
    with torch.no_grad():
        model.model.inner.posterior_head.head.to_stats.bias.normal_()

    _, loss, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics["kl_loss"].ndim == 0 and metrics["kl_loss"] > 0

    loss.backward()
    inner = model.model.inner
    for head in (inner.prior_head, inner.posterior_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_gram_has_no_halt_head_and_runs_every_supervision_step():
    # A halt_threshold this low would make TRM's ACT halt every row after its first step
    model, batch = _gram_head(halt_threshold=-1e9)
    model.train()
    assert not hasattr(model.model.inner, "q_head")
    carry = model.initial_carry(batch)
    for step in range(1, 5):  # halt_max_steps = 4
        carry, loss, metrics, outputs, _ = model(return_keys=["q_halt_logits"], carry=carry, batch=batch)
        assert bool(carry.halted.all()) == (step == 4) and bool(carry.halted.any()) == (step == 4)
    assert torch.isfinite(loss) and set(outputs) <= {"q_halt_logits"} and not any(k.startswith(("q_halt", "halted_q")) for k in metrics)  # q_halt_logits is the LPRM stand-in


def test_gram_inference_loss_head_needs_no_halt_head():
    model, batch = _gram_head()
    head = MaxStepsInferenceLossHead(model.model, loss_type="stablemax_cross_entropy")
    head.eval()
    carry = head.initial_carry(batch)
    with torch.inference_mode():
        for _ in range(4):
            carry, loss, metrics, _, finished = head(return_keys=None, carry=carry, batch=batch)
    assert bool(finished) and torch.isfinite(loss) and metrics["count"] == batch["inputs"].shape[0]
    assert not any(k.startswith("q_halt") for k in metrics)


def test_gram_eval_samples_prior_only():
    model, batch = _gram_head()
    model.eval()
    calls = _count_posterior_calls(model)
    with torch.inference_mode():
        _, outputs = model.model(carry=model.initial_carry(batch), batch=batch)
    assert not calls and outputs["kl_loss"] == 0


@pytest.mark.parametrize("grad_cycles", [0, 3])
def test_gram_rejects_grad_cycles_outside_h_cycles(grad_cycles):
    with pytest.raises(ValueError, match="grad_cycles"):
        _gram_head(H_cycles=2, grad_cycles=grad_cycles)


def test_act_loss_head_total_loss_is_scalar():
    torch.manual_seed(0)
    cfg = _cfg(puzzle_emb_ndim=0)
    with torch.device("cuda"):
        model = ACTLossHead(TRMModel(cfg), loss_type="stablemax_cross_entropy")
    batch = _batch(cfg)
    _, loss, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert loss.ndim == 0 and metrics["lm_loss"].ndim == 0


def test_gram_lprm_head_trains_and_outputs_value():
    model, batch = _gram_head()
    model.train()
    _, loss, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert metrics["lprm_loss"].ndim == 0 and metrics["lprm_loss"] > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.inner.v_head.parameters())

    model.eval()
    with torch.inference_mode():
        _, outputs = model.model(carry=model.initial_carry(batch), batch=batch)
    value = outputs["lprm_value"]
    assert value.shape == (batch["inputs"].shape[0],) and ((value > 0) & (value < 1)).all()


def test_gram_prior_traj_is_drawn_per_puzzle_and_kept_until_reset():
    model, batch = _gram_head(prior_sample_frac=0.5, prior_sample_per_puzzle=True)
    model.train()
    carry = model.initial_carry(batch)
    carry, *_ = model(return_keys=[], carry=carry, batch=batch)
    flags = carry.current_data["prior_traj"].clone()
    for _ in range(model.model.config.halt_max_steps - 1):
        carry, *_ = model(return_keys=[], carry=carry, batch=batch)
        assert torch.equal(carry.current_data["prior_traj"], flags)
