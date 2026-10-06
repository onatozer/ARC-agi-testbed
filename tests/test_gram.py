import pytest
import torch

from models.losses import IGNORE_LABEL_ID
from models.losses.loss_heads import ACTLossHead, ACTLossHeadGRAM
from models.recursive_reasoning.gram import GRAM
from models.recursive_reasoning.merged_arch import TRMModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cfg(**overrides):
    cfg = dict(
        batch_size=4, seq_len=16, vocab_size=6, H_cycles=2, L_cycles=2, H_layers=0, L_layers=1,
        hidden_size=32, expansion=2, num_heads=2, pos_encodings="rope", halt_max_steps=4,
        halt_exploration_prob=0.1, puzzle_emb_ndim=32, puzzle_emb_len=2, num_puzzle_identifiers=3,
        noise_scale=0.0, random_initialization=False, causal=False,
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


def _gram_head():
    torch.manual_seed(0)
    cfg = _cfg()
    with torch.device("cuda"):
        model = ACTLossHeadGRAM(GRAM(cfg), loss_type="stablemax_cross_entropy", kl_grad_prior=0.8, kl_strength=0.07)
    return model, _batch(cfg)


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


def test_gram_eval_samples_prior_only():
    model, batch = _gram_head()
    model.eval()
    with torch.inference_mode():
        _, outputs = model.model(carry=model.initial_carry(batch), batch=batch)
    assert "mu_prior" in outputs and "mu_post" not in outputs


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
