import pytest
import torch

from models.losses.loss_heads import ACTLossHeadRoot
from models.recursive_reasoning.trm_root import TRMRoot
from tests.fixtures import tiny_model_and_batch


def _tiny_root(jacobian_reg_lambda: float, log_jacobian_norm: bool):
    _, batch = tiny_model_and_batch(batch_size=4)
    cfg = dict(
        batch_size=4, seq_len=16, vocab_size=6, H_cycles=2, L_cycles=2, H_layers=0, L_layers=1, hidden_size=32,
        expansion=2, num_heads=2, pos_encodings="rope", halt_max_steps=2, halt_exploration_prob=0.1, puzzle_emb_ndim=0,
        noise_scale=0.0, random_initialization=False, num_puzzle_identifiers=1, causal=False, forward_dtype="bfloat16",
        deq_max_iter=3, deq_min_iter=1, bptt_through=1, jacobian_reg_n_samples=1, log_jacobian_norm=log_jacobian_norm,
    )
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = ACTLossHeadRoot(TRMRoot(cfg), loss_type="stablemax_cross_entropy", jacobian_reg_lambda=jacobian_reg_lambda)
    return model, batch


def _train_step(model, batch):
    model.train()
    carry, loss, metrics, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])
    loss.backward()
    return metrics


def test_logged_without_penalty_when_enabled():
    model, batch = _tiny_root(jacobian_reg_lambda=0.0, log_jacobian_norm=True)
    metrics = _train_step(model, batch)
    assert metrics["fprl/jacobian_norm"].item() > 0
    assert metrics["fprl/jacobian_reg"].item() == 0  # still not in the loss


def test_not_computed_when_off():
    model, batch = _tiny_root(jacobian_reg_lambda=0.0, log_jacobian_norm=False)
    assert "fprl/jacobian_norm" not in _train_step(model, batch)


def test_logged_with_penalty_and_matches_it():
    model, batch = _tiny_root(jacobian_reg_lambda=0.5, log_jacobian_norm=False)
    metrics = _train_step(model, batch)
    assert metrics["fprl/jacobian_reg"].item() == pytest.approx(0.5 * metrics["fprl/jacobian_norm"].item(), rel=1e-5)


def test_penalty_enters_loss_once_per_row():
    # pretrain.py divides the loss by the batch size, so the penalty must be counted per row to keep weight lambda
    model, batch = _tiny_root(jacobian_reg_lambda=0.5, log_jacobian_norm=False)
    model.train()
    _, loss, metrics, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])
    penalty = loss.item() - (metrics["lm_loss"].item() + 0.5 * metrics["q_halt_loss"].item())
    rows = batch["inputs"].shape[0]
    assert penalty == pytest.approx(0.5 * metrics["fprl/jacobian_norm"].item() * rows, rel=1e-3)


def test_not_computed_in_eval():
    model, batch = _tiny_root(jacobian_reg_lambda=0.0, log_jacobian_norm=True)
    model.eval()
    with torch.no_grad():
        _, _, metrics, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])
    assert "fprl/jacobian_norm" not in metrics
