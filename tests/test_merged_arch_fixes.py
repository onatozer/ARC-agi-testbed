import pytest
import torch

from models.losses.loss_heads import ACTLossHead
from models.recursive_reasoning.merged_arch import LatentCarry
from models.recursive_reasoning.trm_root import TRMRoot
from tests.fixtures import tiny_model_and_batch


def _tiny_root(**overrides):
    cfg = dict(
        batch_size=4, seq_len=16, vocab_size=6, H_cycles=2, L_cycles=2, H_layers=0, L_layers=1, hidden_size=32,
        expansion=2, num_heads=2, pos_encodings="rope", halt_max_steps=2, halt_exploration_prob=0.1, puzzle_emb_ndim=0,
        noise_scale=0.0, random_initialization=False, num_puzzle_identifiers=1, causal=False, forward_dtype="bfloat16",
        deq_max_iter=3, deq_min_iter=1, bptt_through=1, jacobian_reg_n_samples=1,
    )
    cfg.update(overrides)
    torch.manual_seed(0)
    with torch.device("cuda"):
        return ACTLossHead(TRMRoot(cfg), loss_type="stablemax_cross_entropy")


def _halt_steps(model, batch, n_steps):
    """Supervision step at which each row first halted (0 = never within n_steps)."""
    carry = model.initial_carry(batch)
    first = torch.zeros(batch["inputs"].shape[0], dtype=torch.int32, device=batch["inputs"].device)
    for _ in range(n_steps):
        carry, _, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
        first = torch.where((first == 0) & carry.halted, carry.steps, first)
    return first


# --- puzzle embedding (ARC) ---

@pytest.mark.parametrize("root", [False, True])
def test_puzzle_embedding_is_prepended_and_trained(root):
    overrides = dict(puzzle_emb_ndim=32, puzzle_emb_len=2, num_puzzle_identifiers=3)
    model = _tiny_root(**overrides) if root else tiny_model_and_batch(batch_size=4, **overrides)[0]
    _, batch = tiny_model_and_batch(batch_size=4)
    batch["puzzle_identifiers"] = torch.tensor([0, 1, 2, 1], dtype=torch.int32, device="cuda")
    model.train()

    assert model.model.puzzle_emb is model.model.inner.puzzle_emb  # what pretrain.py hands to the sparse optimizer
    assert model.model.config.seq_len == 16 + 2

    carry, loss, _, outputs, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=["logits"])
    assert outputs["logits"].shape == (4, 16, 6)  # the 2 prefix slots are stripped, every grid cell kept
    assert carry.inner_carry.z_H.shape == (4, 18, 32)
    loss.backward()
    assert model.model.puzzle_emb.local_weights.grad is not None


def test_puzzle_identifier_changes_the_prediction():
    model, batch = tiny_model_and_batch(batch_size=2, puzzle_emb_ndim=32, puzzle_emb_len=1, num_puzzle_identifiers=2)
    model.eval()
    with torch.no_grad():
        model.model.puzzle_emb.weights.normal_()
    batch = {k: v[:1].repeat(2, *([1] * (v.ndim - 1))) for k, v in batch.items()}  # same grid twice
    batch["puzzle_identifiers"] = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    _, _, _, outputs, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=["logits"])
    assert not torch.allclose(outputs["logits"][0], outputs["logits"][1])


def test_no_prefix_without_puzzle_embedding():
    model, _ = tiny_model_and_batch(puzzle_emb_ndim=0, puzzle_emb_len=16)
    assert model.model.config.puzzle_emb_len == 0 and model.model.config.seq_len == 16
    assert not hasattr(model.model.inner, "puzzle_emb")


# --- z_H update: the damped residual is taken around z_H, with z_L injected ---

@pytest.mark.parametrize("root", [False, True])
def test_z_H_update_residual_is_around_z_H(root):
    # With lambda_ = 0 each L_level call returns its hidden-state argument unchanged, so z_H must stay put
    model = _tiny_root(lambda_=0.0) if root else tiny_model_and_batch(lambda_=0.0)[0]
    _, batch = tiny_model_and_batch(batch_size=4)
    inner = model.model.inner
    shape = (4, inner.config.seq_len, inner.config.hidden_size)
    z_H = torch.randn(shape, device="cuda", dtype=inner.forward_dtype)
    z_L = torch.randn(shape, device="cuda", dtype=inner.forward_dtype)
    model.eval()
    with torch.no_grad():
        new, _ = inner(LatentCarry(z_H.clone(), z_L.clone()), batch)
    assert torch.equal(new.z_H, z_H)


# --- halting config fields ---

def _force_halt_logits(model, q_halt: float):
    with torch.no_grad():
        model.model.inner.q_head.bias[0] = q_halt


def test_defaults_halt_on_q_halt_only_in_training():
    model, batch = tiny_model_and_batch(halt_max_steps=6, halt_exploration_prob=0.0)
    _force_halt_logits(model, 5.0)
    model.train()
    assert (_halt_steps(model, batch, 6) == 1).all()
    model.eval()
    assert (_halt_steps(model, batch, 6) == 6).all()


def test_act_inference_halts_early_in_eval():
    model, batch = tiny_model_and_batch(halt_max_steps=6, halt_exploration_prob=0.0, act_inference=True)
    _force_halt_logits(model, 5.0)
    model.eval()
    assert (_halt_steps(model, batch, 6) == 1).all()


def test_disable_act_halt_runs_to_max_steps():
    model, batch = tiny_model_and_batch(halt_max_steps=6, halt_exploration_prob=0.0, disable_act_halt=True)
    _force_halt_logits(model, 5.0)
    model.train()
    assert (_halt_steps(model, batch, 6) == 6).all()


def test_halt_threshold():
    model, batch = tiny_model_and_batch(halt_max_steps=6, halt_exploration_prob=0.0, halt_threshold=10.0)
    _force_halt_logits(model, 5.0)
    model.train()
    assert (_halt_steps(model, batch, 6) == 6).all()


def test_halt_confirm_steps():
    model, batch = tiny_model_and_batch(halt_max_steps=8, halt_exploration_prob=0.0, halt_confirm_steps=2)
    _force_halt_logits(model, 5.0)
    model.train()
    assert (_halt_steps(model, batch, 8) == 3).all()  # the signal must hold for 1 + halt_confirm_steps steps

