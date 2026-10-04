import pytest
import torch

from models.losses.loss_heads import ACTLossHeadRoot
from models.recursive_reasoning.trm_root import TRMRoot
from tests.fixtures import tiny_model_and_batch


def _tiny_root(**overrides):
    _, batch = tiny_model_and_batch(batch_size=4)
    cfg = dict(
        batch_size=4, seq_len=16, vocab_size=6, H_cycles=2, L_cycles=2, H_layers=0, L_layers=1, hidden_size=32,
        expansion=2, num_heads=2, pos_encodings="rope", halt_max_steps=2, halt_exploration_prob=0.1, puzzle_emb_ndim=0,
        noise_scale=0.0, random_initialization=False, num_puzzle_identifiers=1, causal=False, forward_dtype="bfloat16",
        deq_max_iter=3, bptt_through=1, jacobian_reg_n_samples=1,
    )
    cfg.update(overrides)
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = ACTLossHeadRoot(TRMRoot(cfg), loss_type="stablemax_cross_entropy", jacobian_reg_lambda=0.5)
    return model, batch


def test_spectral_norm_matches_the_exact_largest_singular_value():
    model, _ = _tiny_root(forward_dtype="float32")
    inner = model.model.inner
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator(device="cuda").manual_seed(1)
    z = torch.randn(3, 16, 32, device="cuda", generator=g)
    injection = torch.randn(3, 16, 32, device="cuda", generator=g)
    seq = inner._seq_dict()

    estimate = inner._jacobian_spectral_norm(z, injection, seq, rows=3, iters=200)
    for i in range(3):
        f = lambda zi: inner.L_level(zi[None], injection[i : i + 1], **seq)[0]
        jac = torch.func.jacrev(f)(z[i]).reshape(16 * 32, 16 * 32)
        assert estimate[i].item() == pytest.approx(torch.linalg.matrix_norm(jac, ord=2).item(), rel=1e-3)


def test_logged_in_training_and_leaves_training_unchanged():
    def step(log):
        model, batch = _tiny_root(log_jacobian_spectral_norm=log)
        model.train()
        _, loss, metrics, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])
        loss.backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        return loss.detach(), grads, metrics, torch.cuda.get_rng_state(), torch.get_rng_state()

    loss_off, grads_off, metrics_off, cuda_off, cpu_off = step(False)
    loss_on, grads_on, metrics_on, cuda_on, cpu_on = step(True)
    assert "fprl/jacobian_spectral_norm" not in metrics_off
    assert 0 < metrics_on["fprl/jacobian_spectral_norm"].item() <= metrics_on["fprl/jacobian_spectral_norm_max"].item()
    # a diagnostic only: same loss, gradients and RNG streams as without it
    assert torch.equal(loss_on, loss_off) and torch.equal(grads_on, grads_off)
    assert torch.equal(cuda_on, cuda_off) and torch.equal(cpu_on, cpu_off)


def test_not_computed_in_eval():
    model, batch = _tiny_root(log_jacobian_spectral_norm=True)
    model.eval()
    with torch.no_grad():
        _, _, metrics, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])
    assert "fprl/jacobian_spectral_norm" not in metrics
