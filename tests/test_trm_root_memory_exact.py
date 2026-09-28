"""The TRMRoot memory savers (checkpoint_phantom, jac_two_pass, jac_chunk_size) must not change gradients or RNG use."""
import pytest
import torch

from models.losses.loss_heads import ACTLossHeadRoot
from models.recursive_reasoning.trm_root import TRMRoot
from tests.fixtures import tiny_model_and_batch

BATCH = 8
VARIANTS = {
    "checkpoint_phantom": dict(checkpoint_phantom=True),
    "two_pass": dict(jac_two_pass=True),
    "two_pass_chunked": dict(jac_two_pass=True, jac_chunk_size=3),  # 8 rows -> chunks of 3, 3, 2
    "all": dict(checkpoint_phantom=True, jac_two_pass=True, jac_chunk_size=3),
}
# relative Frobenius error of each parameter's gradient
TOLERANCE = {"float32": 1e-5, "bfloat16": 2e-2}


def _tiny_root(forward_dtype, noise_scale, **flags):
    cfg = dict(
        batch_size=BATCH, seq_len=16, vocab_size=6, H_cycles=3, L_cycles=2, H_layers=0, L_layers=2, hidden_size=32,
        expansion=2, num_heads=2, pos_encodings="rope", halt_max_steps=4, halt_exploration_prob=0.1, puzzle_emb_ndim=0,
        noise_scale=noise_scale, random_initialization=False, num_puzzle_identifiers=1, causal=False,
        forward_dtype=forward_dtype, deq_max_iter=3, bptt_through=3, jacobian_reg_n_samples=2, **flags,
    )
    torch.manual_seed(0)
    with torch.device("cuda"):
        return ACTLossHeadRoot(TRMRoot(cfg), loss_type="stablemax_cross_entropy", jacobian_reg_lambda=1.0)


def _run(model, batch, n_steps):
    """n_steps training steps as in pretrain.train_batch; returns per-step grads, losses, penalties and the RNG states."""
    model.train()
    torch.manual_seed(1)
    carry = model.initial_carry(batch)
    steps = []
    for _ in range(n_steps):
        model.zero_grad(set_to_none=True)
        carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
        (loss / BATCH).backward()
        grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
        steps.append((grads, loss.item(), metrics["fprl/jacobian_reg"].item()))
    return steps, torch.get_rng_state(), torch.cuda.get_rng_state()


@pytest.fixture(autouse=True)
def _no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


@pytest.mark.parametrize("noise_scale", [0.0, 0.01])
@pytest.mark.parametrize("forward_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_memory_savers_match_baseline(variant, forward_dtype, noise_scale):
    _, batch = tiny_model_and_batch(batch_size=BATCH)
    base = _tiny_root(forward_dtype, noise_scale)
    other = _tiny_root(forward_dtype, noise_scale, **VARIANTS[variant])
    other.load_state_dict(base.state_dict())

    base_steps, base_cpu_rng, base_cuda_rng = _run(base, batch, n_steps=3)
    other_steps, other_cpu_rng, other_cuda_rng = _run(other, batch, n_steps=3)

    assert torch.equal(base_cpu_rng, other_cpu_rng) and torch.equal(base_cuda_rng, other_cuda_rng)
    for (base_grads, base_loss, base_jac), (grads, loss, jac) in zip(base_steps, other_steps):
        assert loss == pytest.approx(base_loss, rel=TOLERANCE[forward_dtype])
        assert jac == pytest.approx(base_jac, rel=TOLERANCE[forward_dtype])
        assert grads.keys() == base_grads.keys()
        for name, g in base_grads.items():
            rel = (grads[name] - g).norm() / g.norm().clamp_min(1e-12)
            cos = torch.nn.functional.cosine_similarity(grads[name].flatten(), g.flatten(), dim=0)
            assert rel < TOLERANCE[forward_dtype], (name, rel.item())
            assert cos > 0.9999, (name, cos.item())


def test_gradient_path_runs_in_forward_dtype():
    model = _tiny_root("bfloat16", 0.0)
    _, batch = tiny_model_and_batch(batch_size=BATCH)
    grad_path_dtypes = []
    model.model.inner.L_level.register_forward_hook(
        lambda m, inp, out: grad_path_dtypes.append(inp[0].dtype) if torch.is_grad_enabled() else None
    )
    steps, _, _ = _run(model, batch, n_steps=2)
    assert grad_path_dtypes and set(grad_path_dtypes) == {torch.bfloat16}


@pytest.mark.parametrize("variant", ["checkpoint_phantom", "all"])
def test_memory_savers_match_baseline_compiled(variant):
    """pretrain.py runs the model under torch.compile, which rewrites checkpointed regions; check the flags there too.

    Compilation itself draws from the CPU RNG, and the variants compile different graphs, so both models are compiled
    by a warm-up step first; otherwise the Hutchinson vectors of the measured steps would come from shifted streams.
    """
    _, batch = tiny_model_and_batch(batch_size=BATCH)
    base = _tiny_root("float32", 0.0)
    other = _tiny_root("float32", 0.0, **VARIANTS[variant])
    other.load_state_dict(base.state_dict())
    nets = [torch.compile(m, dynamic=False) for m in (base, other)]
    for net in nets:
        _run(net, batch, n_steps=2)

    (base_steps, base_cpu, base_cuda), (other_steps, cpu, cuda) = (_run(net, batch, n_steps=2) for net in nets)
    assert torch.equal(base_cpu, cpu) and torch.equal(base_cuda, cuda)
    for (base_grads, base_loss, _), (grads, loss, _) in zip(base_steps, other_steps):
        assert loss == pytest.approx(base_loss, rel=TOLERANCE["float32"])
        for name, g in base_grads.items():
            rel = (grads[name] - g).norm() / g.norm().clamp_min(1e-12)
            assert rel < TOLERANCE["float32"], (name, rel.item())
