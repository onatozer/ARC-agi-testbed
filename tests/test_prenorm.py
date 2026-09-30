import pytest
import torch

from models.recursive_reasoning.merged_arch import Blocks
from tests.fixtures import tiny_model_and_batch
from tests.test_merged_arch_fixes import _tiny_root

# The shared fixtures don't set visual_attention yet
MODEL_KW = dict(visual_attention=False)


def _inner(model):
    return model.model.inner


def _L_level(model) -> Blocks:
    return next(m for m in model.modules() if isinstance(m, Blocks))


def test_postnorm_default_has_no_scales():
    model, _ = tiny_model_and_batch(**MODEL_KW)
    assert not any("res_alpha" in name for name, _ in model.named_parameters())


def test_alphas_start_at_their_inits():
    model, _ = tiny_model_and_batch(prenorm=True, res_alpha1_init=0.6, res_alpha2_init=0.3, **MODEL_KW)
    blocks = _L_level(model)
    torch.testing.assert_close(torch.sigmoid(blocks.res_alpha1_logit), torch.full((32,), 0.6, device="cuda"))
    torch.testing.assert_close(torch.sigmoid(blocks.res_alpha2_logit), torch.full((32,), 0.3, device="cuda"))


@pytest.mark.parametrize("L_layers", [1, 2])
def test_zero_sublayers_converge_to_the_injection(L_layers):
    # With every sub-layer output zero, one call maps z to alpha1^(2L) * (alpha2 * z + beta2 * x). Because
    # beta2 = 1 - alpha2 * alpha1^(2L) (FPRM Theorem 1), the post-injection state converges to exactly x,
    # so the returned state converges to alpha1^(2L) * x
    model, _ = tiny_model_and_batch(prenorm=True, L_layers=L_layers, lambda_=1.0, forward_dtype="float32", **MODEL_KW)
    blocks = _L_level(model)
    with torch.no_grad():
        for name, p in blocks.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                p.zero_()
        g = torch.Generator(device="cuda").manual_seed(0)
        z = torch.randn(2, 16, 32, device="cuda", generator=g) * 10
        x = torch.randn(2, 16, 32, device="cuda", generator=g)
        seq_dict = dict(cos_sin=_inner(model).rotary_emb())
        for _ in range(300):
            z = blocks(z, x, **seq_dict)
    torch.testing.assert_close(z, 0.75 ** (2 * L_layers) * x, atol=1e-4, rtol=1e-4)


def test_iterates_stay_bounded():
    model, _ = tiny_model_and_batch(prenorm=True, lambda_=1.0, **MODEL_KW)
    blocks = _L_level(model)
    with torch.no_grad():
        x = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
        z = torch.zeros_like(x)
        seq_dict = dict(cos_sin=_inner(model).rotary_emb())
        for _ in range(500):
            z = blocks(z, x, **seq_dict)
    assert torch.isfinite(z).all()
    assert z.float().norm() < 10 * x.float().norm()


@pytest.mark.parametrize("root", [False, True])
def test_model_trains_with_prenorm(root):
    model = _tiny_root(prenorm=True, **MODEL_KW) if root else tiny_model_and_batch(prenorm=True, **MODEL_KW)[0]
    _, batch = tiny_model_and_batch(batch_size=4, **MODEL_KW)
    carry = model.initial_carry(batch)
    _, loss, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
    loss.backward()
    blocks = _L_level(model)
    for p in (blocks.res_alpha1_logit, blocks.res_alpha2_logit):
        assert p.grad is not None and p.grad.abs().sum() > 0
