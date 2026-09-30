import pytest
import torch

from models.layers import rms_norm
from models.recursive_reasoning.merged_arch import Block
from tests.fixtures import tiny_model_and_batch
from tests.test_merged_arch_fixes import _tiny_root

# The shared fixtures don't set visual_attention yet
MODEL_KW = dict(visual_attention=False)


def _block(model) -> Block:
    return next(m for m in model.modules() if isinstance(m, Block))


def test_block_matches_prenorm_formula():
    model, _ = tiny_model_and_batch(prenorm=True, forward_dtype="float32", **MODEL_KW)
    block, inner = _block(model), model.model.inner
    cos_sin = inner.rotary_emb()
    x = torch.randn(2, 16, 32, device="cuda") * 5
    with torch.no_grad():
        h = x + block.self_attn(cos_sin=cos_sin, hidden_states=rms_norm(x, block.norm_eps))
        expected = h + block.mlp(rms_norm(h, block.norm_eps))
        torch.testing.assert_close(block(cos_sin=cos_sin, hidden_states=x), expected)


@pytest.mark.parametrize("root", [False, True])
def test_model_trains_with_prenorm(root):
    model = _tiny_root(prenorm=True, **MODEL_KW) if root else tiny_model_and_batch(prenorm=True, **MODEL_KW)[0]
    _, batch = tiny_model_and_batch(batch_size=4, **MODEL_KW)
    carry = model.initial_carry(batch)
    _, loss, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for n, p in model.named_parameters() if ".L_level." in n)
