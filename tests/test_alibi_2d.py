"""ViTARC (arXiv 2410.06405) 2D-RPE: the 2D ALiBi attention bias."""

import pytest
import torch

from models.layers import ALiBi2D
from models.losses.loss_heads import ACTLossHeadRoot
from models.recursive_reasoning.trm_root import TRMRoot
from tests.fixtures import tiny_model_and_batch


def _bias(num_heads=8, height=3, width=4, prefix_len=0):
    return ALiBi2D(num_heads, height, width, prefix_len=prefix_len)()


def test_slopes_follow_alibi_geometric_sequence():
    bias = _bias(num_heads=8, height=1, width=2)
    # Token 1 is right after token 0 (distance 1): bias[h, 0, 1] = -r_after, bias[h, 1, 0] = -r_before
    r_before = 2.0 ** -1 * 0.5 ** torch.arange(8)
    r_after = 2.0 ** -0.5 * 0.5 ** torch.arange(8)
    torch.testing.assert_close(bias[:, 1, 0], -r_before)
    torch.testing.assert_close(bias[:, 0, 1], -r_after)


def test_manhattan_distance_and_direction():
    height, width = 3, 4
    bias = _bias(num_heads=2, height=height, width=width)
    idx = lambda r, c: r * width + c
    r_before, r_after = 2.0 ** -1, 2.0 ** -0.5  # head 0

    assert torch.all(bias.diagonal(dim1=1, dim2=2) == 0)
    i = idx(1, 1)
    assert bias[0, i, idx(1, 0)] == pytest.approx(-r_before * 1)  # left: earlier in raster order
    assert bias[0, i, idx(1, 3)] == pytest.approx(-r_after * 2)  # right
    assert bias[0, i, idx(0, 3)] == pytest.approx(-r_before * 3)  # up-right diagonal is still before i
    assert bias[0, i, idx(2, 0)] == pytest.approx(-r_after * 2)  # down-left diagonal is after i


def test_prefix_has_no_bias():
    bias = _bias(num_heads=2, height=2, width=2, prefix_len=3)
    assert bias.shape == (2, 3 + 4, 3 + 4)
    assert torch.all(bias[:, :3] == 0) and torch.all(bias[:, :, :3] == 0)
    assert torch.equal(bias[:, 3:, 3:], _bias(num_heads=2, height=2, width=2))


def test_off_by_default():
    model, _ = tiny_model_and_batch()
    assert model.model.inner._seq_dict()["attn_bias"] is None


@pytest.mark.parametrize("visual_attention", [False, True])
def test_bias_changes_output_and_trains(visual_attention):
    extra = dict(visual_attention=True, patch_height=2, patch_width=2) if visual_attention else {}
    plain, batch = tiny_model_and_batch(**extra)
    model, _ = tiny_model_and_batch(alibi_2d=True, **extra)
    assert model.load_state_dict(plain.state_dict()).missing_keys == []  # the bias is a non-persistent buffer

    model.train()
    carry, loss, _, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])
    loss.backward()
    assert torch.isfinite(loss)

    plain.train()
    _, plain_loss, _, _, _ = plain(carry=plain.initial_carry(batch), batch=batch, return_keys=[])
    assert loss.item() != plain_loss.item()


def test_trm_root_jacobian_penalty_with_bias():
    # The penalty's double backward runs under the MATH SDPA backend, which must accept the additive mask
    _, batch = tiny_model_and_batch(batch_size=4)
    cfg = dict(
        batch_size=4, seq_len=16, vocab_size=6, H_cycles=2, L_cycles=2, H_layers=0, L_layers=1, hidden_size=32,
        expansion=2, num_heads=2, pos_encodings="rope", halt_max_steps=2, halt_exploration_prob=0.1, puzzle_emb_ndim=0,
        noise_scale=0.0, random_initialization=False, num_puzzle_identifiers=1, causal=False, forward_dtype="bfloat16",
        deq_max_iter=3, deq_min_iter=1, bptt_through=1, jacobian_reg_n_samples=1, alibi_2d=True,
    )
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = ACTLossHeadRoot(TRMRoot(cfg), loss_type="stablemax_cross_entropy", jacobian_reg_lambda=0.5)
    model.train()
    _, loss, metrics, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])
    loss.backward()
    assert torch.isfinite(loss) and metrics["fprl/jacobian_norm"].item() > 0
