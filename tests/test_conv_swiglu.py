import pytest
import torch
import torch.nn.functional as F

from models.layers import ConvSwiGLU


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("T", [7, 16])
def test_shape_is_preserved(k, T):
    torch.manual_seed(0)
    mlp = ConvSwiGLU(hidden_size=32, expansion=2, conv_kernel=k)
    assert mlp(torch.randn(3, T, 32)).shape == (3, T, 32)


def test_k2_mixes_only_the_previous_token():
    torch.manual_seed(0)
    mlp = ConvSwiGLU(hidden_size=32, expansion=2, conv_kernel=2)
    x = torch.randn(2, 10, 32)
    x2 = x.clone()
    x2[:, 4] += 1.0
    changed = (mlp(x2) - mlp(x)).abs().amax(dim=(0, 2)) > 0
    assert changed.nonzero().flatten().tolist() == [4, 5]


def test_identity_conv_matches_formula():
    torch.manual_seed(0)
    mlp = ConvSwiGLU(hidden_size=32, expansion=2, conv_kernel=1)
    with torch.no_grad():
        mlp.dwconv.weight.fill_(1.0)
        mlp.dwconv.bias.zero_()
    x = torch.randn(2, 5, 32)
    gate, up = mlp.gate_up_proj(x).chunk(2, dim=-1)
    torch.testing.assert_close(mlp(x), mlp.down_proj(F.silu(F.silu(gate) * up)))

