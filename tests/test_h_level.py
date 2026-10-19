import pytest
import torch

from models.recursive_reasoning.gram import GRAM
from models.recursive_reasoning.merged_arch import TRMModel

from tests.test_gram import _batch, _cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_h_layers_zero_shares_the_block_and_adds_no_keys():
    inner = TRMModel(_cfg()).inner
    assert inner.H_level is inner.L_level
    assert not any("H_blocks" in k for k in inner.state_dict())


def test_h_layers_gives_the_h_update_its_own_block_and_loads_shared_checkpoints():
    with torch.device("cuda"):
        model = TRMModel(_cfg(H_layers=2))
        shared = TRMModel(_cfg())
    inner = model.inner
    assert inner.H_level is inner.H_blocks and len(inner.H_blocks.layers) == 2
    l_ids = {id(p) for p in inner.L_level.parameters()}
    assert inner.H_blocks.parameters() and all(id(p) not in l_ids for p in inner.H_blocks.parameters())

    # A shared-block checkpoint loads: the L block from it, the H block kept at its fresh init
    missing, unexpected = model.load_state_dict(shared.state_dict(), strict=False)
    assert not unexpected and missing and all("H_blocks" in k for k in missing)

    model.eval()
    batch = _batch(_cfg(H_layers=2))
    with torch.inference_mode():
        _, outputs = model(carry=model.initial_carry(batch), batch=batch)
    assert torch.isfinite(outputs["logits"]).all()


def test_gram_h_update_uses_the_h_block():
    cfg = _cfg(H_layers=1, kl_grad_prior=0.8, kl_strength=0.07, grad_cycles=1, posterior_nograd_cycles=False, prior_sample_frac=0.0,
               prior_sample_per_puzzle=False, lprm_prior_rollouts=0, posterior_y_pool="position", lprm_weight=0.5, carry_norm=True,
               kl_reduction="sum", value_bootstrap_weight=0.5, lprm_rollout_cycles="grad", lprm_pool="first", lprm_target="step")
    with torch.device("cuda"):
        model = GRAM(cfg)
    model.eval()
    batch = _batch(cfg)
    with torch.no_grad():
        for p in model.inner.H_blocks.parameters():
            p.zero_()  # a zeroed H block changes the output iff the H update goes through it
        _, out_zero = model(carry=model.initial_carry(batch), batch=batch)
        for p in model.inner.H_blocks.parameters():
            p.normal_()
        _, out_rand = model(carry=model.initial_carry(batch), batch=batch)
    assert not torch.allclose(out_zero["logits"], out_rand["logits"])
