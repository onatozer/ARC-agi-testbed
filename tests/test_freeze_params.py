"""freeze_param_substrings: parameters whose name matches train nowhere; composes with alternation, lr scales and muon."""

import pytest
import torch

from pretrain import create_model, init_train_state, train_batch
from tests.test_freeze_alternation import BATCH, METADATA, _config


def _step_and_diff(config):
    train_state = init_train_state(config, METADATA, rank=0, world_size=1)
    before = {n: p.detach().clone() for n, p in train_state.model.named_parameters()}
    train_batch(config, train_state, BATCH, global_batch_size=4, rank=0, world_size=1)
    changed = {n: not torch.equal(before[n], p) for n, p in train_state.model.named_parameters()}
    return train_state, changed


@pytest.mark.parametrize("optimizer", ["adam_atan2", "muon"])
def test_frozen_substrings_train_nowhere(optimizer):
    config = _config("freeze_param_substrings=[lm_head]", f"optimizer={optimizer}")
    train_state, changed = _step_and_diff(config)
    frozen = {n for n in changed if "lm_head" in n}
    assert frozen and not any(changed[n] for n in frozen)
    assert any(changed[n] for n in changed if n not in frozen)
    for n, p in train_state.model.named_parameters():
        assert p.requires_grad == (n not in frozen)
        if n in frozen:
            assert p.grad is None and all(id(p) not in {id(q) for g in opt.param_groups for q in g["params"]} for opt in train_state.optimizers)


def test_composes_with_freeze_alternation():
    config = _config("freeze_param_substrings=[lm_head]", "freeze_alternation_epochs=1")
    train_state = init_train_state(config, METADATA, rank=0, world_size=1)
    for epoch in range(2):
        before = {n: p.detach().clone() for n, p in train_state.model.named_parameters()}
        emb = train_state.model.model.puzzle_emb.weights.detach().clone()
        train_batch(config, train_state, BATCH, global_batch_size=4, rank=0, world_size=1)
        changed = {n: not torch.equal(before[n], p) for n, p in train_state.model.named_parameters()}
        assert not any(changed[n] for n in changed if "lm_head" in n)
        if epoch == 0:  # embeddings only
            assert not torch.equal(emb, train_state.model.model.puzzle_emb.weights) and not any(changed.values())
        else:  # weights except the frozen ones
            assert torch.equal(emb, train_state.model.model.puzzle_emb.weights) and any(changed.values())


def test_unmatched_or_total_freeze_raises():
    with pytest.raises(ValueError, match="match no parameter"):
        create_model(_config("freeze_param_substrings=[no_such_param]"), METADATA, rank=0, world_size=1)
    with pytest.raises(ValueError, match="every parameter"):
        create_model(_config("freeze_param_substrings=[.]"), METADATA, rank=0, world_size=1)
