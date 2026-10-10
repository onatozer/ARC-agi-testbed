"""freeze_alternation_epochs: alternate between training only the puzzle embeddings and only the weights (arXiv 2609.21181)."""

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from config.schema import PretrainConfig
from dataset.common import PuzzleDatasetMetadata
from pretrain import create_model, init_train_state, train_batch

# One group of 4 examples with a batch of 4, so every step is one epoch
METADATA = PuzzleDatasetMetadata(
    pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=6, seq_len=16, num_puzzle_identifiers=2,
    total_groups=1, mean_puzzle_examples=4.0, sets=["all"],
)
TINY_ARCH = ["arch.hidden_size=32", "arch.num_heads=2", "arch.expansion=2", "arch.L_layers=1", "arch.puzzle_emb_ndim=32",
             "dataset.grid_size=4", "global_batch_size=4", "epochs=10", "lr_warmup_steps=0", "ema=False", "seed=0"]
BATCH = {
    "inputs": torch.randint(1, 6, (4, 16), generator=torch.Generator().manual_seed(0)),
    "labels": torch.randint(2, 6, (4, 16), generator=torch.Generator().manual_seed(1)),
    "puzzle_identifiers": torch.ones(4, dtype=torch.int32),
}


def _config(*overrides) -> PretrainConfig:
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="cfg_pretrain", overrides=[*TINY_ARCH, *overrides])
    return PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]


def _changed_after_step(config, train_state):
    """(embeddings changed, every weight changed) over one train_batch step."""
    emb = train_state.model.model.puzzle_emb.weights.detach().clone()
    weights = [p.detach().clone() for p in train_state.model.parameters()]
    metrics = train_batch(config, train_state, BATCH, global_batch_size=4, rank=0, world_size=1)
    emb_changed = not torch.equal(emb, train_state.model.model.puzzle_emb.weights)
    weights_changed = [not torch.equal(b, a) for b, a in zip(weights, train_state.model.parameters())]
    return emb_changed, weights_changed, metrics


def test_alternates_embeddings_then_weights():
    config = _config("freeze_alternation_epochs=1")
    train_state = init_train_state(config, METADATA, rank=0, world_size=1)

    emb_changed, weights_changed, metrics = _changed_after_step(config, train_state)  # epoch 0
    assert emb_changed and not any(weights_changed)
    assert metrics["train/freeze_phase"] == 0

    emb_changed, weights_changed, metrics = _changed_after_step(config, train_state)  # epoch 1
    assert not emb_changed and any(weights_changed)
    assert metrics["train/freeze_phase"] == 1

    emb_changed, weights_changed, _ = _changed_after_step(config, train_state)  # epoch 2: back to the embeddings
    assert emb_changed and not any(weights_changed)


def test_default_trains_both():
    config = _config()
    train_state = init_train_state(config, METADATA, rank=0, world_size=1)
    emb_changed, weights_changed, metrics = _changed_after_step(config, train_state)
    assert emb_changed and any(weights_changed)
    assert "train/freeze_phase" not in metrics


@pytest.mark.parametrize("overrides", [["arch.puzzle_emb_ndim=0"], ["freeze_weights=True"]])
def test_invalid_combinations_raise(overrides):
    with pytest.raises(ValueError, match="freeze_alternation_epochs"):
        create_model(_config("freeze_alternation_epochs=1", *overrides), METADATA, rank=0, world_size=1)


def test_non_positive_rejected():
    with pytest.raises(ValueError, match="freeze_alternation_epochs"):
        _config("freeze_alternation_epochs=0")
