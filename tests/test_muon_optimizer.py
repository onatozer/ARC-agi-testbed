import pytest
import torch
from adam_atan2_pytorch import AdamAtan2
from hydra import compose, initialize
from omegaconf import OmegaConf

from config.schema import PretrainConfig
from dataset.common import PuzzleDatasetMetadata
from pretrain import create_model

METADATA = PuzzleDatasetMetadata(
    pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=6, seq_len=16, num_puzzle_identifiers=1,
    total_groups=1, mean_puzzle_examples=1.0, sets=["all"],
)
# Tiny TRM built from the real configs, so it goes through the same path as pretrain.py
TINY_ARCH = ["arch.hidden_size=32", "arch.num_heads=2", "arch.expansion=2", "arch.L_layers=1", "arch.puzzle_emb_ndim=0",
             "dataset.grid_size=4", "global_batch_size=4", "seed=0"]


def _create(*overrides):
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="cfg_pretrain", overrides=[*TINY_ARCH, *overrides])
    config = PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    return create_model(config, METADATA, rank=0, world_size=1)


def _params(optim):
    return {id(p) for group in optim.param_groups for p in group["params"]}


def test_default_is_adam_atan2_only():
    model, optimizers, lrs = _create()
    assert [type(o) for o in optimizers] == [AdamAtan2]
    assert _params(optimizers[0]) == {id(p) for p in model.parameters()}
    assert lrs == [1e-4]


def test_muon_takes_the_block_matrices_and_adam_the_rest():
    model, optimizers, lrs = _create("optimizer=muon", "muon_lr=0.02")
    assert [type(o) for o in optimizers] == [AdamAtan2, torch.optim.Muon]
    adam, muon = map(_params, optimizers)
    block_matrices = {id(p) for n, p in model.named_parameters() if ".L_level." in n and p.ndim == 2}
    assert muon == block_matrices
    assert adam == {id(p) for p in model.parameters()} - block_matrices
    assert lrs == [1e-4, 0.02]


def test_muon_lr_defaults_to_lr():
    _, _, lrs = _create("optimizer=muon")
    assert lrs == [1e-4, 1e-4]


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError, match="optimizer"):
        _create("optimizer=sgd")


def test_muon_step_updates_the_block_weights():
    model, optimizers, _ = _create("optimizer=muon")
    muon = optimizers[-1]
    batch = {
        "inputs": torch.randint(1, 6, (4, 16), device="cuda"),
        "labels": torch.randint(2, 6, (4, 16), device="cuda"),
        "puzzle_identifiers": torch.zeros(4, dtype=torch.int32, device="cuda"),
    }
    before = [p.detach().clone() for g in muon.param_groups for p in g["params"]]
    carry = model.initial_carry(batch)  # type: ignore[operator]
    _, loss, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
    loss.backward()
    for o in optimizers:
        for g in o.param_groups:
            g["lr"] = 1e-3
        o.step()
    after = [p for g in muon.param_groups for p in g["params"]]
    assert all(not torch.equal(b, a) for b, a in zip(before, after))
