"""First-order MAML in pretrain.py: task gradients over a puzzle's supervision steps, the inner SGD loop, the meta-step."""
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from config.schema import MAMLConfig, PretrainConfig
from models.losses import IGNORE_LABEL_ID
from models.losses.loss_heads import ACTLossHead
from models.recursive_reasoning.merged_arch import TRMModel
from pretrain import _trainable, maml_adapt, maml_batch, task_grads

from tests.test_gram import _cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")


def _head():
    torch.manual_seed(0)
    cfg = _cfg(puzzle_emb_ndim=0, puzzle_emb_len=0, batch_size=6)
    with torch.device("cuda"):
        model = ACTLossHead(TRMModel(cfg), loss_type="stablemax_cross_entropy")
    model.train()
    return model, cfg


def _two_puzzles(cfg):
    """Six rows: two puzzles of three examples, rows contiguous per puzzle with its own identifier."""
    g = torch.Generator().manual_seed(0)
    inputs = torch.randint(1, cfg["vocab_size"], (6, cfg["seq_len"]), generator=g)
    labels = torch.randint(2, cfg["vocab_size"], (6, cfg["seq_len"]), generator=g)
    labels[:, -3:] = IGNORE_LABEL_ID
    ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    return {"inputs": inputs.cuda(), "labels": labels.cuda(), "puzzle_identifiers": ids.cuda()}


def test_task_grads_cover_every_parameter_and_are_finite():
    model, cfg = _head()
    batch = _two_puzzles(cfg)
    params = _trainable(model)
    grads, loss = task_grads(model, batch, params)
    assert torch.isfinite(loss) and loss > 0
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_maml_adapt_lowers_the_task_loss():
    model, cfg = _head()
    batch = {k: v[:3] for k, v in _two_puzzles(cfg).items()}
    losses = maml_adapt(model, batch, steps=4, lr=1e-2)
    assert len(losses) == 4 and losses[-1] < losses[0]


def test_maml_batch_sums_query_grads_over_tasks_and_restores_the_weights():
    model, cfg = _head()
    batch = _two_puzzles(cfg)
    before = [p.detach().clone() for p in _trainable(model)]
    meta_grads, n_tasks, metrics = maml_batch(model, batch, MAMLConfig(inner_steps=1, inner_lr=1e-2))
    assert n_tasks == 2 and float(metrics["count"]) == 2
    assert all(g is not None and torch.isfinite(g).all() for g in meta_grads) and any(g.abs().sum() > 0 for g in meta_grads)
    assert all(torch.equal(p, p0) for p, p0 in zip(_trainable(model), before))
    assert torch.isfinite(metrics["fprl/query_loss"]) and torch.isfinite(metrics["fprl/support_loss"])
    # a single-row puzzle has no support/query split and is skipped
    one = {k: torch.cat([v[:3], v[3:4]]) for k, v in batch.items()}
    assert maml_batch(model, one, MAMLConfig(inner_steps=1, inner_lr=1e-2))[1] == 1


def test_maml_config_is_a_cli_option():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="cfg_pretrain", overrides=["seed=0", "maml.enabled=true", "maml.inner_steps=2", "maml.inner_lr=0.1"])
        plain = compose(config_name="cfg_pretrain", overrides=["seed=0"])
    assert PretrainConfig(**OmegaConf.to_container(cfg, resolve=True)).maml == MAMLConfig(enabled=True, inner_steps=2, inner_lr=0.1)
    plain_dict = OmegaConf.to_container(plain, resolve=True)
    assert not PretrainConfig(**plain_dict).maml.enabled
    assert not PretrainConfig(**{k: v for k, v in plain_dict.items() if k != "maml"}).maml.enabled  # checkpoints saved before the key
    with pytest.raises(ValueError):
        MAMLConfig(enabled=True, inner_steps=0, inner_lr=0.1)
