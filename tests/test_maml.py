"""First-order MAML in pretrain.py: task gradients over a puzzle's supervision steps, the inner SGD loop (weights and
puzzle-embedding rows), the meta-step over episodes, and the puzzle-level episode sampler."""
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from config.schema import MAMLConfig, PretrainConfig
from models.losses import IGNORE_LABEL_ID
from models.losses.loss_heads import ACTLossHead
from models.recursive_reasoning.merged_arch import TRMModel
from pretrain import EpisodeSampler, _maml_targets, _trainable, maml_adapt, maml_batch, task_grads

from tests.test_gram import _cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")


def _head(**overrides):
    torch.manual_seed(0)
    cfg = _cfg(batch_size=6, **overrides)
    with torch.device("cuda"):
        model = ACTLossHead(TRMModel(cfg), loss_type="stablemax_cross_entropy")
    model.train()
    return model, cfg


def _rows(cfg, n, ids):
    g = torch.Generator().manual_seed(n)
    inputs = torch.randint(1, cfg["vocab_size"], (n, cfg["seq_len"]), generator=g)
    labels = torch.randint(2, cfg["vocab_size"], (n, cfg["seq_len"]), generator=g)
    labels[:, -3:] = IGNORE_LABEL_ID
    return {"inputs": inputs.cuda(), "labels": labels.cuda(), "puzzle_identifiers": torch.tensor(ids, dtype=torch.int32).cuda()}


def test_task_grads_cover_the_weights_and_the_embedding_table():
    model, cfg = _head()
    params = _maml_targets(model)
    assert params[-1] is model.model.puzzle_emb.weights and len(params) == len(_trainable(model)) + 1
    grads, loss = task_grads(model, _rows(cfg, 4, [1, 1, 2, 2]), params)
    assert torch.isfinite(loss) and loss > 0
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    table_grad = grads[-1]
    assert table_grad[1].abs().sum() > 0 and table_grad[2].abs().sum() > 0 and table_grad[0].abs().sum() == 0  # only the rows used


def test_maml_adapt_lowers_the_loss_and_moves_only_the_tasks_embedding_rows():
    model, cfg = _head()
    table_before = model.model.puzzle_emb.weights.detach().clone()
    losses = maml_adapt(model, _rows(cfg, 3, [1, 1, 1]), steps=4, lr=1e-2)
    assert len(losses) == 4 and losses[-1] < losses[0]
    table = model.model.puzzle_emb.weights
    assert not torch.equal(table[1], table_before[1]) and torch.equal(table[0], table_before[0]) and torch.equal(table[2], table_before[2])


def test_maml_adapt_without_puzzle_embeddings():
    model, cfg = _head(puzzle_emb_ndim=0, puzzle_emb_len=0)
    assert len(_maml_targets(model)) == len(_trainable(model))
    losses = maml_adapt(model, _rows(cfg, 3, [0, 0, 0]), steps=3, lr=1e-2)
    assert losses[-1] < losses[0]


def test_maml_batch_sums_query_grads_over_episodes_and_restores_the_weights():
    model, cfg = _head()
    episodes = [(_rows(cfg, 4, [1, 1, 2, 2]), _rows(cfg, 2, [1, 2])), (_rows(cfg, 3, [0, 0, 0]), _rows(cfg, 1, [0]))]
    before = [p.detach().clone() for p in _maml_targets(model)]
    meta_grads, n_tasks, metrics = maml_batch(model, episodes, MAMLConfig(inner_steps=1, inner_lr=1e-2))
    assert n_tasks == 2 and float(metrics["count"]) == 2
    assert all(g is not None and torch.isfinite(g).all() for g in meta_grads) and any(g.abs().sum() > 0 for g in meta_grads)
    assert all(torch.equal(p, p0) for p, p0 in zip(_maml_targets(model), before))
    assert torch.isfinite(metrics["fprl/query_loss"]) and torch.isfinite(metrics["fprl/support_loss"])


def test_episode_sampler_takes_a_puzzles_copies_with_one_query_row_each(tmp_path):
    import json

    import numpy as np

    # A build's train split: 2 groups (puzzles); the first has 3 copies of 2, 2 and 3 rows, the second one copy of 1 row
    split = tmp_path / "train"
    split.mkdir()
    rows = 8
    arrays = {
        "inputs": np.arange(rows * 4).reshape(rows, 4) % 7 + 1,
        "labels": np.full((rows, 4), 9),
        "puzzle_identifiers": np.array([1, 2, 3, 4]),
        "puzzle_indices": np.array([0, 2, 4, 7, 8]),
        "group_indices": np.array([0, 3, 4]),
    }
    for k, v in arrays.items():
        np.save(split / f"all__{k}.npy", v.astype(np.int32))
    (split / "dataset.json").write_text(json.dumps({"ignore_label_id": 9}))

    sampler = EpisodeSampler(str(tmp_path), meta_batch_size=2, copies=2, device=torch.device("cpu"), seed=0)
    episodes = sampler.sample()
    assert len(episodes) == 1  # the single-row puzzle gives no episode
    support, query = episodes[0]
    assert query["inputs"].shape[0] == 2 and len(set(query["puzzle_identifiers"].tolist())) == 2  # one query row per copy
    assert support["inputs"].shape[0] >= 2 and set(support["puzzle_identifiers"].tolist()) == set(query["puzzle_identifiers"].tolist())
    assert (query["labels"] == IGNORE_LABEL_ID).all() and support["inputs"].dtype == torch.int32
    assert not any((support["inputs"] == q).all(-1).any() for q in query["inputs"])  # query rows are not in the support


def test_maml_config_is_a_cli_option():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="cfg_pretrain", overrides=["seed=0", "maml.enabled=true", "maml.inner_steps=2", "maml.inner_lr=0.1", "maml.meta_batch_size=2"])
        plain = compose(config_name="cfg_pretrain", overrides=["seed=0"])
    assert PretrainConfig(**OmegaConf.to_container(cfg, resolve=True)).maml == MAMLConfig(enabled=True, inner_steps=2, inner_lr=0.1, meta_batch_size=2, support_copies=16)
    plain_dict = OmegaConf.to_container(plain, resolve=True)
    assert not PretrainConfig(**plain_dict).maml.enabled
    assert not PretrainConfig(**{k: v for k, v in plain_dict.items() if k != "maml"}).maml.enabled  # checkpoints saved before the key
    with pytest.raises(ValueError):
        MAMLConfig(enabled=True, inner_steps=0, inner_lr=0.1)
