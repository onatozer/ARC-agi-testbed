"""Second-order Meta-SGD (arXiv 1707.09835) as pretrain.py's training step (config.meta_sgd.enabled): every puzzle in the
batch is a task, its validation rows (former test pair) the query set, its other rows the support set."""
import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from config.schema import PretrainConfig
from dataset.common import PuzzleDatasetMetadata
from pretrain import _adapt, _episodes_from_batch, _meta_sgd_targets, _puzzle_emb_name, _supervised_loss, _task_outer_backward, init_train_state, meta_sgd_batch

# 3 puzzles x (2 demos + 1 validation row), one batch of 9 = one epoch
METADATA = PuzzleDatasetMetadata(
    pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=6, seq_len=16, num_puzzle_identifiers=4,
    total_groups=3, mean_puzzle_examples=3.0, sets=["all"],
)
TINY = ["arch.hidden_size=32", "arch.num_heads=2", "arch.expansion=2", "arch.L_layers=1", "arch.puzzle_emb_ndim=32", "dataset.grid_size=4",
        "arch.halt_max_steps=4", "global_batch_size=9", "epochs=10", "lr_warmup_steps=0", "ema=False", "seed=0", "meta_sgd.enabled=true"]
_g = torch.Generator().manual_seed(0)
BATCH = {
    "inputs": torch.randint(1, 6, (9, 16), generator=_g),
    "labels": torch.randint(2, 6, (9, 16), generator=_g),
    "puzzle_identifiers": torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.int32),
    "validation": torch.tensor([0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=torch.int32),
}


def _config(*overrides) -> PretrainConfig:
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="cfg_pretrain", overrides=[*TINY, *overrides])
    return PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]


def _state(*overrides):
    config = _config(*overrides)
    train_state = init_train_state(config, METADATA, rank=0, world_size=1)
    device = next(train_state.model.parameters()).device
    return config, train_state, {k: v.to(device) for k, v in BATCH.items()}


def _episode(model, batch, pid=1):
    """The per-task inputs _task_outer_backward builds for puzzle `pid`: targets, embedding name/leaf/overrides, support, query."""
    emb_name = _puzzle_emb_name(model)
    targets = _meta_sgd_targets(model, [])
    table = model.get_buffer(emb_name)
    emb_rows = table[[pid]].detach().requires_grad_()
    overrides = {emb_name: emb_rows, emb_name[: -len("weights")] + "local_weights": table.new_zeros(0, table.shape[1])}
    rows, val = batch["puzzle_identifiers"] == pid, batch["validation"].bool()
    sub = lambda m: {**{k: v[m] for k, v in batch.items()}, "puzzle_identifiers": torch.zeros(int(m.sum()), dtype=torch.int32, device=table.device)}
    named = [(n, emb_rows if n == emb_name else p) for n, p in targets]
    return named, emb_name, emb_rows, overrides, sub(rows & ~val), sub(rows & val)


def test_episodes_split_each_puzzle_by_the_validation_flag_and_skip_incomplete_ones():
    batch = {"inputs": torch.arange(8).view(8, 1), "puzzle_identifiers": torch.tensor([0, 0, 1, 1, 1, 2, 2, 3]), "validation": torch.tensor([0, 0, 0, 0, 1, 0, 1, 0])}
    episodes = _episodes_from_batch(batch)
    assert [pid for pid, _, _ in episodes] == [1, 2]  # 0 is padding, 3 has no validation row
    (_, s1, q1), (_, s2, q2) = episodes
    assert s1["inputs"].flatten().tolist() == [2, 3] and q1["inputs"].flatten().tolist() == [4]
    assert s2["inputs"].flatten().tolist() == [5] and q2["inputs"].flatten().tolist() == [6]
    with pytest.raises(ValueError):
        _episodes_from_batch({k: v for k, v in batch.items() if k != "validation"})
    with pytest.raises(RuntimeError):
        _episodes_from_batch({**batch, "validation": torch.zeros(8, dtype=torch.int64)})


def test_inner_step_scales_with_alpha():
    config, train_state, batch = _state()
    model = train_state.model
    named, emb_name, emb_rows, overrides, support, query = _episode(model, batch)
    _, theta1 = _adapt(model, named, overrides, support)
    name, p = named[0]
    with torch.no_grad():
        model.meta_sgd.rho[model.meta_sgd.key(name)] -= torch.log(torch.tensor(2.0))
    _, theta1_half = _adapt(model, named, overrides, support)
    torch.testing.assert_close(theta1_half[name] - p, (theta1[name] - p) / 2, rtol=1e-4, atol=1e-6)
    assert all(torch.equal(theta1[n], theta1_half[n]) for n in theta1 if n != name)  # other tensors use their own alpha
    assert theta1[emb_name].shape == (1, 32) and not torch.equal(theta1[emb_name], emb_rows)  # the task's row is adapted too


def test_outer_backward_reaches_theta_rho_and_the_embedding_row_with_second_order_gradients():
    config, train_state, batch = _state()
    model = train_state.model
    named, emb_name, emb_rows, overrides, support, query = _episode(model, batch)
    loss_s, loss_q, row_grad = _task_outer_backward(model, _meta_sgd_targets(model, []), emb_name, 1, support, query, torch.tensor(1.0, device=emb_rows.device))
    assert torch.isfinite(loss_s) and torch.isfinite(loss_q) and row_grad.shape == (1, 32) and row_grad.abs().sum() > 0
    for key, rho in model.meta_sgd.rho.items():
        assert rho.grad is not None and torch.isfinite(rho.grad).all() and rho.grad.abs().sum() > 0, key
    assert model.meta_sgd.rho[model.meta_sgd.key(emb_name)].grad.shape == (1, 32)
    second_order = {n: p.grad.clone() for n, p in model.named_parameters() if not n.startswith("meta_sgd.")}
    assert all(g is not None for g in second_order.values())

    # First-order: the query gradient at theta1 with theta1 cut from theta. It must differ from the second-order one
    model.zero_grad()
    _, theta1 = _adapt(model, named, overrides, support)
    theta1 = {n: t.detach().requires_grad_() for n, t in theta1.items()}
    _supervised_loss(model, theta1, query).backward()
    first_order = {n: theta1[n].grad for n in second_order}
    assert any(not torch.allclose(first_order[n], second_order[n], rtol=1e-3, atol=1e-6) for n in second_order)


def test_meta_sgd_batch_updates_only_through_the_optimizers():
    config, train_state, batch = _state("lr=0", "puzzle_emb_lr=0", "meta_sgd.alpha_lr=0")
    table = train_state.model.model.puzzle_emb.weights.detach().clone()
    weights = {n: p.detach().clone() for n, p in train_state.model.named_parameters()}
    metrics = meta_sgd_batch(config, train_state, batch, global_batch_size=9, rank=0, world_size=1)
    assert torch.equal(table, train_state.model.model.puzzle_emb.weights)
    assert all(torch.equal(w, p) for w, p in zip(weights.values(), train_state.model.parameters()))
    assert all(p.grad is None or not p.grad.any() for p in train_state.model.parameters())
    assert metrics["train/meta/tasks"] == 3 and metrics["train/meta/query_loss"] > 0 and metrics["train/meta/alpha_mean"] == pytest.approx(0.1, rel=1e-4)

    config, train_state, batch = _state("lr=1e-3", "puzzle_emb_lr=1e-2", "meta_sgd.alpha_lr=1e-2")
    table = train_state.model.model.puzzle_emb.weights.detach().clone()
    meta_sgd_batch(config, train_state, batch, global_batch_size=9, rank=0, world_size=1)
    changed = (table != train_state.model.model.puzzle_emb.weights).any(1).nonzero().flatten().tolist()
    assert changed == [1, 2, 3]  # only the batch's puzzles' rows
    meta = train_state.model.meta_sgd
    assert all((meta.lo <= r).all() and (r <= meta.hi).all() for r in meta.rho.values())
    assert meta.summary()[0] != pytest.approx(0.1, rel=1e-6)  # alpha moved


def test_supervised_loss_runs_each_row_until_act_halts_it():
    """Rows are dropped as the model halts them; every row contributes one loss per step of its trajectory, capped at halt_max_steps."""
    config, train_state, batch = _state("arch.halt_max_steps=3")
    model = train_state.model
    named, emb_name, emb_rows, overrides, support, query = _episode(model, batch)
    calls = []
    orig = torch.func.functional_call

    def spy(module, overrides_, args=(), kwargs=None):
        out = orig(module, overrides_, args=args, kwargs=kwargs)
        calls.append((kwargs["batch"]["inputs"].shape[0], out[1].detach().clone()))
        return out

    torch.func.functional_call = spy
    try:
        total = _supervised_loss(model, overrides, support)
    finally:
        torch.func.functional_call = orig
    rows = [n for n, _ in calls]
    assert 1 <= len(calls) <= 3 and rows[0] == 2 and all(a >= b for a, b in zip(rows, rows[1:])) and rows[-1] >= 1
    torch.testing.assert_close(total, sum(l for _, l in calls))


def test_disabled_config_is_the_ordinary_setup():
    config = _config("meta_sgd.enabled=false")
    train_state = init_train_state(config, METADATA, rank=0, world_size=1)
    assert not hasattr(train_state.model, "meta_sgd") and len(train_state.optimizers) == 2


def test_config_is_a_cli_option():
    config = _config("meta_sgd.inner_lr=0.05", "meta_sgd.adapt_substrings=[puzzle_emb]", "meta_sgd.alpha_range=2")
    assert config.meta_sgd.inner_lr == 0.05 and config.meta_sgd.adapt_substrings == ["puzzle_emb"] and config.meta_sgd.alpha_range == 2
    with pytest.raises(ValueError):
        _config("meta_sgd.adapt_substrings=[no_such_module]") and init_train_state(_config("meta_sgd.adapt_substrings=[no_such_module]"), METADATA, rank=0, world_size=1)


def test_adapt_substrings_restrict_the_inner_step():
    config, train_state, batch = _state("meta_sgd.adapt_substrings=[puzzle_emb]")
    model = train_state.model
    assert list(model.meta_sgd.rho) == [model.meta_sgd.key(_puzzle_emb_name(model))]
    weights = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("meta_sgd.")}
    meta_sgd_batch(config, train_state, batch, global_batch_size=9, rank=0, world_size=1)
    assert any(not torch.equal(w, dict(model.named_parameters())[n]) for n, w in weights.items())  # the outer step still trains the weights
