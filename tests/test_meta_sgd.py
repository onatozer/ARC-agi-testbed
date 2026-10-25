"""Second-order Meta-SGD (arXiv 1707.09835) as pretrain.py's training step (config.meta_sgd.enabled): every puzzle in the
batch is a task, its validation rows (former test pair) the query set, its other rows the support set."""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hydra import compose, initialize
from omegaconf import OmegaConf

from config.schema import PretrainConfig
from dataset.common import PuzzleDatasetMetadata
from evaluate_ttt import meta_sgd_adapt
from pretrain import _adapt, _adapt_segment, _adapt_segment_wait, _episode_overrides, _episodes_from_batch, _meta_sgd_targets, _msl_weights, _puzzle_emb_name, _supervised_loss, _task_outer_backward, init_train_state, meta_sgd_batch

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
    # Grouped copies: the validation row of a copy whose demos were cut off at the batch tail is dropped (its id has no row
    # in the support set the query ids are remapped into)
    grouped = {"inputs": torch.arange(4).view(4, 1), "puzzle_identifiers": torch.tensor([1, 1, 2, 2]), "group_ids": torch.tensor([7, 7, 7, 7]), "validation": torch.tensor([0, 1, 1, 1])}
    [(key, s, q)] = _episodes_from_batch(grouped)
    assert key == 7 and s["inputs"].flatten().tolist() == [0] and q["inputs"].flatten().tolist() == [1]


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
    rows, val = batch["puzzle_identifiers"] == 1, batch["validation"].bool()  # the raw rows: the step remaps the ids itself
    loss_s, loss_q, ids, row_grad = _task_outer_backward(model, _meta_sgd_targets(model, []), emb_name, {k: v[rows & ~val] for k, v in batch.items()}, {k: v[rows & val] for k, v in batch.items()}, torch.tensor(1.0, device=emb_rows.device))
    assert torch.isfinite(loss_s) and torch.isfinite(loss_q) and ids.tolist() == [1] and row_grad.shape == (1, 32) and row_grad.abs().sum() > 0
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
    assert all((meta.bounds[k][0] <= r).all() and (r <= meta.bounds[k][1]).all() for k, r in meta.rho.items())
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


def test_episode_overrides_stand_the_batch_rows_in_for_the_table():
    config, train_state, batch = _state()
    model = train_state.model
    named, overrides, remapped, ids = _episode_overrides(model, _meta_sgd_targets(model, []), _puzzle_emb_name(model), batch)
    assert ids.tolist() == [1, 2, 3] and remapped["puzzle_identifiers"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    emb = overrides[_puzzle_emb_name(model)]
    assert emb.shape == (3, 32) and emb.requires_grad and torch.equal(emb, model.model.puzzle_emb.weights[1:4])
    assert dict(named)[_puzzle_emb_name(model)] is emb and overrides[_puzzle_emb_name(model)[: -len("weights")] + "local_weights"].shape == (0, 32)


def test_meta_sgd_adapt_writes_the_inner_steps_into_the_weights_and_the_support_rows():
    """Test-time use of a Meta-SGD checkpoint: the support batch (every aux copy's demos) adapts the weights and the rows' embeddings in place."""
    config, train_state, batch = _state()
    model = train_state.model
    support = {k: v[batch["validation"] == 0] for k, v in batch.items() if k != "validation"}
    table = model.model.puzzle_emb.weights.detach().clone()
    weights = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("meta_sgd.")}
    losses = meta_sgd_adapt(model, support, 2, [])
    assert len(losses) == 2 and all(l > 0 for l in losses)
    assert (table != model.model.puzzle_emb.weights).any(1).nonzero().flatten().tolist() == [1, 2, 3]
    assert all(not torch.equal(w, dict(model.named_parameters())[n]) for n, w in weights.items())
    assert all(torch.equal(r, model.meta_sgd.rho[k]) for k, r in [(k, r.detach().clone()) for k, r in model.meta_sgd.rho.items()])  # alpha is not trained at test time


def test_group_episodes_pool_a_tasks_copies_and_fresh_rows_leave_the_table_alone():
    """copies_per_group > 1: the sampler packs several copies of a task and tags rows with group_ids; the episode is the task."""
    import json
    from dataset.build_arc_dataset import convert_dataset
    from dataset.build_miniarc_dataset import MiniARCDataProcessConfig, load_puzzles_miniarc
    from dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    from test_build_miniarc_heldout import _write_tasks

    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    _write_tasks(tmp)
    out = tmp / "out"
    convert_dataset(MiniARCDataProcessConfig(input_dir=str(tmp), output_dir=str(out), num_aug=3, grid_size=2, heldout_tasks=1, seed=0), load_puzzles=load_puzzles_miniarc)
    # 3 train tasks x 4 copies x 3 rows = 36 rows; two copies of each task per batch -> 18 rows per pass
    cfg = PuzzleDatasetConfig(seed=0, dataset_path=str(out), global_batch_size=18, test_set_mode=False, epochs_per_iter=1, rank=0, num_replicas=1, copies_per_group=2)
    _, batch, _ = next(iter(PuzzleDataset(cfg, split="train")))
    assert "group_ids" in batch and batch["group_ids"].shape == (18,)
    for g in torch.unique(batch["group_ids"]).tolist():
        assert torch.unique(batch["puzzle_identifiers"][batch["group_ids"] == g]).numel() == 2
    episodes = _episodes_from_batch(batch)
    assert len(episodes) == 3 and all(s["inputs"].shape[0] == 4 and q["inputs"].shape[0] == 2 for _, s, q in episodes)  # 2 copies x (2 demos, 1 test pair)
    assert all(torch.unique(s["puzzle_identifiers"]).numel() == 2 for _, s, _ in episodes)

    # fresh_rows: the rows are a leaf at the table mean, the query ids follow the support's remap, and the table is untouched
    config, train_state, _ = _state("meta_sgd.fresh_rows=true", "global_batch_size=6", "lr=1e-3", "puzzle_emb_lr=1e-2", "meta_sgd.alpha_lr=1e-2")
    model = train_state.model
    table = model.model.puzzle_emb.weights.detach().clone()
    _, s, q = episodes[0]
    s = {k: v.to(table.device) for k, v in s.items() if k != "group_ids"}; q = {k: v.to(table.device) for k, v in q.items() if k != "group_ids"}
    # the tiny test model has 4 identifiers; map the build's ids into range
    remap = {pid: i + 1 for i, pid in enumerate(torch.unique(s["puzzle_identifiers"]).tolist())}
    for b in (s, q):
        b["puzzle_identifiers"] = torch.tensor([remap[p] for p in b["puzzle_identifiers"].tolist()], dtype=torch.int32, device=table.device)
        b["inputs"] = torch.randint(1, 6, (b["inputs"].shape[0], 16), device=table.device); b["labels"] = torch.randint(2, 6, (b["inputs"].shape[0], 16), device=table.device)
    _, _, ids, row_grad = _task_outer_backward(model, _meta_sgd_targets(model, []), _puzzle_emb_name(model), s, q, torch.tensor(1.0, device=table.device), fresh_rows=True)
    assert ids.tolist() == [1, 2] and row_grad.shape == (2, 32)
    batch = {k: torch.cat([s[k], q[k]]) for k in s}; batch["validation"] = torch.tensor([0] * 4 + [1] * 2, dtype=torch.int32, device=table.device)
    train_state.model.zero_grad()
    meta_sgd_batch(config, train_state, batch, global_batch_size=6, rank=0, world_size=1)
    assert torch.equal(table, model.model.puzzle_emb.weights)


def test_rows_get_their_own_initial_alpha():
    config, train_state, _ = _state("meta_sgd.inner_lr=0.01", "meta_sgd.inner_lr_rows=1")
    meta = train_state.model.meta_sgd
    key = meta.key(_puzzle_emb_name(train_state.model))
    assert float(meta.alpha(_puzzle_emb_name(train_state.model)).mean()) == pytest.approx(1.0) and meta.bounds[key][1] == pytest.approx(5.0)
    other = [k for k in meta.rho if k != key][0]
    assert float(meta.rho[other].detach().exp().mean()) == pytest.approx(0.01)


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


def test_msl_weights_anneal_from_uniform_to_the_last_step():
    assert _msl_weights(1, 0.0) == [1.0] and _msl_weights(3, 1.0) == [0.0, 0.0, 1.0] and _msl_weights(3, 7.0) == [0.0, 0.0, 1.0]
    w = _msl_weights(4, 0.0)
    assert all(abs(x - 0.25) < 1e-12 for x in w)
    w = _msl_weights(4, 0.5)
    assert abs(sum(w) - 1) < 1e-12 and w[0] == w[1] == w[2] == pytest.approx(0.125) and w[3] == pytest.approx(0.625)


def test_lslr_alpha_is_per_tensor_and_per_step():
    """How to train your MAML: one alpha per adapted tensor (the table keeps one per column) and one per inner step; test
    steps beyond the trained ones reuse the last step's."""
    config, train_state, batch = _state("meta_sgd.alpha_per_tensor=true", "meta_sgd.alpha_per_step=true", "meta_sgd.inner_steps=3")
    model = train_state.model
    emb_name = _puzzle_emb_name(model)
    for name, p in _meta_sgd_targets(model, []):
        rho = model.meta_sgd.rho[model.meta_sgd.key(name)]
        assert rho.shape == ((3, 1, p.shape[1]) if name == emb_name else (3,))
    name = next(n for n, _ in _meta_sgd_targets(model, []) if n != emb_name)
    with torch.no_grad():
        model.meta_sgd.rho[model.meta_sgd.key(name)][1] += 1.0
    a = [float(model.meta_sgd.alpha(name, k)) for k in range(4)]
    assert a[0] == a[2] == a[3] == pytest.approx(config.meta_sgd.inner_lr) and a[1] == pytest.approx(config.meta_sgd.inner_lr * 2.718281828)
    # The per-element default is unchanged
    _, default_state, _ = _state()
    assert default_state.model.meta_sgd.rho[model.meta_sgd.key(name)].shape == dict(model.named_parameters())[name].shape


def test_multi_step_outer_backward_trains_every_steps_alpha_and_first_order_differs():
    config, train_state, batch = _state("meta_sgd.alpha_per_step=true", "meta_sgd.inner_steps=2")
    model = train_state.model
    named, emb_name, emb_rows, overrides, support, query = _episode(model, batch)
    targets = _meta_sgd_targets(model, [])
    loss_s, loss_q, _, _ = _task_outer_backward(model, targets, emb_name, support, query, torch.tensor(1.0), msl_weights=[0.5, 0.5])
    for name, _ in targets:
        g = model.meta_sgd.rho[model.meta_sgd.key(name)].grad
        assert g is not None and torch.isfinite(g).all() and g[0].abs().sum() > 0 and g[1].abs().sum() > 0, name
    second = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None and not n.startswith("meta_sgd.")}
    model.zero_grad(set_to_none=True)
    _task_outer_backward(model, targets, emb_name, support, query, torch.tensor(1.0), msl_weights=[0.5, 0.5], second_order=False)
    first = {n: p.grad for n, p in model.named_parameters() if p.grad is not None and not n.startswith("meta_sgd.")}
    assert any(not torch.allclose(first[n], second[n], rtol=1e-3, atol=1e-6) for n in second)


def test_maml_plus_plus_options_are_cli_options():
    cfg = _config("meta_sgd.inner_steps=2", "meta_sgd.msl_anneal_fraction=0.3", "meta_sgd.alpha_per_tensor=true", "meta_sgd.alpha_per_step=true", "meta_sgd.first_order_steps=10")
    assert (cfg.meta_sgd.inner_steps, cfg.meta_sgd.msl_anneal_fraction, cfg.meta_sgd.alpha_per_tensor, cfg.meta_sgd.alpha_per_step, cfg.meta_sgd.first_order_steps) == (2, 0.3, True, True, 10)
    for bad in ("meta_sgd.inner_steps=0", "meta_sgd.msl_anneal_fraction=0", "meta_sgd.first_order_steps=-1"):
        with pytest.raises(Exception):
            _config(bad)


def test_normalized_inner_step_moves_every_tensor_by_alpha_rms():
    """normalize_grad: the gradient is divided by its RMS per tensor, so the update's RMS per element is alpha."""
    config, train_state, batch = _state("meta_sgd.normalize_grad=true", "meta_sgd.alpha_per_tensor=true", "meta_sgd.inner_lr=0.01")
    model = train_state.model
    named, emb_name, emb_rows, overrides, support, query = _episode(model, batch)
    _, theta1 = _adapt(model, named, overrides, support)
    for name, p in named:
        if name in theta1 and name != emb_name:  # the table's alpha is per column, so its update RMS is not one number
            rms = float((theta1[name] - p).pow(2).mean().sqrt())
            assert rms == pytest.approx(float(model.meta_sgd.alpha(name)), rel=1e-3), name
    assert _config().meta_sgd.normalize_grad is False


def test_stretch_schedule_spreads_the_per_step_alphas_over_a_longer_loop():
    """Two trained alphas, alpha 1 set to ~0: under hold the weights stop moving after step 1, under stretch they move for the
    first half of the loop."""
    def run(schedule, steps):
        torch.manual_seed(0)  # the same initial weights every call
        config, train_state, batch = _state("meta_sgd.alpha_per_step=true", "meta_sgd.inner_steps=2")
        model = train_state.model.eval()  # no dropout: the trajectories must be comparable
        emb_name = _puzzle_emb_name(model)
        name = next(n for n, _ in _meta_sgd_targets(model, []) if n != emb_name)
        with torch.no_grad():
            for key, rho in model.meta_sgd.rho.items():
                rho[1] = -30.0  # alpha_1 ~ 0
        support = {k: v[batch["validation"] == 0] for k, v in batch.items() if k != "validation"}
        meta_sgd_adapt(model, support, steps, [], schedule)
        return dict(model.named_parameters())[name].detach().clone()
    same = lambda a, b: torch.allclose(a, b, atol=1e-6, rtol=0)
    # 4 test steps over 2 trained alphas: hold uses alpha 0, 1, 1, 1; stretch alpha 0, 0, 1, 1 (step k -> k * 2 // 4)
    assert same(run("hold", 1), run("hold", 4))  # steps 2-4 use alpha_1 ~ 0
    assert not same(run("stretch", 4), run("hold", 4))  # stretch takes a second alpha_0 step
    assert same(run("stretch", 3), run("stretch", 4))  # 3 steps: alpha 0, 0, 1 -- the same two real steps
    with pytest.raises(ValueError):
        run("linear", 2)


def test_one_segment_model_is_unaffected_by_the_segment_options():
    """halt_max_steps=1 (VARC-like): a segment step is the full-trajectory step, and last_segment_grad changes nothing."""
    config, train_state, batch = _state("arch.halt_max_steps=1")
    model = train_state.model
    named, emb_name, emb_rows, overrides, support, query = _episode(model, batch)
    torch.manual_seed(0); loss_a, theta_a = _adapt(model, named, overrides, support)
    torch.manual_seed(0); loss_b, theta_b, carry = _adapt_segment(model, named, overrides, support, None)
    torch.manual_seed(0); loss_c, theta_c = _adapt(model, named, overrides, support, last_segment_grad=True)
    torch.testing.assert_close(loss_a, loss_b); torch.testing.assert_close(loss_a, loss_c)
    for n in theta_a:
        torch.testing.assert_close(theta_a[n], theta_b[n]); torch.testing.assert_close(theta_a[n], theta_c[n])


def test_segment_inner_steps_continue_the_carry():
    """Each segment step advances the support rows' ACT state by one segment, as train_batch does."""
    config, train_state, batch = _state()  # halt_max_steps=4
    model = train_state.model.eval()  # no halting exploration
    named, emb_name, emb_rows, overrides, support, query = _episode(model, batch)
    _, theta, carry = _adapt_segment(model, named, overrides, support, None, second_order=False)
    assert carry.steps.tolist() == [1] * support["inputs"].shape[0] and not carry.steps.requires_grad
    named = [(n, theta.get(n, p)) for n, p in named]
    _, theta, carry2 = _adapt_segment(model, named, theta, support, carry, second_order=False)
    assert all(s in (2, 1) for s in carry2.steps.tolist())  # 2, or 1 where the row halted at step 1 and restarted


def test_last_segment_grad_counts_only_final_segments_and_still_trains_alpha():
    config, train_state, batch = _state("meta_sgd.segment_inner_steps=true", "meta_sgd.last_segment_grad=true", "meta_sgd.inner_steps=2", "meta_sgd.alpha_per_step=true")
    model = train_state.model
    named, emb_name, emb_rows, overrides, support, query = _episode(model, batch)
    with torch.no_grad():
        full = float(_supervised_loss(model, overrides, query))
        last = float(_supervised_loss(model, overrides, query, last_segment_grad=True))
    assert last < full  # only each row's final segment loss is summed (the TINY model runs >1 segment per row)
    targets = _meta_sgd_targets(model, [])
    _task_outer_backward(model, targets, emb_name, support, query, torch.tensor(1.0), msl_weights=[0.5, 0.5],
                         segment_inner_steps=True, last_segment_grad=True)
    for name, _ in targets:
        g = model.meta_sgd.rho[model.meta_sgd.key(name)].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name


def test_halt_wait_rows_sit_out_until_the_trajectory_ends_then_all_restart():
    """meta_sgd.halt_wait: a halted row contributes no loss until the trajectory boundary (every halt_max_steps steps),
    where every row starts afresh; with no row running the weights do not move."""
    config, train_state, batch = _state()  # halt_max_steps=4
    model = train_state.model.eval()  # rows halt only at the cap
    named, emb_name, emb_rows, overrides, support, query = _episode(model, batch)
    n = support["inputs"].shape[0]
    _, theta, state = _adapt_segment_wait(model, named, overrides, support, None, 0, 4, second_order=False)
    carry, idx = state
    assert idx.tolist() == list(range(n)) and carry.steps.tolist() == [1] * n
    carry.halted[0] = True  # as if the ACT head had halted row 0 at segment 1
    _, _, (carry1, idx1) = _adapt_segment_wait(model, named, overrides, support, (carry, idx), 1, 4, second_order=False)
    assert idx1.tolist() == list(range(1, n)) and carry1.steps.tolist() == [2] * (n - 1)
    # Boundary: everyone restarts, the halted row included
    _, _, (carry4, idx4) = _adapt_segment_wait(model, named, overrides, support, (carry1, idx1), 4, 4, second_order=False)
    assert idx4.tolist() == list(range(n)) and carry4.steps.tolist() == [1] * n
    # Nobody running mid-trajectory: no loss, no update
    carry4.halted[:] = True
    loss, theta_same, _ = _adapt_segment_wait(model, named, overrides, support, (carry4, idx4), 5, 4, second_order=False)
    assert loss is None and theta_same.keys() == overrides.keys()


def test_outer_backward_with_halt_wait_trains_every_steps_alpha():
    config, train_state, batch = _state("meta_sgd.segment_inner_steps=true", "meta_sgd.halt_wait=true", "meta_sgd.inner_steps=2", "meta_sgd.alpha_per_step=true")
    model = train_state.model
    named, emb_name, emb_rows, overrides, support, query = _episode(model, batch)
    targets = _meta_sgd_targets(model, [])
    _task_outer_backward(model, targets, emb_name, support, query, torch.tensor(1.0), msl_weights=[0.5, 0.5],
                         segment_inner_steps=True, last_segment_grad=True, halt_wait=True, segments=config.arch.halt_max_steps)
    for name, _ in targets:
        g = model.meta_sgd.rho[model.meta_sgd.key(name)].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name
