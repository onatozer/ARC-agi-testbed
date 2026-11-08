"""Meta-SGD (arXiv 1707.09835) as pretrain.py's training step (config.meta_sgd), config.recon's masked-input
reconstruction rows, and evaluate_ttt.py's test-time steps with the learned step sizes."""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hydra import compose, initialize
from omegaconf import OmegaConf

from config.schema import PretrainConfig, ReconConfig
from dataset.common import PuzzleDatasetMetadata
from evaluate_ttt import meta_sgd_adapt
from pretrain import (
    _adapt, _adapt_segment, _episodes_from_batch, _meta_sgd_targets, _puzzle_emb_name, _task_outer_backward, _with_recon_rows,
    init_train_state, meta_sgd_batch,
)

# 3 puzzles x (2 demos + 1 validation row), one batch of 9 = one epoch
METADATA = PuzzleDatasetMetadata(
    pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=6, seq_len=16, num_puzzle_identifiers=4,
    total_groups=3, mean_puzzle_examples=3.0, sets=["all"],
)
TINY = ["arch.hidden_size=32", "arch.num_heads=2", "arch.expansion=2", "arch.L_layers=1", "arch.puzzle_emb_ndim=32", "dataset.grid_size=4",
        "arch.halt_max_steps=4", "global_batch_size=9", "epochs=10", "lr_warmup_steps=0", "ema=False", "seed=0", "meta_sgd.enabled=true",
        "meta_sgd.inner_steps=2", "meta_sgd.copies_per_group=1"]
SEGMENTS = ["meta_sgd.segment_inner_steps=true"]
RECON = ["recon.copies=1", "recon.fraction=0.25", "arch.loss.name=losses.loss_heads@ACTLossHeadRowWeight"]

_g = torch.Generator().manual_seed(0)
BATCH = {
    "inputs": torch.randint(2, 6, (9, 16), generator=_g),
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


def _task(batch, pid=1):
    """(support, query) rows of puzzle `pid`."""
    rows, val = batch["puzzle_identifiers"] == pid, batch["validation"].bool()
    return {k: v[rows & ~val] for k, v in batch.items()}, {k: v[rows & val] for k, v in batch.items()}


def _weights(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("meta_sgd.")}


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

    # Grouped copies: the validation row of a copy whose demos were cut off at the batch tail is dropped
    grouped = {"inputs": torch.arange(4).view(4, 1), "puzzle_identifiers": torch.tensor([1, 1, 2, 2]), "group_ids": torch.tensor([7, 7, 7, 7]), "validation": torch.tensor([0, 1, 1, 1])}
    [(key, s, q)] = _episodes_from_batch(grouped)
    assert key == 7 and s["inputs"].flatten().tolist() == [0] and q["inputs"].flatten().tolist() == [1]


def test_sampler_packs_a_tasks_copies_into_one_episode(tmp_path):
    from dataset.build_arc_dataset import convert_dataset
    from dataset.build_miniarc_dataset import MiniARCDataProcessConfig, load_puzzles_miniarc
    from dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    from test_build_miniarc_heldout import _write_tasks

    _write_tasks(tmp_path)
    out = tmp_path / "out"
    convert_dataset(MiniARCDataProcessConfig(input_dir=str(tmp_path), output_dir=str(out), num_aug=3, grid_size=2, heldout_tasks=1, seed=0), load_puzzles=load_puzzles_miniarc)

    # 3 train tasks x 4 copies x 3 rows; two copies of each task per batch -> 18 rows per pass
    cfg = PuzzleDatasetConfig(seed=0, dataset_path=str(out), global_batch_size=18, test_set_mode=False, epochs_per_iter=1, rank=0, num_replicas=1, copies_per_group=2)
    _, batch, _ = next(iter(PuzzleDataset(cfg, split="train")))
    assert batch["group_ids"].shape == (18,)

    episodes = _episodes_from_batch(batch)
    assert len(episodes) == 3 and all(s["inputs"].shape[0] == 4 and q["inputs"].shape[0] == 2 for _, s, q in episodes)  # 2 copies x (2 demos, 1 test pair)
    assert all(torch.unique(s["puzzle_identifiers"]).numel() == 2 for _, s, _ in episodes)


@pytest.mark.parametrize("overrides", [[], SEGMENTS, SEGMENTS + RECON])
def test_meta_sgd_batch_trains_theta_and_alpha_but_not_the_table(overrides):
    config, train_state, batch = _state("lr=0", "puzzle_emb_lr=0", "meta_sgd.alpha_lr=0", *overrides)
    model = train_state.model
    weights, rho = _weights(model), {k: r.detach().clone() for k, r in model.meta_sgd.rho.items()}
    metrics = meta_sgd_batch(config, train_state, batch, global_batch_size=9, rank=0, world_size=1)
    assert metrics["train/meta/tasks"] == 3 and metrics["train/meta/query_loss"] > 0
    assert metrics["train/meta/alpha_max"] == pytest.approx(3e-2, rel=1e-4)  # the rows' initial step size (the weights': 1e-3)
    assert all(torch.equal(w, dict(model.named_parameters())[n]) for n, w in weights.items())  # lr 0: nothing moves

    config, train_state, batch = _state("lr=1e-3", "puzzle_emb_lr=1e-2", *overrides)
    model = train_state.model
    table, weights, rho = model.model.puzzle_emb.weights.detach().clone(), _weights(model), {k: r.detach().clone() for k, r in model.meta_sgd.rho.items()}
    meta_sgd_batch(config, train_state, batch, global_batch_size=9, rank=0, world_size=1)
    assert torch.equal(table, model.model.puzzle_emb.weights)  # the episodes start from the table mean: it is not trained
    assert any(not torch.equal(w, dict(model.named_parameters())[n]) for n, w in weights.items())
    assert all(not torch.equal(r, model.meta_sgd.rho[k]) for k, r in rho.items() if "L_level" in k)  # the recurrent blocks' alphas learn


def test_inner_step_moves_every_tensor_by_its_alpha_per_step():
    """The gradient is divided by its RMS per tensor, so an update's RMS per element is that tensor's alpha at that step."""
    config, train_state, batch = _state()
    model = train_state.model
    emb_name = _puzzle_emb_name(model)
    assert all(r.shape == ((2, 1, 32) if "puzzle_emb" in k else (2,)) for k, r in model.meta_sgd.rho.items())

    with torch.no_grad():
        for rho in model.meta_sgd.rho.values():
            rho[1] = -3.0
    support, _ = _task(batch)
    named = [(n, p) for n, p in _meta_sgd_targets(model) if n != emb_name]
    for step in (0, 1):
        _, theta = _adapt(model, named, {}, support, step)
        for name, p in named:
            if name in theta:
                rms = float((theta[name] - p).pow(2).mean().sqrt())
                assert rms == pytest.approx(float(model.meta_sgd.alpha(name, step)), rel=1e-3), name


def test_segment_inner_steps_continue_the_carry():
    """Each segment step advances the support rows' ACT state by one segment, as train_batch does."""
    config, train_state, batch = _state(*SEGMENTS)
    model = train_state.model.eval()  # no halting exploration
    support, _ = _task(batch)
    named = [(n, p) for n, p in _meta_sgd_targets(model) if n != _puzzle_emb_name(model)]

    _, theta, carry = _adapt_segment(model, named, {}, support, None, 0)
    assert carry.steps.tolist() == [1] * support["inputs"].shape[0] and not carry.steps.requires_grad

    named = [(n, theta.get(n, p)) for n, p in named]
    _, theta, carry = _adapt_segment(model, named, theta, support, carry, 1)
    assert all(s in (2, 1) for s in carry.steps.tolist())  # 2, or 1 where the row halted at step 1 and restarted


def test_test_time_steps_stretch_the_trained_alphas_over_the_loop():
    """Two trained alphas, the second ~0: 2 test steps use alphas 0, 1 (one real step), 4 use 0, 0, 1, 1 (two)."""
    def run(steps):
        torch.manual_seed(0)  # the same initial weights every call
        config, train_state, batch = _state()
        model = train_state.model.eval()  # no dropout: the trajectories must be comparable
        with torch.no_grad():
            for rho in model.meta_sgd.rho.values():
                rho[1] = -30.0
        meta_sgd_adapt(model, _task(batch)[0], steps, segment_steps=False)
        return _weights(model)

    same = lambda a, b: all(torch.allclose(a[n], b[n], atol=1e-6, rtol=0) for n in a)
    assert same(run(1), run(2))
    assert not same(run(2), run(4))


def test_test_time_rows_frozen_adapts_only_the_weights():
    config, train_state, batch = _state()
    model = train_state.model
    table, weights = model.model.puzzle_emb.weights.detach().clone(), _weights(model)
    meta_sgd_adapt(model, _task(batch)[0], 2, segment_steps=False, rows_frozen=True)
    assert torch.equal(table, model.model.puzzle_emb.weights)
    assert any(not torch.equal(w, dict(model.named_parameters())[n]) for n, w in weights.items())


def test_recon_rows_mask_cells_and_weight_the_copies():
    """Each row gets `copies` masked copies: `fraction` of its color cells (tokens 2-11; never pad 0 or end-of-grid 1) set
    to the mask token, labelled with the full input, on the shared row, weighted 1 / copies; the row-weighting head then
    counts them together as one row."""
    from models.losses.loss_heads import ACTLossHeadRowWeight

    batch = {"inputs": torch.tensor([[2, 3, 4, 5, 1, 0], [6, 7, 8, 9, 10, 11]]), "labels": torch.tensor([[5, 5, 5, 5, 0, 0], [2, 2, 2, 2, 2, 2]]),
             "puzzle_identifiers": torch.tensor([3, 4])}
    out = _with_recon_rows(batch, ReconConfig(copies=5, fraction=0.5), mask_token=12, shared_id=9)
    assert out["inputs"].shape == (12, 6) and out["puzzle_identifiers"].tolist() == [3, 4] + [9] * 10
    assert torch.equal(out["labels"][2:], batch["inputs"].repeat(5, 1)) and torch.equal(out["inputs"][:2], batch["inputs"])
    assert torch.allclose(out["row_weight"], torch.tensor([1.0, 1.0] + [0.2] * 10))

    masked = out["inputs"][2:] == 12
    assert masked.sum(-1).tolist() == [2, 3] * 5  # half of 4 and of 6 color cells
    assert not masked[0::2, 4:].any()  # never the end-of-grid or pad cell
    assert torch.equal(out["inputs"][2:][~masked], batch["inputs"].repeat(5, 1)[~masked])

    # The weighted loss: 5 copies at 1/5 count as one row
    head = ACTLossHeadRowWeight.__new__(ACTLossHeadRowWeight)
    torch.nn.Module.__init__(head)
    head.hsw_lambda, head._row_weight = None, torch.tensor([1.0] + [0.2] * 5)
    head.loss_fn = lambda logits, labels, valid_mask: valid_mask.float()
    labels = torch.ones(6, 4, dtype=torch.long)
    total, _ = head.compute_losses(outputs={"logits": torch.zeros(6, 4, 3), "q_halt_logits": torch.zeros(6)}, labels=labels, mask=labels > 0,
                                   loss_divisor=torch.full((6, 1), 4.0), seq_is_correct=torch.zeros(6, dtype=torch.bool),
                                   step_weight=head._step_weights(torch.ones(6)))
    assert float(total) == pytest.approx(2.0 + 0.5 * 2.0 * float(torch.nn.functional.softplus(torch.tensor(0.0))), rel=1e-5)


def test_recon_needs_a_row_weighting_head():
    with pytest.raises(ValueError, match="row-weighting"):
        _state("recon.copies=1")


def test_inner_steps_reconstruct_masked_demos_on_the_shared_row(monkeypatch):
    """With config.recon the episode's rows include the shared reconstruction row (the table's last) at its learned value."""
    import pretrain

    torch.manual_seed(0)
    config, train_state, batch = _state(*SEGMENTS, *RECON)
    model = train_state.model
    emb_name = _puzzle_emb_name(model)
    table = model.get_buffer(emb_name)
    assert table.shape[0] == METADATA.num_puzzle_identifiers + 1

    seen, real = {}, pretrain._episode_overrides
    def spy(model, targets, emb_name, batch, extra_ids=None):
        out = real(model, targets, emb_name, batch, extra_ids)
        seen["ids"], seen["rows"] = batch["puzzle_identifiers"].tolist(), out[3].tolist()
        return out
    monkeypatch.setattr(pretrain, "_episode_overrides", spy)

    support, query = _task(batch)
    _task_outer_backward(config, model, _meta_sgd_targets(model), emb_name, support, query, torch.tensor(1.0), [0.0, 1.0])
    assert seen["rows"] == [1, table.shape[0] - 1] and seen["ids"] == [1, 1, 4, 4]  # 2 demos, then their masked copies


def test_disabled_config_is_the_ordinary_setup():
    from pretrain import _copies_per_group

    config = _config("meta_sgd.enabled=false", "meta_sgd.copies_per_group=16")
    assert config.recon.copies == 0 and _copies_per_group(config) == 1  # the sampler draws one puzzle per group, as before
    assert not hasattr(init_train_state(config, METADATA, rank=0, world_size=1).model, "meta_sgd")


def test_weights_only_load_restarts_the_ema_from_the_loaded_weights(tmp_path):
    """+load_weights_only without torch.compile (every Meta-SGD fine-tune): the EMA, registered from the random init when the
    model was built, restarts from the loaded weights instead of averaging the init in."""
    from utils.checkpoint import load_training_state

    config, src, _ = _state("ema=true")
    with torch.no_grad():
        for p in src.model.parameters():
            p.add_(1.0)
    path = tmp_path / "ckpt.pth"
    torch.save({"model": src.model.state_dict()}, path)

    config, train_state, _ = _state("ema=true")
    config.load_checkpoint = str(path)
    load_training_state(config, train_state, None, rank=0, world_size=1, load_weights_only=True)
    for name, p in train_state.model.named_parameters():
        if name in train_state.ema_helper.shadow:
            assert torch.equal(train_state.ema_helper.shadow[name], p.data), name


def test_loading_a_checkpoint_without_the_mask_token_grows_the_vocabulary():
    """A plain checkpoint (vocab V) into a config.recon model (vocab V + 1): its rows load, the mask token's row keeps its init."""
    from utils.checkpoint import load_model_state_dict

    _, plain, _ = _state("meta_sgd.enabled=false")
    _, recon, _ = _state("meta_sgd.enabled=false", *RECON)
    src = {k: v.detach().clone() for k, v in plain.model.state_dict().items()}
    keys = [k for k in src if k.endswith(("embed_tokens.embedding_weight", "lm_head.weight"))]
    assert len(keys) == 2

    init = {k: recon.model.state_dict()[k].detach().clone() for k in keys}
    load_model_state_dict(recon.model, dict(src), strict=False)
    for k in keys:
        loaded = recon.model.state_dict()[k]
        assert loaded.shape[0] == src[k].shape[0] + 1
        assert torch.equal(loaded[:-1], src[k].to(loaded.dtype)) and torch.equal(loaded[-1], init[k][-1])


def test_stage_two_learns_only_the_reconstruction_step_sizes():
    """meta_sgd.recon_steps: the episodes add reconstruction steps after the inner steps; only their per-step scales learn."""
    config, train_state, batch = _state(*SEGMENTS, *RECON, "meta_sgd.recon_steps=3", "meta_sgd.recon_scale0=0.01", "lr=1e-3", "puzzle_emb_lr=1e-2")
    model = train_state.model
    scale = model.meta_sgd.recon_log_scale
    assert scale.shape == (3,) and torch.allclose(scale.exp(), torch.full((3,), 0.01, device=scale.device))
    assert len(train_state.optimizers) == 1  # nothing else is trained

    name = next(n for n, _ in _meta_sgd_targets(model) if "L_level" in n)
    assert float(model.meta_sgd.recon_alpha(name, 0)) == pytest.approx(0.01 * 1e-3, rel=1e-4)  # scale x the tensor's initial alpha

    table, weights = model.model.puzzle_emb.weights.detach().clone(), _weights(model)
    rho = {k: r.detach().clone() for k, r in model.meta_sgd.rho.items()}
    before = scale.detach().clone()
    metrics = meta_sgd_batch(config, train_state, batch, global_batch_size=9, rank=0, world_size=1)
    assert not torch.equal(before, model.meta_sgd.recon_log_scale) and "train/meta/recon_scale_mean" in metrics
    assert torch.equal(table, model.model.puzzle_emb.weights)
    assert all(torch.equal(w, dict(model.named_parameters())[n]) for n, w in weights.items())
    assert all(torch.equal(r, model.meta_sgd.rho[k]) for k, r in rho.items())


def test_stage_two_test_time_takes_the_learned_reconstruction_steps():
    from evaluate_ttt import learned_recon_adapt

    config, train_state, batch = _state(*SEGMENTS, *RECON, "meta_sgd.recon_steps=2")
    model = train_state.model
    with torch.no_grad():
        model.meta_sgd.recon_log_scale.fill_(0.0)  # scale 1: a visible move
    weights = _weights(model)
    test = {k: batch[k][:3] for k in ("inputs", "labels", "puzzle_identifiers")}
    learned_recon_adapt(model, test, 4, config.arch.halt_max_steps, config.recon, METADATA.num_puzzle_identifiers)
    assert any(not torch.equal(w, dict(model.named_parameters())[n]) for n, w in weights.items())


def test_stage_two_needs_reconstruction():
    with pytest.raises(ValueError, match="config.recon"):
        _state(*SEGMENTS, "meta_sgd.recon_steps=2")


def test_stage_two_runs_on_a_one_pass_model():
    """VARC has no latent carry to roll out: its reconstruction steps take the loss head's CE of its single forward pass."""
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="cfg_pretrain_miniarc_varc", overrides=[
            "dataset.grid_size=4", "arch.hidden_size=32", "arch.num_layers=1", "arch.mlp_dim=32", "arch.num_heads=2", "global_batch_size=9",
            "epochs=10", "lr_warmup_steps=0", "ema=False", "seed=0", "meta_sgd.enabled=true", "meta_sgd.inner_steps=2", "meta_sgd.copies_per_group=1",
            "recon.copies=1", "recon.fraction=0.25", "arch.loss.name=losses.loss_heads@ACTLossHeadRowWeight", "meta_sgd.recon_steps=2"])
    config = PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    train_state = init_train_state(config, METADATA.model_copy(update={"vocab_size": 12}), rank=0, world_size=1)  # VARC: ARC's tokens (+ the mask)
    model = train_state.model
    assert not hasattr(model.model.inner, "reset_carry")

    before = model.meta_sgd.recon_log_scale.detach().clone()
    device = next(model.parameters()).device
    meta_sgd_batch(config, train_state, {k: v.to(device) for k, v in BATCH.items()}, global_batch_size=9, rank=0, world_size=1)
    assert not torch.equal(before, model.meta_sgd.recon_log_scale)


@pytest.mark.parametrize("overrides", [[], SEGMENTS])
def test_second_order_changes_the_meta_gradient_and_stays_off_at_test_time(overrides):
    grads = {}
    for second in (False, True):
        torch.manual_seed(0)
        config, train_state, batch = _state("lr=1e-3", "meta_sgd.inner_lr=1e-1", f"meta_sgd.second_order={second}", *overrides)
        model = train_state.model
        before = _weights(model)
        meta_sgd_batch(config, train_state, batch, global_batch_size=9, rank=0, world_size=1)
        grads[second] = {n: dict(model.named_parameters())[n].detach() - w for n, w in before.items()}
        assert model.meta_sgd.second_order is second
    assert any(not torch.allclose(grads[False][n], grads[True][n]) for n in grads[False])
    assert not getattr(_state(*overrides)[1].model.meta_sgd, "second_order", False)  # a fresh model (as evaluate_ttt builds): first order


def test_a_list_of_batches_is_one_optimizer_step_over_all_their_episodes():
    """accum_steps: meta_sgd_batch on [batch of tasks 1-2, batch of task 3] updates exactly as on the whole batch."""
    split = [{k: v[:6] for k, v in BATCH.items()}, {k: v[6:] for k, v in BATCH.items()}]
    after = []
    for b in (BATCH, split):
        torch.manual_seed(0)
        config, train_state, _ = _state("lr=1e-3")
        device = next(train_state.model.parameters()).device
        b = [{k: v.to(device) for k, v in part.items()} for part in b] if isinstance(b, list) else {k: v.to(device) for k, v in b.items()}
        metrics = meta_sgd_batch(config, train_state, b, global_batch_size=9, rank=0, world_size=1)
        assert metrics["train/meta/tasks"] == 3 and train_state.step == 1
        after.append(_weights(train_state.model))
    assert all(torch.allclose(after[0][n], after[1][n], atol=1e-6) for n in after[0])


def test_accum_steps_must_be_positive():
    with pytest.raises(Exception):
        _config("meta_sgd.accum_steps=0")


def test_rows_first_inner_loop_moves_rows_then_only_the_backbone(monkeypatch):
    """meta_sgd.rows_steps: the first steps adapt only the rows, the rest only the backbone."""
    import pretrain
    calls = []
    real = pretrain._inner_update
    def spy(model, named, overrides, loss, step, recon=False):
        calls.append((step, sorted({n.split(".")[-1] for n, _ in named})))
        return real(model, named, overrides, loss, step, recon)
    monkeypatch.setattr(pretrain, "_inner_update", spy)
    config, train_state, batch = _state("meta_sgd.rows_steps=2", "meta_sgd.inner_steps=3")
    meta_sgd_batch(config, train_state, batch, global_batch_size=9, rank=0, world_size=1)
    first = calls[:5]  # one task: 2 rows-only steps, then 3 backbone steps
    assert [c[1] == ["weights"] for c in first] == [True, True, False, False, False]
    assert all("weights" not in c[1] for c in first[2:])


def test_rows_first_test_time_adapts_rows_then_weights():
    config, train_state, batch = _state("meta_sgd.rows_steps=2")
    model = train_state.model
    support, _ = _task(batch)
    table, weights = model.model.puzzle_emb.weights.detach().clone(), _weights(model)
    meta_sgd_adapt(model, support, 3, segment_steps=False, rows_only=2)
    assert not torch.equal(table, model.model.puzzle_emb.weights)
    assert all(torch.equal(w, dict(model.named_parameters())[n]) for n, w in weights.items())  # backbone untouched
    table = model.model.puzzle_emb.weights.detach().clone()
    meta_sgd_adapt(model, support, 3, segment_steps=False, rows_frozen=True)
    assert torch.equal(table, model.model.puzzle_emb.weights)
    assert any(not torch.equal(w, dict(model.named_parameters())[n]) for n, w in weights.items())
