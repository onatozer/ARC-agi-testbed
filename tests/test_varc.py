from collections import Counter

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dataset.build_arc_dataset import arc_grid_to_np, grid_hash
from evaluate_varc_ttt import VARCTTTConfig, build_aux_tasks, decode_prediction, encode_pair, rank_votes, score_task
from models.losses import IGNORE_LABEL_ID
from models.losses.loss_heads import ACTLossHead, InferenceLossHead
from models.varc import VARC

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cfg(grid=30, **overrides):
    cfg = dict(
        batch_size=4, seq_len=grid * grid, vocab_size=12, num_puzzle_identifiers=5, puzzle_emb_ndim=32,
        board_height=grid, board_width=grid, hidden_size=32, num_layers=2, num_heads=2, mlp_dim=32, causal=False,
    )
    cfg.update(overrides)
    return cfg


def _batch(grid=30, batch_size=4, seed=0):
    """Dataset-layout pairs: random grids at random translations, EOS row/column after each, PAD labels ignored."""
    rng = np.random.default_rng(seed)
    inputs, labels = [], []
    for _ in range(batch_size):
        h_in, w_in, h_out, w_out = rng.integers(1, grid, size=4)
        r, c = rng.integers(0, grid - max(h_in, h_out) + 1), rng.integers(0, grid - max(w_in, w_out) + 1)
        seqs = []
        for h, w in ((h_in, w_in), (h_out, w_out)):
            frame = np.zeros((grid, grid), dtype=np.int64)
            frame[r:r + h, c:c + w] = rng.integers(2, 12, size=(h, w))
            if r + h < grid:
                frame[r + h, c:c + w] = 1
            if c + w < grid:
                frame[r:r + h, c + w] = 1
            seqs.append(frame.reshape(-1))
        inputs.append(seqs[0])
        labels.append(np.where(seqs[1] == 0, IGNORE_LABEL_ID, seqs[1]))
    return {
        "inputs": torch.tensor(np.stack(inputs)).cuda(),
        "labels": torch.tensor(np.stack(labels)).cuda(),
        "puzzle_identifiers": torch.arange(1, batch_size + 1, dtype=torch.int32).cuda(),
    }


def _model(grid=30, **overrides):
    torch.manual_seed(0)
    with torch.device("cuda"):
        return VARC(_cfg(grid, **overrides))


@cuda
@pytest.mark.parametrize("grid", [30, 10])
def test_canvas_round_trip_recovers_label_tokens(grid):
    inner = _model(grid).inner
    batch = _batch(grid, batch_size=16)
    labels = batch["labels"]
    for _ in range(5):  # several random scales/offsets
        _, raw_index = inner.place_on_canvas(batch["inputs"], labels)
        # Paint each canvas pixel with the one-hot label of the raw cell it came from, then map back
        padded = F.pad(labels.clamp_min(0), (0, 1))
        canvas_onehot = F.one_hot(torch.gather(padded, 1, raw_index), 12).float()
        recovered = inner.canvas_to_raw(canvas_onehot, raw_index).argmax(-1)
        keep = labels != IGNORE_LABEL_ID
        assert torch.equal(recovered[keep], labels[keep])


@cuda
def test_canvas_uses_scales_above_one_and_keeps_input_colors():
    inner = _model().inner
    batch = _batch(batch_size=16)
    canvas, raw_index = inner.place_on_canvas(batch["inputs"], batch["labels"])
    block = torch.stack([torch.bincount(row, minlength=901)[:900].max() for row in raw_index])  # s * s per row
    assert (block > 1).any()  # some rows were scaled up
    # Every input color cell lands fully on the canvas as an s x s block
    assert torch.equal(canvas.view(16, -1).ge(2).sum(1), batch["inputs"].ge(2).sum(1) * block)


@cuda
def test_eval_ignores_labels():
    model = _model().eval()
    batch = _batch()
    corrupted = dict(batch, labels=torch.full_like(batch["labels"], 5))
    with torch.inference_mode():
        torch.manual_seed(1)
        a = model(carry=None, batch=batch)[1]["logits"]
        torch.manual_seed(1)
        b = model(carry=None, batch=corrupted)[1]["logits"]
    assert torch.equal(a, b)


@cuda
def test_train_step_reaches_vit_and_task_tokens():
    model = _model()
    with torch.device("cuda"):
        head = ACTLossHead(model, loss_type="softmax_cross_entropy")
    head.train()
    carry, loss, metrics, _, all_finish = head(return_keys=[], carry=None, batch=_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all_finish and carry.halted.all() and (carry.steps == 1).all()
    loss.backward()
    inner = model.inner
    assert inner.head.weight.grad.abs().sum() > 0
    assert inner.layers[0].attn.qkv.weight.grad.abs().sum() > 0
    assert inner.puzzle_emb.local_weights.grad.abs().sum() > 0


@cuda
def test_inference_head_outputs_preds_and_q_halt():
    model = _model()
    with torch.device("cuda"):
        head = InferenceLossHead(model, loss_type="softmax_cross_entropy").eval()
    batch = _batch()
    with torch.inference_mode():
        carry, _, _, outputs, all_finish = head(return_keys=["preds", "q_halt_logits"], carry=head.initial_carry(batch), batch=batch)
    assert all_finish
    assert outputs["preds"].shape == batch["inputs"].shape
    assert outputs["q_halt_logits"].shape == (4,)


def test_rejects_border_token_vocab_and_bad_task_token_width():
    with pytest.raises(ValueError, match="vocab_size"):
        VARC(_cfg(vocab_size=16))
    with pytest.raises(ValueError, match="puzzle_emb_ndim"):
        VARC(_cfg(puzzle_emb_ndim=16))


TASK = {
    "train": [
        {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
        {"input": [[5, 0, 6]], "output": [[6, 0, 5]]},
    ],
    "test": [{"input": [[7, 8], [0, 9]], "output": [[9, 0], [8, 7]]}],
}


def test_aux_tasks_are_distinct_and_invertible():
    np.random.seed(0)
    aux = build_aux_tasks("task", TASK, num_aux_tasks=6)
    assert len(aux) == 6 and aux[0][0] == "task"
    assert len({name for name, _ in aux}) == 6
    grid = arc_grid_to_np(TASK["test"][0]["output"])
    for name, map_grid in aux:
        inputs, labels = encode_pair(map_grid(grid), map_grid(grid), grid_size=30)
        assert (labels == IGNORE_LABEL_ID).sum() > 0 and (inputs == 0).sum() > 0
        # A perfect prediction in the aux task's frame decodes back to the original grid
        pred = np.where(labels == IGNORE_LABEL_ID, 0, labels)
        assert np.array_equal(decode_prediction(pred, name, 30), grid)


def test_scoring_uses_vote_ranking():
    label = arc_grid_to_np(TASK["test"][0]["output"])
    wrong = arc_grid_to_np([[1]])
    grids = {grid_hash(label): label, grid_hash(wrong): wrong}
    cfg = VARCTTTConfig(
        checkpoint="x", use_ema=True, num_aux_tasks=2, epochs=1, warmup_epochs=0, batch_size=1, lr=1e-3, betas=(0.9, 0.999),
        weight_decay=0.0, ttt_runs_per_task=1, num_views=1, eval_batch_size=1, pass_Ks=[1, 2], submission_K=2, seed=0,
    )
    votes = [Counter({grid_hash(wrong): 3, grid_hash(label): 2})]
    assert rank_votes(votes[0])[0] == grid_hash(wrong)
    result = score_task(TASK, votes, grids, cfg)
    assert result["pass"] == {"1": 0.0, "2": 1.0}
    assert result["test"][0]["attempts"] == [wrong.tolist(), label.tolist()]
