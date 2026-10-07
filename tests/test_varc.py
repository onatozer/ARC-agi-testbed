import numpy as np
import pytest
import torch
import torch.nn.functional as F

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

