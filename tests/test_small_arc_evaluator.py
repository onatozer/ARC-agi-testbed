import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dataset.build_arc_dataset import np_grid_to_seq_translational_augment
from evaluators.small_arc import SmallARC


def _puzzle(seed):
    rng = np.random.default_rng(seed)
    inp, out = (rng.integers(0, 10, size=(3, 4)).tolist() for _ in range(2))
    return {"train": [], "test": [{"input": inp, "output": out}]}


def _evaluator(tmp_path, test_puzzles):
    identifiers = ["<blank>"] + list(test_puzzles)
    (tmp_path / "identifiers.json").write_text(json.dumps(identifiers))
    (tmp_path / "test_puzzles.json").write_text(json.dumps(test_puzzles))
    metadata = SimpleNamespace(blank_identifier_id=0, seq_len=100)
    return SmallARC(str(tmp_path), metadata, pass_Ks=(1, 2)), identifiers


@pytest.mark.parametrize("with_q", [True, False])  # models without a halt head (GRAM) give no q_halt_logits
def test_per_source_pass_at_k(tmp_path, with_q):
    # miniarc puzzles are predicted correctly, arc1_eval puzzles get a wrong grid
    test_puzzles = {"miniarc/a": _puzzle(0), "miniarc/b": _puzzle(1), "arc1_eval/c": _puzzle(2)}
    evaluator, identifiers = _evaluator(tmp_path, test_puzzles)

    inputs, preds, ids = [], [], []
    for name, puzzle in test_puzzles.items():
        pair = puzzle["test"][0]
        target = pair["output"] if name.startswith("miniarc") else (np.array(pair["output"]) + 1) % 10
        inp, pred = np_grid_to_seq_translational_augment(np.array(pair["input"], dtype=np.uint8), np.array(target, dtype=np.uint8),
                                                         do_translation=False, max_grid_size=10)
        inputs.append(inp)
        preds.append(pred)
        ids.append(identifiers.index(name))

    evaluator.begin_eval()
    evaluator.update_batch(
        {"inputs": torch.tensor(np.stack(inputs)), "puzzle_identifiers": torch.tensor(ids)},
        {"preds": torch.tensor(np.stack(preds)), **({"q_halt_logits": torch.zeros(len(ids))} if with_q else {})},
    )
    results = evaluator.result(None, rank=0, world_size=1)

    assert results["ARC/pass@1"] == 2 / 3
    assert results["SmallARC/miniarc/pass@1"] == 1.0
    assert results["SmallARC/arc1_eval/pass@1"] == 0.0
    assert results["SmallARC/miniarc/pass@2"] == 1.0
