import torch

from evaluators.eval_fn import _di_state, _finish_di, _update_di
from models.losses import IGNORE_LABEL_ID


def test_lprm_best_of_n_picks_the_highest_scored_copy():
    labels = torch.tensor([[1, 2, IGNORE_LABEL_ID], [3, 3, 3]])
    # puzzle 0: copies (wrong, right, wrong); puzzle 1: copies (right, wrong, wrong)
    preds = torch.tensor([[0, 0, 0], [1, 2, 9], [1, 0, 0], [3, 3, 3], [0, 3, 3], [3, 0, 3]])
    scores = torch.tensor([0.1, 0.9, 0.2, 0.2, 0.8, 0.1])  # best-of-N right for puzzle 0, wrong for puzzle 1
    state = _di_state(torch.device("cpu"))
    _update_di(state, preds, labels, 3, scores)
    metrics = _finish_di(state, 3, rank=0, world_size=1, group=None)
    assert metrics["different_init/lprm_best_acc"] == 0.5
    assert metrics["different_init/any_correct"] == 1.0
    assert metrics["different_init/majority_vote_acc"] == 0.5  # no copy repeats: ties go to the first seen (right for puzzle 1 only)


def test_validity_scored_vote_and_best_of_n():
    labels = torch.tensor([[1, 2], [3, 4]])
    # puzzle 0: copies (label, other valid, other valid): vote ties -> first seen = label; puzzle 1: (invalid, invalid, valid)
    preds = torch.tensor([[1, 2], [2, 1], [2, 1], [0, 0], [0, 0], [4, 3]])
    valid = torch.tensor([True, True, True, False, False, True])
    scores = torch.tensor([0.1, 0.9, 0.2, 0.9, 0.1, 0.2])  # best-of-N: puzzle 0 picks a valid non-label, puzzle 1 an invalid
    state = _di_state(torch.device("cpu"))
    _update_di(state, preds, labels, 3, scores, valid)
    m = _finish_di(state, 3, rank=0, world_size=1, group=None)
    assert m["different_init/majority_vote_acc"] == 0.0  # puzzle 0's winner is the repeated non-label copy, puzzle 1's an invalid one
    assert m["different_init/majority_valid"] == 0.5
    assert m["different_init/lprm_best_acc"] == 0.0 and m["different_init/lprm_best_valid"] == 0.5
    assert m["different_init/any_valid"] == 1.0 and abs(m["different_init/copy_valid"] - (1.0 + 1 / 3) / 2) < 1e-6


def test_lprm_best_of_n_without_scores_stays_zero():
    labels = torch.tensor([[1, 2]])
    preds = torch.tensor([[1, 2], [1, 2]])
    state = _di_state(torch.device("cpu"))
    _update_di(state, preds, labels, 2)
    metrics = _finish_di(state, 2, rank=0, world_size=1, group=None)
    assert metrics["different_init/lprm_best_acc"] == 0.0 and metrics["different_init/majority_vote_acc"] == 1.0
