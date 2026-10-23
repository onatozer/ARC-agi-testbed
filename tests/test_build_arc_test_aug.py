import numpy as np
import pytest

from dataset.build_arc_dataset import convert_single_arc_puzzle

PUZZLE = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}, {"input": [[5, 0], [0, 6]], "output": [[6, 0], [0, 5]]}],
    "test": [{"input": [[7, 8], [9, 1]], "output": [[1, 9], [8, 7]]}],
}
EVAL_MAPPING = {"train": ("train", "all"), "test": ("test", "all")}  # demo pairs -> train, test pairs -> test


@pytest.mark.parametrize("test_aug", [True, False])
def test_test_aug_controls_only_the_test_split(test_aug):
    np.random.seed(0)
    results = {}
    convert_single_arc_puzzle(results, "p", PUZZLE, 8, EVAL_MAPPING, test_aug=test_aug)
    (train_copies,), (test_copies,) = results["train"]["all"], results["test"]["all"]

    assert len(train_copies) == 9  # the original + 8 augmented copies, either way
    assert len(test_copies) == (9 if test_aug else 1)
    # the test split always contains the original, un-augmented test pair, under the same id as its training demos
    original = test_copies[0]
    assert original.id == "p" == train_copies[0].id
    assert np.array_equal(original.examples[0][0], np.array(PUZZLE["test"][0]["input"]))
    # every test copy's puzzle id also appears in training, so its puzzle embedding is learned from its demos
    assert {c.id for c in test_copies} <= {c.id for c in train_copies}


def test_test_copies_caps_only_the_test_split():
    np.random.seed(0)
    results = {}
    convert_single_arc_puzzle(results, "p", PUZZLE, 8, EVAL_MAPPING, test_aug=True, test_copies=3)
    (train_copies,), (test_copies,) = results["train"]["all"], results["test"]["all"]
    assert len(train_copies) == 9 and len(test_copies) == 4  # the original plus 3 augmented copies
    assert test_copies[0].id == "p" and {c.id for c in test_copies} <= {c.id for c in train_copies}


def test_example_types_missing_from_the_mapping_are_dropped():
    # A held-out task maps only its test pairs: its demos go nowhere, and there is no train split at all
    np.random.seed(0)
    results = {}
    convert_single_arc_puzzle(results, "p", PUZZLE, 2, {"test": ("test", "all")}, test_aug=True)
    assert "train" not in results
    (test_copies,) = results["test"]["all"]
    assert len(test_copies) == 3 and all(len(c.examples) == 1 for c in test_copies)


def test_validation_flags_only_test_pairs_routed_to_the_train_split():
    np.random.seed(0)
    results = {}
    convert_single_arc_puzzle(results, "p", PUZZLE, 1, {"train": ("train", "all"), "test": ("train", "all")}, test_aug=True)
    n_demos, n_tests = len(PUZZLE["train"]), len(PUZZLE["test"])
    assert all(c.validation == [False] * n_demos + [True] * n_tests for c in results["train"]["all"][0])
    results = {}
    convert_single_arc_puzzle(results, "p", PUZZLE, 1, EVAL_MAPPING, test_aug=True)
    assert not any(any(c.validation) for copies in results.values() for group in copies["all"] for c in group)
