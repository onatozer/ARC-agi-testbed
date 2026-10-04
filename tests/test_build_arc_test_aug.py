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
