import torch

from pretrain import _lr_scaled_groups


def test_lr_scaled_groups_split_by_name_substring_first_match_wins():
    params = [(name, torch.nn.Parameter(torch.zeros(1))) for name in ("inner.L_level.w", "inner.candidate_generator.w", "inner.candidate_scorer.w", "inner.lm_head.w")]
    groups = _lr_scaled_groups(params, {"candidate_scorer": 0.1, "candidate_": 0.3})
    by_scale = {g["lr_scale"]: {id(p) for p in g["params"]} for g in groups}
    assert set(by_scale) == {1.0, 0.1, 0.3}
    assert by_scale[0.1] == {id(params[2][1])}
    assert by_scale[0.3] == {id(params[1][1])}
    assert by_scale[1.0] == {id(params[0][1]), id(params[3][1])}


def test_lr_scaled_groups_without_scales_is_one_group():
    params = [("a", torch.nn.Parameter(torch.zeros(1))), ("b", torch.nn.Parameter(torch.zeros(1)))]
    [group] = _lr_scaled_groups(params, {})
    assert group["lr_scale"] == 1.0 and len(group["params"]) == 2
