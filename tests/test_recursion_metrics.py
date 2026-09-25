import torch

from tests.fixtures import tiny_model_and_batch


def _run(model, batch, n_steps):
    carry = model.initial_carry(batch)
    for _ in range(n_steps):
        carry, _, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    return carry, metrics


def test_nfe_residual_and_halt_split_in_training():
    model, batch = tiny_model_and_batch(batch_size=8)
    model.train()
    cfg = model.model.config
    carry, metrics = _run(model, batch, cfg.halt_max_steps)  # every row has halted by now
    count = metrics["count"].item()
    assert count > 0
    # NFE per halted puzzle = supervision steps x L_level calls per step
    per_step = cfg.H_cycles * (cfg.L_cycles + 1)
    assert torch.equal(carry.nfe, carry.steps * per_step)
    assert metrics["nfe"].item() == (carry.steps[carry.halted] * per_step).sum().item()
    for key in ("zH_rel_residual", "zL_rel_residual"):
        assert 0 < metrics[key].item() / count < 10
    split = sum(metrics[f"halt/{s}_{o}"].item() for s in ("quick", "slow") for o in ("correct", "wrong"))
    assert split == count


def test_nfe_resets_for_new_puzzles_and_no_split_in_eval():
    model, batch = tiny_model_and_batch(batch_size=8)
    model.eval()
    cfg = model.model.config
    carry, metrics = _run(model, batch, cfg.halt_max_steps + 1)  # one step into the next puzzle
    assert torch.equal(carry.nfe, torch.full_like(carry.nfe, cfg.H_cycles * (cfg.L_cycles + 1)))
    assert not any(k.startswith("halt/") for k in metrics)
    assert {"nfe", "zH_rel_residual", "zL_rel_residual"} <= set(metrics)
