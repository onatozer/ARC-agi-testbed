import torch

from tests.fixtures import tiny_model_and_batch


def _run(model, batch, n_steps):
    carry = model.initial_carry(batch)
    for _ in range(n_steps):
        carry, _, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    return carry, metrics


def test_nfe_and_residual_per_halted_puzzle():
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
    assert 0 < metrics["zH_rel_residual"].item() / count < 10


def test_nfe_resets_for_new_puzzles():
    model, batch = tiny_model_and_batch(batch_size=8)
    model.eval()
    cfg = model.model.config
    carry, metrics = _run(model, batch, cfg.halt_max_steps + 1)  # one step into the next puzzle
    assert torch.equal(carry.nfe, torch.full_like(carry.nfe, cfg.H_cycles * (cfg.L_cycles + 1)))
    assert "nfe" in metrics and "zH_rel_residual" in metrics
