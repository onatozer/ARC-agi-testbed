"""Hierarchical Supervision Weighting (arXiv 2608.01534 eq. 7): hsw_lambda weights supervision step t's loss by lambda^(t-1) / Z."""

import torch
import torch.nn.functional as F

from tests.fixtures import tiny_model_and_batch


def test_step_weights_form_a_distribution_over_halt_max_steps():
    model, _ = tiny_model_and_batch()  # halt_max_steps=4
    steps = torch.arange(1, 5, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(model._step_weights(steps), torch.ones(4, device="cuda"))  # off: no reweighting
    model.hsw_lambda = 0.5
    w = model._step_weights(steps)
    torch.testing.assert_close(w, torch.tensor([1.0, 0.5, 0.25, 0.125], device="cuda") / 1.875)
    assert w[0] > w[1] > w[2] > w[3]
    model.hsw_lambda = 1.0
    torch.testing.assert_close(model._step_weights(steps), torch.full((4,), 0.25, device="cuda"))


def test_compute_losses_weights_every_row_term():
    model, batch = tiny_model_and_batch()
    model.hsw_lambda = 0.5
    _, _, _, outputs, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=["logits", "q_halt_logits"])
    labels = batch["labels"]
    mask = torch.ones_like(labels, dtype=torch.bool)
    divisor = mask.sum(-1, keepdim=True)
    seq_is_correct = (outputs["logits"].argmax(-1) == labels).all(-1)
    w = model._step_weights(torch.tensor([1, 2, 3, 4, 1, 2], dtype=torch.int32, device="cuda"))
    total, losses = model.compute_losses(
        outputs=outputs, labels=labels, mask=mask, loss_divisor=divisor, seq_is_correct=seq_is_correct, step_weight=w
    )
    lm_rows = (model.loss_fn(outputs["logits"], labels, valid_mask=mask) / divisor).sum(-1)
    q_rows = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.float(), reduction="none")
    torch.testing.assert_close(total, (w * (lm_rows + 0.5 * q_rows)).sum())
    torch.testing.assert_close(losses["lm_loss"], lm_rows.sum())  # logged terms stay unweighted
    torch.testing.assert_close(losses["q_halt_loss"], q_rows.sum())


def test_training_step_uses_the_rows_supervision_step():
    model, batch = tiny_model_and_batch()
    _, loss_off, metrics_off, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])
    model.hsw_lambda = 0.5
    _, loss_on, metrics_on, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])
    # every row is at step 1 after the first forward, whose weight is 1 / Z = 1 / 1.875
    torch.testing.assert_close(loss_on, loss_off / 1.875)
    torch.testing.assert_close(metrics_on["lm_loss"], metrics_off["lm_loss"])
