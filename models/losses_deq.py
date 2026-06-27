"""ACT loss head extended with the Jacobian regulariser term from TRMDEQ.

This is a thin variant of ``losses.ACTLossHead`` -- same lm_loss + q_halt_loss
formula -- that *also* adds ``lambda_jac * outputs['jacobian_reg']`` to the
returned scalar loss. Vanilla TRM training is unaffected (uses the original
ACTLossHead in losses.py).
"""
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .losses import IGNORE_LABEL_ID, softmax_cross_entropy, stablemax_cross_entropy


class ACTLossHeadDEQ(nn.Module):
    def __init__(self, model: nn.Module, *, loss_type: str,
                 jacobian_reg_lambda: float = 0.0):
        super().__init__()
        self.model = model
        self.loss_fn = {
            "stablemax_cross_entropy": stablemax_cross_entropy,
            "softmax_cross_entropy": softmax_cross_entropy,
        }[loss_type]
        self.jacobian_reg_lambda = float(jacobian_reg_lambda)

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        lm_loss = (
            self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID,
                         valid_mask=mask) / loss_divisor
        ).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        metrics.update({"lm_loss": lm_loss.detach(), "q_halt_loss": q_halt_loss.detach()})

        q_continue_loss = torch.zeros((), device=lm_loss.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Jacobian regulariser
        jac_reg = outputs.get("jacobian_reg", torch.zeros((), device=lm_loss.device))
        # Scale by global batch / loss_divisor to keep magnitudes comparable to lm_loss
        jac_loss = self.jacobian_reg_lambda * jac_reg
        metrics["jacobian_reg"] = jac_reg.detach()

        total = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + jac_loss
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, total, metrics, detached_outputs, new_carry.halted.all()
