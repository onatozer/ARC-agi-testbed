from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.losses import IGNORE_LABEL_ID
from models.losses.base import BaseLossHead, softmax_cross_entropy, stablemax_cross_entropy
from utils.training import tree_to_device


def _carry_get(carry: Any, key: str, default: Any = None) -> Any:
    if isinstance(carry, dict):
        return carry.get(key, default)
    return getattr(carry, key, default)


class ACTLossHead(BaseLossHead):

    def __init__(self, model: nn.Module, loss_type: str, **kwargs: Any) -> None:
        del kwargs
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.initial_carry(*args, **kwargs)  # type: ignore[attr-defined]

    def compute_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
        loss_divisor: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:

        del temperature
        lm_loss = (
            self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=valid_mask) / loss_divisor
        ).sum()
        return {"lm_loss": lm_loss}

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs: Any,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:

        new_carry, outputs = self.model(**model_kwargs)
        logits = outputs.get("logits", None)

        if isinstance(logits, torch.Tensor):
            outputs["logits"] = logits.contiguous()

        labels = new_carry.current_data["labels"]
        new_carry = tree_to_device(new_carry, outputs["logits"].device)
        preds = torch.argmax(outputs["logits"], dim=-1)

        prev_halted = outputs.pop("prev_halted", None)
        prev_preds = _carry_get(new_carry, "prev_preds", None)
        if prev_preds is not None:
            mask_source = prev_halted if prev_halted is not None else new_carry.halted
            preds = torch.where(mask_source.unsqueeze(-1), prev_preds.to(preds.dtype), preds)
            new_carry.prev_preds = preds.detach()

        outputs["preds"] = preds
        metrics: Dict[str, torch.Tensor]
        
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid_metrics = new_carry.halted & (loss_counts > 0)
            count = valid_metrics.sum()

            metrics = self._build_step_metrics(
                outputs=outputs,
                carry=new_carry,
                loss_divisor=loss_divisor,
                loss_counts=loss_counts,
                is_correct=is_correct,
                seq_is_correct=seq_is_correct,
                valid_metrics=valid_metrics,
                count=count,
            )

        losses = self.compute_lm_loss(outputs["logits"], labels, valid_mask=mask, loss_divisor=loss_divisor)
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        metrics["q_halt_loss"] = q_halt_loss.detach()
        metrics.update({key: value.detach() for key, value in losses.items()})

        detached_outputs = {key: outputs[key].detach() for key in return_keys if key in outputs}
        total_loss = losses["lm_loss"] + 0.5 * q_halt_loss
        metrics["total_loss"] = total_loss.detach()
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

    def _build_step_metrics(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        carry: Any,
        loss_divisor: torch.Tensor,
        loss_counts: torch.Tensor,
        is_correct: torch.Tensor,
        seq_is_correct: torch.Tensor,
        valid_metrics: torch.Tensor,
        count: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        metrics: Dict[str, torch.Tensor] = {
            "count": count,
            "accuracy": torch.zeros_like(count, dtype=torch.float32),
            "exact_accuracy": torch.zeros_like(count),
            "halted_q_logits": torch.zeros_like(count),
            "q_halt_accuracy": torch.zeros_like(count),
            "q_halt_precision": torch.zeros_like(count, dtype=torch.float32),
            "q_halt_recall": torch.zeros_like(count, dtype=torch.float32),
            "steps": torch.zeros_like(count, dtype=torch.float32),
            "max_steps": torch.zeros_like(count, dtype=torch.float32),
            "min_steps": torch.full_like(count, float("inf"), dtype=torch.float32),
            "raw_steps": torch.zeros_like(carry.steps, dtype=torch.float32),
        }
        if (count > 0).any():
            metrics.update(
                {
                    "accuracy": torch.where(
                        valid_metrics,
                        (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                        0,
                    ).sum(),
                    "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                    "halted_q_logits": torch.where(
                        valid_metrics,
                        outputs["q_halt_logits"],
                        torch.zeros_like(loss_counts, dtype=torch.float32),
                    ).sum(),
                    "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] > 0) == seq_is_correct)).sum(),
                    "q_halt_precision": torch.where(
                        valid_metrics,
                        ((outputs["q_halt_logits"] > 0) & seq_is_correct).to(torch.float32).sum()
                        / ((outputs["q_halt_logits"] >= 0).to(torch.float32).sum().clamp_min(1)),
                        0,
                    ).sum(),
                    "q_halt_recall": torch.where(
                        valid_metrics,
                        ((outputs["q_halt_logits"] > 0) & seq_is_correct).to(torch.float32).sum()
                        / (seq_is_correct.to(torch.float32).sum().clamp_min(1)),
                        0,
                    ).sum(),
                    "steps": torch.where(valid_metrics, carry.steps.to(torch.float32), 0).sum(),
                    "max_steps": torch.where(
                        valid_metrics,
                        carry.steps.to(torch.float32),
                        torch.zeros_like(carry.steps, dtype=torch.float32),
                    ).max(),
                    "min_steps": torch.where(
                        valid_metrics,
                        carry.steps.to(torch.float32),
                        torch.full_like(carry.steps, self.model.config.halt_max_steps, dtype=torch.float32),
                    ).min(),
                    "raw_steps": torch.where(
                        valid_metrics,
                        carry.steps.to(torch.float32),
                        torch.zeros_like(carry.steps, dtype=torch.float32),
                    ),
                }
            )
        return metrics

class InferenceLossHead(ACTLossHead):
    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs: Any,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:

        new_carry, outputs = self.model(**model_kwargs)
        logits = outputs.get("logits", None)
        if isinstance(logits, torch.Tensor):
            outputs["logits"] = logits.contiguous()

        total_loss, metrics, detached_outputs, all_finish = self.compute_metrics_from_outputs(
            outputs=outputs,
            carry=new_carry,
            return_keys=return_keys,
        )
        return new_carry, total_loss, metrics, detached_outputs, all_finish

    def compute_metrics_from_outputs(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        carry: Any,
        return_keys: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:

        labels = carry.current_data["labels"]
        device = labels.device
        preds = torch.argmax(outputs["logits"], dim=-1)

        prev_halted = outputs.pop("prev_halted", None)
        prev_preds = _carry_get(carry, "prev_preds", None)
        if prev_preds is not None:
            mask_source = prev_halted if prev_halted is not None else carry.halted
            preds = torch.where(mask_source.to(device).unsqueeze(-1), prev_preds.to(device=device, dtype=preds.dtype), preds)
            carry.prev_preds = preds.detach().to(carry.steps.device)

        outputs["preds"] = preds
        mask = (labels != IGNORE_LABEL_ID).to(device)
        with torch.no_grad():
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            steps = carry.steps.to(device)
            max_steps = int(getattr(self.model.config, "halt_max_steps", 0) or 0)
            forced_halt = (steps >= max_steps) if max_steps > 0 else carry.halted.to(device)
            valid_metrics = forced_halt & (loss_counts > 0)
            count = valid_metrics.sum()

            metrics = {
                "count": count,
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] > 0) == seq_is_correct)).sum(),
                "q_halt_precision": torch.where(
                    valid_metrics,
                    ((outputs["q_halt_logits"] > 0) & seq_is_correct).to(torch.float32).sum()
                    / ((outputs["q_halt_logits"] >= 0).to(torch.float32).sum().clamp_min(1)),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "q_halt_recall": torch.where(
                    valid_metrics,
                    ((outputs["q_halt_logits"] > 0) & seq_is_correct).to(torch.float32).sum()
                    / (seq_is_correct.to(torch.float32).sum().clamp_min(1)),
                    torch.zeros_like(loss_counts, dtype=torch.float32),
                ).sum(),
                "steps": torch.where(valid_metrics, steps.to(torch.float32), torch.zeros_like(loss_counts, dtype=torch.float32)).sum(),
            }

        lm_loss = (
            self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor
        ).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        metrics["lm_loss"] = lm_loss.detach()
        metrics["q_halt_loss"] = q_halt_loss.detach()

        detached_outputs = {key: outputs[key].detach() for key in return_keys if key in outputs}
        total_loss = lm_loss + 0.5 * q_halt_loss
        metrics["total_loss"] = total_loss.detach()
        finished_mask = carry.halted.to(device)
        if max_steps > 0:
            finished_mask = finished_mask | (steps >= max_steps)
        return total_loss, metrics, detached_outputs, finished_mask.all()

class ACTLossHeadRoot(nn.Module):
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
