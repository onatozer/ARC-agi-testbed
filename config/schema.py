from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import pydantic
from pydantic import BaseModel

from utils.printing import colored_exception


class LossConfig(BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str
    # Loss head used for evaluation (pretrain.py's training-time eval, evaluate.py, evaluate_ttt.py). The default only
    # covers configs saved before this field existed; the arch yamls state it explicitly
    inference_name: str = "losses@InferenceLossHead"


class ArchConfig(BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str
    short_name: Optional[str] = None
    loss: LossConfig


class EvaluatorConfig(BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str


class DatasetConfig(BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str
    data_path: str
    evaluators: List[EvaluatorConfig] = []


class MAMLConfig(BaseModel):
    """First-order MAML (arXiv 1703.03400, section 5.2; cbfinn/maml with stop_grad): each puzzle in a batch is a task,
    adapted by inner_steps SGD steps of size inner_lr on its support rows; the query row's gradient at the adapted
    weights trains the shared weights (pretrain.maml_batch)."""

    enabled: bool = False
    inner_steps: int = 1  # cbfinn's num_updates
    inner_lr: float = 0.1  # cbfinn's update_lr (alpha)

    @pydantic.model_validator(mode="after")
    def _validate(self) -> MAMLConfig:
        if self.enabled and (self.inner_steps < 1 or self.inner_lr <= 0):
            colored_exception(ValueError, "maml.inner_steps must be >= 1 and maml.inner_lr > 0.")
        return self


class PretrainConfig(BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    arch: ArchConfig
    dataset: DatasetConfig
    maml: MAMLConfig = MAMLConfig()  # enabled: meta-train with first-order MAML instead of plain supervision steps
    optimizer_kwargs: Dict[str, Any] = pydantic.Field(default_factory=dict)

    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    gradient_accumulation_steps: int = 1

    weight_decay: float
    optimizer: str = "adam_atan2"  # "adam_atan2" or "muon" (Muon on the looped blocks' 2D weights, AdamAtan2 on the rest)
    muon_lr: Optional[float] = None  # Muon base lr; None uses lr
    param_lr_scales: Dict[str, float] = {}  # parameter-name substring -> lr multiplier (AdamAtan2 params only); first match wins
    beta1: float
    beta2: float

    grad_clip_norm: Optional[float] = None

    target_q_update_every: int

    project_name: Optional[str] = None
    entity: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    resume: str | bool | None = False
    load_checkpoint: Optional[str] = None
    load_weights_only: bool = False
    load_strict: bool = False
    # W&B run for a full resume from load_checkpoint: "continue" the checkpoint's run, start a "new" one, or "fork" it
    # (a new run that copies the old one's history up to the checkpoint step)
    wandb_on_resume: Literal["continue", "new", "fork"] = "continue"

    seed: int = 0
    train_epochs_per_iter: Optional[int] = None
    eval_interval_steps: Optional[int] = None
    max_eval_steps: Optional[int] = None  # cap on eval batches per rank during training; None runs the full test split
    eval_save_outputs: List[str] = []
    eval_save_carry: bool = False
    heavy_metrics_log_interval: Optional[int] = 100
    steps_hist_log_interval_steps: Optional[int] = 100
    steps_hist_plotly: bool = True
    checkpoint_interval_steps: Optional[int] = None
    max_steps: Optional[int] = None

    wandb_mode: Optional[str] = None

    ema: bool = False
    ema_rate: float = 0.999
    # Alternate every N epochs between training only the puzzle embeddings (first) and only the weights; None trains both
    freeze_alternation_epochs: Optional[int] = None

    debug: bool = False
    debug_trace_steps: int = 0
    grad_norm_log_interval_steps: Optional[int] = None
    flops_log_interval_steps: Optional[int] = None  # measure train-step FLOPs every N steps; None disables
    hidden_state_log_interval_steps: Optional[int] = None

    gradient_checkpoint: bool = False
    retain_graph: bool = False

    eval_sync_max_wait_time: float = 300.0
    eval_sync_poll_interval: float = 2.0
    eval_sync_status_log_interval: float = 10.0

    eval_description: Optional[str] = None
    eval_dir: Optional[str] = None

    convergence_window: int = 10
    convergence_top_k: Optional[int] = None
    convergence_vis_plots: List[str] = []

    @pydantic.model_validator(mode="after")
    def _validate_intervals(self) -> PretrainConfig:
        if self.train_epochs_per_iter is not None and self.train_epochs_per_iter <= 0:
            colored_exception(ValueError, "train_epochs_per_iter must be a positive integer when provided.")
        if self.eval_interval_steps is not None and self.eval_interval_steps <= 0:
            colored_exception(ValueError, "eval_interval_steps must be a positive integer when provided.")
        if self.steps_hist_log_interval_steps is not None and self.steps_hist_log_interval_steps <= 0:
            colored_exception(ValueError, "steps_hist_log_interval_steps must be a positive integer when provided.")
        if self.checkpoint_interval_steps is not None and self.checkpoint_interval_steps <= 0:
            colored_exception(ValueError, "checkpoint_interval_steps must be a positive integer when provided.")
        if self.max_steps is not None and self.max_steps <= 0:
            colored_exception(ValueError, "max_steps must be a positive integer when provided.")
        if self.freeze_alternation_epochs is not None and self.freeze_alternation_epochs <= 0:
            colored_exception(ValueError, "freeze_alternation_epochs must be a positive integer when provided.")
        if self.ema_rate <= 0.0 or self.ema_rate > 1.0:
            colored_exception(ValueError, "ema_rate must be in the interval (0, 1].")
        return self


__all__ = [
    "LossConfig",
    "ArchConfig",
    "EvaluatorConfig",
    "DatasetConfig",
    "PretrainConfig",
]
