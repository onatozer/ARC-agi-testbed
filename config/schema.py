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


class MetaSGDConfig(BaseModel):
    """Second-order Meta-SGD (arXiv 1707.09835) in place of train_batch: per puzzle in the batch, one gradient step on its
    demos with a learned per-parameter step size alpha, then theta and alpha are updated from the loss of the adapted weights
    on the puzzle's validation rows (its former test pair; needs a dataset built with heldout_tasks > 0). Every loss runs each
    row's ACT trajectory (up to arch.halt_max_steps supervision steps) from a fresh carry, as the ordinary recipe does."""
    enabled: bool = False
    inner_lr: float = 0.1  # initial alpha
    alpha_lr: float = 1e-3  # Adam lr for alpha (log-space), on the main lr schedule
    alpha_range: float = 5.0  # alpha is clamped to inner_lr * exp(+-alpha_range)
    adapt_substrings: List[str] = []  # parameter-name substrings the inner step adapts; [] = every trainable parameter (+ the puzzle-embedding row)
    inner_lr_rows: Optional[float] = None  # initial alpha for the puzzle-embedding rows (None: inner_lr); rows want ~100x the weights' step
    copies_per_group: int = 1  # augmented copies of a task per batch; > 1 makes the task (all its copies in the batch) the episode, as TTT adapts
    fresh_rows: bool = False  # the episode's embedding rows start from the table mean, as a held-out task's do at test time; the table is not trained
    # "How to train your MAML" (arXiv 1810.09502) components. Inner steps > 1 trains the loop that is run at test time;
    # multi-step loss (MSL) weights the query loss after every inner step, annealing from uniform to the last step only over
    # the first msl_anneal_fraction of training (None: the last step's loss only); LSLR = alpha per tensor (and per step)
    # instead of per element; derivative-order annealing = first-order inner gradients for the first first_order_steps.
    inner_steps: int = 1
    msl_anneal_fraction: Optional[float] = None
    alpha_per_tensor: bool = False  # one alpha per adapted tensor (the embedding table keeps one per column)
    alpha_per_step: bool = False  # a separate alpha for each of the inner_steps (test steps beyond them reuse the last)
    first_order_steps: int = 0
    # Multi-segment (ACT) models such as TRM: inner steps as supervision segments (each continues the carry the previous step
    # left, as train_batch does), and full-trajectory losses with gradient only through each row's final segment (the
    # earlier ones run without gradient, as TRM's own H-cycles). One-segment models (VARC) are unaffected by either.
    segment_inner_steps: bool = False
    last_segment_grad: bool = False
    # With segment_inner_steps: a row the ACT head halts sits out (no loss) until its trajectory ends, every
    # arch.halt_max_steps inner steps, then all rows restart together, so per-step alphas index segment positions exactly
    halt_wait: bool = False
    # Query (outer) loss: "ce", the loss head's loss of the query trajectory; or "trajectory" (multi-segment models): with
    # the adapted weights frozen, optimise latent targets s~_1..s~_N for the query rows (traj_iters Adam steps at traj_lr on
    # CE(decode(s~_N)) + traj_lambda * mean_t ||s~_{t+1} - F(s~_t)||^2, from the model's own rollout), then match the adapted
    # model's free-running per-segment states to them
    query_loss: str = "ce"
    traj_lambda: float = 1.0
    traj_iters: int = 20
    traj_lr: float = 1e-2
    traj_starts: int = 1  # > 1 (cached targets, arch.random_initialization): that many random starts per query row, one optimised path each, the outer loss over all of them
    traj_refresh: float = 0.0  # > 0: targets cached for every training query row from the current weights (before adaptation; each puzzle's own embedding row, or the table mean with fresh_rows, as the episode starts), recomputed every this fraction of training; 0: recomputed per episode at the adapted weights
    traj_lambda_growth: float = 1.0  # consistency weight of link t (s~_t -> s~_{t+1}) ~ growth^t, mean traj_lambda: > 1 makes late links stiffer, so the correction moves to earlier segments
    traj_ce_weight: float = 0.0  # > 0: the outer loss also gets this times the query CE (as query_loss=ce), so lm_head and q_head get a meta-gradient
    normalize_grad: bool = False  # inner step on the gradient divided by its RMS per tensor: a step moves a tensor by ~alpha whatever the gradient's scale (as Adam does; not Meta-SGD proper)

    @pydantic.model_validator(mode="after")
    def _check(self):
        if self.inner_lr <= 0 or self.alpha_range < 0 or self.copies_per_group < 1 or (self.inner_lr_rows is not None and self.inner_lr_rows <= 0):
            raise ValueError("meta_sgd: inner_lr > 0 (inner_lr_rows too), alpha_range >= 0 and copies_per_group >= 1 are required.")
        if self.inner_steps < 1 or self.first_order_steps < 0 or (self.msl_anneal_fraction is not None and not 0 < self.msl_anneal_fraction <= 1):
            raise ValueError("meta_sgd: inner_steps >= 1, first_order_steps >= 0 and msl_anneal_fraction in (0, 1] are required.")
        if self.query_loss not in ("ce", "trajectory") or self.traj_iters < 0 or self.traj_lambda < 0 or self.traj_lr <= 0 or not 0 <= self.traj_refresh <= 1 or self.traj_starts < 1 or (self.traj_starts > 1 and self.traj_refresh == 0) or self.traj_lambda_growth <= 0 or self.traj_ce_weight < 0:
            raise ValueError("meta_sgd: query_loss in {ce, trajectory}, traj_iters >= 0, traj_lambda >= 0, traj_lr > 0, traj_refresh in [0, 1], traj_starts >= 1 (> 1 only with cached targets, traj_refresh > 0), traj_lambda_growth > 0 and traj_ce_weight >= 0 are required.")
        return self


class PretrainConfig(BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    arch: ArchConfig
    dataset: DatasetConfig
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
    meta_sgd: MetaSGDConfig = MetaSGDConfig()

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
