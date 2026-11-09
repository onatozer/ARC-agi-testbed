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
    """Meta-SGD (arXiv 1707.09835) in place of train_batch. Every task in the batch (copies_per_group augmented copies of
    it) is an episode:
      1. its puzzle-embedding rows start from the table mean, as a held-out task's do at test time;
      2. inner_steps steps on its demos, each on the gradient divided by its RMS per tensor (so a step moves a tensor by
         about alpha), with a learned alpha per tensor and per step (LSLR, arXiv 1810.09502);
      3. theta and the alphas learn, first order, from the adapted weights' loss on the task's validation rows (its former
         test pair: a build with heldout_tasks > 0).
    With config.recon on, the demo steps also reconstruct recon.copies masked copies of each demo input."""
    enabled: bool = False
    inner_lr: float = 1e-3  # initial alpha of the weights
    inner_lr_rows: float = 3e-2  # initial alpha of the puzzle-embedding rows (one per column)
    alpha_lr: float = 3e-2  # Adam lr of log alpha, on the main lr schedule
    alpha_range: float = 10.0  # alpha stays within its initial value * exp(+-alpha_range)
    inner_steps: int = 8
    copies_per_group: int = 16
    # True: second-order meta-gradients, through the inner steps' gradients (the inner-loop autograd graph is kept for the
    # outer backward; memory grows ~10x on TRM); False: first order, the gradient enters each step as a constant
    second_order: bool = False
    # Meta-gradient accumulation: this many loader batches' episodes go into one optimizer step (the meta-gradient is
    # the mean over all their query rows), so a memory-limited batch still gives a large effective meta-batch
    accum_steps: int = 1
    second_order_steps: Optional[int] = None  # with second_order: only the last this many backbone inner steps second order, the earlier ones first order (None: all)
    noise_scale: bool = False  # log meta/noise_trace and meta/noise_grad_sq (_noise_scale_terms) from each update's per-episode gradients
    # iMAML (Rajeswaran et al. 2019, arXiv 1909.04630): inner_steps steps on the support loss + lam / 2 ||phi - theta||^2
    # without a graph, then theta's implicit meta-gradient by conjugate gradient (_imaml_task_backward); the step sizes are
    # not learned; memory does not grow with inner_steps, so the inner loop can be as long as test time's
    imaml: bool = False
    imaml_lambda: float = 0.01  # the proximal strength (watch meta/imaml_stationarity and meta/imaml_cg_residual)
    imaml_cg_steps: int = 5
    # The curvature in CG's matrix: "hessian" (the support loss's, indefinite on deep nets: CG can stall at negative curvature)
    # or "gauss_newton" (J^T H_z J, softmax CE's Hessian in the logits: positive semi-definite; one-pass models such as VARC)
    imaml_curvature: Literal["hessian", "gauss_newton"] = "hessian"
    # > 0: hypergradient descent (Baydin et al. 2018, arXiv 1703.04782, multiplicative rule) on every inner step's
    # per-tensor step size (_hd_alpha); evaluate_ttt's hd_beta overrides it at test time
    hd_beta: float = 0.0
    # > 0: the embedding first. Each inner loop starts with this many steps on the task's puzzle-embedding rows alone
    # (the backbone frozen, the rows' per-step sizes 0..rows_steps-1), then the inner_steps on the backbone with the
    # rows frozen; the rows' steps are first order even with second_order. Test time: evaluate_ttt's meta_sgd_rows_steps,
    # then meta_sgd_steps (rows frozen)
    rows_steps: int = 0
    # Multi-segment (ACT) models such as TRM and GRAM: each inner step is one supervision segment continuing the previous
    # step's carry, and the query loss is the final prediction's CE after halt_max_steps segments. Off (VARC): each inner
    # step and the query loss run every row's whole trajectory through the loss head.
    segment_inner_steps: bool = False
    # Multi-step loss (arXiv 1810.09502): the query loss after every inner step, annealed from uniform to the last step only
    # over this fraction of training. None: the last step's only
    msl_anneal_fraction: Optional[float] = None
    # Stage 2 (from a stage-1 checkpoint): > 0 learns only the sizes of that many reconstruction steps after the inner steps,
    # everything else frozen. Each step adapts the weights and rows on the final-prediction reconstruction CE of freshly
    # masked copies of the query inputs, at every tensor's initial step size times one learned scale per step (starting at
    # recon_scale0); the query CE after them is the outer loss. Needs config.recon
    recon_steps: int = 0
    recon_scale0: float = 0.01

    @pydantic.model_validator(mode="after")
    def _check(self):
        if self.recon_steps < 0 or self.recon_scale0 <= 0:
            raise ValueError("meta_sgd: recon_steps >= 0 and recon_scale0 > 0 are required.")
        if self.accum_steps < 1:
            raise ValueError("meta_sgd.accum_steps must be >= 1.")
        if not 0 <= self.rows_steps <= self.inner_steps:
            raise ValueError("meta_sgd.rows_steps must be in [0, inner_steps] (the rows' phase reuses their per-step sizes).")
        if min(self.inner_lr, self.inner_lr_rows) <= 0 or self.alpha_range < 0 or self.inner_steps < 1 or self.copies_per_group < 1:
            raise ValueError("meta_sgd: inner_lr, inner_lr_rows > 0, alpha_range >= 0, inner_steps >= 1 and copies_per_group >= 1 are required.")
        if self.msl_anneal_fraction is not None and not 0 < self.msl_anneal_fraction <= 1:
            raise ValueError("meta_sgd.msl_anneal_fraction must be in (0, 1].")
        return self


class ReconConfig(BaseModel):
    """Masked-input reconstruction as an extra task (after arXiv 1909.13231's test-time training). Each training row gets
    `copies` extra rows:
      - input: the row's input with `fraction` of its color cells, at random, set to a mask token (a new id, the
        dataset's vocab_size: the model's vocabulary grows by one);
      - label: the full input;
      - weight: 1 / copies (batch["row_weight"], read by a *RowWeight loss head);
      - puzzle embedding: one extra learned row shared by every copy (id num_puzzle_identifiers), so a copy is
        reconstructed from its visible cells rather than recalled from its task's row.
    0 copies: off."""
    copies: int = 0
    fraction: float = 0.24  # 6 of a 5x5 MiniARC grid's 25 cells

    @pydantic.model_validator(mode="after")
    def _check(self):
        if self.copies < 0 or not 0 < self.fraction <= 1:
            raise ValueError("recon: copies >= 0 and fraction in (0, 1] are required.")
        return self


class PretrainConfig(BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    arch: ArchConfig
    dataset: DatasetConfig
    recon: ReconConfig = ReconConfig()
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
