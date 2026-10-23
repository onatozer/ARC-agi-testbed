from typing import Any, Dict, List, Optional, Union

import pydantic
from pydantic import Field


class CheckpointEvalConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    checkpoint: str
    param_iter: Dict[str, Any] = Field(default_factory=dict)
    suffix: Optional[str] = None
    description: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_run_name: Optional[str] = None
    init_override: Optional[Dict[str, Any]] = None


class EvalConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow", populate_by_name=True)

    checkpoint: Optional[str] = None
    checkpoint_list: List[CheckpointEvalConfig] = Field(default_factory=list)
    global_batch_size: Optional[int] = None
    seed: Optional[int] = None
    dataset_data_path: Optional[str] = None
    load_strict: Optional[bool] = None
    top_k: Optional[Union[int, List[int]]] = None
    suffix: Optional[str] = None
    save_outputs: List[str] = Field(default_factory=list)
    save_carry: bool = False
    param_iter: Dict[str, Any] = Field(default_factory=dict)
    shared_param_iter: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    max_eval_steps: Optional[int] = None
    force_rerun: bool = False
    wandb_run_name: Optional[str] = Field(default=None, alias="name")
    wandb_tags: Optional[List[str]] = None
    loss_head_name: Optional[str] = None  # None: the checkpoint's arch.loss.inference_name
    different_init: Optional[int] = None
    different_init_reset_std: Optional[float] = 1.0  # None: the copies keep the model's own initial carry (they differ only by their draws)
    convergence_top_k: Optional[int] = None
    convergence_window: int = 3
    convergence_vis_plots: List[str] = Field(default_factory=list)
    wandb_mode: Optional[str] = None  # "disabled" skips logging to the checkpoint's W&B run


class TTTConfig(pydantic.BaseModel):
    """evaluate_ttt.py: per-task test-time training on an ARC checkpoint, then the checkpoint's own evaluators."""

    checkpoint: str
    suffix: str
    output_dir: Optional[str] = None  # None: <run_dir>/eval_preds/step_<N>_<suffix>
    task_ids: Optional[List[str]] = None
    task_ids_file: Optional[str] = None  # a JSON list of task names (e.g. a build's heldout_tasks.json); overrides task_ids
    max_tasks: Optional[int] = None

    num_aux_tasks: int  # the task itself + (num_aux_tasks - 1) dihedral/color augmentations, each with its own puzzle embedding
    epochs: int  # passes over every aux task's demos
    warmup_epochs: int
    ttt_runs_per_task: int  # independent runs whose votes are pooled
    num_views: int  # eval passes per run, pooled; > 1 only helps models with random views (e.g. VARC's canvas)
    train_overrides: Dict[str, Any] = Field(default_factory=dict)  # PretrainConfig fields for the TTT training, e.g. lr
    meta_sgd_steps: int = 0  # > 0: that many Meta-SGD inner steps (the checkpoint's learned alpha, or train_overrides.meta_sgd.*) on the task's demos instead of the TTT epochs
    meta_sgd_schedule: str = "hold"  # per-step alphas over more steps than were trained: "hold" the last one, or "stretch" them over the loop (the k-th trained alpha for the k-th fraction of the steps)
    seed: int = 0
    wandb_mode: Optional[str] = None  # "disabled" skips logging to the checkpoint's W&B run
