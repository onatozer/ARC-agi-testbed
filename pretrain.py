from typing import Optional, Any, List, Dict, Sequence, Tuple
from dataclasses import replace
import json
import os
import resource
import math

# Must be set before CUDA initialises. Lets the allocator grow segments instead of fragmenting across differently
# sized buffers (e.g. TRMRoot's fp32 solver history next to bf16 activations). An explicit user setting wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import multiprocessing as mp
from multiprocessing.managers import SyncManager

from utils.env import get_env
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

import tqdm
import random
import numpy as np
from evaluators.eval_fn import evaluate
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from adam_atan2_pytorch import AdamAtan2
import uuid


torch.set_float32_matmul_precision('high')
try:
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
except Exception:
    pass  # Older PyTorch versions do not have this setting

from dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.losses import IGNORE_LABEL_ID
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed, _sparse_emb_signsgd_dist

from utils.training import (
    build_wandb_run_dir,
    set_deterministic_mode,
    TrainState,
    tree_to_device,
)
from utils.printing import rank_zero_print_warning, rank_zero_print_info, _log_line
from utils.metric_reduce import (
    SPECIAL_METRIC_KEYS,
    partition_metrics,
    reduce_special_metrics,
    reduce_tensor,
)

from utils.checkpoint import (
    load_training_state,
    save_checkpoint,
    save_code_and_config,
)

from datetime import datetime, timedelta
from models.ema import EMAHelper
from utils.metrics import (
    _should_log_grad_norms,
    _should_log_flops,
    TotalFlopCounter,
    _should_log_steps_hist, _collect_raw_steps_for_logging, _build_steps_hist_plotly,
)
from utils.wandb import set_run as wandb_set_run
from utils.wandb import _config_to_wandb_dict
from utils.optimizer import compute_lr
from utils.training import apply_to_tensors
from utils.secrets import apply_wandb_secrets_to_training_config
from config.schema import PretrainConfig
from contextlib import nullcontext, suppress

WANDB_RUN_ID: Optional[str] = None
WANDB_RUN_DIR: Optional[str] = None

_SHARED_STATE_MANAGERS: List[SyncManager] = []


def _resolve_run_root(hydra_config: Optional[DictConfig] = None) -> str:
    """Resolve a stable root directory for outputs.

    Hydra typically changes the process CWD to a per-job directory. If we base
    output paths on os.getcwd(), results end up scattered under the Hydra run
    directory and users won't see `./outputs/<WANDB_RUN_ID>/...` where they
    expect.

    Priority:
      1) OUTPUT_ROOT / RUN_ROOT (explicit override)
      2) hydra.runtime.cwd (original working directory)
      3) directory containing this script (repo root in this project)
      4) current working directory
    """

    env_root = get_env("OUTPUT_ROOT") or get_env("RUN_ROOT")
    if isinstance(env_root, str) and env_root.strip():
        return os.path.abspath(os.path.expanduser(os.path.expandvars(env_root.strip())))

    if hydra_config is not None:
        with suppress(Exception):
            original_cwd = OmegaConf.select(hydra_config, "hydra.runtime.cwd")
            if isinstance(original_cwd, str) and original_cwd.strip():
                return os.path.abspath(os.path.expanduser(os.path.expandvars(original_cwd.strip())))

    with suppress(Exception):
        return os.path.dirname(os.path.abspath(__file__))

    return os.getcwd()


def apply_gradient_checkpointing(model: nn.Module, config: PretrainConfig, rank: int):
    """
    Apply gradient checkpointing to model layers without modifying the original model class.
    This wraps the forward methods of reasoning blocks with torch.utils.checkpoint.
    """
    if not config.gradient_checkpoint:
        return model
    
    if rank == 0:
        rank_zero_print_info("Applying gradient checkpointing to model layers...")
    
    inner_model = model.model if hasattr(model, 'model') else model
    
    if hasattr(inner_model, 'inner'):
        inner_model = inner_model.inner

    if rank == 0 and getattr(getattr(inner_model, "root_config", None), "checkpoint_phantom", False):
        rank_zero_print_warning(
            "gradient_checkpoint and arch.checkpoint_phantom are both on: TRMRoot's phantom iterations get checkpointed "
            "twice, and the solver/Jacobian calls are wrapped too. Use checkpoint_phantom alone."
        )

    if hasattr(inner_model, 'L_level') and hasattr(inner_model.L_level, 'layers'):
        original_layers = inner_model.L_level.layers

        for i, layer in enumerate(original_layers):
            original_forward = layer.forward

            def create_checkpointed_forward(orig_forward):
                def checkpointed_forward(*args, **kwargs):
                    if layer.training:
                        def forward_wrapper(*tensor_args):
                            return orig_forward(*tensor_args, **kwargs)

                        return checkpoint(forward_wrapper, *args, use_reentrant=False)
                    else:
                        return orig_forward(*args, **kwargs)
                return checkpointed_forward
            
            layer.forward = create_checkpointed_forward(original_forward)
        
        if rank == 0:
            rank_zero_print_info(f"Applied gradient checkpointing to {len(original_layers)} layers in L_level")
    
    return model

def create_dataloader(
    config: PretrainConfig,
    split: str,
    rank: int,
    world_size: int,
    **kwargs,
):
    enable_shared_sampler_state = kwargs.pop("enable_shared_sampler_state", True)

    shared_sampler_state = None
    if split == "train" and enable_shared_sampler_state:
        try:
            manager = mp.Manager()
        except Exception as exc:  # pragma: no cover - fallback for sandboxed environments
            rank_zero_print_warning(
                f"Shared sampler state disabled (multiprocessing.Manager unavailable): {exc}"
            )
            enable_shared_sampler_state = False
        else:
            _SHARED_STATE_MANAGERS.append(manager)
            shared_sampler_state = manager.dict()

    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed, dataset_path=config.dataset.data_path, rank=rank, num_replicas=world_size, **kwargs
        ),
        split=split,
        shared_sampler_state=shared_sampler_state,
    )
    
    def worker_init_fn(worker_id: int):
        """Set random seeds for each worker to ensure reproducibility."""
        worker_seed = config.seed + rank + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    loader_kwargs = {
        "batch_size": None,
        "num_workers": 1,
        "persistent_workers": True,
        "pin_memory": torch.cuda.is_available(),
        "prefetch_factor": 8,
    }

    dataloader = DataLoader(
        dataset,
        **loader_kwargs,
        worker_init_fn=worker_init_fn,
    )
    return dataloader, dataset.metadata


def _memory_metrics(prefix: str) -> Dict[str, float]:
    """Peak memory so far: GPU allocated and reserved (what nvidia-smi shows) in MB, and the process's peak host RSS."""
    metrics = {f"{prefix}/host_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10}
    if torch.cuda.is_available():
        metrics[f"{prefix}/gpu_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
        metrics[f"{prefix}/gpu_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
    return metrics


def _param_lr_scales(config: PretrainConfig) -> Dict[str, float]:
    """The lr multipliers in force: the arch's own (`arch.param_lr_scales`, e.g. TreeRM's candidate heads) under the
    train config's `param_lr_scales`, which override them key by key."""
    return {**(getattr(config.arch, "param_lr_scales", None) or {}), **config.param_lr_scales}


def _lr_scaled_groups(named_params: List[Tuple[str, torch.nn.Parameter]], scales: Dict[str, float]) -> List[Dict[str, Any]]:
    """Splits the parameters into optimizer param groups by `scales` (name substring -> lr multiplier; first match wins).
    Each group carries `lr_scale`, which the lr schedule multiplies in; unmatched parameters get 1.0."""
    groups: Dict[float, List[torch.nn.Parameter]] = {}
    for name, param in named_params:
        scale = next((s for substring, s in scales.items() if substring in name), 1.0)
        groups.setdefault(scale, []).append(param)
    return [{"params": params, "lr_scale": scale} for scale, params in groups.items()]


def create_model(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,
    )
    # The canvas width comes from the dataset like seq_len does: grid_size, or grid_size + 1 for an ARC border_tokens build
    if model_cfg.get("board_height") and train_metadata.seq_len % model_cfg["board_height"] == 0:
        model_cfg["board_width"] = train_metadata.seq_len // model_cfg["board_height"]
    print("*"*80)
    print(model_cfg, type(model_cfg))

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    init_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.device(init_device):
        model: nn.Module = model_cls(model_cfg)
        loss_kwargs = dict(config.arch.loss.__pydantic_extra__)
        model = loss_head_cls(model, **loss_kwargs)  # type: ignore
        
        model = apply_gradient_checkpointing(model, config, rank) 
        
        # if "DISABLE_COMPILE" not in os.environ:
        #     model = torch.compile(model)  # type: ignore
        
        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)
             

    # In general, AdamATan2 is used as the optimizer, the other optimizers are for the embedding trick thing that Sapient did in their paper,
    # where they created an embedding for each type of puzzle and trained on those. I think this process can be removed, even for the ARC puzzles, but keeping for now

    # With optimizer=muon, the 2D weight matrices of the looped blocks go to Muon and everything else stays on AdamAtan2
    if config.optimizer == "muon":
        muon_params = [p for name, p in model.named_parameters() if p.ndim == 2 and (".L_level." in name or ".H_blocks." in name)]
        if not muon_params:
            raise ValueError(f"optimizer=muon found no 2D L_level weights in {config.arch.name}.")
        muon_ids = {id(p) for p in muon_params}
        adam_params = _lr_scaled_groups([(n, p) for n, p in model.named_parameters() if id(p) not in muon_ids], _param_lr_scales(config))
    elif config.optimizer == "adam_atan2":
        muon_params, adam_params = [], _lr_scaled_groups(list(model.named_parameters()), _param_lr_scales(config))
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer!r}; expected 'adam_atan2' or 'muon'.")

    if config.freeze_alternation_epochs is not None and (config.arch.puzzle_emb_ndim == 0 or config.freeze_weights):
        raise ValueError("freeze_alternation_epochs needs puzzle embeddings (arch.puzzle_emb_ndim > 0) and freeze_weights=False.")

    # Set to 0 to avoid using the puzzle embedding trick at all
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamAtan2(
                adam_params,
                lr=1e-8,  # Small placeholder; will be overridden by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
        
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=1e-8,  # Small placeholder; will be overridden by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]

    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=1e-8,  # Small placeholder; will be overridden by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamAtan2(
                adam_params,
                lr=1e-8,  # Small placeholder; will be overridden by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    if muon_params and not config.freeze_weights:
        # match_rms_adamw scales Muon's update to AdamW's RMS, so the AdamW lr and weight decay carry over
        optimizers.append(
            torch.optim.Muon(
                muon_params,
                lr=1e-8,  # Small placeholder; will be overridden by scheduler
                weight_decay=config.weight_decay,
                adjust_lr_fn="match_rms_adamw",
            )
        )
        optimizer_lrs.append(config.muon_lr if config.muon_lr is not None else config.lr)
    return model, optimizers, optimizer_lrs

def init_train_state(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
):
    """
    Here, we are actually not calculating the total_samples in the dataset, but the total number of examples to be seen during training.
    say we have 1000 original items, if we generate with 1000 aug per item, then we have 1001 x 1000 examples saved to local
    However, when training, we only sample one item from one group of puzzles. That sampled item contributes however many examples it contains toward filling the batch.
    """

    total_samples = int(
        train_metadata.total_groups * train_metadata.mean_puzzle_examples
    )

    steps_per_epoch = max(
        math.floor(total_samples / max(int(config.global_batch_size), 1)),
        1,
    )
    estimated_steps = int(config.epochs * steps_per_epoch)
    rank_zero_print_info(f"""
epochs: {config.epochs}
total_groups: {train_metadata.total_groups}
mean_puzzle_examples: {train_metadata.mean_puzzle_examples}
puzzle_ids: {train_metadata.num_puzzle_identifiers}
vocab_size: {train_metadata.vocab_size}
seq_len: {train_metadata.seq_len}

total_samples: {total_samples}
global_batch_size: {config.global_batch_size}                         
Estimated total training steps: {estimated_steps}""")
    rank_zero_print_info(f"Steps per epoch: {steps_per_epoch}")
    if config.max_steps is not None:
        total_steps = min(estimated_steps, config.max_steps)
        rank_zero_print_info(f"Using max_steps limit: {config.max_steps}, total_steps set to: {total_steps}")
    else:
        total_steps = estimated_steps

    model, optimizers, optimizer_lrs = create_model(
        config,
        train_metadata,
        rank=rank,
        world_size=world_size,
    )

    ema_helper = None
    if getattr(config, "ema", False):
        ema_helper = EMAHelper(mu=getattr(config, "ema_rate", 0.999))
        ema_helper.register(model)

    return TrainState(
        step=0,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        ema_helper=ema_helper,
    )

# Maybe: Save_train_state, load_checkpoint, compute_lr

def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    evaluators = []
    for cfg in config.dataset.evaluators:
        cls = load_model_class(cfg.name, "evaluators.")(
            data_path=config.dataset.data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
        )  # type: ignore
        evaluators.append(cls)

    return evaluators


def _trainable(model: nn.Module) -> List[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def _puzzle_emb(model: nn.Module) -> Optional[nn.Module]:
    return getattr(model.model, "puzzle_emb", None)


def _maml_targets(model: nn.Module, substrings: Sequence[str] = ()) -> List[torch.Tensor]:
    """What MAML adapts: the trainable parameters plus the puzzle-embedding table when the model has one (a task's
    identifier rows adapt with the weights), restricted to names containing one of `substrings` when given (the
    table's name is model.inner.puzzle_emb.weights, so `puzzle_emb` selects it). CastedSparseEmbedding hands
    gradients to a fixed-size local buffer; its path is bypassed (emptied) so the table itself takes the gradient."""
    match = lambda name: not substrings or any(s in name for s in substrings)
    targets = [p for name, p in model.named_parameters() if p.requires_grad and match(name)]
    emb = _puzzle_emb(model)
    if emb is not None:
        if emb.local_weights.shape[0] != 0:
            emb.local_weights = nn.Buffer(torch.zeros((0, emb.weights.shape[1]), device=emb.weights.device, requires_grad=True), persistent=False)
        emb.weights.requires_grad_(True)
        if match(next(name for name, b in model.named_buffers() if b is emb.weights)):
            targets.append(emb.weights)
    if not targets:
        raise ValueError(f"maml.adapt_substrings {list(substrings)} match no parameter of the model.")
    return targets


def _maml_release(model: nn.Module) -> None:
    """Undoes _maml_targets' grad flag on the embedding table (the SignSGD optimizer tells its tensors apart by it)."""
    emb = _puzzle_emb(model)
    if emb is not None:
        emb.weights.requires_grad_(False)


class EpisodeSampler:
    """MAML episodes from a build's train split. A task is one puzzle (a group); an episode takes `copies` of its
    augmented copies, holds one random row of each out as the query and uses the copies' other rows as the support,
    the structure evaluate_ttt.py's adaptation sees (every aux copy's demos, then the copies' test pairs). Copies
    with a single row are skipped."""

    FIELDS = ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices")

    def __init__(self, data_path: str, meta_batch_size: int, copies: int, device: torch.device, seed: int = 0) -> None:
        split = os.path.join(data_path, "train")
        with open(os.path.join(split, "dataset.json")) as f:
            self.ignore_label_id = json.load(f).get("ignore_label_id")
        self.data = {k: np.load(os.path.join(split, f"all__{k}.npy"), mmap_mode="r") for k in self.FIELDS}
        path = os.path.join(split, "all__object_ids.npy")
        self.object_ids = np.load(path, mmap_mode="r") if os.path.exists(path) else None
        self.row_puzzle = np.repeat(np.arange(len(self.data["puzzle_indices"]) - 1), np.diff(self.data["puzzle_indices"]))
        self.meta_batch_size, self.copies, self.device = meta_batch_size, copies, device
        self.rng = np.random.default_rng(seed)

    def sample(self) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        group_indices, puzzle_indices = self.data["group_indices"], self.data["puzzle_indices"]
        episodes = []
        for g in self.rng.choice(len(group_indices) - 1, size=min(self.meta_batch_size, len(group_indices) - 1), replace=False):
            puzzles = np.arange(group_indices[g], group_indices[g + 1])
            support, query = [], []
            for p in self.rng.choice(puzzles, size=min(self.copies, len(puzzles)), replace=False):
                rows = np.arange(puzzle_indices[p], puzzle_indices[p + 1])
                if len(rows) < 2:
                    continue
                q = self.rng.choice(rows)
                support.extend(int(r) for r in rows if r != q)
                query.append(int(q))
            if support:
                episodes.append((self._batch(support), self._batch(query)))
        return episodes

    def _batch(self, rows: List[int]) -> Dict[str, torch.Tensor]:
        rows = np.array(sorted(rows))
        labels = self.data["labels"][rows].astype(np.int32)
        if self.ignore_label_id is not None:
            labels[labels == self.ignore_label_id] = IGNORE_LABEL_ID
        batch = {
            "inputs": self.data["inputs"][rows].astype(np.int32),
            "labels": labels,
            "puzzle_identifiers": self.data["puzzle_identifiers"][self.row_puzzle[rows]].astype(np.int32),
        }
        if self.object_ids is not None:
            batch["object_ids"] = self.object_ids[rows].astype(np.int32)
        return {k: torch.from_numpy(v).to(self.device) for k, v in batch.items()}


_EPISODES: Dict[str, EpisodeSampler] = {}


def _episodes(config: PretrainConfig, device: torch.device) -> EpisodeSampler:
    key = config.dataset.data_path
    if key not in _EPISODES:
        _EPISODES[key] = EpisodeSampler(key, config.maml.meta_batch_size, config.maml.support_copies, device, seed=config.seed)
    return _EPISODES[key]


def task_grads(model: nn.Module, batch: Dict[str, torch.Tensor], params: List[torch.Tensor]) -> Tuple[List[Optional[torch.Tensor]], torch.Tensor]:
    """Gradient of a task's loss w.r.t. params and the loss itself (per row): the batch run from a fresh carry through
    every supervision step, as eval runs a puzzle; rows that halt early restart on the same rows. Each step's gradient
    is taken on its own and the carry detached, because the inner reads the previous step's state with graph and
    _reset_rows writes into saved tensors in place, so one summed backward would fail."""
    grads: List[Optional[torch.Tensor]] = [None] * len(params)
    total = torch.zeros((), device=batch["inputs"].device)
    carry = model.initial_carry(batch)
    n_steps = model.model.config.halt_max_steps
    scale = 1.0 / (batch["inputs"].shape[0] * n_steps)  # per row and supervision step, so the SGD step size is on the paper's scale
    for _ in range(n_steps):
        carry, loss, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
        for i, g in enumerate(torch.autograd.grad(loss, params, allow_unused=True)):
            if g is not None:
                g = g * scale
                grads[i] = g if grads[i] is None else grads[i] + g
        total += loss.detach() * scale
        carry = apply_to_tensors(carry, torch.Tensor.detach)
    return grads, total


def maml_adapt(model: nn.Module, batch: Dict[str, torch.Tensor], steps: int, lr: float, substrings: Sequence[str] = ()) -> List[float]:
    """MAML's inner loop on one task: `steps` SGD steps of size lr on the adapted tensors (_maml_targets), in place.
    Returns the task loss before each step. Shared by meta-training (maml_batch) and test-time adaptation (evaluate_ttt.py)."""
    params = _maml_targets(model, substrings)
    losses = []
    for _ in range(steps):
        grads, loss = task_grads(model, batch, params)
        losses.append(float(loss))
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is not None:
                    p.sub_(lr * g)
    return losses


def maml_batch(model: nn.Module, episodes: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]], cfg: Any) -> Tuple[List[Optional[torch.Tensor]], int, Dict[str, torch.Tensor]]:
    """First-order MAML over a meta-batch of episodes (EpisodeSampler: a puzzle's augmented copies, support / query):
    per task the support adapts a copy of the weights and embedding rows (maml_adapt; only cfg.adapt_substrings when
    given), the query's gradient at the adapted weights w.r.t. every parameter is summed over the tasks (the paper's
    first-order meta-gradient; the outer loop trains everything), and the weights are restored.
    Returns (meta-gradient sums over _maml_targets(model), number of tasks, metrics for train_batch's logging)."""
    params = _maml_targets(model)
    theta = [p.detach().clone() for p in params]
    meta_grads: List[Optional[torch.Tensor]] = [None] * len(params)
    n_tasks, support_loss, query_loss = 0, 0.0, 0.0
    for support, query in episodes:
        support_loss += maml_adapt(model, support, cfg.inner_steps, cfg.inner_lr, cfg.adapt_substrings)[0]
        grads, loss = task_grads(model, query, params)
        query_loss += float(loss)
        for i, g in enumerate(grads):
            if g is not None:
                meta_grads[i] = g if meta_grads[i] is None else meta_grads[i] + g
        with torch.no_grad():
            for p, p0 in zip(params, theta):
                p.copy_(p0)
        n_tasks += 1
    device = params[0].device
    metrics = {  # fprl/ keys are logged as-is by train_batch (not divided by the batch size)
        "count": torch.tensor(float(n_tasks), device=device),
        "fprl/query_loss": torch.tensor(query_loss / max(n_tasks, 1), device=device),
        "fprl/support_loss": torch.tensor(support_loss / max(n_tasks, 1), device=device),
    }
    return meta_grads, n_tasks, metrics


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
):

    train_state.step += 1
    if train_state.step >= train_state.total_steps:  # At most train_total_steps
        return
    
    # I feel like this try except blocks are unecessary, but can't say fs until I run the code
    try:
        device = next(train_state.model.parameters()).device
    except StopIteration:
        print("*"*80)
        print("Stop iteration reached")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = tree_to_device(batch, device)

        
    try:
        if train_state.carry is not None:  # MAML keeps no carry between steps
            train_state.carry.global_step = int(train_state.step)
    except Exception:
        print("*"*80)
        print("Glob step exception", train_state.step, train_state.carry)
        pass

    steps_per_epoch = max(train_state.steps_per_epoch, 1)
    current_epoch = (train_state.step - 1) // steps_per_epoch


    model_forward_signatures = list(train_state.model.forward.__code__.co_varnames) # This block of logic exists solely to support the transformers_baseline compute_target_q param

    # FLOPs are measured on this step's real forward+backward (works with torch.compile) every flops_log_interval_steps,
    # and the last measurement is carried forward in between (exact when every step has the same cost)
    measure_flops = _should_log_flops(config, train_state.step)
    flop_counter = TotalFlopCounter(display=False) if measure_flops else nullcontext()
    with flop_counter:
        if config.maml.enabled:
            # First-order MAML: the meta-gradient replaces backward; the optimizer / schedule / logging below are shared.
            # The loader's batch only paces the steps; the episodes come from the sampler
            meta_grads, n_tasks, metrics = maml_batch(train_state.model, _episodes(config, device).sample(), config.maml)
            emb = _puzzle_emb(train_state.model)
            for p, g in zip(_maml_targets(train_state.model), meta_grads):
                if emb is not None and p is emb.weights:
                    if g is not None:  # the embedding table: the recipe's SignSGD on the rows the queries touched
                        with torch.no_grad():
                            rows = g.abs().sum(1).nonzero().flatten()
                            _sparse_emb_signsgd_dist(g[rows] / max(n_tasks, 1), rows.to(torch.int32), emb.weights,
                                                     lr=compute_lr(train_state.optimizer_lrs[0], config, train_state),
                                                     weight_decay=config.puzzle_emb_weight_decay, world_size=1)
                else:
                    p.grad = None if g is None else g / max(n_tasks, 1)
            _maml_release(train_state.model)
        elif "compute_target_q" in model_forward_signatures:
            compute_target_q = train_state.step % config.target_q_update_every == 0
            train_state.carry, loss, metrics, _, all_finish = train_state.model(
                carry=train_state.carry, batch=batch, return_keys=[], compute_target_q=compute_target_q
            )
        else:
            train_state.carry, loss, metrics, _, all_finish = train_state.model(
                carry=train_state.carry, batch=batch, return_keys=[]
            )
            # print("$$"*80)
            # dot = make_dot(model(carry=train_state.carry, batch=batch, return_keys=[]), params=dict(model.named_parameters()))
            # dot.render("model_graph", format="png")

        if not config.maml.enabled:
            # Here we scale the loss by the batch size so that when we take the total sum across the GPUs, we end up with the same gradient
            loss_to_backprop = (1 / global_batch_size) * loss

            retain_graph = bool(getattr(config, "retain_graph", False))
            loss_to_backprop.backward(retain_graph=retain_graph)

    if measure_flops:
        train_state.flops_per_step = float(flop_counter.get_total_flops()) * world_size
    train_state.flops_total += train_state.flops_per_step

    if world_size > 1: # Schyncronize the gradients across the GPUs
        for param in train_state.model.parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
            dist.all_reduce(param.grad)
    
    # Grad norms must be read before optim.zero_grad() clears them
    norms = {}
    if _should_log_grad_norms(config, train_state.step):
        total_sq: float = 0.0
        max_norm: float = 0.0
        num_tensors: int = 0
        for param in train_state.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            param_norm = float(torch.norm(grad.to(torch.float32)))
            total_sq += param_norm * param_norm
            if param_norm > max_norm:
                max_norm = param_norm
            num_tensors += 1

        norms["train/grad_norm/total"] = float(math.sqrt(total_sq)) if num_tensors > 0 else 0.0
        norms["train/grad_norm/max"] = float(max_norm)
        norms["train/grad_norm/num_tensors"] = float(num_tensors)

    # Alternating freezing: even blocks of freeze_alternation_epochs step only the puzzle embedding optimizer (always
    # optimizers[0]), odd blocks only the weight optimizers. Gradients are still zeroed for all of them
    train_embeddings = None
    if config.freeze_alternation_epochs is not None:
        train_embeddings = (current_epoch // config.freeze_alternation_epochs) % 2 == 0

    # Backpropogate the loss through the network
    lr_this_step = None
    for i, (optim, base_lr) in enumerate(zip(train_state.optimizers, train_state.optimizer_lrs)):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step * param_group.get("lr_scale", 1.0)

        if train_embeddings is None or (i == 0) == train_embeddings:
            optim.step()
        optim.zero_grad()
    

    
    
    raw_steps = metrics.pop("raw_steps", None)
    if raw_steps is not None and getattr(config, "steps_hist_log_interval_steps", None):
        steps_for_hist = _collect_raw_steps_for_logging(raw_steps, world_size)
        if steps_for_hist and rank == 0:
            train_state.steps_hist_buffer.extend(steps_for_hist)


    if len(metrics):

        metric_keys, special_metric_tensors = partition_metrics(metrics, SPECIAL_METRIC_KEYS)
        if not metric_keys:
            raise RuntimeError("No aggregatable metrics were produced during training.")

        metric_values = torch.stack([metrics[k] for k in metric_keys])
        metric_values = reduce_tensor(metric_values, world_size, None, dist.ReduceOp.SUM)

        special_reduce_results: Dict[str, float] = {}
        if special_metric_tensors:
            reduced_special = reduce_special_metrics(special_metric_tensors, world_size, None)
            if rank == 0:
                for sk, tensor in reduced_special.items():
                    special_reduce_results[sk] = float(tensor.detach().cpu().item())

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            count = max(float(reduced_metrics.get("count", 1.0)), 1.0)
            processed_metrics: Dict[str, float] = {}
            for k, v in reduced_metrics.items():
                if "count" in k:
                    continue
                if k.startswith("fprl/"):
                    processed_metrics[f"train/{k}"] = v / float(world_size)
                else:
                    denom = global_batch_size if k.endswith("loss") else count
                    processed_metrics[f"train/{k}"] = v / denom


            for sk, global_value in special_reduce_results.items():
                processed_metrics[f"train/{sk}"] = global_value

            processed_metrics["train/lr"] = lr_this_step
            processed_metrics.update(_memory_metrics("train"))
            processed_metrics["train/epoch"] = current_epoch
            if train_embeddings is not None:
                processed_metrics["train/freeze_phase"] = 0 if train_embeddings else 1  # 0: embeddings, 1: weights
            if train_state.flops_per_step:
                processed_metrics["train/flops_per_step"] = train_state.flops_per_step
                processed_metrics["train/flops_total"] = train_state.flops_total
            if _should_log_steps_hist(config, train_state.step) and train_state.steps_hist_buffer:
                processed_metrics["train/steps_hist"] = wandb.Histogram(train_state.steps_hist_buffer, num_bins=config.arch.halt_max_steps)
                processed_metrics["train/steps_hist_count"] = len(train_state.steps_hist_buffer)
                if getattr(config, "steps_hist_plotly", False):
                    fig = _build_steps_hist_plotly(train_state.steps_hist_buffer, num_bins = config.arch.halt_max_steps)
                    if fig is not None:
                        processed_metrics["train/steps_hist_plotly"] = fig
                train_state.steps_hist_buffer.clear()

            processed_metrics.update(norms)
            metrics_valid = reduced_metrics.get("count", None)
            if not metrics_valid:
                drop_when_empty = {
                "train/accuracy",
                "train/exact_accuracy",
                "train/q_halt_accuracy",
                "train/q_halt_precision",
                "train/q_halt_recall",
                "train/halted_q_logits",
                "train/steps",
                "train/step_utility",
                "train/max_steps",
                "train/min_steps",
                "train/nfe",
                "train/zH_rel_residual",
                "train/zL_rel_residual",
                "train/solver_abs_residual",
                "train/solver_rel_residual",
                "train/halt/quick_correct",
                "train/halt/quick_wrong",
                "train/halt/slow_correct",
                "train/halt/slow_wrong",
            }
                drop_when_empty.update(k for k in processed_metrics if k.startswith("train/solver_rel_residual@"))
                for key in drop_when_empty:
                    processed_metrics.pop(key, None)
            return processed_metrics 

def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore
        apply_wandb_secrets_to_training_config(config)

        if config.run_name is None:
            model_id = config.arch.short_name if config.arch.short_name else config.arch.name.split('@')[-1]
            config.run_name = f"{model_id}-{config.dataset.name}"
            
        objects = [config]
        
        rank_zero_print_info(f"Run Name: {config.run_name}, Project Name: {config.project_name}")

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl",timeout=timedelta(hours=24))

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo",timeout=timedelta(hours=24))
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Generate seed value if not set before loading in config so program doesn't crash
    if hydra_config.seed is None:
        rng = np.random.default_rng()
        hydra_config.seed = int(rng.integers(low=0, high = 1e8))

    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    set_deterministic_mode(config, rank=RANK)

    if not config.train_epochs_per_iter:
        config.train_epochs_per_iter = config.epochs
    train_epochs_per_iter = config.train_epochs_per_iter
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    if RANK == 0:
        print(f"running on dataset {config.dataset.name} from {config.dataset.data_path}")

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=RANK,
            world_size=WORLD_SIZE,
        )
        evaluators = create_evaluators(config, eval_metadata)
    except FileNotFoundError:
        eval_loader = eval_metadata = None
        evaluators = []

    # Reset after data loading so all ranks initialize identical models.
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    torch.manual_seed(config.seed)
    print('*'*80)
    print(config, type(config))
    train_state = init_train_state(
        config,
        train_metadata,
        rank=RANK,
        world_size=WORLD_SIZE,
    )   

    resume_info = load_training_state(
        config,
        train_state,
        train_loader,
        rank=RANK,
        world_size=WORLD_SIZE,
        load_weights_only=config.load_weights_only,
    )

    start_iteration = 0
    if resume_info is not None:
        start_iteration = min(int(resume_info.get("next_iteration", 0)), total_iters)

    progress_bar = None
    wandb_id_short = None
    if RANK == 0:

        run_start_time = datetime.now()
        wandb_kwargs: Dict[str, Any] = {
            "project": config.project_name,
            "config": _config_to_wandb_dict(config),
            "mode": (
                (config.wandb_mode.lower() if isinstance(config.wandb_mode, str) else config.wandb_mode)
                or "online"
            ),
            "entity": config.entity
        }
        if wandb_kwargs["mode"] == "disable":
            wandb_kwargs["mode"] = "disabled"
        
        continue_run = resume_info is not None and config.wandb_on_resume == "continue"
        if not continue_run:
            wandb_kwargs.update({"name": config.run_name, "tags": [f"{config.dataset.name}"]})

        if resume_info is not None and resume_info.get("wandb") is not None:
            wandb_meta = resume_info["wandb"]
            if continue_run:
                if wandb_meta.get("name"):
                    config.run_name = wandb_meta["name"]
                    wandb_kwargs["name"] = config.run_name

                if (
                    wandb_meta.get("id")
                    and not config.load_weights_only
                ):
                    wandb_kwargs["id"] = wandb_meta["id"]
                    wandb_kwargs["resume"] = "must"
            elif config.wandb_on_resume == "fork" and wandb_meta.get("id"):
                wandb_kwargs["fork_from"] = f"{wandb_meta['id']}?_step={train_state.step}"

        # Honor a user-provided W&B run id when resuming a tracked run.
        env_run_id = os.environ.get("WANDB_RUN_ID")
        if resume_info is None and env_run_id:
            wandb_kwargs.setdefault("id", env_run_id)

        wandb_run_id = wandb_kwargs.get("id") or str(uuid.uuid4())
        wandb_kwargs.setdefault("id", wandb_run_id)

        run_root = _resolve_run_root(hydra_config)
        computed_run_dir = os.path.join(run_root, build_wandb_run_dir(wandb_run_id, timestamp=run_start_time))
        computed_run_dir = os.path.abspath(computed_run_dir)

        os.makedirs(computed_run_dir, exist_ok=True)

        global WANDB_RUN_ID, WANDB_RUN_DIR
        WANDB_RUN_ID = wandb_run_id
        rank_zero_print_info(f"WANDB_RUN_ID: {WANDB_RUN_ID}, {wandb_kwargs.get('id')}") if RANK == 0 else None
        WANDB_RUN_DIR = computed_run_dir
        wandb_set_run(WANDB_RUN_ID, WANDB_RUN_DIR)

        wandb_kwargs["dir"] = computed_run_dir
        rank_zero_print_info(f"Results will be saved in: {WANDB_RUN_DIR}") if RANK == 0 else None

        run = wandb.init(**wandb_kwargs)  # type: ignore[arg-type]
        wandb.define_metric("*", step_metric="step", step_sync=True)

        active_run_dir = WANDB_RUN_DIR
        config.checkpoint_path = os.path.join(active_run_dir, "checkpoints")
        os.makedirs(config.checkpoint_path, exist_ok=True)

        if progress_bar is None:
            wandb_id_short = WANDB_RUN_ID[-3:] if WANDB_RUN_ID else "unknown"
            progress_bar = tqdm.tqdm(
                total=train_state.total_steps,
                initial=train_state.step,
                desc=f"Training [{wandb_id_short}]",
                dynamic_ncols=True,
                leave=True,
                position=0,
                miniters=1,
            )
            progress_bar.set_postfix({"loss": 0.0})

        if run is not None:
            run.config.update({"checkpoint_path": config.checkpoint_path}, allow_val_change=True)

        wandb_meta_for_config = {
            "id": WANDB_RUN_ID,
            "name": config.run_name,
            "project": config.project_name,
            "entity": getattr(run, "entity", None) if run is not None else wandb_kwargs.get("entity"),
            "dir": active_run_dir,
            "url": getattr(run, "url", None) if run is not None else None,
            "mode": wandb_kwargs["mode"],
            "resume": wandb_kwargs.get("resume"),
            "tags": wandb_kwargs.get("tags"),
        }
        should_save_code = (RANK == 0)

        if should_save_code:
            save_code_and_config(
                config,
                hydra_config,
                wandb_meta_for_config,
                run_dir=active_run_dir,
                rank=RANK,
            )
            num_params = sum(x.numel() for x in train_state.model.parameters())
            rank_zero_print_info(f"Model parameters: {num_params}")
            wandb.log({
                "num_params": num_params,
                "step": train_state.step
            })

    if "DISABLE_COMPILE" not in os.environ:
        # Keep compilation deterministic across restarts and ranks.
        compile_kwargs = {
            "dynamic": False,
            "mode": "default",
        }

        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.cache_size_limit = 32
        
        train_state.model = torch.compile(train_state.model, **compile_kwargs)  # type: ignore
        
        if config.ema:
            train_state.ema_helper.register(train_state.model)
            rank_zero_print_info("Model compiled with torch.compile() in deterministic mode. EMA helper re-registered.")
    else:
        rank_zero_print_warning("Model compile is disabled.")


    if config.ema and RANK == 0:
        _log_line("Setup EMA", progress_bar=progress_bar)
        
    eval_interval_steps = config.eval_interval_steps
    checkpoint_interval_steps = config.checkpoint_interval_steps
    has_eval_loader = eval_loader is not None and eval_metadata is not None

    last_ckpt_step: Optional[int] = None
    training_done = False

    for _iter_id in range(start_iteration, total_iters):
        if training_done:
            if RANK == 0:
                _log_line("Training completed.", progress_bar=progress_bar)
            break
        if RANK == 0 and progress_bar is not None:
            wandb_id_display = wandb_id_short if wandb_id_short else "unknown"
            progress_bar.set_description(
                f"Training [{wandb_id_display}] E{_iter_id * train_epochs_per_iter}-{_iter_id * train_epochs_per_iter + train_epochs_per_iter - 1}"
            )
            progress_bar.refresh()

        train_state.model.train()
        for i, (set_name, batch, global_batch_size) in enumerate(train_loader):
            metrics = train_batch(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=RANK,
                world_size=WORLD_SIZE,
            )

            if metrics is not None:
                metrics["step"] = train_state.step
                if RANK == 0:
                    wandb.log(metrics)
                    if progress_bar is not None:
                        progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                        train_loss = metrics.get("train/total_loss", metrics.get("train/lm_loss", 0.0))
                        progress_bar.set_postfix({"loss": f"{train_loss:.6f}"})

            if train_state.ema_helper is not None:
                train_state.ema_helper.update(train_state.model)
                
            if (
                checkpoint_interval_steps is not None
                and train_state.step > 0
                and train_state.step % checkpoint_interval_steps == 0
            ):
                if train_state.step % (train_state.steps_per_epoch * config.train_epochs_per_iter) == 0:
                    next_iteration = (_iter_id + 1)
                else:
                    next_iteration = _iter_id
                save_checkpoint(
                    config,
                    train_state,
                    train_loader,
                    next_iteration=next_iteration,
                    rank=RANK,
                )
                last_ckpt_step = train_state.step

            if (
                has_eval_loader
                and eval_interval_steps is not None
                and train_state.step > 0
                and train_state.step % eval_interval_steps == 0
            ):
                # Swap the EMA weights into the live model rather than deep-copying it: a copy duplicates every
                # buffer (incl. the puzzle-embedding table) and can OOM larger models
                if train_state.ema_helper is not None:
                    if RANK == 0:
                        _log_line("Switching to EMA weights for evaluation", progress_bar=progress_bar)
                    train_state.ema_helper.swap(train_state.model)
                try:
                    # Evaluate the bare model under the inference loss head, which scores each puzzle at the step it halts
                    eval_model = getattr(train_state.model, "_orig_mod", train_state.model).model
                    eval_head_cls = load_model_class(config.arch.loss.inference_name)
                    train_state_eval = replace(train_state, model=eval_head_cls(eval_model, **config.arch.loss.__pydantic_extra__))
                    train_state_eval.model.eval()

                    eval_metrics, eval_time_passed = evaluate(
                        config,
                        train_state_eval,
                        eval_loader,
                        eval_metadata,
                        evaluators,
                        rank=RANK,
                        world_size=WORLD_SIZE,
                        cpu_group=CPU_PROCESS_GROUP,
                        progress_bar=progress_bar,
                        max_eval_steps=config.max_eval_steps,
                    )
                finally:
                    if train_state.ema_helper is not None:
                        train_state.ema_helper.swap(train_state.model)
                del eval_model, train_state_eval
                torch.cuda.empty_cache()

                if RANK == 0 and eval_metrics is not None:
                    eval_metrics["step"] = train_state.step
                    eval_metrics.update(_memory_metrics("eval"))
                    wandb.log(eval_metrics)

                train_state.model.train()
                if config.gradient_checkpoint:
                    train_state.model = apply_gradient_checkpointing(train_state.model, config, RANK)
                if RANK == 0 and progress_bar is not None:
                    progress_bar.refresh()
                    

            if train_state.step >= train_state.total_steps:
                training_done = True
                break

        if RANK == 0 and last_ckpt_step != train_state.step:
            if training_done or (_iter_id == total_iters - 1):
                save_checkpoint(
                    config,
                    train_state,
                    train_loader,
                    next_iteration=_iter_id + 1,
                    rank=RANK,
                )
                last_ckpt_step = train_state.step
                
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
