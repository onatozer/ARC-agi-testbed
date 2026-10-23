from typing import Optional, Any, List, Dict, Sequence, Tuple
from dataclasses import replace
import os
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
from torch.nn.attention import SDPBackend, sdpa_kernel

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
from models.sparse_embedding import CastedSparseEmbedding, CastedSparseEmbeddingSignSGD_Distributed

from utils.training import (
    apply_to_tensors,
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


def _lr_scaled_groups(named_params: List[Tuple[str, torch.nn.Parameter]], scales: Dict[str, float]) -> List[Dict[str, Any]]:
    """Splits the parameters into optimizer param groups by `scales` (name substring -> lr multiplier; first match wins).
    Each group carries `lr_scale`, which the lr schedule multiplies in; unmatched parameters get 1.0."""
    groups: Dict[float, List[torch.nn.Parameter]] = {}
    for name, param in named_params:
        scale = next((s for substring, s in scales.items() if substring in name), 1.0)
        groups.setdefault(scale, []).append(param)
    return [{"params": params, "lr_scale": scale} for scale, params in groups.items()]


class MetaSGD(nn.Module):
    """Meta-SGD (Li et al. 2017, arXiv 1707.09835): a learned step size per element of every adapted tensor, alpha = exp(rho),
    initialised at inner_lr. rho lives on the loss head (checkpointed, EMA'd and all-reduced like any parameter) and is
    trained by its own Adam through the second-order outer loss (meta_sgd_batch)."""

    def __init__(self, named: List[Tuple[str, torch.Tensor]], alpha0: float, alpha_range: float) -> None:
        super().__init__()
        # The puzzle-embedding table gets one step size per column, shared by its rows: a test task's rows are new
        shape = lambda n, p: (1, p.shape[1]) if n.endswith("puzzle_emb.weights") else p.shape
        self.rho = nn.ParameterDict({self.key(n): nn.Parameter(torch.full(shape(n, p), math.log(alpha0), dtype=torch.float32, device=p.device)) for n, p in named})
        self.lo, self.hi = math.log(alpha0) - alpha_range, math.log(alpha0) + alpha_range

    @staticmethod
    def key(name: str) -> str:
        return name.replace(".", "|")

    def alpha(self, name: str) -> torch.Tensor:
        return self.rho[self.key(name)].exp()

    @torch.no_grad()
    def clamp_(self) -> None:
        for rho in self.rho.values():
            rho.clamp_(self.lo, self.hi)

    def summary(self) -> Tuple[float, float]:
        """(geometric mean, max) of the step sizes over every adapted element."""
        rhos = torch.cat([r.detach().flatten() for r in self.rho.values()])
        return float(rhos.mean().exp()), float(rhos.max().exp())


def _puzzle_emb_name(model: nn.Module) -> Optional[str]:
    """Qualified name of the puzzle-embedding table (a buffer); None when the model has no puzzle embeddings."""
    return next((f"{n}.weights" for n, m in model.named_modules() if isinstance(m, CastedSparseEmbedding)), None)


def _meta_sgd_targets(model: nn.Module, substrings: List[str]) -> List[Tuple[str, torch.Tensor]]:
    """The (name, tensor) pairs the inner step adapts: the trainable parameters plus the puzzle-embedding table, those whose
    name contains one of substrings (all of them when substrings is empty)."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("meta_sgd.")]
    emb_name = _puzzle_emb_name(model)
    if emb_name is not None:
        named.append((emb_name, model.get_buffer(emb_name)))
    named = [(n, p) for n, p in named if not substrings or any(s in n for s in substrings)]
    if not named:
        raise ValueError(f"meta_sgd.adapt_substrings {substrings} match no parameter.")
    return named


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

        if config.meta_sgd.enabled:
            if "trm_root" in config.arch.name:
                raise ValueError("meta_sgd needs a double-backward-safe model; TRMRoot's custom autograd is not.")
            model.meta_sgd = MetaSGD(_meta_sgd_targets(model, config.meta_sgd.adapt_substrings), config.meta_sgd.inner_lr, config.meta_sgd.alpha_range)
        
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
    weight_params = [(n, p) for n, p in model.named_parameters() if not n.startswith("meta_sgd.")]  # Meta-SGD's rho has its own Adam (no weight decay)
    if config.optimizer == "muon":
        muon_params = [p for name, p in weight_params if p.ndim == 2 and ".L_level." in name]
        if not muon_params:
            raise ValueError(f"optimizer=muon found no 2D L_level weights in {config.arch.name}.")
        muon_ids = {id(p) for p in muon_params}
        adam_params = _lr_scaled_groups([(n, p) for n, p in weight_params if id(p) not in muon_ids], config.param_lr_scales)
    elif config.optimizer == "adam_atan2":
        muon_params, adam_params = [], _lr_scaled_groups(weight_params, config.param_lr_scales)
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
    if config.meta_sgd.enabled:
        optimizers.append(torch.optim.Adam(model.meta_sgd.parameters(), lr=1e-8))  # placeholder lr; the scheduler sets it
        optimizer_lrs.append(config.meta_sgd.alpha_lr)
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
        if "compute_target_q" in model_forward_signatures:
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


        # Here we scale the loss by the batch size so that when we take the total sum across the GPUs, we end up with the same gradient
        loss_to_backprop = (1 / global_batch_size) * loss

        retain_graph = bool(getattr(config, "retain_graph", False))
        loss_to_backprop.backward(retain_graph=retain_graph)

    return _step_optimizers_and_log(config, train_state, metrics, global_batch_size, rank, world_size, current_epoch, measure_flops, flop_counter)


def _step_optimizers_and_log(
    config: PretrainConfig,
    train_state: TrainState,
    metrics: Dict[str, torch.Tensor],
    global_batch_size: int,
    rank: int,
    world_size: int,
    current_epoch: int,
    measure_flops: bool,
    flop_counter: Any,
) -> Optional[Dict[str, Any]]:
    """The second half of a training step, after the gradients are in .grad: FLOPs bookkeeping, gradient all-reduce, grad
    norms, the optimizer steps on the lr schedule (with freeze alternation), and the reduced metrics (rank 0)."""
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
                if k.startswith(("fprl/", "meta/")):  # already means: summed over ranks, then averaged
                    processed_metrics[f"train/{k}"] = v / float(world_size)
                else:
                    denom = global_batch_size if k.endswith("loss") else count
                    processed_metrics[f"train/{k}"] = v / denom


            for sk, global_value in special_reduce_results.items():
                processed_metrics[f"train/{sk}"] = global_value

            processed_metrics["train/lr"] = lr_this_step
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


def _episodes_from_batch(batch: Dict[str, torch.Tensor]) -> List[Tuple[int, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    """One (puzzle id, support rows, query rows) per puzzle in the batch: query = the rows flagged validation (the task's
    former test pair, see build_miniarc_dataset heldout_tasks), support = its other rows. Padded rows (blank id 0) and
    puzzles missing either kind (the one _sample_batch truncates at the batch tail) are skipped."""
    if "validation" not in batch:
        raise ValueError("meta_sgd needs batch['validation']: build the dataset with heldout_tasks > 0 (dataset/build_miniarc_dataset.py).")
    ids, is_query = batch["puzzle_identifiers"], batch["validation"].bool()
    episodes = []
    for pid in torch.unique(ids).tolist():
        rows = ids == pid
        support, query = rows & ~is_query, rows & is_query
        if pid == 0 or not support.any() or not query.any():
            continue
        episodes.append((pid, {k: v[support] for k, v in batch.items()}, {k: v[query] for k, v in batch.items()}))
    if not episodes:
        raise RuntimeError(f"meta_sgd: no puzzle in the batch has both demo and validation rows ({torch.unique(ids).numel()} ids, {int(is_query.sum())} validation rows).")
    return episodes


def _supervised_loss(model: nn.Module, overrides: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The loss head's loss on `batch` with the module's tensors named in `overrides` substituted (torch.func.functional_call),
    summed over each row's ACT trajectory from a fresh carry: a row contributes every supervision step until the model halts
    it (or halt_max_steps), as in train_batch, where halted rows are replaced by new ones; here they are dropped. The carry
    is detached between steps (deep supervision), so no in-place write touches a tensor saved for backward."""
    carry, total = None, 0.0
    while batch["inputs"].shape[0] > 0:
        carry, loss, _metrics, _, _ = torch.func.functional_call(model, overrides, args=(), kwargs=dict(return_keys=[], carry=carry, batch=batch))
        total = total + loss
        keep = ~carry.halted
        carry = apply_to_tensors(carry, lambda t: t[keep].detach().clone())
        batch = {k: v[keep] for k, v in batch.items()}
    return total


def _adapt(
    model: nn.Module, named: List[Tuple[str, torch.Tensor]], overrides: Dict[str, torch.Tensor], support: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """The inner step: (per-row support loss, theta1 = theta - alpha * its gradient for every tensor in `named`), with the
    gradient's graph kept so the outer loss can differentiate through it. `overrides` are substituted during the forward."""
    loss_s = _supervised_loss(model, overrides, support) / support["inputs"].shape[0]
    grads = torch.autograd.grad(loss_s, [p for _, p in named], create_graph=True, allow_unused=True)
    theta1 = dict(overrides)
    theta1.update({n: p - model.meta_sgd.alpha(n) * g for (n, p), g in zip(named, grads) if g is not None})
    return loss_s, theta1


def _task_outer_backward(
    model: nn.Module, targets: List[Tuple[str, torch.Tensor]], emb_name: Optional[str], pid: int,
    support: Dict[str, torch.Tensor], query: Dict[str, torch.Tensor], loss_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Meta-SGD on one puzzle: theta1 = theta - alpha * grad of the per-row support loss (graph kept), then the query loss
    at theta1, times loss_scale, is backpropagated into theta (second order, through theta1), rho and the puzzle-embedding
    row. Returns (support loss per row, query loss per row, the embedding row's gradient or None)."""
    overrides, emb_rows = {}, None
    if emb_name is not None:
        # The task's row of the table as a leaf in place of the whole table (ids remapped to 0); the forward then takes its
        # differentiable gather path because the row count differs from the (emptied) local_weights buffer
        table = model.get_buffer(emb_name)
        emb_rows = table[[pid]].detach().requires_grad_()
        overrides = {emb_name: emb_rows, emb_name[: -len("weights")] + "local_weights": table.new_zeros(0, table.shape[1])}
        support = {**support, "puzzle_identifiers": torch.zeros_like(support["puzzle_identifiers"])}
        query = {**query, "puzzle_identifiers": torch.zeros_like(query["puzzle_identifiers"])}
    named = [(n, emb_rows if n == emb_name else p) for n, p in targets]

    with sdpa_kernel(SDPBackend.MATH):  # only the math attention kernel has a double backward
        loss_s, theta1 = _adapt(model, named, overrides, support)
        loss_q = _supervised_loss(model, theta1, query)
        (loss_q * loss_scale).backward()
    return loss_s.detach(), loss_q.detach() / query["inputs"].shape[0], None if emb_rows is None else emb_rows.grad


def meta_sgd_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
):
    """train_batch's counterpart for config.meta_sgd.enabled (second-order Meta-SGD, arXiv 1707.09835): every puzzle in the
    batch is a task (_episodes_from_batch), each gets _task_outer_backward, the query losses are averaged over the batch's
    query rows, and the shared optimizer step follows. No persistent ACT carry: every loss runs each row's ACT trajectory
    from a fresh one (_supervised_loss)."""
    train_state.step += 1
    if train_state.step >= train_state.total_steps:
        return
    model = train_state.model
    device = next(model.parameters()).device
    batch = tree_to_device(batch, device)
    current_epoch = (train_state.step - 1) // max(train_state.steps_per_epoch, 1)
    cfg = config.meta_sgd

    episodes = _episodes_from_batch(batch)
    n_query = torch.tensor(float(sum(q["inputs"].shape[0] for _, _, q in episodes)), device=device)
    if world_size > 1:
        dist.all_reduce(n_query)  # the scale makes the all-reduced gradient the mean over every rank's query rows
    emb_name = _puzzle_emb_name(model)
    targets = _meta_sgd_targets(model, cfg.adapt_substrings)
    emb_grad = None if emb_name is None else torch.zeros_like(model.get_buffer(emb_name[: -len("weights")] + "local_weights"))

    measure_flops = _should_log_flops(config, train_state.step)
    flop_counter = TotalFlopCounter(display=False) if measure_flops else nullcontext()
    support_losses, query_losses = [], []
    with flop_counter:
        for pid, support, query in episodes:
            loss_s, loss_q, row_grad = _task_outer_backward(model, targets, emb_name, pid, support, query, 1.0 / n_query)
            support_losses.append(loss_s)
            query_losses.append(loss_q)
            if row_grad is not None:
                emb_grad[batch["puzzle_identifiers"] == pid] = row_grad
    if emb_name is not None:
        # Hand the table's gradient to its SignSGD optimizer the way the ordinary forward does: one row per batch row, plus the ids
        emb = model.get_submodule(emb_name[: -len(".weights")])
        with torch.no_grad():
            emb.local_ids.copy_(batch["puzzle_identifiers"])
        emb.local_weights.grad = emb_grad

    alpha_mean, alpha_max = model.meta_sgd.summary()
    tasks = torch.tensor(float(len(episodes)), device=device)
    metrics = {
        "count": tasks,
        "meta/tasks": tasks,
        "meta/support_loss": torch.stack(support_losses).mean(),
        "meta/query_loss": torch.stack(query_losses).mean(),
        "meta/alpha_mean": torch.tensor(alpha_mean, device=device),
        "meta/alpha_max": torch.tensor(alpha_max, device=device),
    }
    out = _step_optimizers_and_log(config, train_state, metrics, global_batch_size, rank, world_size, current_epoch, measure_flops, flop_counter)
    model.meta_sgd.clamp_()
    return out

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

    if "DISABLE_COMPILE" not in os.environ and not config.meta_sgd.enabled:  # the meta step runs functional_call + double backward eagerly
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
        step_fn = meta_sgd_batch if config.meta_sgd.enabled else train_batch
        for i, (set_name, batch, global_batch_size) in enumerate(train_loader):
            metrics = step_fn(
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
                        train_loss = metrics.get("train/total_loss", metrics.get("train/lm_loss", metrics.get("train/meta/query_loss", 0.0)))
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
