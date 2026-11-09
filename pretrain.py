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
from torch.nn.attention import SDPBackend, sdpa_kernel
from models.losses import IGNORE_LABEL_ID
from models.losses.loss_heads import RowWeightMixin, stablemax_cross_entropy

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


def _checkpoint_layer(layer: nn.Module) -> None:
    """Recompute the layer's activations in the backward pass instead of storing them (training only). Its parameters go
    into the checkpoint as explicit inputs, read at forward time: under Meta-SGD's functional_call they are the episode's
    adapted tensors, which the recomputation (after functional_call returns) would otherwise replace by the originals.
    Non-reentrant checkpointing, so second-order meta-gradients pass through."""
    forward, names, inside = layer.forward, [n for n, _ in layer.named_parameters()], [False]

    def current(name: str) -> torch.Tensor:
        obj = layer
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj

    def checkpointed(*args, **kwargs):
        if inside[0] or not layer.training:
            return forward(*args, **kwargs)

        def run(*flat):
            inside[0] = True
            try:
                return torch.func.functional_call(layer, dict(zip(names, flat[:len(names)])), flat[len(names):], kwargs)
            finally:
                inside[0] = False

        return checkpoint(run, *[current(n) for n in names], *args, use_reentrant=False)

    layer.forward = checkpointed


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

    # TRM's reasoning blocks, VARC's encoder layers
    layers = inner_model.L_level.layers if hasattr(getattr(inner_model, "L_level", None), "layers") else getattr(inner_model, "layers", None)
    if layers is not None:
        for layer in layers:
            _checkpoint_layer(layer)
        if rank == 0:
            rank_zero_print_info(f"Applied gradient checkpointing to {len(layers)} layers")

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
    """Meta-SGD's learned step sizes (arXiv 1707.09835): one per adapted tensor and inner step (LSLR, arXiv 1810.09502),
    and for the puzzle-embedding table one per column, shared by its rows (a test task's rows are new). alpha = exp(rho),
    kept within alpha_range of its initial value. rho lives on the loss head, so it is checkpointed with the model.
    recon_steps > 0 (meta_sgd.recon_steps): also one learned log scale per reconstruction step, on every tensor's initial alpha."""

    def __init__(self, named: List[Tuple[str, torch.Tensor]], steps: int, alpha0: float, alpha0_rows: float, alpha_range: float,
                 recon_steps: int = 0, recon_scale0: float = 1.0) -> None:
        super().__init__()
        is_table = lambda n: n.endswith("puzzle_emb.weights")
        init = lambda n: math.log(alpha0_rows if is_table(n) else alpha0)
        shape = lambda n, p: (steps, 1, p.shape[1]) if is_table(n) else (steps,)

        self.rho = nn.ParameterDict({self.key(n): nn.Parameter(torch.full(shape(n, p), init(n), dtype=torch.float32, device=p.device)) for n, p in named})
        self.bounds = {self.key(n): (init(n) - alpha_range, init(n) + alpha_range) for n, _ in named}
        self.log_init = {self.key(n): init(n) for n, _ in named}
        self.steps = steps

        self.recon_log_scale = None
        if recon_steps:
            self.recon_log_scale = nn.Parameter(torch.full((recon_steps,), math.log(recon_scale0), device=named[0][1].device))

    @staticmethod
    def key(name: str) -> str:
        return name.replace(".", "|")

    def alpha(self, name: str, step: int) -> torch.Tensor:
        return self.rho[self.key(name)][step].exp()

    def recon_alpha(self, name: str, step: int) -> torch.Tensor:
        return (self.recon_log_scale[step] + self.log_init[self.key(name)]).exp()

    @torch.no_grad()
    def clamp_(self) -> None:
        for key, rho in self.rho.items():
            rho.clamp_(*self.bounds[key])

    def summary(self) -> Tuple[float, float]:
        """(geometric mean, max) of the step sizes."""
        rhos = torch.cat([r.detach().flatten() for r in self.rho.values()])
        return float(rhos.mean().exp()), float(rhos.max().exp())


def _puzzle_emb_name(model: nn.Module) -> Optional[str]:
    """Qualified name of the puzzle-embedding table (a buffer); None when the model has no puzzle embeddings."""
    return next((f"{n}.weights" for n, m in model.named_modules() if isinstance(m, CastedSparseEmbedding)), None)


def _meta_sgd_targets(model: nn.Module) -> List[Tuple[str, torch.Tensor]]:
    """The (name, tensor) pairs the inner step adapts: every trainable parameter plus the puzzle-embedding table."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("meta_sgd.")]
    emb_name = _puzzle_emb_name(model)
    if emb_name is not None:
        named.append((emb_name, model.get_buffer(emb_name)))
    return named


def create_model(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        # config.recon appends its masked copies to the ordinary training batch, and adds one shared embedding row
        batch_size=config.global_batch_size // world_size * (1 if config.meta_sgd.enabled else 1 + config.recon.copies),
        vocab_size=train_metadata.vocab_size + int(config.recon.copies > 0),  # config.recon's mask token
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers + int(config.recon.copies > 0),
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

        if config.meta_sgd.enabled and config.meta_sgd.recon_steps and not config.recon.copies:
            raise ValueError("meta_sgd.recon_steps needs config.recon (recon.copies > 0): its steps reconstruct masked inputs on the shared row.")
        if config.recon.copies and (not isinstance(model, RowWeightMixin) or config.arch.puzzle_emb_ndim == 0):
            raise ValueError("recon.copies > 0 needs puzzle embeddings and a row-weighting loss head (arch.loss.name=losses.loss_heads@ACTLossHeadRowWeight or LossHeadGRAMRowWeight).")

        if config.meta_sgd.enabled:
            cfg = config.meta_sgd
            model.meta_sgd = MetaSGD(_meta_sgd_targets(model), cfg.inner_steps, cfg.inner_lr, cfg.inner_lr_rows, cfg.alpha_range,
                                     cfg.recon_steps, cfg.recon_scale0)
            model.meta_sgd.mask_token = train_metadata.vocab_size  # config.recon's
            model.meta_sgd.hd_beta, model.meta_sgd.hd_state, model.meta_sgd.imaml_diag = cfg.hd_beta, {}, []
            if cfg.imaml and config.recon.copies:
                raise ValueError("meta_sgd.imaml does not take config.recon's masked copies: set recon.copies=0.")
        
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
    weight_params = [(n, p) for n, p in model.named_parameters() if not n.startswith("meta_sgd.")]  # Meta-SGD's rho has its own Adam
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

    if config.meta_sgd.enabled and config.meta_sgd.recon_steps:  # stage 2: only the reconstruction step sizes learn
        return model, [torch.optim.Adam([model.meta_sgd.recon_log_scale], lr=1e-8)], [config.meta_sgd.alpha_lr]
    if config.meta_sgd.enabled:
        optimizers.append(torch.optim.Adam(model.meta_sgd.rho.parameters(), lr=1e-8))  # placeholder lr; the scheduler sets it
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
        train_metadata.total_groups * train_metadata.mean_puzzle_examples * _copies_per_group(config)  # a pass draws that many puzzles per group
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


def _copies_per_group(config: PretrainConfig) -> int:
    """Puzzles the training sampler draws per group: a Meta-SGD episode's augmented copies of its task, else 1."""
    return config.meta_sgd.copies_per_group if config.meta_sgd.enabled else 1


def _with_recon_rows(batch: Dict[str, torch.Tensor], cfg: Any, mask_token: int, shared_id: int) -> Dict[str, torch.Tensor]:
    """config.recon: the batch plus cfg.copies masked copies of every row (ReconConfig) and batch["row_weight"]."""
    x, k = batch["inputs"], cfg.copies
    xs = x.repeat(k, 1)

    # round(cfg.fraction x its color cells) random color cells of each copy set to the mask token
    colors = xs >= 2  # the ARC/MiniARC builds' tokens: 0 pad, 1 end of grid, 2-11 colors
    scores = torch.rand(xs.shape).to(xs.device).masked_fill(~colors, -1.0)
    rank = scores.argsort(-1, descending=True).argsort(-1)
    masked = xs.masked_fill(rank < (cfg.fraction * colors.sum(-1, keepdim=True)).round(), mask_token)

    out = {key: torch.cat([v, v.repeat(k, *[1] * (v.dim() - 1))]) for key, v in batch.items()}
    out["inputs"] = torch.cat([x, masked])
    out["labels"] = torch.cat([batch["labels"], xs])
    out["puzzle_identifiers"][x.shape[0]:] = shared_id
    out["row_weight"] = torch.cat([torch.ones(x.shape[0]), torch.full((k * x.shape[0],), 1.0 / k)]).to(x.device)
    return out


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
    """One (key, support rows, query rows) per task in the batch: query = the rows flagged validation (the task's former
    test pair, see build_miniarc_dataset heldout_tasks), support = its other rows. A task is all the augmented copies of
    it the sampler packed together (batch["group_ids"]), or one puzzle without them. Padded rows (blank id 0), tasks
    missing either kind and query rows of a copy without demos (_sample_batch truncates the batch tail) are skipped."""
    if "validation" not in batch:
        raise ValueError("meta_sgd needs batch['validation']: build the dataset with heldout_tasks > 0 (dataset/build_miniarc_dataset.py).")

    ids, is_query = batch["puzzle_identifiers"], batch["validation"].bool()
    keys = batch["group_ids"] if "group_ids" in batch else ids

    episodes = []
    for key in torch.unique(keys[ids != 0]).tolist():
        rows = (keys == key) & (ids != 0)
        support = rows & ~is_query
        query = rows & is_query & torch.isin(ids, ids[support])  # a copy whose demos the batch tail cut off has no row to adapt
        if support.any() and query.any():
            episodes.append((key, {k: v[support] for k, v in batch.items()}, {k: v[query] for k, v in batch.items()}))

    return episodes  # may be empty: a small per-GPU slice can cut every task's query rows off (meta_sgd_batch skips it)


def _segment(model: nn.Module, overrides: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], carry: Any) -> Tuple[Any, torch.Tensor]:
    """One supervision segment of the loss head with `overrides` substituted: (new carry, loss summed over the rows). carry
    None starts every row afresh; rows the carry marks halted restart, as in train_batch."""
    carry, loss, _metrics, _, _ = torch.func.functional_call(model, overrides, args=(), kwargs=dict(return_keys=[], carry=carry, batch=batch))
    return carry, loss


def _supervised_loss(model: nn.Module, overrides: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The loss head's loss on `batch` with `overrides` substituted, summed over each row's ACT trajectory from a fresh
    carry: a row contributes every segment until the model halts it, then drops out. The carry is detached between
    segments (deep supervision)."""
    carry, total = None, 0.0
    while batch["inputs"].shape[0] > 0:
        carry, loss = _segment(model, overrides, batch, carry)
        total = total + loss

        keep = ~carry.halted
        carry = apply_to_tensors(carry, lambda t: t[keep].detach().clone())
        batch = {k: v[keep] for k, v in batch.items()}
    return total


_INNER = "model.inner."  # the multi-segment (TRM, GRAM) inner network under the loss head


def _final_ce(model: nn.Module, overrides: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], segments: int) -> torch.Tensor:
    """The final prediction's CE on `batch`, per token as the loss head normalises, summed over the rows: `segments` calls
    of the inner network from its reset state, no ACT halting, the state detached between segments."""
    from models.recursive_reasoning.merged_arch import LatentCarry

    inner = model.model.inner
    inner_overrides = {k[len(_INNER):]: v for k, v in overrides.items() if k.startswith(_INNER)}
    rows, device = batch["inputs"].shape[0], batch["inputs"].device

    carry = inner.reset_carry(torch.ones(rows, dtype=torch.bool, device=device), inner.empty_carry(rows, device=device), None)
    z_H, z_L = carry.z_H, carry.z_L
    for t in range(segments):
        if t > 0:
            z_H, z_L = z_H.detach(), z_L.detach()
        state = LatentCarry(z_H.to(inner.forward_dtype), z_L.to(inner.forward_dtype))
        carry, info = torch.func.functional_call(inner, inner_overrides, (state, batch))
        z_H, z_L = carry.z_H, carry.z_L

    mask = batch["labels"] != IGNORE_LABEL_ID
    ce = model.loss_fn(info["logits"], batch["labels"], valid_mask=mask) / mask.sum(-1, keepdim=True).clamp_min(1)
    return ce.sum()


def _inner_update(
    model: nn.Module, named: List[Tuple[str, torch.Tensor]], overrides: Dict[str, torch.Tensor], loss: torch.Tensor, step: int,
    recon: bool = False,
) -> Dict[str, torch.Tensor]:
    """theta - alpha * (gradient of loss / its RMS) for every tensor in `named`, alpha the step's per tensor (recon: the
    reconstruction step's). First order: the gradient enters as a constant, so the outer loss reaches theta and alpha but
    not through the gradient; model.meta_sgd.second_order (meta-training with meta_sgd.second_order) keeps its graph."""
    second_order = getattr(model.meta_sgd, "second_order", False) and not recon
    grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True, create_graph=second_order)
    alpha = model.meta_sgd.recon_alpha if recon else model.meta_sgd.alpha

    theta = dict(overrides)
    for (name, p), g in zip(named, grads):
        if g is not None:
            g = g * torch.rsqrt(g.pow(2).mean() + 1e-16)
            theta[name] = p - (alpha(name, step) if recon else _hd_alpha(model, name, g, alpha(name, step))) * g
    return theta


def _hd_alpha(model: nn.Module, name: str, g: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """meta_sgd.hd_beta > 0 (multiplicative hypergradient descent, arXiv 1703.04782): each tensor's step size is scaled online
    by a running multiplier, x (1 + hd_beta * cos(g_t, g_{t-1})), growing while consecutive gradients agree and shrinking
    when they flip; the state (model.meta_sgd.hd_state: previous gradient and multiplier per tensor) restarts per task."""
    beta = getattr(model.meta_sgd, "hd_beta", 0.0)
    if not beta:
        return alpha
    g = g.detach()
    prev = model.meta_sgd.hd_state.get(name)
    mult = 1.0 if prev is None else prev[1] * (1 + beta * float((g * prev[0]).sum() / (g.norm() * prev[0].norm() + 1e-12)))
    model.meta_sgd.hd_state[name] = (g, mult)
    return alpha * mult


def _recon_loss(model: nn.Module, overrides: Dict[str, torch.Tensor], rows: Dict[str, torch.Tensor], segments: int) -> torch.Tensor:
    """The reconstruction steps' loss on masked rows (labels = the inputs), summed over rows: the final prediction's CE after
    `segments` segments on a multi-segment model (TRM, GRAM); on a one-pass model (VARC, no latent carry to reset) the loss
    head's CE of its single forward pass, which is its final prediction."""
    if hasattr(model.model.inner, "reset_carry"):
        return _final_ce(model, overrides, rows, segments)
    return _supervised_loss(model, overrides, rows)


def _recon_steps(
    model: nn.Module, named: List[Tuple[str, torch.Tensor]], theta: Dict[str, torch.Tensor], rows: Dict[str, torch.Tensor],
    recon: Any, shared_id: int, segments: int, steps: int, step0: int = 0, trained: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """meta_sgd.recon_steps' phase: `steps` steps on the final-prediction reconstruction CE of masked copies of rows' inputs
    (fresh masks every step, on the shared row), at the learned reconstruction step sizes, step0 + k of them; trained (fewer
    learned sizes than steps) stretches them over the loop."""
    n = rows["inputs"].shape[0]
    plain = {k: rows[k] for k in ("inputs", "labels", "puzzle_identifiers")}
    for k in range(steps):
        masked = {key: v[n:] for key, v in _with_recon_rows(plain, recon, model.meta_sgd.mask_token, shared_id).items() if key in plain}
        loss = _recon_loss(model, theta, masked, segments) / n
        theta = _inner_update(model, named, theta, loss, step0 + (k * trained // steps if trained else k), recon=True)
        named = [(name, theta.get(name, p)) for name, p in named]
    return theta


def _step_kernel(model: nn.Module):
    """A first-order inner step's graph is not differentiated again, so it may use the fast attention kernels; a
    second-order one needs the math kernel (the only one with a double backward)."""
    if getattr(model.meta_sgd, "second_order", False):
        return sdpa_kernel(SDPBackend.MATH)
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])


def _adapt(
    model: nn.Module, named: List[Tuple[str, torch.Tensor]], overrides: Dict[str, torch.Tensor], support: Dict[str, torch.Tensor], step: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One inner step on every support row's whole trajectory: (per-row support loss, the adapted tensors)."""
    loss = _supervised_loss(model, overrides, support) / support["inputs"].shape[0]
    return loss, _inner_update(model, named, overrides, loss, step)


def _adapt_segment(
    model: nn.Module, named: List[Tuple[str, torch.Tensor]], overrides: Dict[str, torch.Tensor], support: Dict[str, torch.Tensor],
    carry: Any, step: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Any]:
    """One inner step on one supervision segment, the rows continuing `carry` (None: a fresh start), as train_batch does:
    (per-row support loss, the adapted tensors, the next step's carry, detached)."""
    carry, loss = _segment(model, overrides, support, carry)
    loss = loss / support["inputs"].shape[0]
    return loss, _inner_update(model, named, overrides, loss, step), apply_to_tensors(carry, lambda t: t.detach().clone())


def _msl_weights(steps: int, progress: float) -> List[float]:
    """Multi-step loss weights (arXiv 1810.09502): uniform over the inner steps at progress 0, linearly to the last step only
    at progress 1 (and beyond)."""
    early = (1.0 - min(max(progress, 0.0), 1.0)) / steps
    return [early] * (steps - 1) + [1.0 - early * (steps - 1)]


def _episode_overrides(
    model: nn.Module, targets: List[Tuple[str, torch.Tensor]], emb_name: Optional[str], batch: Dict[str, torch.Tensor],
    extra_ids: Optional[torch.Tensor] = None,
) -> Tuple[List[Tuple[str, torch.Tensor]], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """(targets with the embedding rows as a leaf, functional_call overrides, batch with its ids remapped, the ids). The rows
    of the table the batch uses stand in for the whole table, so they are a differentiable leaf and the forward takes its
    gather path (the row count differs from the emptied local_weights buffer). No embeddings: the inputs pass through.
    extra_ids: rows to include although the batch does not use them (config.recon's shared row)."""
    if emb_name is None:
        return list(targets), {}, batch, None

    table = model.get_buffer(emb_name)
    pids = batch["puzzle_identifiers"]
    ids = torch.unique(pids if extra_ids is None else torch.cat([pids, extra_ids.to(pids)]))
    emb_rows = table[ids].detach().requires_grad_()

    overrides = {emb_name: emb_rows, emb_name[: -len("weights")] + "local_weights": table.new_zeros(0, table.shape[1])}
    named = [(n, emb_rows if n == emb_name else p) for n, p in targets]
    batch = {**batch, "puzzle_identifiers": torch.searchsorted(ids, pids).to(pids.dtype)}
    return named, overrides, batch, ids


def _task_outer_backward(
    config: PretrainConfig, model: nn.Module, targets: List[Tuple[str, torch.Tensor]], emb_name: Optional[str],
    support: Dict[str, torch.Tensor], query: Dict[str, torch.Tensor], loss_scale: torch.Tensor, msl_weights: List[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Meta-SGD on one task (MetaSGDConfig): the inner steps on its support rows, then the query losses after them
    (weighted by msl_weights, times loss_scale) backpropagated into theta and the alphas. Returns (support loss per row
    before the first step, query loss per row after the last)."""
    cfg = config.meta_sgd

    # config.recon: the inner steps also reconstruct masked copies of the demo inputs, on the shared row, as pretrained
    shared = None
    if config.recon.copies:
        shared_id = model.get_buffer(emb_name).shape[0] - 1
        support = _with_recon_rows(support, config.recon, model.meta_sgd.mask_token, shared_id)
        shared = torch.tensor([shared_id], device=support["inputs"].device)

    named, overrides, support, ids = _episode_overrides(model, targets, emb_name, support, extra_ids=shared)

    # The task's rows start from the table mean, as a held-out task's do at test time (the shared row keeps its value)
    if emb_name is not None:
        table = model.get_buffer(emb_name)
        rows = table.mean(0, keepdim=True).expand(ids.numel(), -1).clone()
        if shared is not None:
            rows[ids == shared] = table[shared]
        rows.requires_grad_()

        overrides[emb_name] = rows
        named = [(n, rows if n == emb_name else p) for n, p in named]
        query = {**query, "puzzle_identifiers": torch.searchsorted(ids, query["puzzle_identifiers"]).to(query["puzzle_identifiers"].dtype)}

    theta, carry, total, first_loss = overrides, None, 0.0, None
    if cfg.rows_steps and emb_name is not None:  # the embedding first: rows-only steps, then the backbone with the rows frozen
        rows_named = [(n, p) for n, p in named if n == emb_name]
        second_order, model.meta_sgd.second_order = getattr(model.meta_sgd, "second_order", False), False  # rows phase first order: its graph would not fit (H200)
        for r in range(cfg.rows_steps):
            with _step_kernel(model):
                if cfg.segment_inner_steps:
                    loss_r, theta, carry = _adapt_segment(model, rows_named, theta, support, carry, r)
                else:
                    loss_r, theta = _adapt(model, rows_named, theta, support, r)
            first_loss = loss_r if first_loss is None else first_loss
            rows_named = [(n, theta[n]) for n, _ in rows_named]
        named, carry = [(n, p) for n, p in named if n != emb_name], None
        model.meta_sgd.second_order = second_order
    second_order = getattr(model.meta_sgd, "second_order", False)
    first_second_order = 0 if cfg.second_order_steps is None else len(msl_weights) - cfg.second_order_steps
    for k, w in enumerate(msl_weights):
        model.meta_sgd.second_order = second_order and k >= first_second_order  # truncated: only the last steps second order
        with _step_kernel(model):
            if cfg.segment_inner_steps:
                loss_s, theta, carry = _adapt_segment(model, named, theta, support, carry, k)
            else:
                loss_s, theta = _adapt(model, named, theta, support, k)
        first_loss = loss_s if first_loss is None else first_loss
        named = [(n, theta.get(n, p)) for n, p in named]  # the next step differentiates the adapted tensors

        if w > 0:  # the last step's weight is always > 0
            if cfg.recon_steps and k == len(msl_weights) - 1:  # stage 2: the reconstruction steps, on the query's masked inputs
                local = int(torch.searchsorted(ids, shared))
                theta = _recon_steps(model, named, theta, query, config.recon, local, config.arch.halt_max_steps, cfg.recon_steps)
            if cfg.segment_inner_steps:
                loss_q = _final_ce(model, theta, query, config.arch.halt_max_steps)
            else:
                loss_q = _supervised_loss(model, theta, query)
            total = total + w * loss_q

    model.meta_sgd.second_order = second_order
    (total * loss_scale).backward()
    return first_loss.detach(), loss_q.detach() / query["inputs"].shape[0]


def _gauss_newton_product(model: nn.Module, overrides: Dict[str, torch.Tensor], leaves: Dict[str, torch.Tensor], names: List[str],
                          batch: Dict[str, torch.Tensor], n_rows: int):
    """meta_sgd.imaml_curvature=gauss_newton: v -> G v, G = sum over the support rows' segments of J^T H_z J, in the tensors
    `names`, z each segment's logits (the whole ACT trajectory, as _supervised_loss: rows run until they halt, the carry
    detached between segments), H_z = w (diag(p) - p p^T) per labelled token, w the loss head's weight (1 / the row's
    labelled tokens / n_rows), p the probabilities: positive semi-definite, so CG converges. With softmax CE (VARC) this is
    the exact Gauss-Newton matrix of the LM loss; with stablemax CE (TRM, GRAM) p is stablemax's and H_z its softmax-form,
    Fisher-style curvature. The loss head's other terms (q_halt, GRAM's KL / LPRM) are left out of G. J v by the double-
    backward trick, J^T u by a vector-Jacobian product."""
    inner = {k[len("model."):]: v for k, v in {**overrides, **leaves}.items()}
    stablemax = getattr(model, "loss_fn", None) is stablemax_cross_entropy
    params = [leaves[n] for n in names]
    segs, carry = [], None
    while batch["inputs"].shape[0] > 0:
        carry, out = torch.func.functional_call(model.model, inner, args=(), kwargs=dict(carry=carry, batch=batch))
        logits = out["logits"].float()
        labels = carry.current_data["labels"] if hasattr(carry, "current_data") else batch["labels"]
        mask = labels != IGNORE_LABEL_ID
        weight = (mask / mask.sum(-1, keepdim=True).clamp_min(1) / n_rows).unsqueeze(-1)
        if stablemax:
            s = torch.where(logits.detach() < 0, 1 / (1 - logits.detach()), logits.detach() + 1)
            prob = s / s.sum(-1, keepdim=True)
        else:
            prob = torch.softmax(logits.detach(), -1)
        segs.append((logits, weight, prob))
        halted = getattr(carry, "halted", None)
        if halted is None or bool(halted.all()):  # one-pass model, or every row halted
            break
        keep = ~halted
        carry = apply_to_tensors(carry, lambda t: t[keep].detach().clone())
        batch = {k: v[keep] for k, v in batch.items()}
    dummies = [torch.zeros_like(z, requires_grad=True) for z, _, _ in segs]
    jt_dummy = torch.autograd.grad([z for z, _, _ in segs], params, grad_outputs=dummies, create_graph=True, allow_unused=True)
    used = [i for i, g in enumerate(jt_dummy) if g is not None]

    def product(vs: List[torch.Tensor]) -> List[torch.Tensor]:
        jvs = torch.autograd.grad([jt_dummy[i] for i in used], dummies, grad_outputs=[vs[i] for i in used], retain_graph=True)
        hz = [w * (p * jv - p * (p * jv).sum(-1, keepdim=True)) for (_, w, p), jv in zip(segs, jvs)]
        gv = torch.autograd.grad([z for z, _, _ in segs], params, grad_outputs=hz, retain_graph=True, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g for p, g in zip(params, gv)]
    return product


def _imaml_task_backward(
    config: PretrainConfig, model: nn.Module, targets: List[Tuple[str, torch.Tensor]], emb_name: Optional[str],
    support: Dict[str, torch.Tensor], query: Dict[str, torch.Tensor], loss_scale: torch.Tensor,
    regu_coef: float = 1.0, lam_damping: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """meta_sgd.imaml (iMAML, arXiv 1909.04630; reference aravindr93/imaml_dev) on one task. The rows start fresh (the table
    mean) and, with rows_steps, take that many rows-only steps first and are then frozen (part of the task's data). Then
    inner_steps normalised steps (each tensor's alpha, hypergradient descent with hd_beta; one supervision segment per step
    with segment_inner_steps, the rows continuing their carry) on the per-row support loss + lam / 2 ||phi - theta||^2, no
    graph, so phi approaches the proximal problem's minimiser that the implicit gradient assumes. theta's meta-gradient
    A^-1 grad L_query(phi), A = (1 + regu_coef) I + H / (lam + lam_damping) (the reference's damped matrix_evaluator), H the
    full-trajectory support loss's Hessian at phi, by cg_steps conjugate-gradient iterations on Hessian-vector products
    (truncated at negative curvature), is added (times loss_scale) to theta's .grad. Logs meta/imaml_stationarity (the
    proximal objective's gradient norm at phi over the support gradient's at theta) and meta/imaml_cg_residual. Returns
    (support loss per row at theta, query loss per row at phi)."""
    cfg = config.meta_sgd
    lam = cfg.imaml_lambda
    named, overrides, support, ids = _episode_overrides(model, targets, emb_name, support)
    if emb_name is not None:
        table = model.get_buffer(emb_name)
        overrides[emb_name] = table.mean(0, keepdim=True).expand(ids.numel(), -1).clone()
        named = [(n, overrides[emb_name] if n == emb_name else p) for n, p in named]
        query = {**query, "puzzle_identifiers": torch.searchsorted(ids, query["puzzle_identifiers"]).to(query["puzzle_identifiers"].dtype)}
    names = [n for n, _ in named]
    anchor = {n: p.detach() for n, p in named}
    phi = dict(anchor)
    n_support, n_query = support["inputs"].shape[0], query["inputs"].shape[0]
    model.meta_sgd.hd_state = {}  # hypergradient descent restarts with each task, as at test time

    def support_loss(leaves):
        return _supervised_loss(model, {**overrides, **leaves}, support) / n_support

    def step(grads):  # the normalised inner step at each tensor's alpha (its first: iMAML does not learn them), as test time's
        with torch.no_grad():
            for n, g in zip(names, grads):
                if g is not None:
                    g = g * torch.rsqrt(g.pow(2).mean() + 1e-16)
                    phi[n] = phi[n] - _hd_alpha(model, n, g, model.meta_sgd.alpha(n, 0)) * g

    if cfg.rows_steps and emb_name is not None:  # the embedding first: rows-only steps on the fresh rows, then frozen
        for _ in range(cfg.rows_steps):
            rows = phi[emb_name].clone().requires_grad_()
            g = torch.autograd.grad(support_loss({**phi, emb_name: rows}), [rows])[0]
            g = g * torch.rsqrt(g.pow(2).mean() + 1e-16)
            phi[emb_name] = (rows - _hd_alpha(model, emb_name, g, model.meta_sgd.alpha(emb_name, 0)) * g).detach()
        overrides[emb_name] = phi.pop(emb_name)
        anchor.pop(emb_name)
        names = [n for n in names if n != emb_name]

    loss_s0, grad0, carry = None, None, None
    if cfg.segment_inner_steps:  # the stationarity's reference: the full-trajectory support gradient at theta (a segment step's is one segment's)
        at_theta = {n: phi[n].clone().requires_grad_() for n in names}
        grad0 = torch.sqrt(sum(g.pow(2).sum() for g in torch.autograd.grad(support_loss(at_theta), list(at_theta.values()), allow_unused=True) if g is not None))
    for _ in range(cfg.inner_steps):
        leaves = {n: phi[n].clone().requires_grad_() for n in names}
        with _step_kernel(model):
            if cfg.segment_inner_steps:  # TRM / GRAM: one supervision segment per step, as their test-time steps
                carry, loss = _segment(model, {**overrides, **leaves}, support, carry)
                carry, loss = apply_to_tensors(carry, lambda t: t.detach().clone()), loss / n_support
            else:
                loss = support_loss(leaves)
        loss_s0 = loss.detach() if loss_s0 is None else loss_s0
        grads = torch.autograd.grad(loss, [leaves[n] for n in names], allow_unused=True)
        grad0 = torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None)) if grad0 is None else grad0
        step([None if g is None else g + lam * (phi[n] - anchor[n]) for n, g in zip(names, grads)])

    leaves = {n: phi[n].clone().requires_grad_() for n in names}
    with sdpa_kernel(SDPBackend.MATH):  # Hessian-vector products need a double backward
        loss_q = _supervised_loss(model, {**overrides, **leaves}, query)
        g_query = [torch.zeros_like(leaves[n]) if g is None else g
                   for n, g in zip(names, torch.autograd.grad(loss_q, [leaves[n] for n in names], allow_unused=True))]
        g_support = torch.autograd.grad(support_loss(leaves), [leaves[n] for n in names], create_graph=True, allow_unused=True)
        live = [i for i, g in enumerate(g_support) if g is not None]
        curvature_product = _gauss_newton_product(model, overrides, leaves, [names[i] for i in live], support, n_support) \
            if cfg.imaml_curvature == "gauss_newton" else \
            (lambda vs: torch.autograd.grad([g_support[i] for i in live], [leaves[names[i]] for i in live], grad_outputs=vs, retain_graph=True))

        def apply(v):
            hv = curvature_product([v[i] for i in live])
            out = [(1 + regu_coef) * x for x in v]
            for j, i in enumerate(live):
                out[i] = out[i] + hv[j] / (lam + lam_damping)
            return out

        dot = lambda a, b: sum((x * y).sum() for x, y in zip(a, b))
        x = [torch.zeros_like(g) for g in g_query]
        r = [g.clone() for g in g_query]
        p_dir = [g.clone() for g in r]
        rs = dot(r, r)
        for it in range(cfg.imaml_cg_steps):
            ap = apply(p_dir)
            curvature = dot(p_dir, ap)
            if not torch.isfinite(curvature) or curvature <= 0:  # truncated CG: A is not positive definite along p
                if it == 0:
                    x = [g.clone() for g in g_query]
                break
            a = rs / curvature
            x = [xi + a * pi for xi, pi in zip(x, p_dir)]
            r = [ri - a * api for ri, api in zip(r, ap)]
            rs_new = dot(r, r)
            p_dir = [ri + (rs_new / rs.clamp_min(1e-20)) * pi for ri, pi in zip(r, p_dir)]
            rs = rs_new
    with torch.no_grad():
        stationary = torch.sqrt(sum((g_support[i].detach() + lam * (phi[names[i]] - anchor[names[i]])).pow(2).sum() for i in live))
        model.meta_sgd.imaml_diag.append(torch.stack([stationary / grad0.clamp_min(1e-12), torch.sqrt(rs) / torch.sqrt(dot(g_query, g_query)).clamp_min(1e-12)]))
        params = dict(model.named_parameters())
        for n, g in zip(names, x):
            if n in params and params[n].requires_grad:
                params[n].grad = g * loss_scale if params[n].grad is None else params[n].grad + g * loss_scale
    return loss_s0, loss_q.detach() / n_query


def _noise_scale_terms(local_grad: torch.Tensor, episode_sq: List[torch.Tensor], inv_rows: List[float], n_query: torch.Tensor, world_size: int) -> Dict[str, torch.Tensor]:
    """The gradient noise scale's two terms (McCandlish et al. 2018, app. A) from one meta-update: each episode's mean
    gradient over its q query rows has E|g|^2 = |G|^2 + S/q, the update's over all N rows |G|^2 + S/N; solved for |G|^2
    and S = tr(Sigma). B_noise = S / |G|^2 query rows: average both over updates before dividing (each is noisy)."""
    device = local_grad.device
    stats = torch.stack([torch.stack(episode_sq).sum() if episode_sq else torch.zeros((), device=device),
                         torch.tensor(float(sum(inv_rows)), device=device), torch.tensor(float(len(inv_rows)), device=device)])
    if world_size > 1:
        dist.all_reduce(local_grad)
        dist.all_reduce(stats)
    big_sq = local_grad.square().sum()
    small_sq, inv_small = stats[0] / stats[2].clamp_min(1), stats[1] / stats[2].clamp_min(1)
    trace = (small_sq - big_sq) / (inv_small - 1.0 / n_query)
    return {"meta/noise_trace": trace, "meta/noise_grad_sq": big_sq - trace / n_query}


def meta_sgd_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
):
    """train_batch's counterpart with config.meta_sgd.enabled: every task in the batch (_episodes_from_batch) gets
    _task_outer_backward, the query loss averaged over every rank's query rows, then the shared optimizer step. No
    persistent carry: every loss starts its rows afresh. The puzzle-embedding table is not trained (fresh rows).
    `batch` may be a list (meta_sgd.accum_steps loader batches): all their episodes go into the one optimizer step."""
    train_state.step += 1
    if train_state.step >= train_state.total_steps:
        return

    model, cfg = train_state.model, config.meta_sgd
    model.meta_sgd.second_order = cfg.second_order  # meta-training only: test time (evaluate_ttt) keeps first-order steps
    device = next(model.parameters()).device
    current_epoch = (train_state.step - 1) // max(train_state.steps_per_epoch, 1)

    episodes = [e for b in (batch if isinstance(batch, list) else [batch]) for e in _episodes_from_batch(tree_to_device(b, device))]
    n_query = torch.tensor(float(sum(q["inputs"].shape[0] for _, _, q in episodes)), device=device)
    if world_size > 1:
        dist.all_reduce(n_query)  # the scale makes the all-reduced gradient the mean over every rank's query rows
    n_query = n_query.clamp_min(1)

    emb_name = _puzzle_emb_name(model)
    targets = _meta_sgd_targets(model)
    progress = 1.0 if cfg.msl_anneal_fraction is None else train_state.step / (cfg.msl_anneal_fraction * train_state.total_steps)
    msl_weights = _msl_weights(cfg.inner_steps, progress)

    measure_flops = _should_log_flops(config, train_state.step)
    flop_counter = TotalFlopCounter(display=False) if measure_flops else nullcontext()
    support_losses, query_losses = [], []
    params = [p for p in model.parameters() if p.requires_grad]
    flat_grad = lambda: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten() for p in params])
    episode_sq, inv_rows = [], []  # cfg.noise_scale: each episode's |mean query-row gradient|^2 and 1 / its query rows
    with flop_counter:
        for _key, support, query in episodes:
            before = flat_grad() if cfg.noise_scale else None
            if cfg.imaml:
                loss_s, loss_q = _imaml_task_backward(config, model, targets, emb_name, support, query, 1.0 / n_query)
            else:
                with sdpa_kernel(SDPBackend.MATH) if cfg.second_order else nullcontext():  # only the math kernel has a double backward
                    loss_s, loss_q = _task_outer_backward(config, model, targets, emb_name, support, query, 1.0 / n_query, msl_weights)
            support_losses.append(loss_s)
            query_losses.append(loss_q)
            if cfg.noise_scale:
                rows = query["inputs"].shape[0]
                episode_sq.append(((flat_grad() - before) * (n_query / rows)).square().sum())
                inv_rows.append(1.0 / rows)

    alpha_mean, alpha_max = model.meta_sgd.summary()
    tasks = torch.tensor(float(len(episodes)), device=device)
    metrics = {
        "count": tasks,
        "meta/tasks": tasks,
        "meta/support_loss": torch.stack(support_losses).mean() if episodes else torch.zeros((), device=device),
        "meta/query_loss": torch.stack(query_losses).mean() if episodes else torch.zeros((), device=device),
        "meta/alpha_mean": torch.tensor(alpha_mean, device=device),
        "meta/alpha_max": torch.tensor(alpha_max, device=device),
    }
    if model.meta_sgd.imaml_diag:  # iMAML: the inner loop's stationarity and the CG residual (the implicit gradient's two assumptions)
        diag = torch.stack(model.meta_sgd.imaml_diag).mean(0)
        metrics["meta/imaml_stationarity"], metrics["meta/imaml_cg_residual"] = diag[0], diag[1]
        model.meta_sgd.imaml_diag = []
    if model.meta_sgd.recon_log_scale is not None:  # stage 2: the learned reconstruction scales
        metrics["meta/recon_scale_mean"] = model.meta_sgd.recon_log_scale.detach().exp().mean()
    if cfg.noise_scale:
        metrics.update(_noise_scale_terms(flat_grad(), episode_sq, inv_rows, n_query, world_size))

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
        copies_per_group=_copies_per_group(config),
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

    if "DISABLE_COMPILE" not in os.environ and not config.meta_sgd.enabled:  # the meta step runs functional_call eagerly
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
        pending: List[Any] = []  # meta_sgd.accum_steps: the loader batches gathered for the next optimizer step
        for i, (set_name, batch, global_batch_size) in enumerate(train_loader):
            # global_batch_size stays the ordinary rows': the masked copies add to their loss
            if config.recon.copies and not config.meta_sgd.enabled:
                batch = _with_recon_rows(batch, config.recon, train_metadata.vocab_size, train_metadata.num_puzzle_identifiers)

            if config.meta_sgd.enabled and config.meta_sgd.accum_steps > 1:  # gather the batches of one optimizer step
                pending.append(batch)
                if len(pending) < config.meta_sgd.accum_steps:
                    continue
                batch, pending = pending, []

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
