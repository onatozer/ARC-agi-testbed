"""Per-task test-time training (TTT) for any ARC checkpoint, then evaluation with the checkpoint's own evaluators.

For every eval task, starting from the checkpoint's weights (EMA ones if it has them):
  1. build a one-task dataset with dataset/build_arc_dataset.py: the task plus (num_aux_tasks - 1) dihedral/color
     augmentations, each with its own puzzle identifier, demos in the train split and test pairs in the test split;
  2. train on it with pretrain.py's train_batch for `epochs` passes over every aux task's demos;
  3. evaluate num_views times with evaluators/eval_fn.evaluate. The evaluators vote across the aux tasks (after undoing
     each augmentation), the views and the ttt_runs_per_task runs, since ARC's aggregated_voting pools them.

    python evaluate_ttt.py checkpoint=<run_dir>/checkpoints/step_N_<id>.pth [eval=ttt_varc] [epochs=50 train_overrides.lr=1e-4 ...]
    torchrun --nproc_per_node=<gpus> evaluate_ttt.py checkpoint=...
    # Embeddings first, then weights (arXiv 2609.21181), with ttt_varc's 100 epochs; any N alternates every N epochs
    python evaluate_ttt.py checkpoint=... train_overrides.freeze_alternation_epochs=50

Tasks are sharded across ranks with no process group, since tasks are independent and finish at different times.
Each task writes <output_dir>/tasks/<name>.json (existing ones are skipped, so a run can be resumed), and whichever rank
finishes last writes the averaged metrics and the merged submission and logs them to the checkpoint's W&B run.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import shutil
import time
import zlib

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from config.schema import PretrainConfig
from dataset.build_arc_dataset import DataProcessConfig, convert_dataset, convert_single_arc_puzzle
from dataset.build_miniarc_dataset import MiniARCDataProcessConfig
from evaluate import _apply_overrides, _attach_wandb_run, _dataset_path, _load_checkpoint, _seed, _summarize
from evaluators.eval_config import TTTConfig
from evaluators.eval_fn import evaluate
from pretrain import _maml_release, _maml_targets, create_dataloader, create_evaluators, init_train_state, maml_adapt, train_batch
from utils.checkpoint import load_model_state_dict
from utils.functions import load_model_class
from utils.printing import print_info
from utils.training import apply_to_tensors, tree_to_device


torch.set_float32_matmul_precision("high")


def load_tasks(builder: Dict[str, Any], data_path: str) -> Dict[str, dict]:
    """The eval tasks with their test outputs: ARC-AGI challenges with the solutions merged in when the checkpoint was
    trained on an ARC-AGI build, otherwise the build's test_puzzles.json (MiniARC: the evaluators' tasks, outputs included)."""
    prefix, set_name = builder.get("input_file_prefix"), builder.get("test_set_name")
    if not prefix or not set_name:
        with open(os.path.join(data_path, "test_puzzles.json")) as f:
            return json.load(f)
    with open(f"{prefix}_{set_name}_challenges.json") as f:
        tasks = json.load(f)
    with open(f"{prefix}_{set_name}_solutions.json") as f:
        solutions = json.load(f)
    for name, task in tasks.items():
        for pair, output in zip(task["test"], solutions[name]):
            pair["output"] = output
    return tasks


def build_task_dataset(builder: Dict[str, Any], name: str, task: dict, output_dir: str, num_aux_tasks: int, seed: int) -> None:
    """One task as a dataset in the checkpoint's format (grid_size, scale_aug, border_tokens, ... come from its build config,
    ARC-AGI's or MiniARC's). Every aux task gets its own group, so a sampler epoch covers each aux task's demos once."""
    config_cls = DataProcessConfig if "input_file_prefix" in builder else MiniARCDataProcessConfig
    fields = {k: v for k, v in builder.items() if k in config_cls.model_fields}
    config = config_cls(**{**fields, "output_dir": output_dir, "num_aug": num_aux_tasks - 1, "test_aug": True, "seed": seed})

    def load_puzzles(config):
        results: Dict[str, Dict[str, List[list]]] = {}
        convert_single_arc_puzzle(results, name, task, config.num_aug, {"train": ("train", "all"), "test": ("test", "all")}, test_aug=True)
        results["train"]["all"] = [[puzzle] for group in results["train"]["all"] for puzzle in group]
        return results, {name: task}

    convert_dataset(config, load_puzzles=load_puzzles)


def start_state(ckpt: Dict[str, Any], train_cfg: PretrainConfig) -> Dict[str, torch.Tensor]:
    """The checkpoint's weights, with the EMA ones swapped in when it was trained with EMA (as evaluate.py evaluates)."""
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    if train_cfg.ema and ckpt.get("ema") is not None:
        state.update({k.replace("_orig_mod.", ""): v for k, v in ckpt["ema"]["shadow"].items() if "puzzle_emb.local_" not in k})
    return state


def _reinit_state(state: Dict[str, torch.Tensor], substrings: List[str]) -> Dict[str, torch.Tensor]:
    """The checkpoint without the keys containing one of substrings: those modules keep the fresh model's own init."""
    for s in substrings:
        if not any(s in k for k in state):
            raise ValueError(f"reinit_substrings entry {s!r} matches no checkpoint key.")
    return {k: v for k, v in state.items() if not any(s in k for s in substrings)}


def _freeze_except(model: torch.nn.Module, substrings: List[str]) -> int:
    """Only the parameters whose name contains one of substrings keep requires_grad (the optimizers skip the others;
    the puzzle-embedding table is a buffer and is not affected). Empty: everything trains. Returns the trainable count."""
    trainable = 0
    for name, p in model.named_parameters():
        keep = not substrings or any(s in name for s in substrings)
        p.requires_grad_(keep)
        trainable += p.numel() if keep else 0
    if trainable == 0:
        raise ValueError(f"train_only_substrings {substrings} match no parameter.")
    return trainable


def _lprm_phase(config: PretrainConfig, train_state: Any, eval_loader: Any, eval_metadata: Any, passes: int) -> Tuple[float, float]:
    """Label-free TTT on the task's test inputs: `passes` passes over them, every supervision step raising the model's
    LPRM value (its own prediction of its correctness, GRAM's v_head) by gradient on the trainable parameters. The
    model runs in eval mode (the prior's guidance, no posterior, no KL: labels unused) at the optimizers' base lr.
    Returns the mean value at the last supervision step of the first and the last batch."""
    model = train_state.model.model
    model.eval()
    device = next(model.parameters()).device
    values: List[float] = []
    for _ in range(passes):
        for _set_name, batch, _n in eval_loader:
            batch = tree_to_device(batch, device)
            if batch["inputs"].shape[0] == 0:
                continue
            mask = (batch["puzzle_identifiers"] != eval_metadata.blank_identifier_id).float()  # padded rows
            carry = None
            with torch.enable_grad():
                for _ in range(model.config.halt_max_steps):
                    carry, out = model(carry, batch)
                    if "lprm_value" not in out:
                        raise ValueError(f"lprm_steps needs a model with an LPRM value; {config.arch.name} has none.")
                    value = out["lprm_value"].float()
                    (-(torch.log(value.clamp_min(1e-6)) * mask).sum() / mask.sum().clamp_min(1)).backward()
                    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
                        for group in optim.param_groups:
                            group["lr"] = base_lr
                        optim.step()
                        optim.zero_grad()
                    carry = apply_to_tensors(carry, torch.Tensor.detach)  # the graph never spans supervision steps
            values.append(float((value.detach() * mask).sum() / mask.sum().clamp_min(1)))
    return values[0], values[-1]


def _support_batch(config: PretrainConfig, train_metadata: Any) -> Dict[str, torch.Tensor]:
    """Every row of the task's train split (the aux copies' demos) as one batch, in order and without the sampler's
    dropped tails: a sequential (test-mode) pass over the split, minus its padding rows."""
    loader, _ = create_dataloader(config, "train", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1)
    rows = {k: torch.cat([b[k] for _, b, _ in loader]) for k in ("inputs", "labels", "puzzle_identifiers")}
    keep = rows["puzzle_identifiers"] != train_metadata.blank_identifier_id
    if not keep.any():
        raise ValueError("TTT support set is empty: the task's train split has no demos.")
    return {k: v[keep] for k, v in rows.items()}


def ttt_task(cfg: TTTConfig, train_cfg: PretrainConfig, state: Dict[str, torch.Tensor], data_dir: Path, eval_dir: Path, seed: int) -> Dict[str, float]:
    config = train_cfg.model_copy(deep=True)
    _apply_overrides(config, cfg.train_overrides)
    config = PretrainConfig.model_validate(config.model_dump())  # nested overrides (+train_overrides.maml.inner_lr=...) arrive as dicts
    config.dataset.data_path = str(data_dir)
    config.epochs = cfg.epochs
    config.max_steps = None
    config.checkpoint_path = None
    config.eval_dir = str(eval_dir)
    config.seed = seed

    train_loader, train_metadata = create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=cfg.epochs, global_batch_size=config.global_batch_size,
        rank=0, world_size=1, enable_shared_sampler_state=False,
    )
    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1,
    )
    # Reused across runs and views so their votes pool, whatever the checkpoint's dataset config set
    evaluators = create_evaluators(config, eval_metadata)
    for evaluator in evaluators:
        if hasattr(evaluator, "aggregated_voting"):
            evaluator.aggregated_voting = True

    metrics: Optional[Dict[str, Any]] = None
    for run in range(cfg.ttt_runs_per_task):
        _seed(seed + run, 0)
        train_state = init_train_state(config, train_metadata, rank=0, world_size=1)
        # The new aux tasks' puzzle embeddings start from the mean of the checkpoint's (load_model_state_dict's resize)
        load_model_state_dict(train_state.model, _reinit_state(dict(state), cfg.reinit_substrings), strict=False)
        trainable = _freeze_except(train_state.model, cfg.train_only_substrings)
        if train_state.ema_helper is not None:
            train_state.ema_helper.register(train_state.model)
        config.lr_warmup_steps = cfg.warmup_epochs * train_state.steps_per_epoch

        train_state.model.train()
        steps, losses = 0, []
        if config.maml.enabled and cfg.maml_steps != 0:
            # MAML's inner loop on every demo of every aux copy at once (one sequential pass over the train split)
            support = _support_batch(config, train_metadata)
            train_state.ema_helper = None  # the EMA weights are already in state; a swap after would undo the adaptation
            steps = cfg.maml_steps or config.maml.inner_steps
            emb = getattr(train_state.model.model, "puzzle_emb", None)
            table = emb.weights.detach().clone() if emb is not None else None
            losses, _ = maml_adapt(train_state.model, tree_to_device(support, next(train_state.model.parameters()).device), steps, config.maml.inner_lr, config.maml.adapt_substrings)
            moved = int((emb.weights != table).any(1).sum()) if emb is not None else 0
            adapted = sum(p.numel() for p in _maml_targets(train_state.model, config.maml.adapt_substrings) if emb is None or p is not emb.weights)
            _maml_release(train_state.model)
            meta_sgd = getattr(train_state.model, "meta_sgd", None)
            lr_text = "learned step sizes, geometric mean %.3g, max %.3g" % meta_sgd.summary() if meta_sgd is not None else f"lr {config.maml.inner_lr}"
            print_info(f"TTT run {run}: {steps} MAML steps ({lr_text}) on {support['inputs'].shape[0]} demos, {adapted} adapted parameters + {moved} embedding rows, support loss {losses[0]:.3f} -> {losses[-1]:.3f}")
        else:
            config.maml.enabled = False  # maml_steps=0 on a MAML checkpoint: plain supervision steps, not meta-steps per batch
            for _set_name, batch, global_batch_size in train_loader:
                step_metrics = train_batch(config, train_state, batch, global_batch_size, rank=0, world_size=1)
                steps += 1
                if step_metrics and "train/lm_loss" in step_metrics:
                    losses.append(float(step_metrics["train/lm_loss"]))
                if train_state.ema_helper is not None:
                    train_state.ema_helper.update(train_state.model)
                if train_state.step >= train_state.total_steps:
                    break
            if steps == 0:
                # The sampler fills a batch with whole aux tasks and drops under-filled tails: too few aux tasks (or demos) for
                # the batch size means no training at all, which would silently evaluate the untouched checkpoint
                raise ValueError(f"TTT trained for 0 steps: {train_metadata.total_groups} aux tasks x {train_metadata.mean_puzzle_examples:.1f} demos never fill global_batch_size={config.global_batch_size}.")
            print_info(f"TTT run {run}: {steps} steps on {trainable} trainable parameters" + (f", train lm_loss {losses[0]:.3f} -> {losses[-1]:.3f}" if losses else ""))
        if cfg.lprm_steps > 0:
            before, after = _lprm_phase(config, train_state, eval_loader, eval_metadata, cfg.lprm_steps)
            print_info(f"TTT run {run}: {cfg.lprm_steps} LPRM passes over the test inputs, mean LPRM value {before:.3f} -> {after:.3f}")

        if train_state.ema_helper is not None:
            train_state.ema_helper.swap(train_state.model)
        eval_head_cls = load_model_class(config.arch.loss.inference_name)
        train_state_eval = replace(train_state, model=eval_head_cls(train_state.model.model, **config.arch.loss.__pydantic_extra__))
        train_state_eval.model.eval()
        for _ in range(cfg.num_views):
            metrics, _ = evaluate(config, train_state_eval, eval_loader, eval_metadata, evaluators, rank=0, world_size=1, cpu_group=None)
        del train_state, train_state_eval
        torch.cuda.empty_cache()

    return _summarize(metrics or {})[1]


def aggregate(names: List[str], out_dir: Path) -> Optional[Dict[str, float]]:
    """Mean of every per-task metric, and the per-task submissions merged, once every task has a result."""
    paths = [out_dir / "tasks" / f"{name}.json" for name in names]
    if not all(p.exists() for p in paths):
        return None
    results = [json.loads(p.read_text())["metrics"] for p in paths]
    keys = sorted({k for r in results for k in r})
    metrics = {k: sum(r[k] for r in results if k in r) / sum(k in r for r in results) for k in keys}
    submission: Dict[str, Any] = {}
    for path in (out_dir / "tasks").glob("*/evaluator_*/submission.json"):
        submission.update(json.loads(path.read_text()))
    write_json(out_dir / "ttt_metrics.json", {"metrics": metrics, "num_tasks": len(names)})
    write_json(out_dir / "submission.json", submission)
    return metrics


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


@hydra.main(config_path="config", config_name="cfg_eval_ttt", version_base=None)
def launch(hydra_config: DictConfig) -> None:
    cfg = TTTConfig(**OmegaConf.to_container(hydra_config, resolve=True))  # type: ignore[arg-type]
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    ckpt, ckpt_path = _load_checkpoint(cfg.checkpoint)
    train_cfg = PretrainConfig(**ckpt["config"])
    builder = dict(train_cfg.dataset.__pydantic_extra__ or {})
    state = start_state(ckpt, train_cfg)
    step = int(ckpt.get("step", 0))
    out_dir = Path(cfg.output_dir or Path(ckpt_path).parent.parent / "eval_preds" / f"step_{step}_{cfg.suffix}")
    (out_dir / "tasks").mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(builder, _dataset_path(train_cfg.dataset.data_path))
    task_ids = cfg.task_ids
    if cfg.task_ids_file:
        with open(cfg.task_ids_file) as f:
            task_ids = json.load(f)  # e.g. a build's heldout_tasks.json
    names = sorted(task_ids or tasks)[: cfg.max_tasks]
    for name in names[rank::world]:
        result_path = out_dir / "tasks" / f"{name}.json"
        if result_path.exists():
            continue
        start = time.time()
        # Per-task seeds, so a resumed or re-sharded run reproduces the same aux tasks and training order
        seed = (cfg.seed + zlib.crc32(name.encode())) % 2**31
        data_dir = out_dir / "tasks" / name / "data"
        build_task_dataset(builder, name, tasks[name], str(data_dir), cfg.num_aux_tasks, seed)
        metrics = ttt_task(cfg, train_cfg, state, data_dir, out_dir / "tasks" / name, seed)
        shutil.rmtree(data_dir)
        write_json(result_path, {"name": name, "metrics": metrics, "time_sec": time.time() - start})
        print_info(f"[rank {rank}] {name}: {json.dumps({k: v for k, v in metrics.items() if 'pass@' in k})} ({time.time() - start:.1f}s)")

    metrics = aggregate(names, out_dir)
    if metrics is None:
        return
    print_info(f"[rank {rank}] {len(names)} tasks: {json.dumps(metrics)} -> {out_dir}")
    run = _attach_wandb_run(cfg, ckpt, str(Path(ckpt_path).parent.parent))  # type: ignore[arg-type]
    if run is not None:
        run.log({**{f"eval/{cfg.suffix}/{k}": v for k, v in metrics.items()}, "step": step})
        run.finish()


if __name__ == "__main__":
    launch()
