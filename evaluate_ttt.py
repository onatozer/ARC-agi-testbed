"""Per-task test-time training (TTT) for any ARC checkpoint, then evaluation with the checkpoint's own evaluators.

For every eval task, starting from the checkpoint's weights (EMA ones if it has them):
  1. build a one-task dataset with dataset/build_arc_dataset.py: the task plus (num_aux_tasks - 1) dihedral/color
     augmentations, each with its own puzzle identifier, demos in the train split and test pairs in the test split;
  2. train on it with pretrain.py's train_batch for `epochs` passes over every aux task's demos, or, with meta_sgd_steps,
     take Meta-SGD steps with a meta-trained checkpoint's learned step sizes (meta_sgd_ttt);
  3. evaluate num_views times with evaluators/eval_fn.evaluate. The evaluators vote across the aux tasks (after undoing
     each augmentation), the views and the ttt_runs_per_task runs, since ARC's aggregated_voting pools them.

    python evaluate_ttt.py checkpoint=<run_dir>/checkpoints/step_N_<id>.pth [eval=ttt_varc] [epochs=50 train_overrides.lr=1e-4 ...]
    torchrun --nproc_per_node=<gpus> evaluate_ttt.py checkpoint=...
    # Embeddings first, then weights (arXiv 2609.21181), with ttt_varc's 100 epochs; any N alternates every N epochs
    python evaluate_ttt.py checkpoint=... train_overrides.freeze_alternation_epochs=50
    # Held-out MiniARC with a Meta-SGD + reconstruction checkpoint (the recipe in config/schema.py's MetaSGDConfig)
    python evaluate_ttt.py checkpoint=... eval=ttt_mini task_ids_file=data/miniarc5-aug-31-ho30/heldout_tasks.json meta_sgd_steps=1000 recon_steps=20

Tasks are sharded across ranks with no process group, since tasks are independent and finish at different times.
Each task writes <output_dir>/tasks/<name>.json (existing ones are skipped, so a run can be resumed), and whichever rank
finishes last writes the averaged metrics and the merged submission and logs them to the checkpoint's W&B run.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math
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
from pretrain import (
    _adapt, _adapt_segment, _episode_overrides, _meta_sgd_targets, _puzzle_emb_name, _with_recon_rows,
    create_dataloader, create_evaluators, init_train_state, train_batch,
)
from utils.checkpoint import load_model_state_dict
from utils.functions import load_model_class
from utils.printing import print_info
from utils.training import tree_to_device


torch.set_float32_matmul_precision("high")


def load_tasks(builder: Dict[str, Any], data_path: str) -> Dict[str, dict]:
    """The eval tasks with their test outputs (only the evaluators read them): ARC-AGI challenges with the solutions merged
    in for an ARC-AGI build, otherwise the build's test_puzzles.json (MiniARC)."""
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


def _split_rows(config: PretrainConfig, split: str, metadata: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    """Every row of the task's `split` (its aux copies' demos, or test pairs) as one batch, minus the padding rows."""
    loader, _ = create_dataloader(config, split, test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1)
    rows = {k: torch.cat([b[k] for _, b, _ in loader]) for k in ("inputs", "labels", "puzzle_identifiers")}
    keep = rows["puzzle_identifiers"] != metadata.blank_identifier_id
    if not keep.any():
        raise ValueError(f"TTT: the task's {split} split is empty.")
    return tree_to_device({k: v[keep] for k, v in rows.items()}, device)


def meta_sgd_adapt(
    model: torch.nn.Module, support: Dict[str, torch.Tensor], steps: int, segment_steps: bool,
    rows_frozen: bool = False, on_step: Optional[Any] = None,
) -> List[float]:
    """`steps` Meta-SGD inner steps on `support` with the checkpoint's learned alphas, written into the weights and (unless
    rows_frozen) the task's puzzle-embedding rows. The trained per-step alphas are stretched over the loop: the k-th of
    them for the k-th fraction of the steps. on_step(k) runs before step k. Returns the support loss before each step."""
    targets, emb_name, carry, losses = _meta_sgd_targets(model), _puzzle_emb_name(model), None, []
    if rows_frozen:
        targets = [(n, p) for n, p in targets if n != emb_name]

    for k in range(steps):
        if on_step is not None:
            on_step(k)

        named, overrides, batch, ids = _episode_overrides(model, targets, emb_name, support)
        alpha_step = k * model.meta_sgd.steps // steps
        if segment_steps:  # one supervision segment per step, the rows continuing their carry (TRM, GRAM)
            loss, theta, carry = _adapt_segment(model, named, overrides, batch, carry, alpha_step)
        else:
            loss, theta = _adapt(model, named, overrides, batch, alpha_step)
        losses.append(float(loss))

        with torch.no_grad():
            for name, p in targets:
                if name == emb_name:
                    model.get_buffer(emb_name)[ids] = theta[name]
                elif name in theta:
                    p.copy_(theta[name])
    return losses


def meta_sgd_ttt(cfg: TTTConfig, config: PretrainConfig, model: torch.nn.Module, train_metadata: Any, eval_metadata: Any) -> None:
    """Test-time Meta-SGD on one task, as meta-trained (config/schema.py MetaSGDConfig):
      1. meta_sgd_steps steps on the demos (with config.recon, plus masked copies of their inputs);
      2. with config.recon, recon_steps reconstruction steps on masked copies of the demo and test inputs (labels: the
         inputs, never the test outputs), the weights only, at the step sizes of 1. times recon_alpha_scale."""
    if getattr(model, "meta_sgd", None) is None:
        raise ValueError("meta_sgd_steps needs a checkpoint meta-trained with meta_sgd.enabled.")

    device = next(model.parameters()).device
    segment_steps, segments = config.meta_sgd.segment_inner_steps, config.arch.halt_max_steps
    recon_args = (config.recon, train_metadata.vocab_size, train_metadata.num_puzzle_identifiers)

    demos = _split_rows(config, "train", train_metadata, device)
    support = _with_recon_rows(demos, *recon_args) if config.recon.copies else demos
    losses = meta_sgd_adapt(model, support, cfg.meta_sgd_steps, segment_steps)
    print_info(f"TTT: {cfg.meta_sgd_steps} Meta-SGD steps on {demos['inputs'].shape[0]} demos, support loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    if not (config.recon.copies and cfg.recon_steps):
        return

    test = _split_rows(config, "test", eval_metadata, device)
    rows = {k: torch.cat([demos[k], test[k]]) for k in ("inputs", "puzzle_identifiers")}
    rows["labels"] = rows["inputs"]
    n = rows["inputs"].shape[0]
    draw = lambda: {k: v[n:] for k, v in _with_recon_rows(rows, *recon_args).items() if k in ("inputs", "labels", "puzzle_identifiers")}
    recon = draw()

    def redraw(k: int) -> None:
        # fresh masks at every trajectory start, written in place so the loop's batch sees them
        if 0 < k and k % segments == 0:
            for key, v in draw().items():
                recon[key].copy_(v)

    with torch.no_grad():  # rho = log alpha
        for rho in model.meta_sgd.rho.values():
            rho.add_(math.log(cfg.recon_alpha_scale))

    losses = meta_sgd_adapt(model, recon, cfg.recon_steps, segment_steps, rows_frozen=True, on_step=redraw)
    print_info(f"TTT: {cfg.recon_steps} reconstruction steps on {recon['inputs'].shape[0]} masked inputs, loss {losses[0]:.3f} -> {losses[-1]:.3f}")


def ttt_task(cfg: TTTConfig, train_cfg: PretrainConfig, state: Dict[str, torch.Tensor], data_dir: Path, eval_dir: Path, seed: int) -> Dict[str, float]:
    config = train_cfg.model_copy(deep=True)
    _apply_overrides(config, cfg.train_overrides)
    config.meta_sgd.enabled = cfg.meta_sgd_steps > 0  # 0: the ordinary TTT loop, also from a Meta-SGD checkpoint
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
        load_model_state_dict(train_state.model, dict(state), strict=False)
        if config.recon.copies:  # the resize above averaged the table's new rows: put the shared reconstruction (last) row back
            name = _puzzle_emb_name(train_state.model)
            with torch.no_grad():
                train_state.model.get_buffer(name)[-1] = state[name][-1]
        if train_state.ema_helper is not None:
            train_state.ema_helper.register(train_state.model)
        config.lr_warmup_steps = cfg.warmup_epochs * train_state.steps_per_epoch

        train_state.model.train()
        if config.meta_sgd.enabled:
            train_state.ema_helper = None  # the EMA weights are already in state; a swap after would undo the adaptation
            meta_sgd_ttt(cfg, config, train_state.model, train_metadata, eval_metadata)
        else:
            for _set_name, batch, global_batch_size in train_loader:
                train_batch(config, train_state, batch, global_batch_size, rank=0, world_size=1)
                if train_state.ema_helper is not None:
                    train_state.ema_helper.update(train_state.model)
                if train_state.step >= train_state.total_steps:
                    break

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
            task_ids = json.load(f)  # e.g. a held-out build's heldout_tasks.json
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
