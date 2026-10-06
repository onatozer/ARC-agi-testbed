"""Per-task test-time training (TTT) and multi-view voting for a VARC checkpoint (arXiv 2511.14761, Sec. 4).

For every eval task, starting from the offline (pretrain.py) weights:
  1. build aux tasks: the task itself plus distinct dihedral/color augmentations of its demos, each with a fresh task token,
  2. fine-tune all weights on the aux tasks' demos (Adam, cosine schedule with warmup),
  3. predict each test input under every aux task x num_views random scale/translation views, undo the augmentation,
     and majority-vote; the top submission_K grids are the attempts.
ttt_runs_per_task independent runs pool their votes. Test outputs are only read for scoring.

    python evaluate_varc_ttt.py checkpoint=<path/to/step_N.pth> [key=value ...]
    torchrun --nproc_per_node=<gpus> evaluate_varc_ttt.py checkpoint=...

Tasks are sharded across ranks with no process group; each finished task writes <output_dir>/tasks/<name>.json
(existing ones are skipped, so a run can be resumed), and whichever rank finishes last writes metrics.json and
submission.json.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import json
import math
import os
import time
import zlib

import numpy as np
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel

from config.schema import PretrainConfig
from dataset.build_arc_dataset import ARCPuzzle, arc_grid_to_np, aug, grid_hash, inverse_aug, np_grid_to_seq_translational_augment, puzzle_hash
from evaluators.arc import _crop
from models.layers import CastedEmbedding
from models.losses import IGNORE_LABEL_ID
from utils.functions import load_model_class
from utils.optimizer import cosine_schedule_with_warmup_lr_lambda

DEFAULT_YAML = Path(__file__).parent / "config" / "eval" / "varc_ttt.yaml"


class VARCTTTConfig(BaseModel):
    checkpoint: str
    output_dir: Optional[str] = None
    use_ema: bool
    input_file_prefix: Optional[str] = None
    test_set_name: Optional[str] = None
    task_ids: Optional[List[str]] = None
    max_tasks: Optional[int] = None

    num_aux_tasks: int
    epochs: int
    warmup_epochs: int
    batch_size: int
    lr: float
    betas: Tuple[float, float]
    weight_decay: float
    ttt_runs_per_task: int

    num_views: int
    eval_batch_size: int
    pass_Ks: List[int]
    submission_K: int
    seed: int


def load_config(cli_args: Optional[Sequence[str]] = None) -> VARCTTTConfig:
    cli = OmegaConf.from_cli(list(cli_args) if cli_args is not None else None)
    base = OmegaConf.load(cli.pop("eval_yaml", DEFAULT_YAML))
    return VARCTTTConfig(**OmegaConf.to_container(OmegaConf.merge(base, cli), resolve=True))  # type: ignore[arg-type]


def load_tasks(prefix: str, set_name: str) -> Dict[str, dict]:
    """ARC-AGI challenges with the solutions merged in as test outputs, like build_arc_dataset.load_puzzles_arcagi."""
    with open(f"{prefix}_{set_name}_challenges.json") as f:
        tasks = json.load(f)
    with open(f"{prefix}_{set_name}_solutions.json") as f:
        solutions = json.load(f)
    for name, task in tasks.items():
        for pair, output in zip(task["test"], solutions[name]):
            pair["output"] = output
    return tasks


def build_aux_tasks(name: str, task: dict, num_aux_tasks: int) -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    """The task itself plus up to num_aux_tasks - 1 distinct augmentations of its demos, named as build_arc_dataset names them.
    Seed numpy's global RNG first: aug() draws from it."""
    demos = [(arc_grid_to_np(p["input"]), arc_grid_to_np(p["output"])) for p in task["train"]]
    aux: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [(name, lambda grid: grid)]
    hashes = {puzzle_hash({"train": ARCPuzzle(name, demos)})}
    for _ in range(10 * num_aux_tasks):  # same retry budget as build_arc_dataset (ARCAugmentRetriesFactor)
        if len(aux) >= num_aux_tasks:
            break
        aug_name, map_grid = aug(name)
        h = puzzle_hash({"train": ARCPuzzle(aug_name, [(map_grid(i), map_grid(o)) for i, o in demos])})
        if h not in hashes:
            hashes.add(h)
            aux.append((aug_name, map_grid))
    return aux


def encode_pair(inp: np.ndarray, out: np.ndarray, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Token sequences in the dataset's layout (no translation; VARC places grids on its canvas itself), PAD labels ignored."""
    inputs, labels = np_grid_to_seq_translational_augment(inp, out, do_translation=False, max_grid_size=grid_size)
    inputs, labels = inputs.astype(np.int64), labels.astype(np.int64)
    return inputs, np.where(labels == 0, IGNORE_LABEL_ID, labels)


def decode_prediction(pred: np.ndarray, aux_name: str, grid_size: int) -> np.ndarray:
    """Crop at the first non-color token, then undo the aux task's augmentation."""
    return inverse_aug(aux_name)[1](_crop(pred, grid_size, grid_size))


def rank_votes(votes: Counter) -> List[str]:
    return [h for h, _ in votes.most_common()]


def build_model(train_cfg: PretrainConfig, base_state: Dict[str, torch.Tensor], cfg: VARCTTTConfig, device: torch.device):
    """The checkpoint's loss head + VARC with the offline weights and a dense task embedding with one row per aux task."""
    arch_extra = dict(train_cfg.arch.__pydantic_extra__)  # type: ignore[arg-type]
    height = arch_extra["board_height"]
    width = arch_extra.get("board_width") or height
    model_cfg = dict(
        **arch_extra, batch_size=cfg.batch_size, vocab_size=12, seq_len=height * width,
        num_puzzle_identifiers=cfg.num_aux_tasks + 1, causal=False,
    )
    with torch.device(device):
        model = load_model_class(train_cfg.arch.loss.name)(load_model_class(train_cfg.arch.name)(model_cfg), **train_cfg.arch.loss.__pydantic_extra__)  # type: ignore[arg-type]
        inner = model.model.inner
        if not hasattr(inner, "puzzle_emb"):
            raise ValueError("VARC TTT needs a checkpoint trained with task tokens (arch.puzzle_emb_ndim = hidden_size).")
        inner.puzzle_emb = CastedEmbedding(cfg.num_aux_tasks + 1, inner.config.hidden_size, init_std=inner.config.task_emb_init_std, cast_to=torch.float32)

    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if unexpected or any("puzzle_emb" not in k for k in missing):
        raise ValueError(f"checkpoint does not match the VARC model: missing {missing}, unexpected {unexpected}")
    return model


def load_offline_state(cfg: VARCTTTConfig) -> Tuple[PretrainConfig, Dict[str, torch.Tensor]]:
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    train_cfg = PretrainConfig(**ckpt["config"])
    state = dict(ckpt["model"])
    if cfg.use_ema and ckpt.get("ema") is not None:
        state.update(ckpt["ema"]["shadow"])
    # Task tokens are re-initialized per task; compiled models prefix keys with _orig_mod.
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items() if "puzzle_emb" not in k}
    return train_cfg, state


def reset_model(model: torch.nn.Module, base_state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(base_state, strict=False)
    emb = model.model.inner.puzzle_emb
    with torch.no_grad():
        torch.nn.init.trunc_normal_(emb.embedding_weight, std=model.model.inner.config.task_emb_init_std)


def ttt_run(model, base_state, aux, task: dict, cfg: VARCTTTConfig, grid_size: int, device: torch.device, votes: List[Counter], grids: Dict[str, np.ndarray]) -> float:
    reset_model(model, base_state)

    # Train on every aux task's demos; aux task i uses task token i + 1 (0 is the dataset's blank id)
    train = [(*encode_pair(map_grid(arc_grid_to_np(p["input"])), map_grid(arc_grid_to_np(p["output"])), grid_size), i + 1)
             for i, (_, map_grid) in enumerate(aux) for p in task["train"]]
    inputs = torch.tensor(np.stack([x for x, _, _ in train]), device=device)
    labels = torch.tensor(np.stack([y for _, y, _ in train]), device=device)
    ids = torch.tensor([i for _, _, i in train], dtype=torch.int32, device=device)

    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    model.train()
    step, last_loss = 0, torch.zeros(())
    for _ in range(cfg.epochs):
        for idx in torch.randperm(len(train), device=device).split(cfg.batch_size):
            for group in optimizer.param_groups:
                group["lr"] = cosine_schedule_with_warmup_lr_lambda(
                    step, base_lr=cfg.lr, num_warmup_steps=cfg.warmup_epochs * steps_per_epoch, num_training_steps=total_steps
                )
            batch = {"inputs": inputs[idx], "labels": labels[idx], "puzzle_identifiers": ids[idx]}
            _, loss, _, _, _ = model(return_keys=[], carry=None, batch=batch)
            optimizer.zero_grad(set_to_none=True)
            last_loss = loss / len(idx)
            last_loss.backward()
            optimizer.step()
            step += 1

    # Inference: every test input under every aux task, num_views random canvas views each
    queries = [(t, a) for t in range(len(task["test"])) for a in range(len(aux)) for _ in range(cfg.num_views)]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(queries), cfg.eval_batch_size):
            chunk = queries[start:start + cfg.eval_batch_size]
            seqs = []
            for t, a in chunk:
                grid = aux[a][1](arc_grid_to_np(task["test"][t]["input"]))
                seqs.append(encode_pair(grid, grid, grid_size)[0])
            batch = {
                "inputs": torch.tensor(np.stack(seqs), device=device),
                "puzzle_identifiers": torch.tensor([a + 1 for _, a in chunk], dtype=torch.int32, device=device),
            }
            _, outputs = model.model(carry=None, batch=batch)
            preds = outputs["logits"].argmax(-1).cpu().numpy()
            for (t, a), pred in zip(chunk, preds):
                grid = decode_prediction(pred, aux[a][0], grid_size)
                h = grid_hash(grid)
                grids.setdefault(h, grid)
                votes[t][h] += 1
    return float(last_loss.detach())


def score_task(task: dict, votes: List[Counter], grids: Dict[str, np.ndarray], cfg: VARCTTTConfig) -> dict:
    """Per-test-pair pass@K over the vote ranking; the task score is the mean over its test pairs, as in evaluators/arc.py."""
    tests = []
    for pair, counter in zip(task["test"], votes):
        ranked = rank_votes(counter)
        label_hash = grid_hash(arc_grid_to_np(pair["output"]))
        tests.append({
            "correct": {str(k): label_hash in ranked[:k] for k in cfg.pass_Ks},
            "attempts": [grids[h].tolist() for h in ranked[:cfg.submission_K]],
            "top_votes": [[grids[h].tolist(), counter[h]] for h in ranked[:10]],
        })
    return {
        "pass": {str(k): float(np.mean([t["correct"][str(k)] for t in tests])) for k in cfg.pass_Ks},
        "test": tests,
    }


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def aggregate(names: List[str], out_dir: Path, cfg: VARCTTTConfig) -> Optional[Dict[str, float]]:
    paths = [out_dir / "tasks" / f"{name}.json" for name in names]
    if not all(p.exists() for p in paths):
        return None
    results = {name: json.loads(p.read_text()) for name, p in zip(names, paths)}
    metrics = {f"ARC/pass@{k}": float(np.mean([r["pass"][str(k)] for r in results.values()])) for k in cfg.pass_Ks}
    metrics["num_tasks"] = len(results)
    submission = {
        name: [{f"attempt_{i + 1}": grid for i, grid in enumerate(t["attempts"] + t["attempts"][:1] * (cfg.submission_K - len(t["attempts"])))} for t in r["test"]]
        for name, r in results.items()
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "submission.json", submission)
    return metrics


def main() -> None:
    cfg = load_config()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_cfg, base_state = load_offline_state(cfg)
    dataset_extra = train_cfg.dataset.__pydantic_extra__ or {}
    prefix = cfg.input_file_prefix or dataset_extra.get("input_file_prefix")
    set_name = cfg.test_set_name or dataset_extra.get("test_set_name")
    if not prefix or not set_name:
        raise ValueError("The checkpoint's dataset is not an ARC-AGI build (no input_file_prefix/test_set_name); pass input_file_prefix= and test_set_name=.")
    grid_size = int(train_cfg.arch.__pydantic_extra__["board_height"])  # type: ignore[index]
    out_dir = Path(cfg.output_dir or Path(cfg.checkpoint).parent / "varc_ttt")
    (out_dir / "tasks").mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(prefix, set_name)
    names = sorted(cfg.task_ids or tasks)[: cfg.max_tasks]
    model = build_model(train_cfg, base_state, cfg, device)
    base_state = {k: v.to(device) for k, v in base_state.items()}

    for name in names[rank::world]:
        path = out_dir / "tasks" / f"{name}.json"
        if path.exists():
            continue
        start = time.time()
        task = tasks[name]
        # Per-task seeds so a resumed or re-sharded run reproduces the same aux tasks and training order
        task_seed = cfg.seed + zlib.crc32(name.encode())
        np.random.seed(task_seed % 2**32)
        aux = build_aux_tasks(name, task, cfg.num_aux_tasks)
        votes: List[Counter] = [Counter() for _ in task["test"]]
        grids: Dict[str, np.ndarray] = {}
        for run in range(cfg.ttt_runs_per_task):
            torch.manual_seed(task_seed + run)
            final_loss = ttt_run(model, base_state, aux, task, cfg, grid_size, device, votes, grids)
        result = score_task(task, votes, grids, cfg)
        write_json(path, {"name": name, "num_aux_tasks": len(aux), **result})
        print(f"[rank {rank}] {name}: pass {result['pass']}, final loss {final_loss:.4f}, aux tasks {len(aux)}, {time.time() - start:.1f}s", flush=True)

    metrics = aggregate(names, out_dir, cfg)
    if metrics is not None:
        print(f"[rank {rank}] {json.dumps(metrics)} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
