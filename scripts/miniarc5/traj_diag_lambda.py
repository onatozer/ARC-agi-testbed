"""(scripts/miniarc5/traj_diag_lambda.py: the lambda x K grid of 2026-10-26.) Diagnostic for meta_sgd.query_loss=trajectory: trm_ho3 frozen, the query rows (former test pairs) of 10 training batches,
target optimisation over a lambda x K grid. Prints CE / exact match of decode(s_N) before and of decode(s~_N) after, the
mean transition residual R vs the rollout's own step size, and Delta_t at t = 1, N-1, N."""
import glob, itertools, sys, time, torch
from config.schema import PretrainConfig
from pretrain import create_dataloader, init_train_state, _optimize_trajectory
from utils.checkpoint import load_model_state_dict

ckpt = sorted(glob.glob("autoresearch_runs/miniarc5/trm_ho3/outputs/*/*/*/checkpoints/*.pth"))[-1]
obj = torch.load(ckpt, map_location="cpu", weights_only=False)
config = PretrainConfig(**obj["config"])
config.meta_sgd.copies_per_group = 16; config.ema = False
loader, meta = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=256, rank=0, world_size=1, copies_per_group=16)
ts = init_train_state(config, meta, rank=0, world_size=1)
load_model_state_dict(ts.model, obj["model"], strict=False)
model = ts.model.train()
dev = next(model.parameters()).device
torch.manual_seed(0)
batches = []
for i, (_, b, _) in enumerate(loader):
    q = b["validation"].bool() & (b["puzzle_identifiers"] != 0)
    rows = {k: b[k][q].to(dev) for k in ("inputs", "labels", "puzzle_identifiers")}
    for j in range(0, rows["inputs"].shape[0], 16):  # one task's query rows, as one meta-training episode has
        batches.append({k: v[j : j + 16] for k, v in rows.items()})
    if i == 9: break
n = config.arch.halt_max_steps
print(f"{sum(b['inputs'].shape[0] for b in batches)} query rows, N = {n}")
keys = ["ce_before", "ce_after", "exact_before", "exact_after", "residual", "step", "delta_first", "delta_prev", "delta_last"]
print("lambda K | " + " | ".join(keys) + " | s/episode")
for lam, k in itertools.product((0.1, 1.0, 10.0), (5, 20)):
    t0, acc = time.time(), {key: 0.0 for key in keys}
    for b in batches:
        _, d = _optimize_trajectory(model, {}, b, n, lam, k, 1e-2)
        for key in keys: acc[key] += d[key] / len(batches)
    print(f"{lam} {k} | " + " | ".join(f"{acc[key]:.4g}" for key in keys) + f" | {(time.time() - t0) / len(batches):.2f}", flush=True)
