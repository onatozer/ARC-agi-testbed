"""(scripts/miniarc5/traj_diag_fixes.py; log autoresearch_runs/miniarc5/traj_diag_fixes.log.) Round 26 post-mortem checks and
the round 29 fixes, trm_ho3 frozen, the query rows of 10 training batches (traj_diag_budget.py's), lr 0.1, K 20, lambda 10:
1. the row gap: per-segment squared distance between the rollout with the table-mean row (an episode's fresh start) and
   with each puzzle's own row, next to the own-row targets' squared change from their rollout;
2. targets built from the mean row (fresh rows) for traj_lambda_growth in GROWTHS: CE / exact match after, and the
   per-segment change Delta_t and residual R_t.
Prints the chosen growth last: the largest Delta_1 / Delta_N with exact match after >= 95%."""
import glob, sys, time, torch
from config.schema import PretrainConfig
from pretrain import create_dataloader, init_train_state, _optimize_trajectory, _transition, _initial_state, _puzzle_emb_name, _INNER
from utils.checkpoint import load_model_state_dict

GROWTHS = (1.0, 1.5, 2.0, 3.0)
LR, K, LAM = 0.1, 20, 10.0
ckpt = sorted(glob.glob("autoresearch_runs/miniarc5/trm_ho3/outputs/*/*/*/checkpoints/*.pth"))[-1]
obj = torch.load(ckpt, map_location="cpu", weights_only=False)
config = PretrainConfig(**obj["config"])
config.meta_sgd.copies_per_group = 16; config.ema = False
loader, meta = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=256, rank=0, world_size=1, copies_per_group=16)
ts = init_train_state(config, meta, rank=0, world_size=1)
load_model_state_dict(ts.model, obj["model"], strict=False)
model = ts.model.train()
inner = model.model.inner
dev = next(model.parameters()).device
torch.manual_seed(0)
batches = []
for i, (_, b, _) in enumerate(loader):
    q = b["validation"].bool() & (b["puzzle_identifiers"] != 0)
    rows = {k: b[k][q].to(dev) for k in ("inputs", "labels", "puzzle_identifiers")}
    for j in range(0, rows["inputs"].shape[0], 16):
        batches.append({k: v[j : j + 16] for k, v in rows.items()})
    if len(batches) >= 10: break
batches = batches[:10]
n = config.arch.halt_max_steps
emb = _puzzle_emb_name(model)
table = model.get_buffer(emb)
mean_ov = {emb: table.mean(0, keepdim=True), emb[: -len("weights")] + "local_weights": table.new_zeros(0, table.shape[1])}
fresh = lambda b: {**b, "puzzle_identifiers": torch.zeros_like(b["puzzle_identifiers"])}


def dist(a, b):
    return (a[0].float() - b[0].float()).square().mean((1, 2)) + (a[1].float() - b[1].float()).square().mean((1, 2))


@torch.no_grad()
def rollout(ov, b):
    ov = {k[len(_INNER):]: v for k, v in ov.items()}
    states = [_initial_state(inner, b["inputs"].shape[0], dev)]
    for _ in range(n):
        states.append(_transition(inner, ov, states[-1], b)[0])
    return states


def profile(ov, b, targets):
    """Per-segment squared change ||s~_t - s_t||^2 and residual ||s~_t - F(s~_{t-1})||^2, means over rows."""
    states = rollout(ov, b)
    iov = {k[len(_INNER):]: v for k, v in ov.items()}
    traj = [states[0]] + targets
    with torch.no_grad():
        change = [float(dist(targets[t], states[t + 1]).mean()) for t in range(n)]
        resid = [float(dist(traj[t + 1], _transition(inner, iov, traj[t], b)[0]).mean()) for t in range(n)]
    return change, resid


fmt = lambda xs: " ".join(f"{x:.3f}" for x in xs)
print(f"{sum(b['inputs'].shape[0] for b in batches)} query rows, N = {n}, lr {LR}, K {K}, lambda {LAM}", flush=True)

# 1. the row gap
gap, own_change = [0.0] * n, [0.0] * n
for b in batches:
    own, mean = rollout({}, b), rollout(mean_ov, fresh(b))
    targets, _ = _optimize_trajectory(model, {}, b, n, LAM, K, LR)
    ch, _ = profile({}, b, targets)
    for t in range(n):
        gap[t] += float(dist(mean[t + 1], own[t + 1]).mean()) / len(batches)
        own_change[t] += ch[t] / len(batches)
print("row gap ||mean-row rollout - own-row rollout||^2 per segment:", fmt(gap), f"| mean {sum(gap) / n:.3f}")
print("own-row target change ||s~_t - s_t||^2 per segment:         ", fmt(own_change), f"| mean {sum(own_change) / n:.3f}", flush=True)

# 2. fresh-row targets by growth
best = None
for g in GROWTHS:
    t0, acc, ch, rs = time.time(), {k: 0.0 for k in ("ce_before", "ce_after", "exact_before", "exact_after")}, [0.0] * n, [0.0] * n
    for b in batches:
        targets, d = _optimize_trajectory(model, mean_ov, fresh(b), n, LAM, K, LR, growth=g)
        c, r = profile(mean_ov, fresh(b), targets)
        for k in acc: acc[k] += d[k] / len(batches)
        for t in range(n): ch[t] += c[t] / len(batches); rs[t] += r[t] / len(batches)
    ratio = (ch[0] / ch[-1]) ** 0.5
    print(f"growth {g}: ce {acc['ce_before']:.3f} -> {acc['ce_after']:.3f}, exact {acc['exact_before']:.3f} -> {acc['exact_after']:.3f}, "
          f"Delta_1/Delta_N {ratio:.3f}, {(time.time() - t0) / len(batches):.1f} s/episode", flush=True)
    print(f"  change^2 per segment: {fmt(ch)}\n  residual^2 per link:  {fmt(rs)}", flush=True)
    if acc["exact_after"] >= 0.95 and (best is None or ratio > best[0]):
        best = (ratio, g)
print(f"chosen growth: {best[1] if best else 1.0}")
