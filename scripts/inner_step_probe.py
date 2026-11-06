"""What the Meta-SGD inner steps do to a pretrained checkpoint, before any meta-training: for each (weight alpha, row alpha)
pair, every episode of the first BATCHES meta-training batches takes the inner steps at that fixed step size (all steps),
and the probe reports the demo loss at the starting weights and the query loss after the steps (the outer loss), averaged
over episodes. No parameter is updated (the outer loss is scaled by 0).
    PYTHONPATH=. python scripts/inner_step_probe.py <pretrain ckpt> <meta data_path> [batches 4] [rows per batch 110]
"""
import math
import sys

import torch

from config.schema import PretrainConfig
from pretrain import _episodes_from_batch, _meta_sgd_targets, _puzzle_emb_name, _task_outer_backward, create_dataloader, init_train_state
from utils.checkpoint import load_model_state_dict
from utils.training import tree_to_device

CKPT, DATA = sys.argv[1], sys.argv[2]
BATCHES = int(sys.argv[3]) if len(sys.argv) > 3 else 4
ROWS = int(sys.argv[4]) if len(sys.argv) > 4 else 110
GRID = [(0.0, 0.0), (0.0, 3e-2), (0.0, 1e-1), (1e-5, 3e-2), (1e-4, 3e-2), (3e-4, 3e-2), (1e-3, 3e-2)]

obj = torch.load(CKPT, map_location="cpu", weights_only=False)
config = PretrainConfig(**obj["config"])
config.dataset.data_path, config.global_batch_size, config.ema = DATA, ROWS, False
config.meta_sgd.enabled, config.meta_sgd.segment_inner_steps = True, True

loader, meta = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=ROWS, rank=0, world_size=1,
                                 copies_per_group=config.meta_sgd.copies_per_group, enable_shared_sampler_state=False)
model = init_train_state(config, meta, rank=0, world_size=1).model
load_model_state_dict(model, {k.replace("_orig_mod.", ""): v for k, v in obj["model"].items()}, strict=False)  # raw weights, as meta-training loads
model.train()

device = next(model.parameters()).device
batches = []
for _, batch, _ in loader:
    batches.append(tree_to_device(batch, device))
    if len(batches) == BATCHES:
        break
episodes = [e for b in batches for e in _episodes_from_batch(b)]
targets, emb_name = _meta_sgd_targets(model), _puzzle_emb_name(model)
print(f"{len(episodes)} episodes, {config.meta_sgd.inner_steps} inner steps each")

for w_alpha, row_alpha in GRID:
    with torch.no_grad():
        for key, rho in model.meta_sgd.rho.items():
            alpha = row_alpha if "puzzle_emb" in key else w_alpha
            rho.fill_(math.log(alpha) if alpha > 0 else -60.0)  # exp(-60) ~ no step

    torch.manual_seed(0)  # the same masks and halting draws for every pair
    support, query = [], []
    for _key, s, q in episodes:
        loss_s, loss_q = _task_outer_backward(config, model, targets, emb_name, s, q, torch.tensor(0.0, device=device), [0.0] * 7 + [1.0])
        support.append(float(loss_s))
        query.append(float(loss_q))
    model.zero_grad(set_to_none=True)
    print(f"weights alpha {w_alpha:g}, rows alpha {row_alpha:g}: demo loss at theta {sum(support) / len(support):.3f}, "
          f"query CE after the steps {sum(query) / len(query):.3f}")
