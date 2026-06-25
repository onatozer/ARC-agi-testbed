"""Quick debug: build TRM-DEQ model, run 3 train batches + 5 eval batches."""
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from experiments.eqlm_sudoku.data.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from experiments.eqlm_sudoku.trm.trm_deq import TRMDEQ
from experiments.eqlm_sudoku.trm.losses_deq import ACTLossHeadDEQ
from torch.utils.data import DataLoader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "/scratch1/feinashl/data/sudoku-extreme-1k-aug-1000"
    batch_size = 64

    cfg_dict = dict(
        batch_size=batch_size, seq_len=81, vocab_size=11,
        num_puzzle_identifiers=1,
        H_cycles=3, L_cycles=6, H_layers=0, L_layers=2,
        hidden_size=512, expansion=4.0, num_heads=8,
        pos_encodings="rope", halt_max_steps=16,
        halt_exploration_prob=0.1, forward_dtype="bfloat16",
        mlp_t=False, puzzle_emb_len=16, no_ACT_continue=True,
        puzzle_emb_ndim=512,
        deq_inner=True, deq_max_iter=8, deq_min_iter=4, deq_tol=1e-3,
        deq_anderson_m=5, deq_anderson_beta=1.0,
        bptt_through=2, jacobian_reg_lambda=0.0, jacobian_reg_n_samples=1,
    )

    print("Building model...")
    with torch.device(device):
        inner = TRMDEQ(cfg_dict)
        model = ACTLossHeadDEQ(inner, loss_type="stablemax_cross_entropy",
                               jacobian_reg_lambda=0.0)
    n = sum(p.numel() for p in model.parameters())
    print(f"Params: {n / 1e6:.3f}M")

    # Train loader (3 batches)
    print("\n--- TRAIN (3 batches) ---")
    train_ds = PuzzleDataset(PuzzleDatasetConfig(
        seed=0, dataset_paths=[data_dir], global_batch_size=batch_size,
        test_set_mode=False, epochs_per_iter=1, rank=0, num_replicas=1,
    ), split="train")
    train_dl = DataLoader(train_ds, batch_size=None)
    model.train()
    carry = None
    for i, (set_name, batch, gbs) in enumerate(train_dl):
        if i >= 3:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        if carry is None:
            with torch.device(device):
                carry = model.initial_carry(batch)
        t0 = time.time()
        carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
        ((1 / batch_size) * loss).backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        count = max(float(metrics.get("count", 1)), 1)
        print(f"  batch {i}: loss={float(loss)/batch_size:.4f}  "
              f"acc={float(metrics.get('accuracy', 0))/count:.4f}  "
              f"exact={float(metrics.get('exact_accuracy', 0))/count:.4f}  "
              f"dt={dt:.2f}s")

    # Eval loader (5 batches)
    print("\n--- EVAL (5 batches) ---")
    test_ds = PuzzleDataset(PuzzleDatasetConfig(
        seed=0, dataset_paths=[data_dir], global_batch_size=batch_size,
        test_set_mode=True, epochs_per_iter=1, rank=0, num_replicas=1,
    ), split="test")
    test_dl = DataLoader(test_ds, batch_size=None)
    model.eval()
    max_steps = 16
    for i, (set_name, batch, gbs) in enumerate(test_dl):
        if i >= 5:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.device(device):
            carry = model.initial_carry(batch)
        t0 = time.time()
        steps = 0
        with torch.inference_mode():
            while steps < max_steps:
                carry, _loss, metrics, _preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=[])
                steps += 1
                if all_finish:
                    break
        torch.cuda.synchronize()
        dt = time.time() - t0
        count = max(float(metrics.get("count", 1)), 1)
        print(f"  batch {i}: steps={steps}  "
              f"acc={float(metrics.get('accuracy', 0))/count:.4f}  "
              f"exact={float(metrics.get('exact_accuracy', 0))/count:.4f}  "
              f"dt={dt:.2f}s  all_finish={bool(all_finish)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
