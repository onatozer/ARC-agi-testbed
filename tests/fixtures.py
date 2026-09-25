"""Tiny randomly-initialised TRM + batch for exercising the dynamics code without a dataset."""

import torch

from models.losses.loss_heads import ACTLossHead
from models.recursive_reasoning.merged_arch import TRMModel


def tiny_model_and_batch(batch_size: int = 6, seq_len: int = 16, vocab: int = 6, device: str = "cuda", **overrides):
    cfg = dict(
        batch_size=batch_size, seq_len=seq_len, vocab_size=vocab, H_cycles=2, L_cycles=2, H_layers=0, L_layers=1,
        hidden_size=32, expansion=2, num_heads=2, pos_encodings="rope", halt_max_steps=4,
        halt_exploration_prob=0.1, puzzle_emb_ndim=0, noise_scale=0.0, random_initialization=False,
        num_puzzle_identifiers=1, causal=False,
    )
    cfg.update(overrides)
    torch.manual_seed(0)  # deterministic weights, independent of test order
    with torch.device(device):
        model = ACTLossHead(TRMModel(cfg), loss_type="stablemax_cross_entropy")
    g = torch.Generator().manual_seed(0)
    batch = {
        "inputs": torch.randint(1, vocab, (batch_size, seq_len), generator=g).to(device),
        "labels": torch.randint(2, vocab, (batch_size, seq_len), generator=g).to(device),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.int32).to(device),
    }
    return model, batch
