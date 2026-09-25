import torch

from models.losses.loss_heads import ACTLossHeadRoot
from models.recursive_reasoning.trm_root import TRMRoot
from tests.fixtures import tiny_model_and_batch


def test_nfe_is_per_puzzle_solver_iterations():
    _, batch = tiny_model_and_batch(batch_size=8)
    cfg = dict(
        batch_size=8, seq_len=16, vocab_size=6, H_cycles=2, L_cycles=2, H_layers=0, L_layers=1, hidden_size=32,
        expansion=2, num_heads=2, pos_encodings="rope", halt_max_steps=2, halt_exploration_prob=0.1, puzzle_emb_ndim=0,
        noise_scale=0.0, random_initialization=False, num_puzzle_identifiers=1, causal=False, forward_dtype="bfloat16",
        deq_max_iter=8, bptt_through=1,
    )
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = ACTLossHeadRoot(TRMRoot(cfg), loss_type="stablemax_cross_entropy", jacobian_reg_lambda=0.0)
    model.eval()
    inner = model.model.inner

    def step():
        with torch.no_grad():
            carry, _, _, _, _ = model(carry=model.initial_carry(batch), batch=batch, return_keys=[])
        return carry.nfe, inner._last_info["rel_trace"][:, 1:]

    # Tolerance never met: every puzzle runs the full solve (one solve, one z_H update, bptt_through + 1 evals)
    inner.root_config.deq_tol = 0.0
    nfe, rel = step()
    assert torch.equal(nfe, torch.full_like(nfe, 8 + 1 + 2))

    # Tolerance between rows' residuals: puzzles that converge sooner are charged fewer evals
    inner.root_config.deq_tol = rel[:, -1].median().item()
    nfe, rel = step()
    below = rel < inner.root_config.deq_tol
    needed = torch.where(below.any(1), below.int().argmax(1) + 3, 8)
    assert torch.equal(nfe, (needed + 3).to(nfe.dtype))
    assert nfe.unique().numel() > 1
