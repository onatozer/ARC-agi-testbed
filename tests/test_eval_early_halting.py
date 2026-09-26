from types import SimpleNamespace

import pytest
import torch

from evaluators.eval_fn import _finalize_state, _accumulate, _metric_state, evaluate
from models.losses.loss_heads import InferenceLossHead
from models.recursive_reasoning.trm_root import TRMRoot
from tests.fixtures import tiny_model_and_batch


class _Recorder:
    """Evaluator that keeps the per-row outputs evaluate() hands to evaluators."""
    required_outputs = {"preds", "q_halt_logits"}

    def begin_eval(self):
        self.batches = []

    def update_batch(self, batch, preds):
        self.batches.append({k: v.clone() for k, v in preds.items()})

    def result(self, *args, **kwargs):
        return {}


def _run_evaluate(model, batches):
    recorder = _Recorder()
    config = SimpleNamespace(eval_save_outputs=[], eval_dir=None, eval_save_carry=False, convergence_top_k=None)
    metrics, _ = evaluate(
        config=config,
        train_state=SimpleNamespace(model=model),
        eval_loader=[("test", b, None) for b in batches],
        eval_metadata=SimpleNamespace(sets=["test"]),
        evaluators=[recorder],
        rank=0,
        world_size=1,
        cpu_group=None,
    )
    return metrics["test"], recorder.batches


def _heads(**overrides):
    act, batch = tiny_model_and_batch(batch_size=8, **overrides)
    root_cfg = dict(act.model.config.model_dump(), seq_len=16, deq_max_iter=3, deq_min_iter=1, bptt_through=1)  # seq_len before any prefix
    torch.manual_seed(0)
    with torch.device("cuda"):
        root = TRMRoot(root_cfg)
    return {
        "trm": InferenceLossHead(act.model, loss_type="stablemax_cross_entropy"),
        "trm_root": InferenceLossHead(root, loss_type="stablemax_cross_entropy"),
    }, batch


def _second_batch(batch):
    g = torch.Generator().manual_seed(1)
    other = dict(batch)
    other["inputs"] = torch.randint(1, 6, batch["inputs"].shape, generator=g).to(batch["inputs"].device)
    return other


@pytest.mark.parametrize("head", ["trm", "trm_root"])
def test_default_eval_matches_running_every_row_to_max_steps(head):
    heads, batch = _heads()
    model = heads[head]
    model.eval()
    batches = [batch, _second_batch(batch)]

    # The pre-change loop: step the loss head until the whole batch halts, and score that last step
    state = _metric_state(batch["inputs"].device, 1)
    with torch.inference_mode():
        carry = None
        for b in batches:
            carry = model.initial_carry(b) if carry is None else carry
            while True:
                carry, _, metrics, _, all_finish = model(carry=carry, batch=b, return_keys=[])
                if all_finish:
                    break
            _accumulate(state, metrics, 0)
    expected = _finalize_state(state, {"test": 0}, 0, 1, None)["test"]

    got, _ = _run_evaluate(model, batches)
    assert got == pytest.approx(expected)
    assert got["steps"] == model.model.config.halt_max_steps


def _first_halt_reference(model, batch):
    """Steps the bare model and records, for each row, its step, prediction and q_halt when it first halts."""
    n = batch["inputs"].shape[0]
    steps = torch.zeros(n, dtype=torch.int32, device="cuda")
    preds = torch.zeros_like(batch["labels"])
    q = torch.zeros(n, device="cuda")
    torch.manual_seed(0)  # same noise as the evaluate() run below
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        for _ in range(model.model.config.halt_max_steps):
            carry, out = model.model(carry=carry, batch=batch)
            first = carry.halted & (steps == 0)
            steps = torch.where(first, carry.steps, steps)
            preds = torch.where(first[:, None], out["logits"].argmax(-1), preds)
            q = torch.where(first, out["q_halt_logits"], q)
    assert (steps > 0).all()
    return steps, preds, q


@pytest.mark.parametrize("head", ["trm", "trm_root"])
def test_act_inference_holds_each_row_at_its_first_halt(head):
    # With noise a restarted row doesn't retrace its first run, so its first-halt outputs must be held, not recomputed
    heads, batch = _heads(halt_max_steps=6, halt_exploration_prob=0.0, act_inference=True, noise_scale=0.1)
    model = heads[head]
    model.eval()
    batches = [batch]
    for seed in range(10):  # find a q_head that makes rows halt at different steps
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed)
            model.model.inner.q_head.weight.copy_(torch.randn(model.model.inner.q_head.weight.shape, generator=g) * 3)
            model.model.inner.q_head.bias.zero_()
        refs = [_first_halt_reference(model, b) for b in batches]
        all_steps = torch.cat([r[0] for r in refs])
        if len(set(all_steps.tolist())) > 1:
            break
    assert len(set(all_steps.tolist())) > 1, "test needs rows that halt at different steps"

    torch.manual_seed(0)
    got, recorded = _run_evaluate(model, batches)

    for (steps, preds, q), out in zip(refs, recorded):
        assert torch.equal(out["preds"], preds)
        assert torch.equal(out["q_halt_logits"], q)
    cfg = model.model.config
    assert got["steps"] == pytest.approx(all_steps.float().mean().item())
    assert got["nfe"] == pytest.approx(all_steps.float().mean().item() * cfg.H_cycles * (cfg.L_cycles + 1))
