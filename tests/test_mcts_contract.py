"""Contract tests every models.mcts.MCTS_IMPLS implementation must pass (the contract is the module docstring of
models/mcts.py). Exact implementations (impl.exact) must reproduce the baseline's visit counts and callback
traffic; semantics-changing ones only keep the invariants."""
import copy

import pytest
import torch

from models.mcts import MCTS, MCTS_IMPLS, Node
from models.recursive_reasoning.treerm import TreeRM, _compiled_step_core

from tests.test_gram import _cfg
from tests.test_treerm import _treerm_head

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
IMPLS = sorted(MCTS_IMPLS)
EXACT_IMPLS = [name for name in IMPLS if MCTS_IMPLS[name].exact]


# Toy domain: a state is the tuple of child indices taken from the root; a value function maps states to floats

class Trace:
    """Runs one search over a batch of toy trees and records every callback: the states each round evaluated and
    expanded (as tuples per tree), so the tests can check the traffic, not just the final counts."""

    def __init__(self, impl: str, values, depths, num_children, num_simulations, c_puct=1.25, priors=None):
        self.values = values  # callable(tree, state) -> float
        self.evaluated = []  # per round: list of (tree, state, allowed_depth)
        self.expanded = []  # per round: list of (tree, state)
        self.roots = []
        for tree, depth in enumerate(depths):
            root = Node(latent=(tree, ()), allowed_depth=depth)
            tree_priors = priors[tree] if priors is not None else [1 / num_children] * num_children
            root.expand([(tree, (k,)) for k in range(num_children)], tree_priors)
            self.roots.append(root)
        self.num_children = num_children
        self.search = MCTS_IMPLS[impl](num_simulations=num_simulations, c_puct=c_puct)
        self.counts = self.search.run(self.roots, self.expand_fn, self.evaluate_fn)

    def expand_fn(self, states):
        self.expanded.append([state for state in states])
        children = [[(tree, path + (k,)) for k in range(self.num_children)] for tree, path in states]
        priors = [[1 / self.num_children] * self.num_children for _ in states]
        return children, priors

    def evaluate_fn(self, states, allowed_depths):
        self.evaluated.append([(tree, path, depth) for (tree, path), depth in zip(states, allowed_depths)])
        return [self.values(tree, path) for tree, path in states]


def _by_first_child(table):
    return lambda tree, path: table[path[0]]


@pytest.mark.parametrize("impl", IMPLS)
def test_counts_sum_to_simulations_and_every_root_child_is_visited(impl):
    trace = Trace(impl, _by_first_child([0.0, 1.0, 0.2]), depths=[1, 2], num_children=3, num_simulations=12)
    for counts in trace.counts:
        assert sum(counts) == 12 and all(c >= 1 for c in counts)
        assert counts.index(max(counts)) == 1  # the best child gets the most visits


@pytest.mark.parametrize("impl", IMPLS)
def test_leaves_at_depth_zero_are_evaluated_but_never_expanded(impl):
    trace = Trace(impl, _by_first_child([0.3, 0.9]), depths=[2], num_children=2, num_simulations=10)
    seen_states = [state for round_ in trace.evaluated for _, state, _ in round_]
    assert max(len(state) for state in seen_states) == 2  # never deeper than the root's allowed depth
    for round_ in trace.expanded:
        for _, state in round_:
            assert len(state) < 2  # only leaves with depth left are expanded
    for round_ in trace.evaluated:
        for _, state, depth in round_:
            assert depth == 2 - len(state)  # evaluate_fn sees each leaf's own remaining depth
    assert any(len(state) == 2 for state in seen_states)  # depth-0 leaves do get evaluated


@pytest.mark.parametrize("impl", IMPLS)
def test_rows_with_different_allowed_depths_search_independently(impl):
    trace = Trace(impl, _by_first_child([0.0, 1.0]), depths=[1, 3], num_children=2, num_simulations=8)
    for round_ in trace.expanded:
        for tree, state in round_:
            assert tree == 1  # tree 0's root children have no depth left, so nothing of tree 0 is ever expanded
    for round_ in trace.evaluated:
        for tree, state, depth in round_:
            assert depth == [1, 3][tree] - len(state)
    assert all(sum(c) == 8 for c in trace.counts)


@pytest.mark.parametrize("impl", IMPLS)
def test_values_are_minmax_normalised_per_tree(impl):
    values = {(0,): 0.5, (1,): 0.2, (0, 0): 0.9, (0, 1): 0.1, (1, 0): 0.4, (1, 1): 0.3}
    base = Trace(impl, lambda tree, path: values[path], depths=[2], num_children=2, num_simulations=7)
    affine = Trace(impl, lambda tree, path: 3 * values[path] + 7, depths=[2], num_children=2, num_simulations=7)
    assert base.counts == affine.counts


@pytest.mark.parametrize("impl", IMPLS)
def test_one_tree_never_moves_another_trees_counts(impl):
    table = {0: [0.1, 0.9, 0.5], 1: [0.9, 0.1, 0.5]}
    alone = Trace(impl, _by_first_child(table[0]), depths=[2], num_children=3, num_simulations=10)
    together = Trace(impl, lambda tree, path: table[tree][path[0]], depths=[2, 2], num_children=3, num_simulations=10)
    assert together.counts[0] == alone.counts[0]
    assert together.counts[1] != together.counts[0]  # and the second tree followed its own values


@pytest.mark.parametrize("impl", IMPLS)
def test_search_is_deterministic(impl):
    runs = [Trace(impl, _by_first_child([0.4, 0.6, 0.5]), depths=[2, 1], num_children=3, num_simulations=9).counts for _ in range(2)]
    assert runs[0] == runs[1]


# Exact semantics: these pin the baseline (hand-computed) and anything that claims to reproduce it

def _evaluated(trace):
    """Every evaluated (tree, state) in order, rounds flattened: an implementation may batch several of the baseline's
    rounds into one call, but the order of the work it hands the domain is the baseline's."""
    return [(tree, state) for round_ in trace.evaluated for tree, state, _ in round_]


@pytest.mark.parametrize("impl", EXACT_IMPLS)
def test_first_k_simulations_visit_each_root_child_once_in_order(impl):
    trace = Trace(impl, _by_first_child([0.0, 1.0, 0.5, 0.2]), depths=[1, 2], num_children=4, num_simulations=6)
    assert _evaluated(trace)[:8] == [(tree, (k,)) for k in range(4) for tree in range(2)]


@pytest.mark.parametrize("impl", EXACT_IMPLS)
def test_hand_computed_depth_two_search(impl):
    # c_puct 1.25, uniform priors 0.5. Rounds: (0,) v=.5 and (1,) v=.2 are forced; round 2 picks child 0 (Q norm 1 vs
    # 0) then its first child (unvisited, tied, first max) (0,0) v=.9; round 3: child 0 again (Q .714 + .361 beats
    # 0 + .541), inside it (0,0) (1 + .442 beats 0 + .884). Depth-0 leaves are not expanded.
    values = {(0,): 0.5, (1,): 0.2, (0, 0): 0.9, (0, 1): 0.1, (1, 0): 0.4, (1, 1): 0.3}
    trace = Trace(impl, lambda tree, path: values[path], depths=[2], num_children=2, num_simulations=4)
    assert trace.counts == [[3, 1]]
    assert [s for _, s in _evaluated(trace)] == [(0,), (1,), (0, 0), (0, 0)]
    assert [s for r in trace.expanded for _, s in r] == [(0,), (1,)]
    assert [d for r in trace.evaluated for _, _, d in r] == [1, 1, 0, 0]


@pytest.mark.parametrize("impl", EXACT_IMPLS)
def test_puct_ties_go_to_the_first_maximal_child(impl):
    # Equal values and priors: after the forced round-robin the exploration term alone orders the children, so visits
    # rotate and every tie resolves to the lowest index: [1,1,1] -> child 0 -> children 1, 2 tie -> 1 -> 2 -> 0
    trace = Trace(impl, _by_first_child([0.5, 0.5, 0.5]), depths=[1], num_children=3, num_simulations=7)
    assert trace.counts == [[3, 2, 2]]
    assert [s for _, s in _evaluated(trace)] == [(0,), (1,), (2,), (0,), (1,), (2,), (0,)]


@pytest.mark.parametrize("impl", EXACT_IMPLS)
def test_priors_order_unvisited_children(impl):
    # Child 1 has the larger prior; among unvisited siblings (Q = 0) PUCT follows the prior
    values = {(0,): 0.5, (1,): 0.5, (0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 0.5}
    trace = Trace(impl, lambda tree, path: values[path], depths=[2], num_children=2, num_simulations=4, priors=[[0.3, 0.7]])
    # Round 2: child 1 (prior .7) beats child 0; its own children have uniform priors, so (1, 0) is first
    assert [s for _, s in _evaluated(trace)][2] == (1, 0)


@pytest.mark.parametrize("impl", EXACT_IMPLS)
def test_every_simulation_evaluates_one_leaf_per_tree_and_expands_those_with_depth(impl):
    trace = Trace(impl, _by_first_child([0.1, 0.9]), depths=[2, 3, 1], num_children=2, num_simulations=6)
    evaluated = _evaluated(trace)
    assert len(evaluated) == 6 * 3
    assert [t for t, _ in evaluated] == [0, 1, 2] * 6  # tree-major within every simulation
    growing = sum(len(round_) for round_ in trace.expanded)
    assert growing == sum(1 for r in trace.evaluated for _, _, d in r if d > 0)


@pytest.mark.parametrize("impl", EXACT_IMPLS)
def test_counts_match_the_baseline_on_random_trees(impl):
    if impl == "python":
        pytest.skip("the baseline is the reference")
    g = torch.Generator().manual_seed(0)
    for case in range(5):
        table = {}

        def values(tree, path, table=table):
            if (tree, path) not in table:
                table[(tree, path)] = torch.rand((), generator=g).item()
            return table[(tree, path)]

        depths = [int(d) for d in torch.randint(1, 4, (4,), generator=g)]
        priors = torch.rand(4, 3, generator=g).softmax(-1).tolist()
        kwargs = dict(depths=depths, num_children=3, num_simulations=10, priors=priors)
        reference = Trace("python", lambda t, p: values(t, p), **kwargs)
        table_ref = dict(table)
        table.clear(); table.update(table_ref)
        other = Trace(impl, lambda t, p: table_ref.setdefault((t, p), values(t, p)), **kwargs)
        assert other.counts == reference.counts, f"case {case}"


# Real-model parity: the posterior logits of TreeRMInner._search are what the model sees

def _search_inputs(model, batch, seed=0):
    inner = model.model.inner
    torch.manual_seed(seed)
    shape = (batch["inputs"].shape[0], inner.config.seq_len, inner.config.hidden_size)
    u_t, z_L, x = (torch.randn(shape, device="cuda", dtype=inner.forward_dtype) for _ in range(3))
    cand = inner._candidates(z_L, u_t)
    prior = inner.candidate_scorer(u_t, cand, inner._cos_sin())
    return cand, prior, z_L, x


# (impl, search_compile): every implementation (all exact), and the fused / cpp ones with their compiled round
VARIANTS = [(impl, False) for impl in IMPLS] + [("fused", True), ("cpp", True)]


def _posterior(impl, compile_, training):
    """The posterior logits of one search on the tiny model (same seed, so the same weights) under impl. float32: the
    parity under test is the search's decisions given the same values; in bf16 the compiled round rounds a last bit
    differently from eager, which can flip a PUCT pick (the bench's parity field measures that on the real model)."""
    model, batch = _treerm_head(mcts_impl=impl, search_compile=compile_, mcts_simulations=8, num_candidates=3, forward_dtype="float32")
    model.train(training)
    inner = model.model.inner
    remaining = [inner.config.H_cycles] * batch["inputs"].shape[0]
    with torch.no_grad():
        cand, prior, z_L, x = _search_inputs(model, batch)
        return inner._search(cand, prior, z_L, x, batch["labels"], remaining)


@needs_cuda
@pytest.mark.parametrize("impl,compile_", VARIANTS)
@pytest.mark.parametrize("training", [True, False])
def test_search_posterior_matches_the_baseline_on_the_model(impl, compile_, training):
    reference = _posterior("python", False, training)
    posterior = _posterior(impl, compile_, training)
    torch.testing.assert_close(posterior, reference)


@needs_cuda
def test_compiled_search_runs_in_the_module_copy_that_calls_it():
    # The EMA eval model is a deepcopy of the training model; its search must use the copy's mode and weights
    model, batch = _treerm_head(mcts_impl="fused", search_compile=True, mcts_simulations=8, num_candidates=3)
    model.train()
    copied = copy.deepcopy(model)
    copied.eval()
    inner = copied.model.inner
    inner.rollout_loss_fn = None  # as under the inference loss head: an eval search must never need the label CE
    with torch.no_grad():
        inner.candidate_generator.gate.bias.fill_(0.5)  # the copy's weights differ from the original's
        cand, prior, z_L, x = _search_inputs(copied, batch)
        n = cand.shape[0]
        args = (cand[:, 0].contiguous(), z_L, x, batch["labels"], torch.full((n,), 2, device="cuda"), 2)
        compiled = _compiled_step_core()(inner, *args)  # a copy in eval mode: no label CE, the copy's weights
        eager = inner._step_core(*args)
    # Inductor's fused kernels round differently from eager: ~1e-2 on bf16 latents after 10 passes, nothing like the
    # O(1) difference the original's weights (gate 1.0 vs 0.5: sqrt(H)/2 per element) would give
    for got, want in zip(compiled, eager):  # values, targets, leaf tokens, candidates, priors, next z_L
        torch.testing.assert_close(got.float(), want.float(), rtol=5e-2, atol=0.25 if got.dtype == torch.bfloat16 else 5e-2)


@needs_cuda
@pytest.mark.parametrize("impl,compile_", VARIANTS)
def test_treerm_trains_and_evals_with_the_impl(impl, compile_):
    model, batch = _treerm_head(mcts_impl=impl, search_compile=compile_)
    model.train()
    _, loss, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert torch.isfinite(loss) and metrics["kl_loss"] > 0
    loss.backward()
    model.eval()
    with torch.inference_mode():
        _, outputs = model.model(carry=model.initial_carry(batch), batch=batch)
    assert torch.isfinite(outputs["logits"]).all()
    # Training resumes after an in-training eval on the same instance (pretrain.py swaps the EMA weights in place):
    # the search's arena made under inference_mode must not be written to outside it
    model.train()
    _, loss, _, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert torch.isfinite(loss)


def test_cpp_falls_back_to_fused_without_a_toolchain(monkeypatch):
    import warnings

    from models.mcts import CppMCTS, FusedMCTS

    def unbuildable(self, *args, **kwargs):
        raise ImportError("no g++")

    monkeypatch.setattr(CppMCTS, "__init__", unbuildable)
    cfg = _cfg(**{
        "kl_grad_prior": 0.8, "kl_strength": 0.07, "grad_cycles": 1, "posterior_nograd_cycles": False, "prior_sample_frac": 0.0,
        "prior_sample_per_puzzle": False, "num_candidates": 3, "mcts_simulations": 6, "mcts_c_puct": 1.25, "mcts_impl": "cpp",
        "search_compile": False, "candidate_loss": "none", "candidate_loss_eps": 0.05, "value_bootstrap_weight": 0.0, "eval_mcts_simulations": None, "eval_pick": "argmax", "mcts_horizon": "step", "lprm_pool": "first", "lprm_target": "step", "lprm_prior_rollouts": 0, "posterior_y_pool": "position", "lprm_weight": 0.5, "carry_norm": True, "kl_reduction": "mean", "lprm_rollout_cycles": "grad",
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = TreeRM(cfg)
    assert type(model.inner.mcts) is FusedMCTS and any("falling back" in str(w.message) for w in caught)


def test_unknown_impl_is_rejected():
    cfg = _cfg(**{
        "kl_grad_prior": 0.8, "kl_strength": 0.07, "grad_cycles": 1, "posterior_nograd_cycles": False, "prior_sample_frac": 0.0,
        "prior_sample_per_puzzle": False, "num_candidates": 3, "mcts_simulations": 6, "mcts_c_puct": 1.25, "mcts_impl": "nope",
        "search_compile": False, "candidate_loss": "none", "candidate_loss_eps": 0.05, "value_bootstrap_weight": 0.0, "eval_mcts_simulations": None, "eval_pick": "argmax", "mcts_horizon": "step", "lprm_pool": "first", "lprm_target": "step", "lprm_prior_rollouts": 0, "posterior_y_pool": "position", "lprm_weight": 0.5, "carry_norm": True, "kl_reduction": "mean", "lprm_rollout_cycles": "grad",
    })
    with pytest.raises(ValueError, match="mcts_impl"):
        TreeRM(cfg)
    cfg.update(mcts_impl="python", search_compile=True)
    with pytest.raises(ValueError, match="search_compile"):
        TreeRM(cfg)  # only the fused round has the fixed shapes a compiled callee needs
