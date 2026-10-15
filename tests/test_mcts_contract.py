"""Contract tests every models.mcts.MCTS_IMPLS implementation must pass (the contract is the module docstring of
models/mcts.py). Exact implementations (impl.exact) must reproduce the baseline's visit counts and callback
traffic; semantics-changing ones only keep the invariants."""
import pytest
import torch

from models.mcts import MCTS, MCTS_IMPLS, Node
from models.recursive_reasoning.treerm import TreeRM

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

@pytest.mark.parametrize("impl", EXACT_IMPLS)
def test_first_k_simulations_visit_each_root_child_once_in_order(impl):
    trace = Trace(impl, _by_first_child([0.0, 1.0, 0.5, 0.2]), depths=[1, 2], num_children=4, num_simulations=6)
    for k in range(4):
        assert [state for _, state, _ in trace.evaluated[k]] == [(k,), (k,)]


@pytest.mark.parametrize("impl", EXACT_IMPLS)
def test_hand_computed_depth_two_search(impl):
    # c_puct 1.25, uniform priors 0.5. Rounds: (0,) v=.5 and (1,) v=.2 are forced; round 2 picks child 0 (Q norm 1 vs
    # 0) then its first child (unvisited, tied, first max) (0,0) v=.9; round 3: child 0 again (Q .714 + .361 beats
    # 0 + .541), inside it (0,0) (1 + .442 beats 0 + .884). Depth-0 leaves are not expanded.
    values = {(0,): 0.5, (1,): 0.2, (0, 0): 0.9, (0, 1): 0.1, (1, 0): 0.4, (1, 1): 0.3}
    trace = Trace(impl, lambda tree, path: values[path], depths=[2], num_children=2, num_simulations=4)
    assert trace.counts == [[3, 1]]
    assert [[s for _, s, _ in r] for r in trace.evaluated] == [[(0,)], [(1,)], [(0, 0)], [(0, 0)]]
    assert [[s for _, s in r] for r in trace.expanded] == [[(0,)], [(1,)]]
    assert [[d for _, _, d in r] for r in trace.evaluated] == [[1], [1], [0], [0]]


@pytest.mark.parametrize("impl", EXACT_IMPLS)
def test_puct_ties_go_to_the_first_maximal_child(impl):
    # Equal values and priors: after the forced round-robin the exploration term alone orders the children, so visits
    # rotate and every tie resolves to the lowest index: [1,1,1] -> child 0 -> children 1, 2 tie -> 1 -> 2 -> 0
    trace = Trace(impl, _by_first_child([0.5, 0.5, 0.5]), depths=[1], num_children=3, num_simulations=7)
    assert trace.counts == [[3, 2, 2]]
    assert [[s for _, s, _ in r] for r in trace.evaluated] == [[(0,)], [(1,)], [(2,)], [(0,)], [(1,)], [(2,)], [(0,)]]


@pytest.mark.parametrize("impl", EXACT_IMPLS)
def test_priors_order_unvisited_children(impl):
    # Child 1 has the larger prior; among unvisited siblings (Q = 0) PUCT follows the prior
    values = {(0,): 0.5, (1,): 0.5, (0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 0.5}
    trace = Trace(impl, lambda tree, path: values[path], depths=[2], num_children=2, num_simulations=4, priors=[[0.3, 0.7]])
    # Round 2: child 1 (prior .7) beats child 0; its own children have uniform priors, so (1, 0) is first
    assert [[s for _, s, _ in r] for r in trace.evaluated][2] == [(1, 0)]


@pytest.mark.parametrize("impl", EXACT_IMPLS)
def test_one_evaluate_call_per_simulation_over_every_tree(impl):
    trace = Trace(impl, _by_first_child([0.1, 0.9]), depths=[2, 3, 1], num_children=2, num_simulations=6)
    assert len(trace.evaluated) == 6 and all(len(round_) == 3 for round_ in trace.evaluated)
    assert [[t for t, _, _ in r] for r in trace.evaluated] == [[0, 1, 2]] * 6
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


@needs_cuda
@pytest.mark.parametrize("impl", EXACT_IMPLS)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("horizon", ["step", "full"])
def test_search_posterior_matches_the_baseline_on_the_model(impl, training, horizon):
    model, batch = _treerm_head(mcts_horizon=horizon, mcts_simulations=8, num_candidates=3, candidate_mode="flags")
    model.train(training)
    inner = model.model.inner
    inner.steps_done = torch.zeros(batch["inputs"].shape[0], dtype=torch.long)
    remaining = inner._remaining_updates(0, batch["inputs"].shape[0])
    posteriors = {}
    for name in ("python", impl):
        inner.mcts = MCTS_IMPLS[name](num_simulations=inner.config.mcts_simulations, c_puct=inner.config.mcts_c_puct)
        with torch.no_grad():
            cand, prior, z_L, x = _search_inputs(model, batch)
            posteriors[name] = inner._search(cand, prior, z_L, x, batch["labels"], remaining)
    torch.testing.assert_close(posteriors[impl], posteriors["python"])


@needs_cuda
@pytest.mark.parametrize("impl", IMPLS)
def test_treerm_trains_and_evals_with_the_impl(impl):
    model, batch = _treerm_head(mcts_impl=impl, candidate_mode="flags")
    model.train()
    _, loss, metrics, _, _ = model(return_keys=[], carry=model.initial_carry(batch), batch=batch)
    assert torch.isfinite(loss) and metrics["kl_loss"] > 0
    loss.backward()
    model.eval()
    with torch.inference_mode():
        _, outputs = model.model(carry=model.initial_carry(batch), batch=batch)
    assert torch.isfinite(outputs["logits"]).all()


def test_unknown_impl_is_rejected():
    cfg = _cfg(**{
        "kl_grad_prior": 0.8, "kl_strength": 0.07, "grad_cycles": 1, "posterior_nograd_cycles": False, "prior_sample_frac": 0.0,
        "prior_sample_per_puzzle": False, "num_candidates": 3, "mcts_simulations": 6, "mcts_c_puct": 1.25, "mcts_impl": "nope",
        "mcts_horizon": "step", "gumbel_temp": 1.0, "candidate_mode": "flags", "eval_search": True, "posterior_smoothing": 0.0,
        "candidate_diversity_weight": 0.0, "value_bootstrap_weight": 0.0, "rollout_policy": "trm", "argmax_pick": False, "eval_mcts_simulations": None,
    })
    with pytest.raises(ValueError, match="mcts_impl"):
        TreeRM(cfg)
