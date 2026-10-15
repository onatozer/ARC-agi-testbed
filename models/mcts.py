"""Per-puzzle MCTS over TreeRM's z_H candidates. MCTS_IMPLS (bottom) lists the searches TreeRM can run (config
mcts_impl); the baseline MCTS is the reference every other implementation is compared against.

Contract of an implementation, Impl(num_simulations, c_puct):
- attribute num_simulations: TreeRM overwrites it before every search (eval may search deeper than training);
- run(roots, expand_fn, evaluate_fn) -> each root's children's visit counts, in child order. The roots are Nodes already
  expanded with their K child states and priors; root.allowed_depth >= 1 is how many levels its tree may grow, a child
  has one less (so a root child may have 0). expand_fn(states) -> (child states per leaf, child priors per leaf) is the
  transition, evaluate_fn(states, allowed_depths) -> one value per leaf, higher is better; both batch over the leaves of
  every tree so the domain can batch its work. Nothing reads the Node tree after run returns;
- class attribute exact: True when the search reproduces the baseline's counts given the same inputs and callback
  values; a search that changes the semantics sets False, and tests/test_mcts_contract.py skips its parity tests.

Invariants every implementation keeps (tests/test_mcts_contract.py):
- each root's counts sum to num_simulations and, with num_simulations >= K, every root child is visited; the baseline's
  first K simulations visit child 0..K-1 once each, in order, before PUCT takes over;
- a leaf with allowed_depth 0 is evaluated but never expanded: expand_fn only sees leaves with depth left, no state is
  created below the root's allowed depth, evaluate_fn gets every leaf with its own allowed_depth;
- Q is min-max normalised per tree (MinMaxStats): an affine rescaling of a tree's values changes nothing, and one tree
  never affects another's counts; PUCT ties go to the first maximal child;
- deterministic: identical inputs give identical counts.
"""

import math
from typing import Any, Callable, Dict, List, Sequence, Tuple

#NOTE: Worth considering if we think it's worth it to store all the latents in the nodes or not, might be overkill in terms of memory
class Node:
    def __init__(self, latent: Any, allowed_depth: int, prior: float = 1.0, parent = None):
        self.parent = parent
        self.state = latent
        self.prior = prior
        self.children: List["Node"] = []
        self.is_expanded = False
        self.num_visits = 0
        self.total_value = 0
        self.allowed_depth = allowed_depth # Since there's no defined 'terminal state', instead we just cap the depth

    @property
    def q(self) -> float:
        return self.total_value / self.num_visits if self.num_visits else 0.0

    def expand(self, latents: Sequence[Any], priors: Sequence[float]):
        """Given a set of tensors the correspond to the next states of the node and their prior probabilities,
          expand the node's internal representation to include said nodes"""

        if self.is_expanded or self.allowed_depth == 0:
            return None

        self.children = [Node(latent=latent, allowed_depth=self.allowed_depth-1, prior=prior, parent = self) for latent, prior in zip(latents, priors)]
        self.is_expanded = True

    def backpropagate(self, reward):
        #NOTE: Could also write this recursively, but imo don't think it makes any difference (might be slower too ??)
        current_parent = self.parent
        self.total_value += reward
        self.num_visits += 1

        while(current_parent is not None):
            current_parent.total_value += reward
            current_parent.num_visits += 1
            current_parent = current_parent.parent

        return current_parent


class MinMaxStats:
    """MuZero's Q normalisation: rescales values to [0, 1] by the min / max seen in the tree, so PUCT's c_puct doesn't
    depend on the reward's scale."""

    def __init__(self):
        self.minimum, self.maximum = math.inf, -math.inf

    def update(self, value: float) -> None:
        self.minimum, self.maximum = min(self.minimum, value), max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTS:
    """PUCT search over already expanded roots, one tree per root, run in lockstep so the domain can batch its work over
    the trees. The domain comes in as callbacks over the trees' current leaves:
    expand_fn(states) -> (child states per leaf, child priors per leaf) and
    evaluate_fn(states, allowed_depths) -> one value per leaf (higher is better)."""

    exact = True  # the reference semantics (see the module docstring)

    def __init__(self, num_simulations: int, c_puct: float):
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def select(self, node: Node, stats: MinMaxStats) -> Node:
        # max(., 1) so the priors order the children before the node has any visits
        sqrt_visits = math.sqrt(max(node.num_visits, 1))

        def puct(child: Node) -> float:
            q = stats.normalize(child.q) if child.num_visits else 0.0
            return q + self.c_puct * child.prior * sqrt_visits / (1 + child.num_visits)

        return max(node.children, key=puct)

    def run(
        self,
        roots: Sequence[Node],
        expand_fn: Callable[[List[Any]], Tuple[Sequence[Sequence[Any]], Sequence[Sequence[float]]]],
        evaluate_fn: Callable[[List[Any], List[int]], Sequence[float]],
    ) -> List[List[int]]:
        """Runs num_simulations select -> evaluate -> expand -> backpropagate passes on every tree; returns each root's
        children's visit counts."""
        stats = [MinMaxStats() for _ in roots]
        num_children = max(len(root.children) for root in roots)
        for simulation in range(self.num_simulations):
            leaves = []
            for root, tree_stats in zip(roots, stats):
                # The first len(children) simulations evaluate every root child once: PUCT alone would keep picking the
                # first-visited child (unvisited ones score Q = 0, and Q isn't normalised until two values are seen)
                if simulation < num_children:
                    leaves.append(root.children[simulation % len(root.children)])
                    continue
                node = root
                while node.is_expanded:
                    node = self.select(node, tree_stats)
                leaves.append(node)

            # Each leaf is scored by simulating the rest of the horizon from its state, then grows its tree by one level
            values = evaluate_fn([leaf.state for leaf in leaves], [leaf.allowed_depth for leaf in leaves])
            growing = [leaf for leaf in leaves if leaf.allowed_depth > 0]
            if growing:
                for leaf, child_states, priors in zip(growing, *expand_fn([leaf.state for leaf in growing])):
                    leaf.expand(child_states, priors)

            for leaf, value, tree_stats in zip(leaves, values, stats):
                tree_stats.update(value)
                leaf.backpropagate(value)

        return [[child.num_visits for child in root.children] for root in roots]


class FusedMCTS(MCTS):
    """The baseline search with its per-round domain work fused. run takes an optional step_fn(states, allowed_depths)
    -> (values, children per leaf, priors per leaf) that evaluates and expands the round's leaves in one call (children /
    priors are None for leaves with no depth left), so the domain computes the transition once instead of inside both
    callbacks; without it the two callbacks are composed. The forced first-K root-child visits, which are independent
    of one another, run as one batched round (K x B leaves in the baseline's order) instead of K rounds. Visit counts
    are the baseline's (exact)."""

    exact = True
    fused = True  # TreeRM passes step_fn to implementations with this attribute

    def run(self, roots, expand_fn, evaluate_fn, step_fn=None) -> List[List[int]]:
        if step_fn is None:
            step_fn = compose_step(expand_fn, evaluate_fn)
        stats = [MinMaxStats() for _ in roots]
        num_children = max(len(root.children) for root in roots)
        simulation = 0
        while simulation < self.num_simulations:
            if simulation == 0:
                # The baseline's rounds 0..min(K, sims)-1 (root child k of every tree) in one batch, round-major
                forced = min(num_children, self.num_simulations)
                leaves = [(tree, root.children[k % len(root.children)]) for k in range(forced) for tree, root in enumerate(roots)]
                simulation = forced
            else:
                leaves = []
                for tree, (root, tree_stats) in enumerate(zip(roots, stats)):
                    node = root
                    while node.is_expanded:
                        node = self.select(node, tree_stats)
                    leaves.append((tree, node))
                simulation += 1

            values, children, priors = step_fn([leaf.state for _, leaf in leaves], [leaf.allowed_depth for _, leaf in leaves])
            for (tree, leaf), value, child_states, child_priors in zip(leaves, values, children, priors):
                if leaf.allowed_depth > 0:
                    leaf.expand(child_states, child_priors)
                stats[tree].update(value)
                leaf.backpropagate(value)

        return [[child.num_visits for child in root.children] for root in roots]


def compose_step(expand_fn, evaluate_fn):
    """A step_fn from the baseline's two callbacks: evaluate every leaf, expand those with depth left."""

    def step_fn(states, allowed_depths):
        values = evaluate_fn(states, allowed_depths)
        growing = [i for i, depth in enumerate(allowed_depths) if depth > 0]
        children, priors = [None] * len(states), [None] * len(states)
        if growing:
            for i, child_states, child_priors in zip(growing, *expand_fn([states[i] for i in growing])):
                children[i], priors[i] = child_states, child_priors
        return values, children, priors

    return step_fn


# The searches TreeRM can run, by config mcts_impl; every entry honours the contract in the module docstring
MCTS_IMPLS: Dict[str, type] = {"python": MCTS, "fused": FusedMCTS}
