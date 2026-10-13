import math
from typing import Any, Callable, List, Sequence, Tuple

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
