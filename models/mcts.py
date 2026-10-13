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
    """PUCT search over an already expanded root. The domain comes in as callbacks:
    expand_fn(state) -> (child states, child priors) and evaluate_fn(state, allowed_depth) -> value (higher is better)."""

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
        root: Node,
        expand_fn: Callable[[Any], Tuple[Sequence[Any], Sequence[float]]],
        evaluate_fn: Callable[[Any, int], float],
    ) -> List[int]:
        """Runs num_simulations select -> evaluate -> expand -> backpropagate passes; returns the root children's visit counts."""
        stats = MinMaxStats()
        for _ in range(self.num_simulations):
            node = root
            while node.is_expanded:
                node = self.select(node, stats)

            # The leaf is scored by simulating the rest of the horizon from its state, then grows the tree by one level
            value = evaluate_fn(node.state, node.allowed_depth)
            if node.allowed_depth > 0:
                node.expand(*expand_fn(node.state))

            stats.update(value)
            node.backpropagate(value)

        return [child.num_visits for child in root.children]
