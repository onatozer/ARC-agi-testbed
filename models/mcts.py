import torch

#NOTE: Worth considering if we think it's worth it to store all the latents in the nodes or not, might be overkill in terms of memory
class Node:
    def __init__(self, latent: torch.Tensor, allowed_depth: int, parent = None):
        self.parent = parent
        self.state = latent
        self.is_expanded = False
        self.num_visits = 0
        self.total_value = 0
        self.allowed_depth = allowed_depth # Since there's no defined 'terminal state', instead we just cap the depth

    def expand(self, latents: list[torch.Tensor]):
        """Given a set of tensors the correspond to the next states of the node,
          expand the node's internal representation to include said nodes"""

        if self.is_expanded or self.allowed_depth == 0:
            return None

        self.children = [Node(latent=latent, allowed_depth=self.allowed_depth-1, parent = self) for latent in latents]
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



class MCTSBase:

    def __init__(self):
        self.nodes
        pass

    def select(self):
        raise NotImplementedError

    def expand(self):
        raise NotImplementedError

    