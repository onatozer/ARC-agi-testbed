"""TreeGRAM: TreeRM's search over GRAM's learned prior instead of a candidate head (REPORTS_SCALING.md). The K
candidates for a z_H update are samples of GRAM's prior p(eps | u_t): slot 0 is the prior mean (so K=1 is GRAM's
eval_prior_mean inference exactly), the others add sqrt(var) * noise. The search prior over them is uniform, as in
Sampled MuZero (the samples already carry the prior), so there is no scorer; the pick, the candidate loss and the
value bootstrap train the prior head through the samples."""
import torch

from .gram import GRAM, GaussianHead
from .merged_arch import _add_puzzle_emb_positions
from .treerm import TreeRMInner


class TreeGRAMInner(TreeRMInner):
    def __init__(self, config: dict):
        super().__init__(config)
        del self.candidate_generator, self.candidate_scorer
        self.prior_head = GaussianHead(dim=self.config.hidden_size, expansion=self.config.expansion, norm_eps=self.config.rms_norm_eps)
        if self.config.prior_target not in ("visits", "none"):
            raise ValueError(f"prior_target must be 'visits' or 'none', got {self.config.prior_target!r}.")

    def _candidates(self, z_L: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """[B, K, S, H]: u_t + mu, then u_t + mu + sqrt(var) * noise for the other K - 1 slots."""
        mu, var = self.prior_head(u_t)
        noise = torch.randn((u_t.shape[0], self.config.num_candidates) + u_t.shape[1:], device=u_t.device, dtype=u_t.dtype)
        noise[:, 0] = 0
        return (u_t + mu).unsqueeze(1) + var.sqrt().to(u_t.dtype).unsqueeze(1) * noise

    def _prior_logits(self, u_t: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        return torch.zeros(cand.shape[:2], device=cand.device, dtype=torch.float32)

    def _prior_loss(self, u_t: torch.Tensor, cand: torch.Tensor, prior_logits: torch.Tensor, post_logits: torch.Tensor) -> torch.Tensor:
        """prior_target=visits: the search's visit-weighted Gaussian NLL of the (detached) samples under the prior, so
        mu moves toward the samples the search favoured and var toward their spread (Sampled MuZero's policy target
        for a sampled continuous action space); 'none' = pathwise only. Only slots 1..K-1 are draws: slot 0 is the
        mean, whose zero residual would drive var to its floor and the mean's 1/var gradient to infinity (it did).
        Summed over the hidden dim, averaged over tokens, summed over the batch and weighted by kl_strength like
        GRAM's KL."""
        if self.config.prior_target == "none" or cand.shape[1] == 1:
            return prior_logits.new_zeros(())
        mu, var = (t.float() for t in self.prior_head(u_t))
        diff = cand[:, 1:].detach().float() - mu.unsqueeze(1)  # [B, K-1, S, H]
        nll = 0.5 * (diff.pow(2) / var.unsqueeze(1) + var.log().unsqueeze(1)).sum(-1).mean(-1)  # [B, K-1]
        weights = post_logits[:, 1:].float().softmax(dim=-1) * (1 - post_logits.float().softmax(dim=-1)[:, :1])  # the draws' visit share
        return (weights * nll).sum() * self.config.kl_strength


class TreeGRAM(GRAM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.inner = TreeGRAMInner(_add_puzzle_emb_positions(config))
