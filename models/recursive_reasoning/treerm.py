from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.losses import IGNORE_LABEL_ID
from models.losses.base import kl_balanced_logits
from models.mcts import MCTS, Node
from ..layers import (
    CastedLinear,
    SwiGLU,
    rms_norm,
)
from .merged_arch import ModelCarry, LatentCarry, _add_puzzle_emb_positions
from .gram import GRAM, GRAM_Inner
from .grampd import CandidateScorer


class CandidateGenerator(nn.Module):
    """Proposes num_candidates full z_H vectors from the new z_H and z_L: [B, S, H] x [B, S, H] -> [B, K, S, H]."""

    def __init__(self, hidden_size: int, num_candidates: int, expansion: float):
        super().__init__()

        self.num_candidates = num_candidates

        # Not zero-initialised: identical candidates would all get the same gradient direction and never separate
        self.head = nn.Sequential(
            CastedLinear(hidden_size*2, hidden_size, bias=True),
            SwiGLU(hidden_size, expansion),
            CastedLinear(hidden_size, hidden_size*num_candidates, bias=True),
        )

    def forward(self, z_L: torch.Tensor, z_H: torch.Tensor) -> torch.Tensor:
        x = self.head(torch.cat([z_L, z_H], dim=-1))
        return x.unflatten(-1, (self.num_candidates, -1)).transpose(1, 2)


class TreeRMInner(GRAM_Inner):
    """GRAM_Inner where each z_H update is one of num_candidates proposals, picked by a per-puzzle MCTS. The prior p(k)
    is the candidate scorer; the posterior is the search's root visit distribution, valued by the label CE of a rollout
    in training and by the LPRM head at eval."""

    def __init__(self, config: dict):
        super().__init__(config)

        # Replace GRAM's Gaussian prior / posterior heads: the prior is the scorer, the posterior comes from the search
        del self.prior_head, self.posterior_head
        self.candidate_generator = CandidateGenerator(
            hidden_size=self.config.hidden_size,
            num_candidates=self.config.num_candidates,
            expansion=self.config.expansion,
        )
        self.candidate_scorer = CandidateScorer(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            norm_eps=self.config.rms_norm_eps,
        )
        self.mcts = MCTS(num_simulations=self.config.mcts_simulations, c_puct=self.config.mcts_c_puct)

        if self.config.mcts_horizon not in ("step", "full"):
            raise ValueError(f"mcts_horizon must be 'step' or 'full', got {self.config.mcts_horizon!r}.")

        # Token-wise CE the training rollouts are valued with; LossHeadTreeRM sets it to the loss head's loss_fn
        self.rollout_loss_fn: Optional[Callable[..., torch.Tensor]] = None
        # Per-puzzle supervision steps already run, set by TreeRM.forward; the 'full' horizon needs it
        self.steps_done: Optional[torch.Tensor] = None

    def _decode(self, z_H: torch.Tensor) -> torch.Tensor:
        # Clip out the puzzle embedding dims from the decoder
        if self.config.puzzle_emb_ndim != 0:
            return self.lm_head(z_H[:, self.config.puzzle_emb_len :]).contiguous()
        return self.lm_head(z_H).contiguous()

    def _lprm_value(self, z_H: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.v_head(z_H[:, 0]).to(torch.float32)).squeeze(-1)

    def _candidates(self, z_L: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """The K proposals for the new z_H: the generator's output, or with candidate_residual u_t plus it."""
        cand = self.candidate_generator(z_L, u_t)
        if self.config.candidate_residual:
            cand = cand + u_t.unsqueeze(1)
        return cand

    def _remaining_updates(self, cycle: int, batch_size: int) -> List[int]:
        """z_H updates left from this cycle's on (inclusive), per puzzle: to the end of this supervision step, or with
        mcts_horizon = 'full' to the end of the last one."""
        left = self.config.H_cycles - cycle
        if self.config.mcts_horizon == "step":
            return [left] * batch_size
        return [left + self.config.H_cycles * (self.config.halt_max_steps - int(s) - 1) for s in self.steps_done.tolist()]

    # A search state is (batch row, z_H [S, H], z_L [S, H]); the callbacks below batch over the leaves of every tree

    def _expand(self, states: List[Tuple[int, torch.Tensor, torch.Tensor]], x: torch.Tensor) -> Tuple[List[List[Tuple[int, torch.Tensor, torch.Tensor]]], List[List[float]]]:
        """The transition: one L-cycle pass from each (z_H, z_L), then the candidates for the new z_H and their prior."""
        rows = [row for row, _, _ in states]
        z_H = torch.stack([s[1] for s in states])
        z_L = torch.stack([s[2] for s in states])
        x = x[rows]

        seq_dict = self._seq_dict()
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + x, **seq_dict)
        u_t = self.L_level(z_H, z_L, **seq_dict)

        cand = self._candidates(z_L, u_t)
        priors = self.candidate_scorer(u_t, cand, seq_dict["cos_sin"]).softmax(dim=-1).tolist()
        children = [[(row, cand[i, k], z_L[i]) for k in range(cand.shape[1])] for i, row in enumerate(rows)]
        return children, priors

    def _leaf_values(self, z_H: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-row value of a final z_H: the label's -mean token CE in training, the LPRM value at eval."""
        z_H = rms_norm(z_H, self.config.rms_norm_eps)
        if not self.training:
            return self._lprm_value(z_H)

        if self.rollout_loss_fn is None:
            raise RuntimeError("TreeRM's training search needs rollout_loss_fn; wrap the model in LossHeadTreeRM.")
        mask = y != IGNORE_LABEL_ID
        ce = self.rollout_loss_fn(self._decode(z_H), y, valid_mask=mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        return -ce

    def _rollout_values(self, states: List[Tuple[int, torch.Tensor, torch.Tensor]], depths: List[int], x: torch.Tensor, y: torch.Tensor) -> List[float]:
        """The simulation: from each leaf, its allowed depth's worth of plain TRM updates (no candidates), then its value."""
        rows = [row for row, _, _ in states]
        z_H = torch.stack([s[1] for s in states])
        z_L = torch.stack([s[2] for s in states])
        x, y = x[rows], y[rows]
        depths_t = torch.tensor(depths, device=z_H.device)
        values = torch.zeros(len(states), device=z_H.device)

        seq_dict = self._seq_dict()
        for n_updates in range(max(depths) + 1):
            done = depths_t == n_updates
            if done.any():
                values[done] = self._leaf_values(z_H[done], y[done]).float()
            if n_updates == max(depths):
                break
            for _ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + x, **seq_dict)
            z_H = self.L_level(z_H, z_L, **seq_dict)

        return values.tolist()

    @torch.compiler.disable
    @torch.no_grad()
    def _search(self, cand: torch.Tensor, prior_logits: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, y: torch.Tensor, remaining: List[int]) -> torch.Tensor:
        """One MCTS per puzzle from the root candidates, run in lockstep; returns the posterior logits log(visit
        distribution) [B, K]."""
        priors = prior_logits.float().softmax(dim=-1).tolist()
        roots = []
        for b in range(cand.shape[0]):
            root = Node(latent=None, allowed_depth=remaining[b])
            root.expand([(b, cand[b, k], z_L[b]) for k in range(cand.shape[1])], priors[b])
            roots.append(root)

        counts = self.mcts.run(
            roots,
            expand_fn=lambda states: self._expand(states, x),
            evaluate_fn=lambda states, depths: self._rollout_values(states, depths, x, y),
        )
        visits = torch.tensor(counts, dtype=torch.float32, device=cand.device)
        posterior = visits / visits.sum(dim=-1, keepdim=True)
        # Mix in the uniform so a one-hot visit distribution can't drive the prior's logits to infinity through the KL
        eps = self.config.posterior_smoothing
        posterior = (1 - eps) * posterior + eps / posterior.shape[-1]
        return torch.log(posterior.clamp_min(1e-8))

    def _mcts_guidance(self, u_t: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, y: torch.Tensor, remaining: List[int]) -> Tuple[torch.Tensor, torch.Tensor | int]:
        """Picks the next z_H as one candidate sampled from the MCTS posterior, with its beta-weighted balanced KL to the
        scorer's prior (only when gradients are on)."""
        cand = self._candidates(z_L, u_t)
        prior_logits = self.candidate_scorer(u_t, cand, self._cos_sin())
        if self.training or self.config.eval_search:
            post_logits = self._search(cand.detach(), prior_logits.detach(), z_L.detach(), x.detach(), y, remaining)
        else:
            # Diagnostic: no search at eval, the prior alone picks the candidate (as GRAMDP does at eval)
            post_logits = prior_logits.detach()

        kl_loss = 0
        # Only the grad cycles pay the KL; the posterior has no parameters, so only the prior-update term trains anything
        if torch.is_grad_enabled():
            kl_loss = kl_balanced_logits(
                prior_logits=prior_logits,
                post_logits=post_logits,
                alpha=self.config.kl_grad_prior,
            ).sum() * self.config.kl_strength
            # Diversity penalty (folded into the guidance loss): mean pairwise cosine between a puzzle's candidates,
            # summed over the batch. Without it the candidates collapse onto one another (see REPORTS.md)
            if self.config.candidate_diversity_weight > 0:
                kl_loss = kl_loss + self.config.candidate_diversity_weight * self._pairwise_cosine(cand).sum()

        # Straight-through Gumbel-softmax as in GRAMDP: the forward value of the logits is the posterior, so the one-hot
        # sample is an exact draw from the visit distribution, while the backward pass reaches the scorer's prior logits
        logits = prior_logits + (post_logits - prior_logits).detach()
        weights = F.gumbel_softmax(logits.float(), tau=self.config.gumbel_temp, hard=True, dim=-1)

        z_H = torch.einsum("bk,bksh->bsh", weights, cand.float()).to(u_t.dtype)
        diag = self._search_diagnostics(cand, prior_logits, post_logits)
        with torch.no_grad():  # candidate norm relative to the TRM update it replaces
            diag["search/cand_norm_ratio"] = cand.float().flatten(2).norm(dim=-1).mean(-1) / u_t.float().flatten(1).norm(dim=-1).clamp_min(1e-6)
        return z_H, kl_loss, diag

    @staticmethod
    def _pairwise_cosine(cand: torch.Tensor) -> torch.Tensor:
        """Per-puzzle [B] mean cosine between each pair of distinct candidates, over their flattened tokens."""
        unit = F.normalize(cand.float().flatten(2), dim=-1)
        cos = unit @ unit.transpose(1, 2)
        num_candidates = cand.shape[1]
        return (cos.sum(dim=(1, 2)) - cos.diagonal(dim1=1, dim2=2).sum(-1)) / max(num_candidates * (num_candidates - 1), 1)

    @torch.no_grad()
    def _search_diagnostics(self, cand: torch.Tensor, prior_logits: torch.Tensor, post_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-puzzle [B] diagnostics of one search: how different the candidates are (mean pairwise 1 - cosine over
        their flattened tokens), the posterior's entropy, its unweighted KL from the prior, and whether the prior and
        posterior agree on the best candidate."""
        log_post, log_prior = post_logits.float().log_softmax(-1), prior_logits.float().log_softmax(-1)
        p_post = log_post.exp()
        return {
            "search/cand_div": 1 - self._pairwise_cosine(cand),
            "search/post_entropy": -(p_post * log_post).sum(-1),
            "search/kl_nats": (p_post * (log_post - log_prior)).sum(-1),
            "search/argmax_agree": (post_logits.argmax(-1) == prior_logits.argmax(-1)).float(),
        }

    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, dict]:
        z_H, z_L = carry.z_H, carry.z_L
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"], batch.get("object_ids"))
        y = batch["labels"]
        seq_dict = self._seq_dict()
        batch_size = z_H.shape[0]
        diagnostics: Dict[str, torch.Tensor] = {}

        def accumulate(diag: Dict[str, torch.Tensor]) -> None:
            for key, value in diag.items():
                diagnostics[key] = diagnostics.get(key, 0) + value / self.config.H_cycles

        # Run the recursion cycle for H_cycles - grad_cycles times without the gradient; every z_H update is searched
        with torch.no_grad():
            for cycle in range(self.config.H_cycles - self.config.grad_cycles):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + x, **seq_dict)

                u_t = self.L_level(z_H, z_L, **seq_dict)
                z_H, _, diag = self._mcts_guidance(u_t, z_L, x, y, self._remaining_updates(cycle, batch_size))
                accumulate(diag)

        # The last grad_cycles cycles carry the gradient; each pays the KL of its search's posterior from the prior
        kl_loss = 0
        for cycle in range(self.config.H_cycles - self.config.grad_cycles, self.config.H_cycles):
            for __ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + x, **seq_dict)

            u_t = self.L_level(z_H, z_L, **seq_dict)
            z_H, guidance_loss, diag = self._mcts_guidance(u_t, z_L, x, y, self._remaining_updates(cycle, batch_size))
            kl_loss += guidance_loss
            accumulate(diag)

        z_H = rms_norm(z_H, self.config.rms_norm_eps)

        info = {
            "lprm_value": self._lprm_value(z_H),
            "logits": self._decode(z_H),
            "kl_loss": kl_loss,
            **diagnostics,  # per-puzzle, averaged over the H_cycles searches
        }

        return LatentCarry(z_H, z_L), info


class TreeRM(GRAM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.inner = TreeRMInner(_add_puzzle_emb_positions(config))

    def forward(self, carry: ModelCarry, batch: Dict[str, torch.Tensor], **kwargs: Any) -> Tuple[ModelCarry, Dict[str, torch.Tensor]]:
        """GRAM.forward, after handing the inner each puzzle's supervision steps so far (halted puzzles restart at 0)."""
        if carry is None:
            carry = self.initial_carry(batch)
        self.inner.steps_done = torch.where(carry.halted, 0, carry.steps).cpu()
        return super().forward(carry, batch, **kwargs)
