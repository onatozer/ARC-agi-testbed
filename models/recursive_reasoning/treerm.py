import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.losses import IGNORE_LABEL_ID
from models.losses.base import kl_balanced_logits
from models.mcts import MCTS_IMPLS, Node
from ..layers import CastedLinear, rms_norm
from .merged_arch import Blocks, LatentCarry, _add_puzzle_emb_positions
from .gram import GRAM, GRAM_Inner
from .grampd import CandidateScorer


class FlagCandidates(nn.Module):
    """Candidates that can't collapse onto one another: u_t plus, per slot k, a fixed random unit direction e_k scaled by
    a learned per-token gate, cand_k = u_t + g_k(z_L, u_t) * sqrt(H) * e_k. The directions are orthonormal and frozen,
    so the slots differ by construction; the network downstream can learn what each "flag" means, and the gates can
    only shrink the flags (collapse point: plain TRM). Slot 0 is the identity (e_0 = 0): the plain update is always
    among the candidates, so a larger K only adds options, and K=1 is plain TRM. [B, S, H] x [B, S, H] -> [B, K, S, H]."""

    def __init__(self, hidden_size: int, num_candidates: int):
        super().__init__()
        self.scale = math.sqrt(hidden_size)
        # Drawn on the CPU with a fixed seed so every instance gets the same directions, whatever device the model is built on
        self.gate = CastedLinear(2 * hidden_size, num_candidates, bias=True)
        directions, _ = torch.linalg.qr(torch.randn(hidden_size, num_candidates, generator=torch.Generator().manual_seed(0), device="cpu"))
        directions = directions.T.contiguous()  # [K, H], orthonormal rows
        directions[0].zero_()  # slot 0 = u_t itself (its gate row is dead weight)
        self.register_buffer("directions", directions.to(self.gate.weight.device), persistent=True)
        # Start with every flag at full strength (gate = 1) and no input dependence
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.fill_(1.0)

    def forward(self, z_L: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([z_L, u_t], dim=-1)).transpose(1, 2).unsqueeze(-1)  # [B, K, S, 1]
        flags = (self.scale * self.directions).to(u_t.dtype)[None, :, None, :]  # [1, K, 1, H]
        return u_t.unsqueeze(1) + gate.to(u_t.dtype) * flags


class NoiselessBlocks(Blocks):
    """Blocks whose forward skips the noise draw when noise_scale is 0: the same value (x + 0 * noise is x) without
    a philox kernel per pass, which matters to TreeRM because its search replays this block hundreds of times per
    step. Only TreeRM uses it: other models keep the draw (TRMRoot's memory savers align their RNG streams on it)."""

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.noise_scale:
            return super().forward(hidden_states, input_injection, **kwargs)
        updated_states = hidden_states + input_injection
        for layer in self.layers:
            updated_states = layer(hidden_states=updated_states, **kwargs)
        return (1 - self.lambda_) * hidden_states + self.lambda_ * updated_states


class TreeRMInner(GRAM_Inner):
    """GRAM_Inner where each z_H update is one of num_candidates proposals, picked by a per-puzzle MCTS. The prior p(k)
    is the candidate scorer; the posterior is the search's root visit distribution, valued by the label CE of a rollout
    in training and by the LPRM head at eval."""

    def __init__(self, config: dict):
        super().__init__(config)

        # Replace GRAM's Gaussian prior / posterior heads: the prior is the scorer, the posterior comes from the search
        del self.prior_head, self.posterior_head
        # The same layers (state-dict keys unchanged) in a block that skips the zero-scale noise draw the search replays
        self.L_level = NoiselessBlocks(layers=list(self.L_level.layers), config=self.config)
        self.candidate_generator = FlagCandidates(hidden_size=self.config.hidden_size, num_candidates=self.config.num_candidates)
        if self.config.candidate_loss not in ("none", "wta", "posterior"):
            raise ValueError(f"candidate_loss must be 'none', 'wta' or 'posterior', got {self.config.candidate_loss!r}.")
        if not 0.0 <= self.config.candidate_loss_eps <= 1.0:
            raise ValueError(f"candidate_loss_eps must be in [0, 1], got {self.config.candidate_loss_eps!r}.")
        if self.config.eval_pick not in ("argmax", "sample"):
            raise ValueError(f"eval_pick must be 'argmax' or 'sample', got {self.config.eval_pick!r}.")
        if self.config.mcts_horizon not in ("step", "full"):
            raise ValueError(f"mcts_horizon must be 'step' or 'full', got {self.config.mcts_horizon!r}.")
        self._rollout_steps_left = 0  # supervision steps a simulation rolls through after this one (mcts_horizon=full, eval)
        self.candidate_scorer = CandidateScorer(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            norm_eps=self.config.rms_norm_eps,
        )
        if self.config.mcts_impl not in MCTS_IMPLS:
            raise ValueError(f"mcts_impl must be one of {sorted(MCTS_IMPLS)}, got {self.config.mcts_impl!r}.")
        try:
            self.mcts = MCTS_IMPLS[self.config.mcts_impl](num_simulations=self.config.mcts_simulations, c_puct=self.config.mcts_c_puct)
        except ImportError as exc:
            if self.config.mcts_impl != "cpp":
                raise
            # The default search needs a C++ toolchain; without one the same search runs with its tree in Python
            warnings.warn(f"mcts_impl=cpp could not be built, falling back to mcts_impl=fused (same visit counts, slower eval): {exc}")
            self.mcts = MCTS_IMPLS["fused"](num_simulations=self.config.mcts_simulations, c_puct=self.config.mcts_c_puct)

        if self.config.search_compile and not getattr(self.mcts, "fused", False):
            raise ValueError("search_compile needs a fused search (mcts_impl=fused or cpp): only its round has fixed shapes.")

        # Token-wise CE the training rollouts are valued with; LossHeadTreeRM sets it to the loss head's loss_fn
        self.rollout_loss_fn: Optional[Callable[..., torch.Tensor]] = None
        self._bootstrap: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def _decode(self, z_H: torch.Tensor) -> torch.Tensor:
        # Clip out the puzzle embedding dims from the decoder
        if self.config.puzzle_emb_ndim != 0:
            return self.lm_head(z_H[:, self.config.puzzle_emb_len :]).contiguous()
        return self.lm_head(z_H).contiguous()

    def _candidates(self, z_L: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """The K proposals for the new z_H: u_t plus a gated flag per slot."""
        return self.candidate_generator(z_L, u_t)

    def _prior_logits(self, u_t: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """The search prior over the candidates, as logits [B, K]: the scorer's."""
        return self.candidate_scorer(u_t, cand, self._cos_sin())

    def _rollout_next(self, u_t: torch.Tensor) -> torch.Tensor:
        """A rollout's next z_H from the plain update u_t: the plain update itself (TreeGRAM adds the prior's mean)."""
        return u_t

    def _rollout_boundary(self, z_H: torch.Tensor, left: torch.Tensor, steps_left: int) -> torch.Tensor:
        """With mcts_horizon=full a rollout crosses supervision steps: rms-norm the state where inference does, i.e.
        the rows with `left` updates to go at a step boundary (a positive multiple of H_cycles within the steps left)."""
        if not steps_left:
            return z_H
        boundary = (left > 0) & (left <= self.config.H_cycles * steps_left) & (left % self.config.H_cycles == 0)
        return torch.where(boundary[:, None, None], rms_norm(z_H, self.config.rms_norm_eps), z_H)

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
        priors = self._prior_logits(u_t, cand).softmax(dim=-1).tolist()
        children = [[(row, cand[i, k], z_L[i]) for k in range(cand.shape[1])] for i, row in enumerate(rows)]
        return children, priors

    def _leaf_values(self, z_H: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-row value of a final z_H (the label's -mean token CE in training, the LPRM value at eval) and the
        rollout's token accuracy (the value head's bootstrap target; meaningless at eval)."""
        z_H = rms_norm(z_H, self.config.rms_norm_eps)
        if not self.training:
            value = self._lprm_value(z_H)
            return value, value

        if self.rollout_loss_fn is None:
            raise RuntimeError("TreeRM's training search needs rollout_loss_fn; wrap the model in LossHeadTreeRM.")
        mask = y != IGNORE_LABEL_ID
        logits = self._decode(z_H)
        ce = self.rollout_loss_fn(logits, y, valid_mask=mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        token_acc = (mask & (logits.argmax(-1) == y)).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        return -ce, token_acc.float()

    def _rollout_values(self, states: List[Tuple[int, torch.Tensor, torch.Tensor]], depths: List[int], x: torch.Tensor, y: torch.Tensor) -> List[float]:
        """The simulation: from each leaf, its allowed depth's worth of plain TRM updates (no candidates), then its value."""
        rows = [row for row, _, _ in states]
        z_H = torch.stack([s[1] for s in states])
        z_L = torch.stack([s[2] for s in states])
        x, y = x[rows], y[rows]
        depths_t = torch.tensor(depths, device=z_H.device)
        values = torch.zeros(len(states), device=z_H.device)
        targets = torch.zeros(len(states), device=z_H.device)
        # The value head reads the leaf's own state (rms-normed as in forward); the rollout gives its target
        leaf_token = self._lprm_token(rms_norm(z_H, self.config.rms_norm_eps))

        seq_dict = self._seq_dict()
        for n_updates in range(max(depths) + 1):
            done = depths_t == n_updates
            if done.any():
                values[done], targets[done] = (t.float() for t in self._leaf_values(z_H[done], y[done]))
            if n_updates == max(depths):
                break
            for _ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + x, **seq_dict)
            z_H = self._rollout_boundary(self._rollout_next(self.L_level(z_H, z_L, **seq_dict)), depths_t - (n_updates + 1), self._rollout_steps_left)

        if self.training and self.config.value_bootstrap_weight > 0:
            self._bootstrap.append((leaf_token, targets))
        return values.tolist()

    def _step_core(self, z_H: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, y: torch.Tensor, depths: torch.Tensor, max_depth: int, steps_left: int = 0):
        """The fixed-shape tensor work of one fused search round over n leaves: max_depth plain TRM updates, each
        leaf's value at its own depth (computed for every row at every stage and selected by depth, so nothing is
        data-dependent), and from the first update the K candidates and their prior for every leaf. Returns
        (values [n], bootstrap targets [n], leaf tokens [n, H], candidates [n, K, S, H], priors [n, K], z_L after the
        first update [n, S, H]); the last three are None when max_depth is 0. Compiled when search_compile is on."""
        values = torch.zeros(z_H.shape[0], device=z_H.device)
        targets = torch.zeros(z_H.shape[0], device=z_H.device)
        leaf_token = self._lprm_token(rms_norm(z_H, self.config.rms_norm_eps))
        cand = priors = z_L_next = None
        seq_dict = self._seq_dict()
        for n_updates in range(max_depth + 1):
            value, target = (t.float() for t in self._leaf_values(z_H, y))
            done = depths == n_updates
            values, targets = torch.where(done, value, values), torch.where(done, target, targets)
            if n_updates == max_depth:
                break
            for _ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + x, **seq_dict)
            u_t = self.L_level(z_H, z_L, **seq_dict)
            if n_updates == 0:
                # The transition: the rollout's first update is exactly the expand's u_t
                cand = self._candidates(z_L, u_t)
                priors, z_L_next = self._prior_logits(u_t, cand).softmax(dim=-1), z_L
            z_H = self._rollout_boundary(self._rollout_next(u_t), depths - (n_updates + 1), steps_left)
        return values, targets, leaf_token, cand, priors, z_L_next

    # A fused search's state is (batch row, arena index): the latents of every node of this search live in two
    # preallocated arenas [capacity, S, H], so a round gathers its leaves with one index_select and appends all its
    # children with one copy, instead of stacking per-node views and slicing K views per leaf in Python

    def _arena_reset(self, cand: torch.Tensor, z_L: torch.Tensor, num_simulations: int) -> None:
        """Sizes the arenas for this search (root children, then K children per leaf per round, plus one round of
        slack: B*K*(sims + 2) nodes; reused across searches) and writes the root children in."""
        B, K, S, H = cand.shape
        capacity = B * K * (num_simulations + 2)
        arena = getattr(self, "_arena", None)
        # A tensor made under inference_mode (the in-training eval) cannot be written outside it (the next training
        # step) and vice versa, so an arena is only reused on the same side of that boundary
        if arena is None or arena[0].shape[0] < capacity or arena[0].shape[1:] != (S, H) or arena[0].device != cand.device or arena[0].dtype != cand.dtype \
                or arena[0].is_inference() != torch.is_inference_mode_enabled():
            arena = tuple(torch.empty(capacity, S, H, dtype=cand.dtype, device=cand.device) for _ in range(2))
            self._arena = arena
        arena[0][: B * K].copy_(cand.flatten(0, 1))
        arena[1][: B * K].copy_(z_L.repeat_interleave(K, dim=0))
        self._arena_next = B * K

    def _step(self, states: List[Tuple[int, int]], depths: List[int], x: torch.Tensor, y: torch.Tensor):
        """_rollout_values and _expand in one pass (FusedMCTS's step_fn) over arena states: the rollout's first TRM
        update is exactly the expand's u_t, so the candidates and priors of the leaves with depth left come from it
        instead of a second update; one host sync carries both the values and the priors."""
        arena_H, arena_L = self._arena
        n = len(states)
        rows = [row for row, _ in states]
        indices = [index for _, index in states]
        padded = depths
        if self.config.search_compile and n % SEARCH_BATCH_QUANTUM:
            # The compiled round specialises on its batch, so a partial batch (the test set's last 40 puzzles) is padded
            # with copies of its first leaf up to the next multiple of the quantum instead of compiling a new shape
            pad = SEARCH_BATCH_QUANTUM - n % SEARCH_BATCH_QUANTUM
            rows, indices, padded = rows + rows[:1] * pad, indices + indices[:1] * pad, depths + depths[:1] * pad
        ids = torch.tensor(indices, device=arena_H.device)
        args = (arena_H[ids], arena_L[ids], x[rows], y[rows], torch.tensor(padded, device=arena_H.device), max(padded), self._rollout_steps_left)
        outputs = _compiled_step_core()(self, *args) if self.config.search_compile else self._step_core(*args)
        values, targets, leaf_token, cand, priors, z_L_next = (t[:n] if t is not None else None for t in outputs)

        if self.training and self.config.value_bootstrap_weight > 0:
            # Kept until the loss: the compiled core's outputs live in CUDA-graph memory that the next replay overwrites
            self._bootstrap.append((leaf_token.clone(), targets.clone()) if self.config.search_compile else (leaf_token, targets))
        children: List[Optional[List[Tuple[int, int]]]] = [None] * len(states)
        prior_lists: List[Optional[List[float]]] = [None] * len(states)
        if cand is None:
            return values.tolist(), children, prior_lists
        n, k = cand.shape[:2]
        start, self._arena_next = self._arena_next, self._arena_next + n * k
        arena_H[start : start + n * k].copy_(cand.flatten(0, 1))
        arena_L[start : start + n * k].copy_(z_L_next.repeat_interleave(k, dim=0))
        flat = torch.cat([values, priors.flatten()]).tolist()
        for i, depth in enumerate(depths):
            if depth > 0:
                children[i] = [(rows[i], start + i * k + j) for j in range(k)]
                prior_lists[i] = flat[n + i * k : n + (i + 1) * k]
        return flat[:n], children, prior_lists

    @torch.compiler.disable
    @torch.no_grad()
    def _search(self, cand: torch.Tensor, prior_logits: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, y: torch.Tensor, remaining: List[int]) -> torch.Tensor:
        """One MCTS per puzzle from the root candidates, run in lockstep; returns the posterior logits log(visit
        distribution) [B, K]."""
        if not torch.isfinite(prior_logits).all():  # the C++ tree segfaults on NaN priors; a diverged run should say so
            raise FloatingPointError("TreeRM search: the candidate scorer's logits are not finite (the run diverged).")
        priors = prior_logits.float().softmax(dim=-1).tolist()
        self._bootstrap = []  # (leaf first-token states, rollout targets) this search's rollouts record for the value head
        # Eval may search deeper than training (test-time compute); the value head is what it relies on
        self.mcts.num_simulations = self.config.mcts_simulations if self.training else (self.config.eval_mcts_simulations or self.config.mcts_simulations)
        fused = getattr(self.mcts, "fused", False)
        B, K = cand.shape[:2]
        if fused:
            self._arena_reset(cand, z_L, self.mcts.num_simulations)
        roots = []
        for b in range(B):
            root = Node(latent=None, allowed_depth=remaining[b])
            root.expand([(b, b * K + k) if fused else (b, cand[b, k], z_L[b]) for k in range(K)], priors[b])
            roots.append(root)

        counts = self.mcts.run(
            roots,
            expand_fn=lambda states: self._expand(states, x),
            evaluate_fn=lambda states, depths: self._rollout_values(states, depths, x, y),
            **({"step_fn": lambda states, depths: self._step(states, depths, x, y)} if fused else {}),
        )
        visits = torch.tensor(counts, dtype=torch.float32, device=cand.device)
        posterior = visits / visits.sum(dim=-1, keepdim=True)
        return torch.log(posterior.clamp_min(1e-8))

    def _mcts_guidance(self, u_t: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, y: torch.Tensor, remaining: List[int]) -> Tuple[torch.Tensor, torch.Tensor | int]:
        """Picks the next z_H as one candidate sampled from the MCTS posterior, with its beta-weighted balanced KL to the
        scorer's prior (only when gradients are on)."""
        cand = self._candidates(z_L, u_t)
        prior_logits = self._prior_logits(u_t, cand)
        # The label-informed search only where its KL is paid (the grad cycles); elsewhere it would steer the
        # trajectory with label information for free. The prior picks the candidate on the no-grad cycles
        search = not self.training or torch.is_grad_enabled()
        post_logits = self._search(cand.detach(), prior_logits.detach(), z_L.detach(), x.detach(), y, remaining) if search else prior_logits.detach()

        kl_loss = 0
        # Only the grad cycles pay the prior's loss; the posterior has no parameters, so only the prior trains from it
        if torch.is_grad_enabled():
            kl_loss = self._prior_loss(u_t, cand, prior_logits, post_logits)
            # Value bootstrapping (AlphaZero-style): the value head predicts each visited leaf's rollout token accuracy.
            # Summed over puzzles, averaged over this search's leaves (one leaf per puzzle per simulation, however the
            # search batches them), so it scales like the loss head's LPRM term
            if self.config.value_bootstrap_weight > 0 and self._bootstrap:
                leaf_tokens = torch.cat([t for t, _ in self._bootstrap])
                targets = torch.cat([v for _, v in self._bootstrap])
                predicted = torch.sigmoid(self.v_head(leaf_tokens).to(torch.float32)).squeeze(-1)
                bootstrap = F.mse_loss(predicted, targets, reduction="sum") / self.mcts.num_simulations
                kl_loss = kl_loss + self.config.value_bootstrap_weight * bootstrap
            # Every candidate (its gate) gets a gradient from its own rollout (_candidate_loss); without it only the picked
            # one does: +14 points on N-Queens (REPORTS_CANDIDATES.md section 0)
            if self.config.candidate_loss != "none":
                cand_loss, cand_diag = self._candidate_loss(cand, z_L, x, y, post_logits, remaining[0])
                kl_loss = kl_loss + cand_loss

        if self.training or self.config.eval_pick == "sample":
            # Straight-through Gumbel-softmax as in GRAMDP: the forward value of the logits is the posterior, so the one-hot
            # sample is an exact draw from the visit distribution, while the backward pass reaches the scorer's prior logits
            logits = prior_logits + (post_logits - prior_logits).detach()
            weights = F.gumbel_softmax(logits.float(), tau=1.0, hard=True, dim=-1)
        else:
            # The most-visited candidate (ties to the first): the draw's K/sims uniform floor made a larger K a more
            # random pick (REPORTS_SCALING.md section 1)
            weights = F.one_hot(post_logits.argmax(dim=-1), cand.shape[1]).float()

        z_H = torch.einsum("bk,bksh->bsh", weights, cand.float()).to(u_t.dtype)
        diag = self._search_diagnostics(cand, prior_logits, post_logits)
        if torch.is_grad_enabled() and self.config.candidate_loss != "none":
            diag.update(cand_diag)
        if not self.training and cand.shape[1] > 1 and self.rollout_loss_fn is not None:  # a bare model has no CE to score with
            diag.update(self._pick_diagnostics(cand, z_L, x, y, weights, remaining[0]))
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

    def _prior_loss(self, u_t: torch.Tensor, cand: torch.Tensor, prior_logits: torch.Tensor, post_logits: torch.Tensor) -> torch.Tensor:
        """What a grad cycle pays to pull the prior toward the search: the beta-weighted balanced KL of the search's
        posterior from the scorer's prior over the candidates."""
        return kl_balanced_logits(prior_logits=prior_logits, post_logits=post_logits, alpha=self.config.kl_grad_prior).sum() * self.config.kl_strength

    def _candidate_rollout(self, cand: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, y: torch.Tensor, left: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Every candidate rolled out exactly as the search's root-child rollouts are (left - 1 plain TRM updates to the
        end of this supervision step, then the decode): its label CE and its LPRM value, both [B, K]."""
        B, K = cand.shape[:2]
        z_H = cand.flatten(0, 1)
        z_L = z_L.repeat_interleave(K, dim=0)
        x, y = x.repeat_interleave(K, dim=0), y.repeat_interleave(K, dim=0)
        seq_dict = self._seq_dict()
        for _ in range(left - 1):
            for _ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + x, **seq_dict)
            z_H = self._rollout_next(self.L_level(z_H, z_L, **seq_dict))
        z_H = rms_norm(z_H, self.config.rms_norm_eps)
        mask = y != IGNORE_LABEL_ID
        ce = self.rollout_loss_fn(self._decode(z_H), y, valid_mask=mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        return ce.float().view(B, K), self._lprm_value(z_H).view(B, K)

    @torch.no_grad()
    def _pick_diagnostics(self, cand: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, left: int) -> Dict[str, torch.Tensor]:
        """Per-puzzle [B] eval diagnostics of the pick against the label: the picked candidate's CE above the CE-best
        candidate, whether it is the CE-best, the CE gap between the worst and best candidate, and how well the LPRM
        value head ranks the candidates (Spearman correlation of its values with -CE over the K candidates)."""
        ce, value = self._candidate_rollout(cand, z_L, x, y, left)
        picked = weights.argmax(dim=-1, keepdim=True)
        ranks = lambda t: t.argsort(dim=-1).argsort(dim=-1).float()
        r_value, r_ce = ranks(value), ranks(-ce)
        r_value, r_ce = r_value - r_value.mean(-1, keepdim=True), r_ce - r_ce.mean(-1, keepdim=True)
        return {
            "search/pick_ce_regret": ce.gather(1, picked).squeeze(1) - ce.min(dim=-1).values,
            "search/pick_is_best": (picked.squeeze(1) == ce.argmin(dim=-1)).float(),
            "search/cand_ce_gap": ce.max(dim=-1).values - ce.min(dim=-1).values,
            "search/value_rank_corr": (r_value * r_ce).sum(-1) / (r_value.norm(dim=-1) * r_ce.norm(dim=-1)).clamp_min(1e-6),
        }

    def _candidate_loss(self, cand: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, y: torch.Tensor, post_logits: torch.Tensor, left: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """A gradient for every candidate of a grad cycle, not just the picked one. Each candidate is rolled out with
        gradient (_candidate_rollout) and its label CE is weighted per candidate_loss: 'wta' = relaxed
        winner-take-all (the lowest-CE candidate gets 1 - candidate_loss_eps, the others share it; Rupprecht et al. 2017,
        the standard cure for a multiple-choice head whose untrained slots never win), 'posterior' = the search's visit
        distribution. Returns the batch-summed loss and per-puzzle diagnostics: the weighted CE, the CE gap between the
        worst and best candidate, and whether the search's argmax is the CE-best candidate (REPORTS_CANDIDATES.md)."""
        B, K = cand.shape[:2]
        ce, _ = self._candidate_rollout(cand, z_L, x, y, left)
        if self.config.candidate_loss == "wta":
            eps = self.config.candidate_loss_eps
            weights = (1 - eps) * F.one_hot(ce.argmin(dim=-1), K).float() + eps / K
        else:
            weights = post_logits.float().softmax(dim=-1)
        loss = (weights.detach() * ce).sum(dim=-1)  # [B]
        diag = {
            "search/cand_loss": loss.detach(),
            "search/cand_ce_gap": (ce.max(dim=-1).values - ce.min(dim=-1).values).detach(),
            "search/cand_best_agree": (ce.argmin(dim=-1) == post_logits.argmax(dim=-1)).float(),
        }
        return loss.sum(), diag

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
        counts: Dict[str, int] = {}
        # mcts_horizon=full at eval: a simulation also rolls through the supervision steps after this one (the outer
        # model passes steps_left; a batch's rows share the step at eval), so the search values the trajectory's end
        self._rollout_steps_left = 0
        if not self.training and self.config.mcts_horizon == "full" and "steps_left" in batch:
            self._rollout_steps_left = int(batch["steps_left"].max().item())
        horizon = self._rollout_steps_left * self.config.H_cycles

        def accumulate(diag: Dict[str, torch.Tensor]) -> None:
            # Each key is averaged over the cycles that report it: the search diagnostics over every cycle, the
            # candidate-loss ones (cand_loss, cand_ce_gap, cand_best_agree) over the grad cycles only
            for key, value in diag.items():
                diagnostics[key] = diagnostics.get(key, 0) + value
                counts[key] = counts.get(key, 0) + 1

        # Run the recursion cycle for H_cycles - grad_cycles times without the gradient; every z_H update is searched
        with torch.no_grad():
            for cycle in range(self.config.H_cycles - self.config.grad_cycles):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + x, **seq_dict)

                u_t = self.L_level(z_H, z_L, **seq_dict)
                z_H, _, diag = self._mcts_guidance(u_t, z_L, x, y, [self.config.H_cycles - cycle + horizon] * batch_size)
                accumulate(diag)

        # The last grad_cycles cycles carry the gradient; each pays the KL of its search's posterior from the prior
        kl_loss = 0
        for cycle in range(self.config.H_cycles - self.config.grad_cycles, self.config.H_cycles):
            for __ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + x, **seq_dict)

            u_t = self.L_level(z_H, z_L, **seq_dict)
            z_H, guidance_loss, diag = self._mcts_guidance(u_t, z_L, x, y, [self.config.H_cycles - cycle + horizon] * batch_size)
            kl_loss += guidance_loss
            accumulate(diag)

        z_H = rms_norm(z_H, self.config.rms_norm_eps)

        info = {
            "lprm_value": self._lprm_value(z_H),
            # No halt head: the LPRM value (the model's own estimate that its answer is right) stands in for ACT's halt
            # logit where the evaluators want a confidence, e.g. the ARC vote's tie-break (sigmoid of this is the value)
            "q_halt_logits": torch.logit(self._lprm_value(z_H).clamp(1e-6, 1 - 1e-6)),
            "logits": self._decode(z_H),
            "kl_loss": kl_loss,
            **{key: value / counts[key] for key, value in diagnostics.items()},  # per-puzzle, averaged over the cycles reporting them
        }

        return LatentCarry(z_H, z_L), info


_STEP_CORE_COMPILED: Optional[Callable[..., Any]] = None
SEARCH_BATCH_QUANTUM = 64  # a compiled search round's batch is padded to a multiple of this (see _step)


def _compiled_step_core() -> Callable[..., Any]:
    """TreeRMInner._step_core under torch.compile(mode="reduce-overhead"), built once for the process. The search is
    launch-bound and outside the model's compile (data-dependent control flow), but a compiled callee still runs
    compiled from inside it: the fixed-shape round becomes one CUDA graph replay per (batch, depth, mode). The
    unbound function is compiled, with the module as its first argument, so dynamo guards on the module that calls
    it (its mode and weights); a compiled bound method stored on the instance would survive the deepcopy that makes
    the EMA eval model and keep running the original instance."""
    global _STEP_CORE_COMPILED
    if _STEP_CORE_COMPILED is None:
        # One cache entry per (batch, depth, mode) specialisation, so the default limit of 8 is too small
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        # Static shapes: the backbone specialises the batch dimension anyway (mark_dynamic fails inside Blocks.forward),
        # so _step pads a round's batch to a multiple of SEARCH_BATCH_QUANTUM to keep the number of shapes small
        _STEP_CORE_COMPILED = torch.compile(TreeRMInner._step_core, mode="reduce-overhead", dynamic=False)
    return _STEP_CORE_COMPILED


class TreeRM(GRAM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.inner = TreeRMInner(_add_puzzle_emb_positions(config))
