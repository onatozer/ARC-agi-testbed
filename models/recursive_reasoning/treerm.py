import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.losses import IGNORE_LABEL_ID
from models.losses.base import kl_balanced_logits
from models.mcts import MCTS_IMPLS, Node
from ..layers import (
    CastedLinear,
    SwiGLU,
    rms_norm,
)
from .merged_arch import Blocks, ModelCarry, LatentCarry, _add_puzzle_emb_positions
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


class FlagCandidates(nn.Module):
    """Candidates that can't collapse onto one another: u_t plus, per slot k, a fixed random unit direction e_k scaled by
    a learned per-token gate, cand_k = u_t + g_k(z_L, u_t) * sqrt(H) * e_k. The directions are orthonormal and frozen,
    so the slots differ by construction; the network downstream can learn what each "flag" means, and the gates can
    only shrink the flags (collapse point: plain TRM). [B, S, H] x [B, S, H] -> [B, K, S, H]."""

    def __init__(self, hidden_size: int, num_candidates: int):
        super().__init__()
        self.scale = math.sqrt(hidden_size)
        # Drawn on the CPU with a fixed seed so every instance gets the same directions, whatever device the model is built on
        self.gate = CastedLinear(2 * hidden_size, num_candidates, bias=True)
        directions, _ = torch.linalg.qr(torch.randn(hidden_size, num_candidates, generator=torch.Generator().manual_seed(0), device="cpu"))
        self.register_buffer("directions", directions.T.contiguous().to(self.gate.weight.device), persistent=True)  # [K, H], orthonormal rows
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
        if self.config.candidate_mode not in ("full", "residual", "flags", "flags_residual"):
            raise ValueError(f"candidate_mode must be 'full', 'residual', 'flags' or 'flags_residual', got {self.config.candidate_mode!r}.")
        if self.config.candidate_mode in ("flags", "flags_residual"):
            self.candidate_generator = FlagCandidates(hidden_size=self.config.hidden_size, num_candidates=self.config.num_candidates)
        if self.config.candidate_mode == "flags_residual":
            # The learned per-slot content on top of the flags: slot k = u_t + flag_k + delta_k(z_L, u_t)
            self.candidate_residual = CandidateGenerator(
                hidden_size=self.config.hidden_size,
                num_candidates=self.config.num_candidates,
                expansion=self.config.expansion,
            )
        if self.config.candidate_mode in ("full", "residual"):
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

        if self.config.mcts_horizon not in ("step", "full"):
            raise ValueError(f"mcts_horizon must be 'step' or 'full', got {self.config.mcts_horizon!r}.")
        if self.config.rollout_policy not in ("trm", "prior"):
            raise ValueError(f"rollout_policy must be 'trm' or 'prior', got {self.config.rollout_policy!r}.")
        if self.config.candidate_loss not in ("none", "wta", "posterior"):
            raise ValueError(f"candidate_loss must be 'none', 'wta' or 'posterior', got {self.config.candidate_loss!r}.")
        if not 0.0 <= self.config.candidate_loss_eps <= 1.0:
            raise ValueError(f"candidate_loss_eps must be in [0, 1], got {self.config.candidate_loss_eps!r}.")
        if self.config.candidate_loss_eps_anneal_steps < 0:
            raise ValueError("candidate_loss_eps_anneal_steps must be >= 0.")
        if self.config.candidate_recover_weight > 0:
            # Slot recoverability (InfoGAN-style): a K-way classifier reads a candidate's decoded distribution and must say
            # which slot made it; its CE trains the generator, so identical candidates (unrecoverable slots) are penalised
            decoded_len = self.config.seq_len - (self.config.puzzle_emb_len if self.config.puzzle_emb_ndim != 0 else 0)  # what _decode keeps
            self.candidate_recover_head = nn.Linear(decoded_len * self.config.vocab_size, self.config.num_candidates)
        # Training forwards so far (one per optimizer step: a batch is one forward); drives the eps anneal. Not saved,
        # so a resumed run anneals again from the start
        self.register_buffer("_train_forwards", torch.zeros((), dtype=torch.long), persistent=False)
        if self.config.search_compile and not getattr(self.mcts, "fused", False):
            raise ValueError("search_compile needs a fused search (mcts_impl=fused or wu): only its round has fixed shapes.")
        if self.config.mcts_leaves_per_round != 1:
            if not hasattr(self.mcts, "leaves_per_round"):
                raise ValueError("mcts_leaves_per_round > 1 needs the WU-UCT search (mcts_impl=wu).")
            self.mcts.leaves_per_round = self.config.mcts_leaves_per_round
        if self.config.search_memo and not getattr(self.mcts, "memo", False):
            raise ValueError("search_memo needs a search that can continue from a subtree (mcts_impl=cpp).")
        # (previous search's forest and states, chosen root children) of this step's last eval search, see _search
        self._memo: Optional[Tuple[Any, List[Any], List[int]]] = None
        self._pending_memo: Optional[Tuple[Any, List[Any]]] = None

        # Token-wise CE the training rollouts are valued with; LossHeadTreeRM sets it to the loss head's loss_fn
        self.rollout_loss_fn: Optional[Callable[..., torch.Tensor]] = None
        # Per-puzzle supervision steps already run, set by TreeRM.forward; the 'full' horizon needs it
        self.steps_done: Optional[torch.Tensor] = None
        self._bootstrap: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def _decode(self, z_H: torch.Tensor) -> torch.Tensor:
        # Clip out the puzzle embedding dims from the decoder
        if self.config.puzzle_emb_ndim != 0:
            return self.lm_head(z_H[:, self.config.puzzle_emb_len :]).contiguous()
        return self.lm_head(z_H).contiguous()

    def _lprm_value(self, z_H: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.v_head(z_H[:, 0]).to(torch.float32)).squeeze(-1)

    def _candidates(self, z_L: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """The K proposals for the new z_H (see candidate_mode)."""
        cand = self.candidate_generator(z_L, u_t)
        if self.config.candidate_mode == "residual":
            cand = cand + u_t.unsqueeze(1)
        elif self.config.candidate_mode == "flags_residual":
            cand = cand + self.candidate_residual(z_L, u_t)
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

    def _rollout_depths(self, depths: List[int]) -> List[int]:
        """The number of TRM updates each leaf's rollout runs: its allowed depth, capped at eval by eval_rollout_cap
        (at least 1, since the transition needs the first update). A cap changes the search's values (the value head
        scores a state short of the step's end), so it is a measured option, off by default."""
        cap = self.config.eval_rollout_cap
        if self.training or cap is None:
            return depths
        return [min(depth, max(int(cap), 1)) for depth in depths]

    def _rollout_values(self, states: List[Tuple[int, torch.Tensor, torch.Tensor]], depths: List[int], x: torch.Tensor, y: torch.Tensor) -> List[float]:
        """The simulation: from each leaf, its allowed depth's worth of plain TRM updates (no candidates), then its value."""
        rows = [row for row, _, _ in states]
        z_H = torch.stack([s[1] for s in states])
        z_L = torch.stack([s[2] for s in states])
        x, y = x[rows], y[rows]
        depths = self._rollout_depths(depths)
        depths_t = torch.tensor(depths, device=z_H.device)
        values = torch.zeros(len(states), device=z_H.device)
        targets = torch.zeros(len(states), device=z_H.device)
        # The value head reads the leaf's own state (first token, rms-normed as in forward); the rollout gives its target
        leaf_token = rms_norm(z_H, self.config.rms_norm_eps)[:, 0]

        seq_dict = self._seq_dict()
        for n_updates in range(max(depths) + 1):
            done = depths_t == n_updates
            if done.any():
                values[done], targets[done] = (t.float() for t in self._leaf_values(z_H[done], y[done]))
            if n_updates == max(depths):
                break
            for _ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + x, **seq_dict)
            u_t = self.L_level(z_H, z_L, **seq_dict)
            if self.config.rollout_policy == "prior":
                # Follow the policy being trained: the scorer's argmax candidate, as the no-grad cycles and eval do
                cand = self._candidates(z_L, u_t)
                best = self.candidate_scorer(u_t, cand, seq_dict["cos_sin"]).argmax(dim=-1)
                z_H = cand[torch.arange(cand.shape[0], device=cand.device), best]
            else:
                z_H = u_t

        if self.training and self.config.value_bootstrap_weight > 0:
            self._bootstrap.append((leaf_token, targets))
        return values.tolist()

    def _step_core(self, z_H: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, y: torch.Tensor, depths: torch.Tensor, max_depth: int):
        """The fixed-shape tensor work of one fused search round over n leaves: max_depth plain TRM updates, each
        leaf's value at its own depth (computed for every row at every stage and selected by depth, so nothing is
        data-dependent), and from the first update the K candidates and their prior for every leaf. Returns
        (values [n], bootstrap targets [n], leaf tokens [n, H], candidates [n, K, S, H], priors [n, K], z_L after the
        first update [n, S, H]); the last three are None when max_depth is 0. Compiled when search_compile is on."""
        values = torch.zeros(z_H.shape[0], device=z_H.device)
        targets = torch.zeros(z_H.shape[0], device=z_H.device)
        leaf_token = rms_norm(z_H, self.config.rms_norm_eps)[:, 0]
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
            if n_updates == 0 or self.config.rollout_policy == "prior":
                step_cand = self._candidates(z_L, u_t)
                step_scores = self.candidate_scorer(u_t, step_cand, seq_dict["cos_sin"])
                if n_updates == 0:
                    # The transition: the rollout's first update is exactly the expand's u_t
                    cand, priors, z_L_next = step_cand, step_scores.softmax(dim=-1), z_L
            if self.config.rollout_policy == "prior":
                z_H = step_cand[torch.arange(step_cand.shape[0], device=step_cand.device), step_scores.argmax(dim=-1)]
            else:
                z_H = u_t
        return values, targets, leaf_token, cand, priors, z_L_next

    # A fused search's state is (batch row, arena index): the latents of every node of this search live in two
    # preallocated arenas [capacity, S, H], so a round gathers its leaves with one index_select and appends all its
    # children with one copy, instead of stacking per-node views and slicing K views per leaf in Python

    def _arena_reset(self, cand: torch.Tensor, z_L: torch.Tensor, num_simulations: int, keep: bool = False) -> None:
        """Sizes the arenas for this search (root children, then K children per leaf per round: B*K*(sims + 1) nodes,
        plus a round's worth of slack for a WU-UCT search's wider rounds; reused across searches) and writes the root
        children in. keep: continue a memoised search (its nodes stay where they are; this step's searches share the
        arena, hence H_cycles times the capacity)."""
        B, K, S, H = cand.shape
        per_search = B * K * (num_simulations + getattr(self.mcts, "leaves_per_round", 1) + 1)
        capacity = per_search * (self.config.H_cycles if self.config.search_memo else 1)
        arena = getattr(self, "_arena", None)
        # A tensor made under inference_mode (the in-training eval) cannot be written outside it (the next training
        # step) and vice versa, so an arena is only reused on the same side of that boundary
        if arena is None or arena[0].shape[0] < capacity or arena[0].shape[1:] != (S, H) or arena[0].device != cand.device or arena[0].dtype != cand.dtype \
                or arena[0].is_inference() != torch.is_inference_mode_enabled():
            arena = tuple(torch.empty(capacity, S, H, dtype=cand.dtype, device=cand.device) for _ in range(2))
            self._arena = arena
            keep = False
        if keep and self._arena_next + per_search <= capacity:
            return
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
        rollout = self._rollout_depths(padded)  # how far each leaf is rolled out (eval_rollout_cap); depths still say what gets expanded
        args = (arena_H[ids], arena_L[ids], x[rows], y[rows], torch.tensor(rollout, device=arena_H.device), max(rollout))
        outputs = _compiled_step_core(self.config.search_compile_mode)(self, *args) if self.config.search_compile else self._step_core(*args)
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
        # search_memo (eval): the previous cycle's search of this step chose root child k*; its subtree (children
        # states in the arena, priors, rollout values at the depths this search needs) is exactly this search's
        # start, so the search continues from it instead of re-rolling what it already valued
        memo = self.config.search_memo and not self.training and getattr(self.mcts, "memo", False)  # the bench swaps impls
        reuse = self._memo if memo else None
        self._memo = self._pending_memo = None
        if fused:
            self._arena_reset(cand, z_L, self.mcts.num_simulations, keep=reuse is not None)
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
            **({"reuse": reuse} if reuse is not None else {}),
            **({"cache": True} if memo else {}),
        )
        if memo:
            self._pending_memo = self.mcts.last  # completed by _mcts_guidance once the candidate is picked
        visits = torch.tensor(counts, dtype=torch.float32, device=cand.device)
        posterior = visits / visits.sum(dim=-1, keepdim=True)
        # Mix in the uniform so a one-hot visit distribution can't drive the prior's logits to infinity through the KL
        eps = self.config.posterior_smoothing
        posterior = (1 - eps) * posterior + eps / posterior.shape[-1]
        return torch.log(posterior.clamp_min(1e-8))

    def _mcts_guidance(self, u_t: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, y: torch.Tensor, remaining: List[int], left: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor | int, Dict[str, torch.Tensor]]:
        """Picks the next z_H as one candidate sampled from the MCTS posterior, with its beta-weighted balanced KL to the
        scorer's prior (only when gradients are on). left: z_H updates left in this supervision step, this one included
        (what the candidate loss rolls out); None = remaining[0], exact for mcts_horizon=step."""
        cand = self._candidates(z_L, u_t)
        prior_logits = self.candidate_scorer(u_t, cand, self._cos_sin())
        if self.training:
            # The label-informed search only where its KL is paid (the grad cycles); elsewhere it would steer the
            # trajectory with label information for free. The prior picks the candidate on the no-grad cycles
            search = torch.is_grad_enabled()
        else:
            search = self.config.eval_search  # eval_search=false: the prior alone picks at eval (diagnostic)
        post_logits = self._search(cand.detach(), prior_logits.detach(), z_L.detach(), x.detach(), y, remaining) if search else prior_logits.detach()

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
            # Value bootstrapping (AlphaZero-style): the value head predicts each visited leaf's rollout token accuracy.
            # Summed over puzzles, averaged over this search's leaves (one leaf per puzzle per simulation, however the
            # search batches them), so it scales like the loss head's LPRM term
            if self.config.value_bootstrap_weight > 0 and self._bootstrap:
                leaf_tokens = torch.cat([t for t, _ in self._bootstrap])
                targets = torch.cat([v for _, v in self._bootstrap])
                predicted = torch.sigmoid(self.v_head(leaf_tokens).to(torch.float32)).squeeze(-1)
                bootstrap = F.mse_loss(predicted, targets, reduction="sum") / self.mcts.num_simulations
                kl_loss = kl_loss + self.config.value_bootstrap_weight * bootstrap
            # Every candidate gets a gradient from its own rollout (see _candidate_loss); without this only the picked
            # one does, and learned candidates collapse (REPORTS.md section 4, REPORTS_CANDIDATES.md)
            if self.config.candidate_loss != "none" or self.config.candidate_recover_weight > 0:
                cand_loss, cand_diag = self._candidate_loss(cand, z_L, x, y, post_logits, left if left is not None else remaining[0])
                kl_loss = kl_loss + cand_loss

        # Straight-through Gumbel-softmax as in GRAMDP: the forward value of the logits is the posterior, so the one-hot
        # sample is an exact draw from the visit distribution, while the backward pass reaches the scorer's prior logits
        logits = prior_logits + (post_logits - prior_logits).detach()
        if self.config.argmax_pick and not torch.is_grad_enabled():
            weights = F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).float()
        else:
            weights = F.gumbel_softmax(logits.float(), tau=self.config.gumbel_temp, hard=True, dim=-1)

        if self.config.candidate_soft_grad and torch.is_grad_enabled():
            # Forward: the sampled candidate, with the straight-through path to the scorer. Backward into the candidates:
            # the posterior-weighted mixture, so the unpicked ones learn from this pick in proportion to their mass
            mix = torch.einsum("bk,bksh->bsh", logits.float().softmax(dim=-1).detach(), cand.float())
            z_H = torch.einsum("bk,bksh->bsh", weights, cand.detach().float()) + (mix - mix.detach())
        else:
            z_H = torch.einsum("bk,bksh->bsh", weights, cand.float())
        z_H = z_H.to(u_t.dtype)
        if torch.is_grad_enabled() and self.config.candidate_balance_weight > 0:
            # Load balancing as for a mixture-of-experts router (Switch Transformer): K * sum_k f_k * P_k, with f_k the
            # fraction of this batch's picks that went to slot k and P_k the scorer's mean prior on k; minimal (1) when
            # the prior is spread over the slots and the picks follow it, so no slot can take every puzzle
            picks = weights.detach().mean(dim=0)
            mean_prior = prior_logits.float().softmax(dim=-1).mean(dim=0)
            balance = logits.shape[-1] * (picks * mean_prior).sum()
            kl_loss = kl_loss + self.config.candidate_balance_weight * balance * logits.shape[0]
            cand_diag = {**(cand_diag if self.config.candidate_loss != "none" or self.config.candidate_recover_weight > 0 else {}),
                         "search/balance": balance.detach().expand(logits.shape[0]), "search/pick_entropy": -(picks * picks.clamp_min(1e-8).log()).sum().expand(logits.shape[0])}
        if self._pending_memo is not None:
            self._memo = (*self._pending_memo, weights.argmax(dim=-1).tolist())
            self._pending_memo = None
        diag = self._search_diagnostics(cand, prior_logits, post_logits)
        if torch.is_grad_enabled() and (self.config.candidate_loss != "none" or self.config.candidate_recover_weight > 0 or self.config.candidate_balance_weight > 0):
            diag.update(cand_diag)
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

    def _candidate_loss(self, cand: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, y: torch.Tensor, post_logits: torch.Tensor, left: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """A gradient for every candidate of a grad cycle, not just the picked one. Each candidate is rolled out with
        gradient exactly as the search's root-child rollouts are (plain TRM updates to the end of this supervision step,
        then the decode) and its label CE is weighted per candidate_loss: 'wta' = relaxed winner-take-all (the lowest-CE
        candidate gets 1 - eps, the others share eps; Rupprecht et al. 2017, the standard cure for a multiple-choice head
        whose untrained slots never win), 'posterior' = the search's visit distribution. Returns the batch-summed loss and
        per-puzzle diagnostics: the weighted CE, the CE gap between the best and worst candidate, and whether the search's
        argmax is the CE-best candidate."""
        B, K = cand.shape[:2]
        z_H = cand.flatten(0, 1)
        z_L = z_L.repeat_interleave(K, dim=0)
        x, y = x.repeat_interleave(K, dim=0), y.repeat_interleave(K, dim=0)
        seq_dict = self._seq_dict()
        for _ in range(left - 1):
            for _ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + x, **seq_dict)
            z_H = self.L_level(z_H, z_L, **seq_dict)
        z_H = rms_norm(z_H, self.config.rms_norm_eps)
        mask = y != IGNORE_LABEL_ID
        logits = self._decode(z_H)
        ce = self.rollout_loss_fn(logits, y, valid_mask=mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        ce = ce.float().view(B, K)
        diag = {
            "search/cand_ce_gap": (ce.max(dim=-1).values - ce.min(dim=-1).values).detach(),
            "search/cand_best_agree": (ce.argmin(dim=-1) == post_logits.argmax(dim=-1)).float(),
        }
        total = ce.new_zeros(())
        if self.config.candidate_loss != "none":
            if self.config.candidate_loss == "wta":
                eps = self._candidate_loss_eps()
                weights = (1 - eps) * F.one_hot(ce.argmin(dim=-1), K).float() + eps / K
                diag["search/cand_eps"] = ce.new_full((B,), eps)
            else:
                weights = post_logits.float().softmax(dim=-1)
            loss = (weights.detach() * ce).sum(dim=-1)  # [B]
            diag["search/cand_loss"] = loss.detach()
            total = total + self.config.candidate_loss_weight * loss.sum()
        if self.config.candidate_recover_weight > 0:
            # Which slot made this outcome? Trained on every candidate's decoded distribution (gradient into the generator
            # through the decode); recover_acc is a direct measure of diversity in outcome space
            slot_logits = self.candidate_recover_head(logits.float().softmax(dim=-1).flatten(1))
            slots = torch.arange(K, device=ce.device).repeat(B)
            recover = F.cross_entropy(slot_logits, slots, reduction="none").view(B, K).mean(dim=-1)  # [B]
            diag["search/recover_loss"] = recover.detach()
            diag["search/recover_acc"] = (slot_logits.argmax(dim=-1) == slots).float().view(B, K).mean(dim=-1)
            total = total + self.config.candidate_recover_weight * recover.sum()
        return total, diag

    def _candidate_loss_eps(self) -> float:
        """The wta eps in force: annealed linearly from 1 (every candidate trains equally) at the first training forward
        to candidate_loss_eps at candidate_loss_eps_anneal_steps, so no slot locks in before the others have trained."""
        steps = self.config.candidate_loss_eps_anneal_steps
        if steps == 0:
            return self.config.candidate_loss_eps
        frac = min(int(self._train_forwards) / steps, 1.0)
        return 1.0 + (self.config.candidate_loss_eps - 1.0) * frac

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
        self._memo = self._pending_memo = None  # a search memo only carries across the cycles of one step
        if self.training and torch.is_grad_enabled():
            self._train_forwards += 1

        def accumulate(diag: Dict[str, torch.Tensor]) -> None:
            # Each key is averaged over the cycles that report it: the search diagnostics over every cycle, the
            # candidate-loss ones (cand_loss, cand_ce_gap, ...) over the grad cycles only
            for key, value in diag.items():
                diagnostics[key] = diagnostics.get(key, 0) + value
                counts[key] = counts.get(key, 0) + 1

        # Run the recursion cycle for H_cycles - grad_cycles times without the gradient; every z_H update is searched
        with torch.no_grad():
            for cycle in range(self.config.H_cycles - self.config.grad_cycles):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + x, **seq_dict)

                u_t = self.L_level(z_H, z_L, **seq_dict)
                z_H, _, diag = self._mcts_guidance(u_t, z_L, x, y, self._remaining_updates(cycle, batch_size), left=self.config.H_cycles - cycle)
                accumulate(diag)

        # The last grad_cycles cycles carry the gradient; each pays the KL of its search's posterior from the prior
        kl_loss = 0
        for cycle in range(self.config.H_cycles - self.config.grad_cycles, self.config.H_cycles):
            for __ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + x, **seq_dict)

            u_t = self.L_level(z_H, z_L, **seq_dict)
            z_H, guidance_loss, diag = self._mcts_guidance(u_t, z_L, x, y, self._remaining_updates(cycle, batch_size), left=self.config.H_cycles - cycle)
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


def _compiled_step_core(mode: str = "reduce-overhead") -> Callable[..., Any]:
    """TreeRMInner._step_core under torch.compile(mode), built once for the process (the first caller's mode). The search is
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
        _STEP_CORE_COMPILED = torch.compile(TreeRMInner._step_core, mode=mode, dynamic=False)
    return _STEP_CORE_COMPILED


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
