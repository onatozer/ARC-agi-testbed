"""Paper-style figures built from a ProbeResult. Each function returns a matplotlib Figure (caller closes it)."""

import numpy as np

from utils.dynamics.probe import ProbeResult

GREEN, RED, GREY, BLUE, ORANGE = "#2a9d4b", "#d1495b", "#9a9a9a", "#2c6fbb", "#e08a1e"
MODE_NAMES = ("Quick success", "Delayed success", "Failure")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    return plt


def _steps(r: ProbeResult) -> np.ndarray:
    return np.arange(1, r.depth + 1)


def _mark_train_depth(ax, r: ProbeResult) -> None:
    if r.depth > r.train_depth:
        ax.axvline(r.train_depth, color=GREY, ls=":", lw=1)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC; nan if only one class present."""
    labels = labels.astype(bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def halt_prob(q_logit: np.ndarray) -> np.ndarray:
    """The ACT halt head's predicted probability that the current decoded answer is exactly correct."""
    return 1 / (1 + np.exp(-q_logit))


HALT_LABEL = "ACT halt head: P(answer correct)"


def q_and_acc_by_correctness(r: ProbeResult):
    """PTRM Fig. 3: the halt head's confidence (solid) and cell accuracy (dashed) per step, split by final correctness at T.

    PTRM plots the raw q_halt logit; here it is sigmoid(q_halt) so both curves share a [0, 1] axis. Training halts when
    it crosses 0.5 (q_halt > 0)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    t = _steps(r)
    final = r.exact[r.train_depth - 1]
    for mask, color, name in ((final, GREEN, "Correct"), (~final, RED, "Wrong")):
        if mask.any():
            ax.plot(t, halt_prob(r.q_logit[:, mask]).mean(1), color=color, lw=2, label=f"{name} at step {r.train_depth} ({mask.sum()}): {HALT_LABEL}")
            ax.plot(t, r.token_acc[:, mask].mean(1), color=color, lw=1.5, ls="--", label=f"{name}: cell accuracy")
    ax.axhline(0.5, color=GREY, lw=0.8, ls="--")
    ax.text(t[-1], 0.51, "halt threshold", ha="right", va="bottom", fontsize=7, color=GREY)
    _mark_train_depth(ax, r)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Supervision step")
    ax.set_ylabel("Probability / accuracy")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set_title("Does the halt head know when the answer is right? (eval)")
    fig.tight_layout()
    return fig


def pca_trajectory_modes(r: ProbeResult):
    """PTRM Fig. 2: one example puzzle per trajectory mode; top = PCA of pooled z_H, bottom = Q and cell accuracy."""
    plt = _plt()
    modes = r.modes()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    t = _steps(r)
    for col, name in enumerate(MODE_NAMES):
        top, bot = axes[0, col], axes[1, col]
        idx = np.flatnonzero(modes == col)
        top.set_title(f"{name} ({len(idx)}/{len(modes)})")
        if len(idx) == 0:
            top.text(0.5, 0.5, "no example", ha="center", va="center", transform=top.transAxes)
            top.axis("off")
            bot.set_visible(False)
            continue
        # most representative: for delayed/failure prefer the one with the most answer switches
        b = idx[np.argmax(r.answer_changed[:, idx].sum(0))] if col else idx[0]
        # skip the init state: the first jump away from the fixed z_H init otherwise dominates the PCA
        traj = r.pooled_zH[1:, b]  # [D, H]
        centered = traj - traj.mean(0)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        var = s**2 / max((s**2).sum(), 1e-12)
        xy = centered @ vt[:2].T
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(xy)))
        for i in range(len(xy) - 1):
            top.plot(xy[i : i + 2, 0], xy[i : i + 2, 1], color=colors[i + 1], lw=1.8)
        end_color = GREEN if r.exact[r.train_depth - 1, b] else RED  # same criterion as the mode label
        top.scatter(*xy[0], marker="o", s=50, color="white", edgecolor="k", zorder=3, label="Start")
        top.scatter(*xy[-1], marker="s", s=50, color=end_color, edgecolor="k", zorder=3, label="End")
        top.set_xlabel(f"PC 1 ({var[0]:.0%} of trajectory variance)")
        top.set_ylabel(f"PC 2 ({var[1]:.0%} of trajectory variance)")
        top.legend(frameon=False, fontsize=7)

        bot.plot(t, halt_prob(r.q_logit[:, b]), color=end_color, lw=2, label=HALT_LABEL)
        bot.plot(t, r.token_acc[:, b], color="k", ls="--", lw=1.2, label="cell accuracy")
        bot.axhline(0.5, color=GREY, lw=0.8, ls="--")
        _mark_train_depth(bot, r)
        bot.set_ylim(0, 1.02)
        bot.set_xlabel("Supervision step")
        bot.set_ylabel("Probability / accuracy")
        bot.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("TRM trajectory modes (eval; PCA of latent z_H, token-mean pooled, fit per puzzle)")
    fig.tight_layout()
    return fig


def residual_over_rollout(r: ProbeResult):
    """EqR Fig. 11 / Loopscape Fig. 3C: step residual ||z_t - z_{t-1}|| and distance to the final latent."""
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    t = _steps(r)
    final = r.exact[-1]
    for ax, data, ylabel, x in (
        (a, r.res, "Residual ||z_t - z_{t-1}||", t),
        (b, r.dist_to_final[:-1], "||z_t - z_D|| / ||z_0 - z_D||", np.arange(0, r.depth)),  # drop t=D (=0 on log axis)
    ):
        for i in range(data.shape[1]):
            ax.plot(x, data[:, i], color=GREEN if final[i] else RED, alpha=0.15, lw=0.8)
        ax.plot(x, np.median(data, 1), color="k", lw=2, label="median")
        _mark_train_depth(ax, r)
        ax.set_xlabel("Supervision step")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
    a.set_title("Fixed-point residual over rollout")
    b.set_title("Convergence to final latent (green = correct)")
    a.legend(frameon=False)
    fig.tight_layout()
    return fig


def acc_vs_residual_depth(r: ProbeResult):
    """EqR Fig. 1: exact accuracy vs mean residual, one point per depth, coloured by iteration."""
    plt = _plt()
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    res, acc, t = r.res.mean(1), r.exact.mean(1), _steps(r)
    ax.plot(res, acc, color=GREY, lw=1, zorder=1)
    sc = ax.scatter(res, acc, c=t, cmap="plasma", norm=LogNorm(1, max(t[-1], 2)), s=28, zorder=2)
    T = r.train_depth - 1
    ax.scatter(res[T], acc[T], marker="^", s=120, facecolor="none", edgecolor="k", lw=1.5, zorder=3, label=f"trained depth ({r.train_depth})")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Residual ||f(z) - z||  (← larger | smaller →)")
    ax.set_ylabel("Exact accuracy")
    fig.colorbar(sc, ax=ax, label="Iterations")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Accuracy vs residual as eval depth grows (eval)")
    fig.tight_layout()
    return fig


def settling_and_switch(r: ProbeResult):
    """Loopscape: settling-time distribution and answer-switch frequency vs settling time."""
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    final = r.exact[-1]
    bins = np.arange(0.5, r.depth + 1.5)
    a.hist([r.settle_step[final], r.settle_step[~final]], bins=bins, stacked=True, color=[GREEN, RED],
           label=[f"correct ({final.sum()})", f"incorrect ({(~final).sum()})"])
    if r.depth > r.train_depth:
        a.axvline(r.train_depth + 0.5, color=GREY, ls=":", lw=1)
    a.set_xlabel("Loops to converge L (decoded answer stops changing)")
    a.set_ylabel("Puzzles")
    a.legend(frameon=False)
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(final))
    b.scatter(r.settle_step + jitter, r.switch_freq, c=np.where(final, GREEN, RED), s=14, alpha=0.7)
    b.set_xlabel("Loops to converge L")
    b.set_ylabel("Answer-switch frequency")
    a.set_title("Settling time")
    b.set_title("Slow puzzles oscillate between answers")
    fig.tight_layout()
    return fig


ALL_PROBE_FIGURES = {
    "q_acc_by_correctness": q_and_acc_by_correctness,
    "trajectory_modes": pca_trajectory_modes,
    "residual_over_rollout": residual_over_rollout,
    "acc_vs_residual_depth": acc_vs_residual_depth,
    "settling_and_switch": settling_and_switch,
}
