"""Basin maps over a random 2D slice of the initial latent space (Lai, Bao, Quinn, Gilpin, "Fractal basins trap
latent reasoning", arXiv 2609.04963; reference code github.com/GilpinLab/loopscape).

For one puzzle, every pixel (a, b) of a G x G grid starts the recursion from z0 + a*u + b*v and is rolled out for
`depth` outer steps (one outer step = one `inner(carry, batch)` call). Per pixel we record the settling time, final
correctness, a fast Lyapunov indicator and the answer-switch frequency; per image the basin entropy and uncertainty
exponent.

Slice scale: u, v are orthonormal in the flattened (z_H, z_L) space and then multiplied by sqrt(N) (N = latent dim),
as in loopscape's `plane_basis`, so `span` is measured in per-coordinate RMS units (the default z_H/z_L init has
per-coordinate std ~1). Without this rescaling a unit step would be invisible in a ~1e5-dim latent.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from models.losses import IGNORE_LABEL_ID
from models.recursive_reasoning.merged_arch import LatentCarry
from utils.dynamics.probe import initial_latent


def basin_entropy(img: np.ndarray, box: int = 5) -> Tuple[float, float]:
    """(S_b, S_bb): mean Shannon entropy (nats) of the values in non-overlapping box x box tiles, over all tiles and
    over boundary tiles (those holding >1 distinct value). Trailing rows/cols that don't fill a tile are dropped."""
    H, W = (img.shape[0] // box) * box, (img.shape[1] // box) * box
    tiles = img[:H, :W].reshape(H // box, box, W // box, box).transpose(0, 2, 1, 3).reshape(-1, box * box)
    ent = []
    for t in tiles:
        _, counts = np.unique(t, return_counts=True)
        p = counts / counts.sum()
        ent.append(float(-(p * np.log(p)).sum()))
    ent = np.array(ent)
    boundary = ent > 0
    return float(ent.mean()), float(ent[boundary].mean()) if boundary.any() else 0.0


def uncertainty_fraction(img: np.ndarray, eps: Sequence[int] = (1, 2, 3, 4, 5)) -> np.ndarray:
    """f(eps): fraction of horizontal + vertical pixel pairs at separation eps whose values differ."""
    f = []
    for e in eps:
        dh = img[:, e:] != img[:, :-e]
        dv = img[e:, :] != img[:-e, :]
        f.append((dh.sum() + dv.sum()) / (dh.size + dv.size))
    return np.array(f, dtype=float)


def uncertainty_exponent(img: np.ndarray, eps: Sequence[int] = (1, 2, 3, 4, 5)) -> float:
    """alpha from the fit log f(eps) = alpha log eps + c (boundary dimension = 2 - alpha). NaN if some f(eps) == 0."""
    f = uncertainty_fraction(img, eps)
    if (f <= 0).any():
        return float("nan")
    return float(np.polyfit(np.log(np.asarray(eps, float)), np.log(f), 1)[0])


def _plane(numel: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(numel, 2, generator=g, dtype=torch.float64))
    return (Q * numel**0.5).float()  # [N, 2], orthogonal columns with per-coordinate RMS 1


def _fli(preds: np.ndarray) -> np.ndarray:
    """preds [D, G, G, M] -> max_t mean_{4-nbrs} ||onehot(p) - onehot(q)||, normalised to [0, 1] over the image.

    For one-hot decoded states the Euclidean distance is sqrt(2 * #differing cells)."""
    D, G1, G2, _ = preds.shape
    tot = np.zeros((D, G1, G2))
    cnt = np.zeros((G1, G2))
    for ax in (1, 2):
        n = preds.shape[ax]
        a = np.take(preds, np.arange(n - 1), axis=ax)
        b = np.take(preds, np.arange(1, n), axis=ax)
        d = np.sqrt(2.0 * (a != b).sum(-1))  # [D, ...] distance between neighbours along this axis
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[ax], sl_hi[ax] = slice(0, n - 1), slice(1, n)
        tot[tuple(sl_lo)] += d
        tot[tuple(sl_hi)] += d
        cnt[tuple(sl_lo[1:])] += 1
        cnt[tuple(sl_hi[1:])] += 1
    fli = (tot / cnt).max(0)
    lo, hi = fli.min(), fli.max()
    return (fli - lo) / (hi - lo) if hi > lo else np.zeros_like(fli)


@torch.no_grad()
def basin_map(
    inner: torch.nn.Module,
    batch_row: Dict[str, torch.Tensor],
    grid: int = 64,
    span: float = 1.0,
    depth: int = 32,
    chunk: int = 512,
    seed: int = 0,
    random_z0: bool = False,
    box: int = 5,
) -> Dict[str, object]:
    """Basin map of one puzzle (`batch_row` tensors have leading dim 1) over the slice z0 + a*u + b*v,
    (a, b) in [-span, span]^2. Image row i <-> b = ys[i] (ascending), column j <-> a = xs[j].

    Settling time L = first step from which the decoded answer (argmax over non-ignored label cells) never changes
    again within `depth` (1 if it never changes); `capped` marks pixels whose answer still changed at the last step.
    """
    device = batch_row["inputs"].device
    was_training = inner.training
    inner.eval()
    try:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            z0 = initial_latent(inner, 1, device)
            if random_z0:
                z0 = LatentCarry(torch.randn_like(z0.z_H.float()).to(z0.z_H.dtype), torch.randn_like(z0.z_L.float()).to(z0.z_L.dtype))
        dtype = z0.z_H.dtype
        nH = z0.z_H.numel()
        base = torch.cat((z0.z_H.flatten(), z0.z_L.flatten())).float()
        uv = _plane(base.numel(), seed).to(device)

        coords = torch.linspace(-span, span, grid, dtype=torch.float32)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        ab = torch.stack((gx.flatten(), gy.flatten()), 1).to(device)  # [P, 2]
        P = ab.shape[0]

        labels = batch_row["labels"][0]
        mask = labels != IGNORE_LABEL_ID
        vocab = inner.config.vocab_size
        store_dtype = np.uint8 if vocab <= 255 else np.int16
        preds_all = np.empty((depth, P, int(mask.sum())), dtype=store_dtype)

        for s in range(0, P, chunk):
            B = min(chunk, P - s)
            z = base + ab[s : s + B] @ uv.T  # [B, N]
            carry = LatentCarry(
                z[:, :nH].reshape(B, *z0.z_H.shape[1:]).to(dtype), z[:, nH:].reshape(B, *z0.z_L.shape[1:]).to(dtype)
            )
            batch = {k: v.expand(B, *v.shape[1:]).contiguous() for k, v in batch_row.items()}
            for t in range(depth):
                carry, info = inner(carry, batch)
                preds_all[t, s : s + B] = info["logits"][:, mask].argmax(-1).cpu().numpy().astype(store_dtype)
    finally:
        inner.train(was_training)

    changed = np.zeros((depth, P), dtype=bool)
    changed[1:] = (preds_all[1:] != preds_all[:-1]).any(-1)
    steps = np.arange(1, depth + 1)[:, None]
    settle = np.where(changed, steps, 1).max(0)
    capped = changed[-1] if depth > 1 else np.zeros(P, dtype=bool)
    switch_freq = changed.sum(0) / settle
    correct = (preds_all[-1] == labels[mask].cpu().numpy()).all(-1)

    G = grid
    settle_img = settle.reshape(G, G)
    s_b, s_bb = basin_entropy(settle_img, box)
    alpha = uncertainty_exponent(settle_img)
    return {
        "settle": settle_img,
        "correct": correct.reshape(G, G),
        "capped": capped.reshape(G, G),
        "fli": _fli(preds_all.reshape(depth, G, G, -1)),
        "switch_freq": switch_freq.reshape(G, G),
        "basin_entropy": s_b,
        "boundary_basin_entropy": s_bb,
        "uncertainty_exponent": alpha,
        "boundary_dimension": 2.0 - alpha,
        "uncertainty_f": uncertainty_fraction(settle_img),
        "grid": G,
        "span": float(span),
        "depth": int(depth),
        "box": int(box),
    }


def plot_basin(result: Dict[str, object], title: str = ""):
    """Settling time / correctness / fast Lyapunov indicator / switch frequency panels."""
    from matplotlib.colors import ListedColormap

    from utils.dynamics.plots import _plt

    plt = _plt()
    sp = result["span"]
    ext = (-sp, sp, -sp, sp)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    kw = dict(origin="lower", extent=ext, interpolation="nearest")

    im = axes[0].imshow(result["settle"], cmap="viridis", **kw)
    fig.colorbar(im, ax=axes[0], label="Loops to converge (L)")
    capped = np.ma.masked_where(~result["capped"], np.ones_like(result["settle"], dtype=float))
    if result["capped"].any():
        axes[0].imshow(capped, cmap=ListedColormap(["#ff4d4d"]), **kw)
    axes[0].set_title(f"Settling time (red: capped at D={result['depth']})")

    im = axes[1].imshow(result["correct"].astype(float), cmap=ListedColormap(["#2b2d42", "#f2cc8f"]), vmin=0, vmax=1, **kw)
    cb = fig.colorbar(im, ax=axes[1], ticks=[0.25, 0.75])
    cb.ax.set_yticklabels(["wrong", "correct"])
    axes[1].set_title(f"Final answer correct ({result['correct'].mean():.0%})")

    im = axes[2].imshow(result["fli"], cmap="magma", vmin=0, vmax=1, **kw)
    fig.colorbar(im, ax=axes[2], label="λ_F (normalised)")
    axes[2].set_title("Fast Lyapunov indicator")

    im = axes[3].imshow(result["switch_freq"], cmap="cividis", **kw)
    fig.colorbar(im, ax=axes[3], label="answer changes / L")
    axes[3].set_title("Answer-switch frequency")

    for a in axes:
        a.set_xlabel("a")
        a.set_ylabel("b")
    fig.suptitle(
        f"{title}   S_b={result['basin_entropy']:.3f}  S_bb={result['boundary_basin_entropy']:.3f}  "
        f"α={result['uncertainty_exponent']:.3f}  (dim={result['boundary_dimension']:.3f})"
    )
    fig.tight_layout()
    return fig
