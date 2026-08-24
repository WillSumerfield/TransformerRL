"""The shared landscape both optimizers minimize, plus the precomputed basin tables.

`f(da) -> v` maps a point in (design, action) space to a value. Both optimizers minimize it.
Neither ever sees it whole: the designer only observes a design's realized value, the controller
only the slice for the design in front of it.

The basin tables are the reason the whole sweep stays a batched tensor program. A "climb" descends
to the minimum of the basin the sample landed in, and that target is a pure function of position,
so it can be precomputed once on a grid and then gathered in O(1) per sample:

    T_a[i, j]  action index of the basin min along the action axis, within row i (fixed design)
    T_d[i, j]  design index of the basin min along the design axis, within column j (fixed action)

Both are built by iterated steepest descent with pointer doubling, so they converge exactly in
ceil(log2(n)) passes rather than one pass per basin width.
"""

from __future__ import annotations

import math

import torch

BOUNDS = (-2.0, 2.0)

# --- the landscape --------------------------------------------------------------------------
M = 2.04
W = 0.55
S = 1.07
P = 2.0
F_md, F_ma = 4.08, 3.0
F_wd, F_wa = 8.44, 15.1
F_sd, F_sa = 0.63, 2.64
F_nd, F_na = 20, 19
F_pd, F_pa = 0.8, 3.5


def f(da: torch.Tensor) -> torch.Tensor:
    """Landscape, (..., 2) -> (...). Index 0 is the design, index 1 is the action.

    Four superposed terms. F_p is a sum of two sines rather than a product, so it spans +-2 and
    the normalization does not bound f to [-1, 1]; only relative scale matters here.

      F_m  broad sinusoid   -- the coarse basins a wide searcher can see
      F_w  fine sinusoid    -- local structure only a narrow searcher resolves, and much finer
                               along the action axis than the design axis
      F_s  separable spike  -- cos^F_nd(d) * sin^F_na(a), near-zero except in a small region
      F_p  periodic pattern -- sin(F_pd d) + sin(F_pa a), separable, adds complexity

    Both optimizers minimize this.
    """
    d, a = da[..., 0], da[..., 1]
    F_m = torch.sin(F_md * d - 1) * torch.cos(F_ma * a + 1.5)
    F_w = torch.sin(F_wd * d + 1) * torch.cos(F_wa * a)
    F_s = torch.abs(torch.pow(torch.cos(F_sd * d), F_nd) * torch.pow(torch.sin(F_sa * a), F_na))
    F_p = torch.sin(F_pd * d + 1.9) + torch.sin(F_pa * a)
    return (M * F_m + W * F_w + S * F_s + P * F_p) / (M + W + S + P)


# --- grid and basin tables ------------------------------------------------------------------


def axis(n: int, device="cpu") -> torch.Tensor:
    """The shared coordinate axis, `n` points spanning BOUNDS."""
    return torch.linspace(*BOUNDS, n, device=device)


def grid(n: int = 512, device="cpu") -> torch.Tensor:
    """Dense (n, n, 2) sweep of the domain. Index [i, j] is (design_i, action_j)."""
    ax = axis(n, device)
    dd, aa = torch.meshgrid(ax, ax, indexing="ij")
    return torch.stack([dd, aa], dim=-1)


def _descend(z: torch.Tensor) -> torch.Tensor:
    """Index of each point's basin minimum, descending along the last axis of `z`.

    One steepest-descent step gives a functional graph whose fixed points are the local minima;
    pointer doubling then resolves every point to its basin floor in ceil(log2(n)) passes.
    """
    n = z.shape[-1]
    left = torch.roll(z, 1, dims=-1)
    right = torch.roll(z, -1, dims=-1)
    left[..., 0] = torch.inf  # edges cannot descend outward
    right[..., -1] = torch.inf

    idx = torch.arange(n, device=z.device).expand_as(z).clone()
    idx = torch.where(left < z, idx - 1, idx)
    # a strictly lower right neighbour wins only if it also beats the left one
    go_right = (right < z) & (right < left)
    idx = torch.where(go_right, torch.arange(n, device=z.device).expand_as(z) + 1, idx)

    for _ in range(math.ceil(math.log2(n))):
        idx = torch.gather(idx, -1, idx)
    return idx


def basin_tables(n: int = 512, device="cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute `(Z, T_d, T_a)` on an n x n grid.

    Z    (n, n)  landscape values, [design, action]
    T_d  (n, n)  design index of the basin min along the design axis, per fixed action
    T_a  (n, n)  action index of the basin min along the action axis, per fixed design
    """
    Z = f(grid(n, device))
    T_a = _descend(Z)  # descend along the action axis (last dim)
    T_d = _descend(Z.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()
    return Z, T_d, T_a


def to_index(x: torch.Tensor, n: int) -> torch.Tensor:
    """Map coordinates in BOUNDS to nearest grid indices, clamped."""
    lo, hi = BOUNDS
    return (((x - lo) / (hi - lo)) * (n - 1)).round().long().clamp_(0, n - 1)


def to_coord(i: torch.Tensor, n: int) -> torch.Tensor:
    """Map grid indices back to coordinates in BOUNDS."""
    lo, hi = BOUNDS
    return lo + (hi - lo) * i.to(torch.float32) / (n - 1)
