"""Joint-optimization toy: two coupled optimizations over the same 2D domain.

A **designer** picks a point (x, y). A **predictor** predicts L2 at that point, seeing only
(x, y). The loss *both* minimize is the quality landscape gated by the predictor's fidelity:

    L1 = min(f(x, y) - c, 0)                    <= 0, the quality landscape
    L2 = g(x, y, L1)                                  the target landscape
    A  = exp(-(pred - L2)^2 / 2 sigma^2)        in (0, 1], fidelity
    R  = L1 * A                                 <= 0, the loss both agents minimize

Because L1 <= 0 and A in (0, 1], an inaccurate predictor drags R toward 0, i.e. makes good
points look bad. Where f >= c the clamp leaves L1 == 0 (the **dead zone**): points so bad that
no amount of fidelity helps. Raising c shrinks the dead zone and deepens the wells.

Note the units: c and sigma are in f's and g's raw units. There is no normalization, so f must
actually rise above c somewhere for a dead zone to exist.

SIGN WARNING: R is a LOSS, returned in gym's reward slot. Negate it before handing it to a
maximizing RL algorithm.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

BOUNDS = (-1.0, 1.0)
DEFAULT_C = 0.25
DEFAULT_SIGMA = 1.0

M = 0.766
W = 0.406
S = 0.224
F_mx = 5
F_my = 6
F_wx = 13.5
F_wy = 20
F_sx = 6.2
F_sy = 4.6
S_n = 12


def f(xy: torch.Tensor) -> torch.Tensor:
    """Raw quality landscape, (..., 2) -> (...). Three superposed scales, weights summing to 1:

      F_m  medium sinusoid -- the broad basins a coarse searcher can see
      F_w  fine sinusoid   -- local structure only a narrow searcher resolves
      F_s  sharp ridges    -- cos^S_n, near-zero except on thin diagonal bands

    Negated so wells are minima; range is about [-1, 1], so c is read against that scale.
    """
    norm = M + W + S
    F_m = torch.sin(F_mx * xy[..., 0]) * torch.cos(F_my * xy[..., 1])
    F_w = torch.sin(F_wx * xy[..., 0] + 1) * torch.cos(F_wy * xy[..., 1])
    F_s = torch.abs(torch.pow(torch.cos(F_sx * xy[..., 0] + torch.sin(F_sy * xy[..., 1])), S_n))
    return -(M * F_m + W * F_w + S * F_s) / norm


def g(xy: torch.Tensor, l1: torch.Tensor) -> torch.Tensor:
    """Target landscape, (..., 2) + (...) -> (...). PLACEHOLDER: smooth, mildly L1-coupled.

    The predictor sees only (x, y), so it must implicitly learn f to fit the l1 term.
    Replace with the real formula.
    """
    return torch.pow(l1, 3) - (torch.cos(5 * xy[..., 0]) + torch.sin(12 * xy[..., 1]))/10.0 - 1


def evaluate(
    xy: torch.Tensor,
    pred: torch.Tensor,
    *,
    c: float = DEFAULT_C,
    sigma: float = DEFAULT_SIGMA,
) -> dict[str, torch.Tensor]:
    """Every term of the loss, batched. `xy` is (..., 2), `pred` is (...); out-of-bounds xy is
    clipped to BOUNDS."""
    xy = xy.clamp(*BOUNDS)
    f_raw = f(xy)
    l1 = torch.clamp(f_raw - c, max=0.0)
    l2 = g(xy, l1)
    err = pred - l2
    fid = torch.exp(-0.5 * (err / sigma) ** 2)
    return {"f_raw": f_raw, "L1": l1, "L2": l2, "err": err, "fidelity": fid, "reward": l1 * fid}


def grid(n: int = 201, device: torch.device | str = "cpu") -> torch.Tensor:
    """Dense (n, n, 2) sweep of the domain, for plots and oracle scans."""
    ax = torch.linspace(*BOUNDS, n, device=device)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    return torch.stack([xx, yy], -1)


class JointOptimizationEnv(gym.Env):
    """One-step bandit wrapper over `evaluate`.

    obs    zeros(1) -- the designer is unconditional; a model-based designer fits its own
           surrogate from the (x, y, R) tuples in `info`.
    action [x, y, pred]; x, y clipped to BOUNDS, pred unbounded.
    reward R, a LOSS (see module docstring).

    The caller enforces the ordering the coupling assumes: pick (x, y) first, then predict from
    it. Nothing here stops a caller from feeding the predictor the true L2.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        c: float = DEFAULT_C,
        sigma: float = DEFAULT_SIGMA,
        device: torch.device | str = "cpu",
    ) -> None:
        self.c = c
        self.sigma = sigma
        self.device = torch.device(device)
        self.observation_space = spaces.Box(0.0, 0.0, (1,), np.float32)
        self.action_space = spaces.Box(
            np.array([BOUNDS[0], BOUNDS[0], -np.inf], np.float32),
            np.array([BOUNDS[1], BOUNDS[1], np.inf], np.float32),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {}

    def step(self, action):
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(3)
        out = evaluate(a[:2], a[2], c=self.c, sigma=self.sigma)
        info = {k: float(v) for k, v in out.items()}
        info["x"], info["y"] = (float(v) for v in a[:2].clamp(*BOUNDS))
        return np.zeros(1, np.float32), info["reward"], True, False, info
