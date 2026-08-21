"""Vectorized runner: advances an entire config grid x seed set as one tensor program.

One loop over controller steps, with the designer updating every `k`-th step. Because a step
always costs `P_d * P_a` evaluations, holding `total_steps` fixed holds the evaluation budget
fixed across every sampling ratio automatically -- `k` only decides how many designer updates fit
inside that budget. That is the comparison Experiment 4 needs: high ratio buys the controller
adaptation time and costs the designer updates, rather than buying more compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from landscape import BOUNDS, basin_tables, f
from optimizers import controller_act, designer_propose, recombine

PARAMS = ("sig_d", "sig_c", "e_d", "e_c", "g_d", "g_c")

DATA = Path(__file__).parent / "data"

# Spread and generalization are radii on a domain of width 2, so they are log-spaced: 0.01 is
# finer than any landscape feature, 1.0 spans the whole space. Exploration is a probability.
SPREAD = np.geomspace(0.01, 1.0, 11)
GEN = np.geomspace(0.01, 1.0, 11)
EXPLORE = np.linspace(0.0, 1.0, 11)

RATIOS = (1, 8, 64, 256, 2048)


def pins() -> dict[str, np.ndarray]:
    """The pinned values for un-swept params, as chosen by `pilot.py`."""
    z = np.load(DATA / "pilot.npz")
    return {p: np.array([float(z[f"pin_{p}"])]) for p in PARAMS}


@dataclass
class Params:
    """The six swept configuration axes. Each field is a (R,) tensor, one entry per run."""

    sig_d: torch.Tensor
    sig_c: torch.Tensor
    e_d: torch.Tensor
    e_c: torch.Tensor
    g_d: torch.Tensor
    g_c: torch.Tensor

    @staticmethod
    def from_grid(axes: dict[str, np.ndarray], seeds: int, device="cpu") -> "Params":
        """Cartesian product of the named axes x seeds, flattened to a run dimension.

        Run index unravels as (*axis_lengths, seeds), matching the order of `PARAMS`.
        """
        cols = [torch.as_tensor(axes[p], dtype=torch.float32, device=device) for p in PARAMS]
        mesh = torch.meshgrid(*cols, torch.arange(seeds, device=device, dtype=torch.float32), indexing="ij")
        return Params(**{p: m.reshape(-1).contiguous() for p, m in zip(PARAMS, mesh[:-1])})

    def __len__(self) -> int:
        return self.sig_d.numel()


@dataclass
class Result:
    curves: torch.Tensor  # (R, C) best design fitness so far, at each checkpoint
    evals: np.ndarray  # (C,) evaluation count at each checkpoint
    best: torch.Tensor  # (R,) best design fitness over the whole run
    paths: dict | None  # joint trajectory, only when record_paths


def run(
    p: Params,
    k: int,
    *,
    total_steps: int,
    P_d: int = 16,
    P_a: int = 16,
    alpha: float = 0.5,
    n: int = 512,
    checkpoints: int = 256,
    seed: int = 0,
    device: str = "cpu",
    record_paths: bool = False,
) -> Result:
    """Advance every run in `p` for `total_steps` controller steps at sampling ratio `k`."""
    R = len(p)
    gen = torch.Generator(device=device).manual_seed(seed)
    _, T_d, T_a = basin_tables(n, device)

    lo, hi = BOUNDS
    mu_d = torch.empty(R, device=device).uniform_(lo, hi, generator=gen)
    mu_a = torch.empty(R, device=device).uniform_(lo, hi, generator=gen)
    mu_d_prev = mu_d.clone()

    best = torch.full((R,), torch.inf, device=device)
    marks = np.unique(np.geomspace(1, total_steps, checkpoints).round().astype(int))
    curves = torch.full((R, len(marks)), torch.inf, device=device)
    path = {"mu_d": [], "mu_a": [], "d": [], "a": []} if record_paths else None

    D = acc = None
    ci = 0
    for t in range(total_steps):
        if t % k == 0:
            if D is not None:
                mu_d_prev = mu_d
                mu_d = recombine(mu_d, D, acc / k, alpha)
            D = designer_propose(mu_d, mu_a, p.sig_d, p.e_d, p.g_d, T_d, n, P_d, gen)
            acc = torch.zeros(R, P_d, device=device)

        A = controller_act(D, mu_a, mu_d_prev, p.sig_c, p.e_c, p.g_c, T_a, n, P_a, gen)
        v = f(torch.stack([D.unsqueeze(-1).expand_as(A), A], dim=-1))

        acc += v.mean(-1)
        mu_a = recombine(mu_a, A.flatten(1), v.flatten(1), alpha)

        # a design's fitness is only known once its whole adaptation window has closed
        if (t + 1) % k == 0:
            best = torch.minimum(best, (acc / k).min(-1).values)

        if ci < len(marks) and t + 1 == marks[ci]:
            curves[:, ci] = best
            ci += 1

        if record_paths:
            path["mu_d"].append(mu_d.clone())
            path["mu_a"].append(mu_a.clone())
            path["d"].append(D.clone())
            path["a"].append(A.reshape(R, -1).clone())

    if record_paths:
        path = {key: torch.stack(val).cpu() for key, val in path.items()}
    return Result(curves.cpu(), marks * P_d * P_a, best.cpu(), path)


def convergence_evals(
    curves: torch.Tensor, evals: np.ndarray, eps_frac: float = 0.01
) -> torch.Tensor:
    """Evaluations until best-so-far comes within eps of the run's *own* final value.

    Nothing is censored, so every cell is finite and the surface has no holes -- but a run that
    collapses instantly onto a bad point scores as maximally fast. This metric must always be read
    alongside the best-fitness metric, never on its own.

    `eps` is a fraction of the run's total improvement, so it is scale-free.
    """
    final = curves[:, -1:]
    eps = eps_frac * (curves[:, :1] - final).clamp_min(1e-12)
    first = (curves <= final + eps).float().argmax(dim=1)
    return torch.as_tensor(evals)[first]


def save(path: str, res: Result, axes: dict[str, np.ndarray], seeds: int, **meta) -> None:
    """Persist a result with its axis values, so the notebook never reconstructs axis order."""
    shape = tuple(len(axes[p]) for p in PARAMS) + (seeds,)
    np.savez_compressed(
        path,
        curves=res.curves.numpy().reshape(*shape, -1),
        best=res.best.numpy().reshape(*shape),
        evals=res.evals,
        param_order=np.array(PARAMS),
        **{f"axis_{p}": axes[p] for p in PARAMS},
        **meta,
    )
