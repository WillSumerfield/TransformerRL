"""Vectorized runner: advances an entire config grid x seed set as one tensor program.

Starting positions use common random numbers -- every config sees the same set of `n_seeds`
starting points -- so cell-to-cell differences reflect the parameters rather than the draw. The
per-step sampling noise is still independent across configs. `run(start=...)` overrides the draw
entirely, which is what the path figures use to put every archetype on a chosen patch of landscape.

The run dimension is chunked to a fixed memory budget, so a sweep is bounded by wall-clock rather
than by VRAM and the seed count is a free parameter. Chunk size is derived from `P_d * P_a` and a
constant budget rather than from free VRAM, so it depends only on this code: two machines large
enough to hold one chunk produce the same numbers. Starting positions are drawn once, before any
chunking, and indexed by global run index -- chunk boundaries cannot move where a run starts.

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

from landscape import BOUNDS, peak_tables
from landscape import f_fused as f
from optimizers import controller_act, designer_propose, recombine

PARAMS = ("sig_d", "sig_c", "e_d", "e_c", "g_d", "g_c")

DATA = Path(__file__).parent / "data"

# Spread and generalization are radii on the domain, so they are log-spaced: the top of the range
# spans the whole width-4 domain and the bottom is ~13 grid cells at n=512. Exploration is a
# probability.
SPREAD = np.geomspace(0.1, 4, 11)
GEN = np.geomspace(0.1, 4, 11)
EXPLORE = np.linspace(0.0, 1.0, 11)

RATIOS = (1, 8, 64, 256, 2048)

# Peak working set is dominated by the (R, P_d, P_a) tensors inside a step -- the action cloud,
# the fused landscape evaluation, the peak-table gathers -- so it is linear in R * P_d * P_a, plus
# a small per-run term for the best-so-far curve. Coefficients are measured (RTX 5080, fp32, with
# `f` and `climb` fused); the budget is a constant rather than a query of free VRAM so that chunk
# size, and therefore the RNG stream, is a property of the code and not of what else is on the GPU.
CHUNK_BUDGET = 8 * 2**30  # bytes of GPU working set allowed per chunk
BYTES_PER_SAMPLE = 40  # per run, per (P_d * P_a) evaluation
BYTES_PER_RUN = 1024  # per run, independent of population size


def chunk_size(P_d: int, P_a: int) -> int:
    """How many runs fit in one chunk at this population size."""
    return max(int(CHUNK_BUDGET // (BYTES_PER_SAMPLE * P_d * P_a + BYTES_PER_RUN)), 1)


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
    n_seeds: int = 1

    @staticmethod
    def from_grid(axes: dict[str, np.ndarray], seeds: int, device="cpu") -> "Params":
        """Cartesian product of the named axes x seeds, flattened to a run dimension.

        Run index unravels as (*axis_lengths, seeds), matching the order of `PARAMS`.
        """
        cols = [torch.as_tensor(axes[p], dtype=torch.float32, device=device) for p in PARAMS]
        mesh = torch.meshgrid(*cols, torch.arange(seeds, device=device, dtype=torch.float32), indexing="ij")
        flat = {p: m.reshape(-1).contiguous() for p, m in zip(PARAMS, mesh[:-1])}
        return Params(**flat, n_seeds=seeds)

    def __len__(self) -> int:
        return self.sig_d.numel()

    def slice(self, lo: int, hi: int) -> "Params":
        """Runs `[lo, hi)` as their own Params. `n_seeds` is carried through unchanged -- it
        describes the seed-start set, which is global, not the chunk's own length."""
        return Params(**{p: getattr(self, p)[lo:hi] for p in PARAMS}, n_seeds=self.n_seeds)


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
    P_d: int = 64,
    P_a: int = 64,
    alpha: float = 0.005,
    top_percent: float = 0.01,
    n: int = 512,
    checkpoints: int = 256,
    seed: int = 0,
    device: str = "cpu",
    record_paths: bool = False,
    start: tuple | None = None,
    chunk: int | None = None,
) -> Result:
    """Advance every run in `p` for `total_steps` controller steps at sampling ratio `k`.

    `start` pins the initial `(mu_d, mu_a)` instead of drawing them. Each element is a scalar
    (every run starts in the same place) or an (R,) array (one start per run). Sweeps leave it
    None and take the common random draw; the path figures set it, so which landscape a
    trajectory is walking is a stated choice rather than a property of the seed.

    The run dimension is processed in chunks of `chunk` runs (default `chunk_size(P_d, P_a)`), so
    the seed count is bounded by wall-clock rather than by VRAM. Results are identical to a single
    chunk except for the per-step noise stream, which follows the chunk layout.
    """
    R = len(p)
    gen = torch.Generator(device=device).manual_seed(seed)
    tables = peak_tables(n, device)[1:]
    marks = np.unique(np.geomspace(1, total_steps, checkpoints).round().astype(int))

    # Common random numbers: seeds is the last axis of the run index, so run `r` starts at
    # seed-start `r % n_seeds` and every config sees the *same* set. A difference between two
    # cells is then attributable to their parameters rather than to where they happened to land,
    # which is what makes the cells a paired comparison. Drawn here, once, before any chunking:
    # indexing by global run index means a chunk boundary cannot change where a run starts.
    if start is None:
        b_lo, b_hi = BOUNDS
        s = max(p.n_seeds, 1)
        seed_d = torch.empty(s, device=device).uniform_(b_lo, b_hi, generator=gen)
        seed_a = torch.empty(s, device=device).uniform_(b_lo, b_hi, generator=gen)
        idx = torch.arange(R, device=device) % s
        mu_d0, mu_a0 = seed_d[idx], seed_a[idx]
    else:
        mu_d0, mu_a0 = (torch.as_tensor(x, dtype=torch.float32, device=device).expand(R).clone()
                        for x in start)

    cs = chunk or chunk_size(P_d, P_a)
    if record_paths and R > cs:
        raise ValueError(
            f"record_paths needs one chunk but {R} runs exceed {cs}; the trajectories are the "
            "dominant allocation and would not fit anyway -- run fewer configs at a time."
        )

    curves, best, path = [], [], None
    for c, c_lo in enumerate(range(0, R, cs)):
        c_hi = min(c_lo + cs, R)
        if R > cs:
            print(f"  chunk {c + 1}/{-(-R // cs)}: runs [{c_lo}, {c_hi})", flush=True)
        cu, be, path = _run_chunk(
            p.slice(c_lo, c_hi), mu_d0[c_lo:c_hi], mu_a0[c_lo:c_hi], k, total_steps, P_d, P_a,
            alpha, top_percent, n, marks, tables, gen, device, record_paths,
        )
        curves.append(cu.cpu())
        best.append(be.cpu())
    return Result(torch.cat(curves), marks * P_d * P_a, torch.cat(best), path)


def _run_chunk(
    p, mu_d, mu_a, k, total_steps, P_d, P_a, alpha, top_percent, n, marks, tables, gen,
    device, record_paths,
):
    """One chunk of the run dimension, start to finish. `gen` is shared across chunks, so the
    per-step noise stream depends on the chunk layout -- which is why `chunk_size` is derived
    from constants rather than from whatever VRAM happens to be free."""
    T_d, T_a = tables
    R = len(p)
    mu_d, mu_a = mu_d.clone(), mu_a.clone()
    mu_d_prev = mu_d.clone()

    best = torch.full((R,), -torch.inf, device=device)
    curves = torch.full((R, len(marks)), -torch.inf, device=device)
    path = {"mu_d": [], "mu_a": [], "d": [], "a": []} if record_paths else None
    start = (mu_d.clone(), mu_a.clone()) if record_paths else None

    D = acc = None
    ci = 0
    for t in range(total_steps):
        if t % k == 0:
            if D is not None:
                mu_d_prev = mu_d
                mu_d = recombine(mu_d, D, acc / k, alpha, top_percent=top_percent)
            D = designer_propose(mu_d, mu_a, p.sig_d, p.e_d, p.g_d, T_d, n, P_d, gen)
            acc = torch.zeros(R, P_d, device=device)

        A = controller_act(D, mu_a, mu_d_prev, p.sig_c, p.e_c, p.g_c, T_a, n, P_a, gen)
        v = f(torch.stack([D.unsqueeze(-1).expand_as(A), A], dim=-1))

        acc += v.mean(-1)
        mu_a = recombine(mu_a, A.flatten(1), v.flatten(1), alpha, top_percent=top_percent)

        # a design's fitness is only known once its whole adaptation window has closed
        if (t + 1) % k == 0:
            best = torch.maximum(best, (acc / k).max(-1).values)

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
        # the true pre-loop means: mu_a has already moved once by the time step 0 is recorded
        path["start_d"], path["start_a"] = start[0].cpu(), start[1].cpu()
    return curves, best, path


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
    eps = eps_frac * (final - curves[:, :1]).clamp_min(1e-12)
    first = (curves >= final - eps).float().argmax(dim=1)
    return torch.as_tensor(evals)[first]


def save(path: str, res: Result, axes: dict[str, np.ndarray], seeds: int, **meta) -> None:
    """Persist a result with its axis values, so the figure script never reconstructs axis order."""
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
