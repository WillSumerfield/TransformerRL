"""Experiment 4 -- Sampling ratios.

Sweeps all six params at every sampling ratio. Unlike Experiments 1-3 this keeps the full grid but
persists only per-run summary scalars: full curves at 5^6 x 5 ratios x seeds would be gigabytes,
and this experiment asks coarse questions (which params matter, how the optimum shifts) that need
grid coverage more than they need re-definable metrics. Changing a metric here means re-running.

The evaluation budget is identical across ratios by construction -- a controller step always costs
P_d * P_a evaluations, so fixing `total_steps` fixes the budget and `k` only decides how many
designer updates fit inside it. Without that, ratio 1:2048 would simply buy 2048x the compute and
win trivially.
"""

from __future__ import annotations

import numpy as np
import torch

import sweep
from sweep import DATA, GEN, PARAMS, RATIOS, SPREAD, Params

COARSE = 5
SEEDS = 128
DESIGNER_ITERS_AT_MAX = 25
STEPS = max(RATIOS) * DESIGNER_ITERS_AT_MAX
NAME = "exp4_ratios"


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Endpoints taken from SPREAD/GEN rather than restated, at coarse resolution: the two must not
    # drift apart again. They did once -- this grid was left on the pre-BOUNDS-change range
    # (0.01, 1.0), which stops below the pilot's own optimum, so the sweep could not contain the
    # good region and the two radii archetypes snapped to a single cell.
    axes = {
        "sig_d": np.geomspace(SPREAD[0], SPREAD[-1], COARSE),
        "sig_c": np.geomspace(SPREAD[0], SPREAD[-1], COARSE),
        "e_d": np.linspace(0.0, 1.0, COARSE),
        "e_c": np.linspace(0.0, 1.0, COARSE),
        "g_d": np.geomspace(GEN[0], GEN[-1], COARSE),
        "g_c": np.geomspace(GEN[0], GEN[-1], COARSE),
    }
    p = Params.from_grid(axes, SEEDS, dev)
    shape = (COARSE,) * len(PARAMS) + (SEEDS,)
    best, speed = [], []

    for k in RATIOS:
        print(f"{NAME}: k={k}, {len(p)} runs, {STEPS} steps, {STEPS // k} designer iters")
        res = sweep.run(p, k=k, total_steps=STEPS, P_d=8, P_a=8, device=dev)
        best.append(res.best.reshape(shape).numpy())
        speed.append(sweep.convergence_evals(res.curves, res.evals).reshape(shape).numpy())

    DATA.mkdir(exist_ok=True)
    np.savez_compressed(
        DATA / f"{NAME}.npz",
        best=np.stack(best, axis=-2),  # (..., ratio, seed)
        speed=np.stack(speed, axis=-2),
        ratios=np.array(RATIOS),
        param_order=np.array(PARAMS),
        **{f"axis_{k}": v for k, v in axes.items()},
        steps=STEPS,
        seeds=SEEDS,
    )
    print(f"saved -> {DATA / (NAME + '.npz')}")


if __name__ == "__main__":
    main()
