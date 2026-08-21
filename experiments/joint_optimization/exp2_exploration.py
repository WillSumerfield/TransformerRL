"""Experiment 2 -- Exploration.

Sweeps designer x controller exploration with the other four params pinned by `pilot.py`. Saves full
best-so-far curves rather than metrics, so the notebook can redefine the convergence threshold
without re-running: those definitions are exactly what looks wrong once you see the first plot.

Note when reading the result: exploration gates generalization rather than sitting beside it. At
e = 1 nothing ever climbs, so the pinned generalization values have no effect at all along that
edge of the grid. The degeneracy is real and is reported, not smoothed away (ADR-0023).
"""

from __future__ import annotations

import torch

import sweep
from sweep import DATA, EXPLORE, Params

SEEDS = 32
STEPS = 4000
NAME = "exp2_exploration"


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    axes = sweep.pins() | {"e_d": EXPLORE, "e_c": EXPLORE}
    p = Params.from_grid(axes, SEEDS, dev)
    print(f"{NAME}: {len(p)} runs, {STEPS} steps, k=1 on {dev}")
    res = sweep.run(p, k=1, total_steps=STEPS, device=dev)
    DATA.mkdir(exist_ok=True)
    sweep.save(str(DATA / f"{NAME}.npz"), res, axes, SEEDS, ratio=1, steps=STEPS)
    print(f"saved -> {DATA / (NAME + '.npz')}   best {float(res.best.min()):.4f}")


if __name__ == "__main__":
    main()
