"""Experiment 3 -- Generalization.

Sweeps designer x controller generalization with the other four params pinned by `pilot.py`. Saves full
best-so-far curves rather than metrics, so `analysis.py` can redefine the convergence threshold
without re-running: those definitions are exactly what looks wrong once you see the first plot.

Note when reading the result: this is the experiment the matching-radii hypothesis lives in. The
controller's radius is measured in the joint space, so a designer ranging beyond it gets its good
designs played badly and scored badly -- the off-diagonal cells are where that shows up. At
g_c -> inf the controller overwrites its own sample entirely and controller spread stops mattering.
"""

from __future__ import annotations

import torch

import sweep
from sweep import DATA, GEN, Params

SEEDS = 2048
STEPS = 4000
NAME = "exp3_generalization"


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    axes = sweep.pins() | {"g_d": GEN, "g_c": GEN}
    p = Params.from_grid(axes, SEEDS, dev)
    print(f"{NAME}: {len(p)} runs, {STEPS} steps, k=1 on {dev}")
    res = sweep.run(p, k=1, total_steps=STEPS, device=dev)
    DATA.mkdir(exist_ok=True)
    sweep.save(str(DATA / f"{NAME}.npz"), res, axes, SEEDS, ratio=1, steps=STEPS)
    print(f"saved -> {DATA / (NAME + '.npz')}   best {float(res.best.max()):.4f}")


if __name__ == "__main__":
    main()
