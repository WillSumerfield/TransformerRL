"""Experiment 1 -- Spread.

Sweeps designer x controller spread with the other four params pinned by `pilot.py`. Saves full
best-so-far curves rather than metrics, so `analysis.py` can redefine the convergence threshold
without re-running: those definitions are exactly what looks wrong once you see the first plot.

Note when reading the result: the update steps toward the sample cloud, so wider spread produces
larger steps. That coupling is deliberate (ADR-0023) -- wide sampling genuinely buying faster
coarse progress is the trade-off this experiment is about.
"""

from __future__ import annotations

import torch

import sweep
from sweep import DATA, SPREAD, Params

SEEDS = 2048
STEPS = 4000
NAME = "exp1_spread"


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    axes = sweep.pins() | {"sig_d": SPREAD, "sig_c": SPREAD}
    p = Params.from_grid(axes, SEEDS, dev)
    print(f"{NAME}: {len(p)} runs, {STEPS} steps, k=1 on {dev}")
    res = sweep.run(p, k=1, total_steps=STEPS, device=dev)
    DATA.mkdir(exist_ok=True)
    sweep.save(str(DATA / f"{NAME}.npz"), res, axes, SEEDS, ratio=1, steps=STEPS)
    print(f"saved -> {DATA / (NAME + '.npz')}   best {float(res.best.max()):.4f}")


if __name__ == "__main__":
    main()
