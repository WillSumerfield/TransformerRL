"""Coarse six-parameter sweep at ratio 1:1, to choose where Experiments 1-3 pin their four
un-swept params.

Pinning matters more than it looks: "generalization beats spread" can flip entirely depending on
where exploration sits, since maximal exploration makes generalization inert. Pinning at the
best-performing region means every later plot reads as "how does this param matter when everything
else is set well", which is what a reader assumes anyway.

The pin is the centroid of the top-N configs, not the single argmin, so it is stable against the
seed noise that makes a raw argmin jump between neighbouring cells. It runs at the same population
size as Experiments 1-3, so the pins are chosen under the selection pressure they are pinned for.
"""

from __future__ import annotations

import numpy as np
import torch

import sweep
from sweep import DATA, PARAMS, Params

COARSE = 5
TOP_N = 32
SEEDS = 256
STEPS = 2000


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    axes = {
        "sig_d": np.geomspace(0.1, 4.0, COARSE),
        "sig_c": np.geomspace(0.1, 4.0, COARSE),
        "e_d": np.linspace(0.0, 1.0, COARSE),
        "e_c": np.linspace(0.0, 1.0, COARSE),
        # single-element axes, not COARSE copies of one value: the cartesian product would
        # otherwise run 25 identical copies of every cell. Experiment 3 sweeps these two.
        "g_d": np.array([2.0]),
        "g_c": np.array([2.0]),
    }
    p = Params.from_grid(axes, SEEDS, dev)
    print(f"pilot: {len(p)} runs, {STEPS} steps, k=1 on {dev}")
    res = sweep.run(p, k=1, total_steps=STEPS, device=dev)

    # mean over seeds, then centroid of the best TOP_N configs
    shape = tuple(len(axes[q]) for q in PARAMS) + (SEEDS,)
    best = res.best.reshape(shape).mean(-1)
    order = best.flatten().argsort(descending=True)[:TOP_N]
    idx = np.unravel_index(order.numpy(), best.shape)
    pin = {p_: float(np.mean(axes[p_][idx[i]])) for i, p_ in enumerate(PARAMS)}

    DATA.mkdir(exist_ok=True)
    np.savez(
        DATA / "pilot.npz",
        best=best.numpy(),
        **{f"axis_{k}": v for k, v in axes.items()},
        **{f"pin_{k}": v for k, v in pin.items()},
        top_n=TOP_N,
        seeds=SEEDS,
        steps=STEPS,
    )
    print("pins:", {k: round(v, 4) for k, v in pin.items()})
    print(f"best config value {float(best.max()):.4f}")


if __name__ == "__main__":
    main()
