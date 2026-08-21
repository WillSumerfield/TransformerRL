"""Coarse six-parameter sweep at ratio 1:1, to choose where Experiments 1-3 pin their four
un-swept params.

Pinning matters more than it looks: "generalization beats spread" can flip entirely depending on
where exploration sits, since maximal exploration makes generalization inert. Pinning at the
best-performing region means every later plot reads as "how does this param matter when everything
else is set well", which is what a reader assumes anyway.

The pin is the centroid of the top-N configs, not the single argmin, so it is stable against the
seed noise that makes a raw argmin jump between neighbouring cells.
"""

from __future__ import annotations

import numpy as np
import torch

import sweep
from sweep import DATA, PARAMS, Params

COARSE = 5
TOP_N = 32
SEEDS = 8
STEPS = 2000


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    axes = {
        "sig_d": np.geomspace(0.01, 1.0, COARSE),
        "sig_c": np.geomspace(0.01, 1.0, COARSE),
        "e_d": np.linspace(0.0, 1.0, COARSE),
        "e_c": np.linspace(0.0, 1.0, COARSE),
        "g_d": np.geomspace(0.01, 1.0, COARSE),
        "g_c": np.geomspace(0.01, 1.0, COARSE),
    }
    p = Params.from_grid(axes, SEEDS, dev)
    print(f"pilot: {len(p)} runs, {STEPS} steps, k=1 on {dev}")
    res = sweep.run(p, k=1, total_steps=STEPS, P_d=8, P_a=8, device=dev)

    # mean over seeds, then centroid of the best TOP_N configs
    best = res.best.reshape(*(COARSE,) * len(PARAMS), SEEDS).mean(-1)
    order = best.flatten().argsort()[:TOP_N]
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
    print(f"best config value {float(best.min()):.4f}")


if __name__ == "__main__":
    main()
