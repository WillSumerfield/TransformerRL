"""Named example configurations, run individually with their full trajectories recorded.

No sweep output contains a trajectory -- the sweeps keep only best-so-far curves and summaries --
so the path-taken figure needs its own artifact. All four archetypes start from the same stated
`START` rather than a random draw, so the panels differ only by configuration. `analysis.py`
windows the landscape around that point and shades the surface itself red where the pair went.

The archetypes exist to be legible rather than optimal: each isolates one failure mode, and the
matched/mismatched pair is the direct visual test of the hypothesis that the two optimizers need
comparable radii.
"""

from __future__ import annotations

import numpy as np
import torch

import sweep
from sweep import DATA, PARAMS, Params

STEPS = 2000
SUBSAMPLE = 8  # keep every Nth step's sampled points; a full run is millions

# Every archetype starts here, rather than wherever the seed happened to land: comparing four
# configurations only means something if they are handed the same problem. Chosen to be a hard
# but not hopeless spot -- 2nd percentile of the landscape, 1.4 away in design from the global
# max at (-0.20, +0.59), and at an action where the slice the designer climbs is *anti*-correlated
# with the marginal it is scored on, so a pair that cannot escape stays badly wrong.
START = (-1.5, 1.5)

ARCHETYPES = {
    # narrow and greedy: converges immediately, onto whichever hill it started on
    "narrow_greedy": dict(sig_d=0.03, sig_c=0.03, e_d=0.05, e_c=0.05, g_d=0.3, g_c=0.3),
    # wide and undirected: covers the space, commits to nothing
    "wide_exploratory": dict(sig_d=4, sig_c=4, e_d=0.8, e_c=0.8, g_d=1.0, g_c=1.0),
    # comparable radii on both optimizers
    "matched_radii": dict(sig_d=2, sig_c=2, e_d=0.35, e_c=0.35, g_d=2.0, g_c=2.0),
    # designer ranges far beyond what its controller can handle
    "mismatched_radii": dict(sig_d=4, sig_c=2, e_d=0.35, e_c=0.35, g_d=1.0, g_c=2.0),
}


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = list(ARCHETYPES)
    axes = {p: np.array([ARCHETYPES[n][p] for n in names]) for p in PARAMS}
    # one run per archetype, so build Params directly rather than as a cartesian product
    p = Params(**{k: torch.tensor(v, dtype=torch.float32, device=dev) for k, v in axes.items()})

    print(f"paths: {len(names)} archetypes, {STEPS} steps, k=1 from {START} on {dev}")
    res = sweep.run(p, k=1, total_steps=STEPS, device=dev, record_paths=True, start=START)

    DATA.mkdir(exist_ok=True)
    np.savez_compressed(
        DATA / "paths.npz",
        names=np.array(names),
        mu_d=res.paths["mu_d"].numpy(),
        mu_a=res.paths["mu_a"].numpy(),
        start_d=res.paths["start_d"].numpy(),
        start_a=res.paths["start_a"].numpy(),
        d=res.paths["d"][::SUBSAMPLE].numpy(),
        a=res.paths["a"][::SUBSAMPLE].numpy(),
        subsample=SUBSAMPLE,
        best=res.best.numpy(),
        **{f"param_{k}": v for k, v in axes.items()},
    )
    for n, b in zip(names, res.best.tolist()):
        print(f"  {n:18s} best {b:.4f}")
    print(f"saved -> {DATA / 'paths.npz'}")


if __name__ == "__main__":
    main()
