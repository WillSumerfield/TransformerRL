"""Joint-optimization toy: two optimizers maximizing reward on one shared landscape.

A **designer** chooses a design without seeing anything; a **controller** chooses how to act on
that design. Both maximize reward on the same landscape. Nothing in the landscape couples them --
the coupling is emergent, because a design is only worth what the controller manages to achieve
on it.

The experiments measure how the two optimizers' spread, exploration, and generalization trade off
against each other, and how that changes with the sampling ratio between them. See
`experiments/CONTEXT.md` for the vocabulary and ADR-0023 for how the three axes are mechanized.

    landscape.py    f, the grid, and the precomputed peak tables
    optimizers.py   the Simple-ES primitives both optimizers share
    sweep.py        vectorized runner: a whole config grid advances as one tensor program
    pilot.py        coarse sweep that chooses where exps 1-3 pin their un-swept params
    exp[1-4]_*.py   the four experiments
    paths.py        named archetypes with full trajectories recorded
    run_all.py      runs the series in dependency order; `--skip` to leave stages out
    analysis.py     every figure (`# %%` cells, run interactively)

Experiment scripts produce data only; all plotting lives in `analysis.py`.
"""

from landscape import BOUNDS, axis, f, grid, peak_tables, to_coord, to_index
from optimizers import climb, controller_act, designer_propose, recombine
from sweep import PARAMS, RATIOS, Params, Result, convergence_evals, run, save

__all__ = [
    "BOUNDS", "PARAMS", "RATIOS", "Params", "Result", "axis", "climb",
    "controller_act", "convergence_evals", "designer_propose", "f", "grid", "peak_tables",
    "recombine", "run",
    "save", "to_coord", "to_index",
]
