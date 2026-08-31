#!/usr/bin/env python3
"""Run the experiment series in dependency order.

Experiments 1-3 read their pinned parameters from `pilot.npz`, so the pilot has to have run first.
That ordering is enforced here rather than left to be remembered, and a selection that needs pins
which do not exist yet is refused up front instead of failing 27 minutes in.

Everything runs in one process: `landscape.f` and `optimizers.climb` are compiled on first use, so
sharing a process pays that warmup once for the series rather than once per stage. A stage that
raises is reported and the series continues -- losing the remaining hours to one failure is worse
than finishing with a gap.

    python run_all.py                     # everything, in order
    python run_all.py --skip exp4         # everything except exp4
    python run_all.py pilot exp1 exp2     # only these, still in canonical order
    python run_all.py --list              # the stages and what they cost
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # flat imports, as the modules expect

from sweep import DATA

# name -> (module, measured minutes on an RTX 5080). Iteration order is the run order.
STAGES = {
    "pilot": ("pilot", 3),
    "exp1": ("exp1_spread", 27),
    "exp2": ("exp2_exploration", 27),
    "exp3": ("exp3_generalization", 27),
    "exp4": ("exp4_ratios", 276),
    "paths": ("paths", 1),
}
NEEDS_PINS = ("exp1", "exp2", "exp3")


def fmt(minutes: float) -> str:
    return f"{minutes:.0f} min" if minutes < 90 else f"{minutes / 60:.1f} h"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stages", nargs="*", help=f"stages to run (default: all of {', '.join(STAGES)})")
    ap.add_argument("--skip", nargs="+", default=[], metavar="STAGE", help="stages to leave out")
    ap.add_argument("--list", action="store_true", help="show the stages and exit")
    ap.add_argument("--dry-run", action="store_true", help="show the plan without running it")
    a = ap.parse_args(argv)

    if bad := sorted((set(a.stages) | set(a.skip)) - set(STAGES)):
        ap.error(f"unknown stage(s): {', '.join(bad)}   known: {', '.join(STAGES)}")

    if a.list:
        for name, (mod, mins) in STAGES.items():
            print(f"  {name:6s} {mod + '.py':24s} {fmt(mins):>7s}")
        return 0

    sel = [s for s in STAGES if s in (a.stages or STAGES) and s not in a.skip]
    if not sel:
        ap.error("nothing selected")

    # refuse up front rather than after the earlier stages have already burned their hours
    if any(s in NEEDS_PINS for s in sel) and "pilot" not in sel and not (DATA / "pilot.npz").exists():
        ap.error(f"{', '.join(s for s in sel if s in NEEDS_PINS)} need pins, but no {DATA}/pilot.npz"
                 " -- include `pilot` or run it first")

    total = sum(STAGES[s][1] for s in sel)
    print(f"plan: {' -> '.join(sel)}   estimated {fmt(total)}", flush=True)
    if a.dry_run:
        return 0

    t0 = time.time()
    failed = []
    for i, s in enumerate(sel, 1):
        el = (time.time() - t0) / 60
        print(f"\n=== [{i}/{len(sel)}] {s}  (est {fmt(STAGES[s][1])}, elapsed {fmt(el)}) "
              f"{'=' * 20}", flush=True)
        t = time.time()
        try:
            importlib.import_module(STAGES[s][0]).main()
        except Exception:
            traceback.print_exc()
            failed.append(s)
            print(f"!!! {s} FAILED after {fmt((time.time() - t) / 60)} -- continuing", flush=True)
        else:
            print(f"--- {s} done in {fmt((time.time() - t) / 60)}", flush=True)

    print(f"\nfinished {len(sel) - len(failed)}/{len(sel)} in {fmt((time.time() - t0) / 60)}"
          + (f"   FAILED: {', '.join(failed)}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
