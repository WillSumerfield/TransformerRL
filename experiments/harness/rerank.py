"""Turn the random-design arm's shortlist into a committed body, by scoring it properly.

    python experiments/harness/rerank.py baselines
    python experiments/harness/rerank.py baselines --seeds 42,43 --envs-per-body 128
    python experiments/harness/rerank.py baselines --dry-run

A learned generator is *asked* for its best body. A coin has no opinion, so the random-design
baseline's answer can only be the best body it happened to see -- and the run is in no position to
say which that was. `num_morphs` defaults to `num_actors`, so every body it trains on gets one env
and one episode, and the argmax over ~200k single-episode returns is a draw from the noise's upper
tail rather than a good design (experiments/CONTEXT.md, "Selection noise"). Committing that body
would hand the baseline a number built out of its own measurement error, in the one direction that
flatters it.

So the run carries a `Shortlist` -- its top 32 bodies by that noisy score -- in its checkpoint, and
this pass re-scores those 32 at `--envs-per-body` environments each and commits the winner. The
shortlist is a candidate set, deliberately not a verdict; this is where the verdict is made.

**The rollout is the package's, not the harness's.** `codesigner.evaluate` reconstructs the library
from the checkpoint's provenance, verifies it against the task being scored on, and drives the Task
through `ControlPolicy.act()`, freezing each env at its FIRST completed episode. That last property
is the reason not to write a rollout here: bodies terminate at different rates, so averaging over
"however many episodes fit" would weight the fragile bodies by their worst behaviour -- exactly the
bias a re-evaluation exists to remove.

Only the random-design arm has anything to rerank, and that is read off the checkpoints rather than
off the arm's name: a run is reranked iff its payload carries a shortlist. `fixed_body` has none --
its committed body was its input -- so it is skipped, and a future arm that grows a shortlist is
picked up with no edit here.

`raw` and `score` are both in the sidecar and are NOT on the same scale: `raw` is what the run
measured while training -- a stochastic policy, mid-window, on however much of an episode the body
survived -- and `score` is a deterministic full-episode rollout by the finished controller. Only
their ORDER is comparable, which is all a shortlist is for. `score` is the reportable number.

Output is one sidecar per study, `data/paper/rerank_<study>.npz`, alongside the `spec_`/`ladder_`
sidecars the scrape merges. `committed_bodies()` reads it back, and is what the specialization pass
takes its body from.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from codesigner import checkpoint as _checkpoint                # noqa: E402
from codesigner.evaluate import evaluate                        # noqa: E402
from codesigner.rundir import holds_run, resolve                # noqa: E402

from experiments.harness.evalpass import EVAL_SEED              # noqa: E402
from experiments.harness.launch import STUDIES, Run, study_runs  # noqa: E402
from experiments.harness.baselines import algorithm_for, overrides_from_sets  # noqa: E402
from transformer_rl.morphology import designs_from_arrays       # noqa: E402
from transformer_rl.random_body import Shortlist                # noqa: E402

ENVS_PER_BODY = 128
"""32 bodies x 128 envs x one episode is about one training window's frames -- the budget the
shortlist was sized against, so a rerank costs a forty-eighth of the run it corrects. Fixed rather
than derived from the run's `num_actors`, so a short run with fewer shortlist entries is scored at
the same per-body precision as a full one and the two remain comparable."""

PREFER = "latest"
"""Which checkpoint's controller does the scoring. `latest`, NOT the package's default `best`,
and only for this arm: `best` is the checkpoint whose window scored highest, and under a population
redrawn at random a window scores highest partly because its DRAW was lucky. Selecting on that would
pick the controller by the same noise the rerank exists to remove. `latest` is unambiguously the
controller the run finished with."""


def sidecar(study: str) -> Path:
    return _ROOT / "data" / "paper" / f"rerank_{study}.npz"


# ---- one run ----------------------------------------------------------------------

def _shortlist_of(ckpt: dict):
    """The run's shortlist, decoded against the library the checkpoint was written with."""
    state = ckpt["payload"].get("shortlist")
    if state is None:
        return None
    sl = Shortlist()
    sl.load(ckpt["library"], state)
    return sl


def rerank_run(run: Run, *, envs_per_body: int = ENVS_PER_BODY, seed: int = EVAL_SEED,
               prefer: str = PREFER) -> dict | None:
    """Re-score one run's shortlist. `None` if the run has no shortlist to score."""
    ckpt_path = resolve(run.run_dir, prefer=prefer)
    ckpt = _checkpoint.load(ckpt_path, map_location="cpu")
    sl = _shortlist_of(ckpt)
    if sl is None or not sl.entries:
        return None

    bodies = sl.bodies()
    raw = np.array([e["score"] for e in sl.entries], dtype=np.float64)

    # Reconstituted from the overrides the launcher passed, not from defaults: the network shape a
    # checkpoint's weights load into is a function of the config that trained it.
    algorithm = algorithm_for(run.meta["arm"], _ROOT / run.config, name=run.name,
                              seed=run.meta["seed"], overrides=overrides_from_sets(run.sets))
    task = algorithm.make_task()
    # `evaluate` loads the checkpoint again itself, verifying it against THIS task's registry key --
    # which is the check worth paying a second cpu-mapped read for.
    scored = np.array(evaluate(ckpt_path, task, algorithm, bodies, envs_per_body=envs_per_body,
                               seed=seed), dtype=np.float64)

    arrays = sl.state(ckpt["library"])
    return {"run": run.name, "arm": run.meta["arm"], "seed": run.meta["seed"],
            "ckpt": os.path.relpath(ckpt_path, _ROOT), "raw": raw, "score": scored,
            **{k: arrays[k].numpy() for k in ("counts", "eff_sub", "cap_sub")},
            "winner": int(scored.argmax())}


# ---- the study --------------------------------------------------------------------

def _pad(arrays, k: int, fill):
    """Stack per-run arrays whose leading axis is the shortlist, padded out to a common `k`."""
    out = np.full((len(arrays), k, *arrays[0].shape[1:]), fill, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i, :len(a)] = a
    return out


def write(study: str, results: list[dict]) -> Path:
    """One sidecar for the study, on a (run x shortlist entry) grid."""
    k = max(len(r["raw"]) for r in results)
    path = sidecar(study)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        runs=np.array([r["run"] for r in results]),
        arms=np.array([r["arm"] for r in results]),
        seeds=np.array([r["seed"] for r in results]),
        ckpts=np.array([r["ckpt"] for r in results]),
        n_entries=np.array([len(r["raw"]) for r in results]),
        winner=np.array([r["winner"] for r in results]),
        raw=_pad([r["raw"] for r in results], k, np.nan),
        score=_pad([r["score"] for r in results], k, np.nan),
        counts=_pad([r["counts"] for r in results], k, 0),
        eff_sub=_pad([r["eff_sub"] for r in results], k, -1),
        cap_sub=_pad([r["cap_sub"] for r in results], k, -1),
    )
    return path


def committed_bodies(study: str, library) -> dict:
    """`{run_name: Morphology}` -- what each run commits to, for the specialization pass.

    Decoded against the library passed in rather than the one the arrays were written with, so a
    caller that has already built a library for its own Task gets bodies in that vocabulary. The
    two are the same library by construction; `designs_from_arrays` would raise if they were not.
    """
    d = np.load(sidecar(study))
    out = {}
    for i, name in enumerate(d["runs"]):
        w = int(d["winner"][i])
        body, = designs_from_arrays(library,
                                    torch.as_tensor(d["counts"][i, w:w + 1]).long(),
                                    torch.as_tensor(d["eff_sub"][i, w:w + 1]).long(),
                                    torch.as_tensor(d["cap_sub"][i, w:w + 1]).long())
        out[str(name)] = body
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("study", choices=sorted(STUDIES))
    ap.add_argument("--arms", default=None, help="comma-separated; default every arm in the study")
    ap.add_argument("--seeds", default=None, help="comma-separated; default the study's seeds")
    ap.add_argument("--envs-per-body", type=int, default=ENVS_PER_BODY, dest="envs_per_body")
    ap.add_argument("--eval-seed", type=int, default=EVAL_SEED, dest="eval_seed")
    ap.add_argument("--prefer", choices=("latest", "best"), default=PREFER,
                    help=f"which checkpoint's controller scores the shortlist (default {PREFER})")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = ap.parse_args()

    arms = args.arms.split(",") if args.arms else None
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else None
    # Reranking is a CodesignAlgorithm shortlist pass.  The native methods in
    # the shared baseline study deliberately have no compatible shortlist.
    runs = [r for r in study_runs(args.study, arms, seeds)
            if r.driver == "codesigner" and holds_run(r.run_dir)]
    if not runs:
        raise SystemExit(f"[rerank] {args.study}: no run directories to read")

    if args.dry_run:
        for r in runs:
            print(f"  {r.name:44s} {resolve(r.run_dir, prefer=args.prefer).name}")
        print(f"[rerank] {len(runs)} run(s); those carrying a shortlist would be scored at "
              f"{args.envs_per_body} envs/body")
        return

    results, skipped = [], []
    for run in runs:
        out = rerank_run(run, envs_per_body=args.envs_per_body, seed=args.eval_seed,
                         prefer=args.prefer)
        if out is None:
            skipped.append(run.name)
            continue
        results.append(out)
        # The number that says whether this pass was needed: where the raw leader lands once every
        # candidate has been measured at the same precision. A rerank that always returned rank 1
        # would mean the single-episode score was already trustworthy.
        rank = int((out["score"] > out["score"][0]).sum()) + 1
        print(f"  {run.name:44s} raw top {out['raw'][0]:8.1f} -> reranks {rank}/"
              f"{len(out['raw'])} | committed {out['score'][out['winner']]:8.1f} "
              f"(raw {out['raw'][out['winner']]:8.1f})", flush=True)

    if skipped:
        print(f"[rerank] no shortlist, nothing to commit: {', '.join(skipped)}")
    if not results:
        raise SystemExit("[rerank] no run carried a shortlist")
    print(f"[rerank] wrote {write(args.study, results)}")


if __name__ == "__main__":
    main()
