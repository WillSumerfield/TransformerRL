"""Metrics 2 and 3: the spread ladder, and the two curves measured on it.

    python experiments/harness/ladder.py backbone            # -> data/paper/ladder_backbone.npz
    python experiments/harness/ladder.py clone --seeds 42 --episodes 8
    python experiments/harness/ladder.py aux --dry-run

One pass, no training: for every (run x ladder checkpoint) it builds a one-parameter family of body
distributions around that checkpoint's committed body, rolls the SAME control policy out on each
level, and reads GenCrit's opinion of the same bodies. `scrape.py` merges the sidecar this writes.

**The ladder is `net.sample`'s `beta` knob, re-indexed.** `beta` is an inverse temperature on the
generator's raw logits, and the three body sources `eval.py` already has are three of its values --
`beta=inf` is `greedy` (the committed body), `beta=1` is `stochastic` (the trained distribution),
`beta=0` is `uniform` (a random policy on the same grammar). So the family is not a new sampler; it
is the existing three-point comparison filled in continuously, which is the only reason it can be
bisected on at all.

**Levels are indexed by perturbation distance, never by beta.** Level `k` is the distribution whose
MEAN `d_struct` from the committed body is `k` modules, found by bisecting `beta`. Two runs' betas
mean nothing to each other -- one generator's logits are not another's -- while `k = 3` is three
modules of body change in both. `k_max` is set by the grammar's uniform draw, so the ladders are
comparable end to end. A level is a *distribution* whose mean distance is `k`, not a set of bodies
exactly `k` out.

**Level 0 is the anchor that makes metric 3 mean anything.** Every one of its `envs` bodies is the
identical committed body, so its across-body spread is pure episode noise -- the metric's own floor,
measured in-band per seed -- and `bias(0)` is GenCrit's in-distribution offset. Subtracting it
removes both of metric 3's confounds at once (GenCrit regressed returns under the *sampled* policy
by *earlier* control; the ladder rolls the final policy at mu), neither of which is constant across
conditions. The notebook plots `pred - G` anchored at level 0, not raw.

**Each level reports its skeleton share.** Per the free_entropy finding the cheap subtype axis moves
first, so a level can accumulate distance while every body plan stays identical. `skel_share` is the
same mean distance on the subtype-collapsed skeleton over the typed one: a flat
control-generalization curve at near-zero skeleton share means the ladder never tested a new body
plan, and the null is about subtypes rather than about generalization.

Bisection is sampling-only -- no simulator, no rollouts -- so the whole schedule costs a few seconds
per checkpoint and the budget is spent entirely on the rollouts. It is also run under a FIXED seed
per probe, which makes the sampled mean distance a deterministic function of `beta` and the
bisection a bisection rather than a random walk.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from experiments.harness.committance import encode_population, population_to_repr   # noqa: E402
from experiments.harness.evalpass import (                                          # noqa: E402
    EVAL_SEED, install, load_net, modal_design, open_task, rollout,
)
from experiments.harness.launch import STUDIES                                      # noqa: E402
from experiments.harness.specialize import _targets, ckpt_windows                   # noqa: E402

ENVS = 256              # bodies per level; one env each, so across-env spread IS across-body spread
EPISODES = 32           # K, episodes per body. Read level 0's band to tell whether it is enough
POP = 512               # greedy draws the committed body is the mode of (== specialize.py's)
DIST_N = 512            # bodies per bisection probe -- sampling only, no sim
BISECT_ITERS = 16
BISECT_TOL = 0.02       # modules; well inside the integer spacing the axis is read at
LOG_BETA = (-3.0, 3.0)  # bisection bracket. 1e-3 is uniform to within sampling noise, 1e3 argmax
LEVEL_CAP = 32          # allocation only; the axis is trimmed to the deepest level any run reached


def sidecar(study: str) -> Path:
    return _ROOT / "data" / "paper" / f"ladder_{study}.npz"


# ---- the distance the axis is in ---------------------------------------------------

def _arrays(pop) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (pop["counts"].cpu().numpy().astype(int), pop["eff_sub"].cpu().numpy().astype(int),
            pop["cap_sub"].cpu().numpy().astype(int))


def mean_distance(pop_arrays, ref_arrays, collapse: bool = False) -> float:
    """Mean `d_struct` from a population to ONE reference body, in modules.

    Computed as Hamming over `encode_population`'s packing, which IS `d_struct` at `W_OVERHANG == 1`
    (the identity is proved in that function's docstring and self-tested in `travel.py`). The
    reference is encoded in the SAME call as the population, because the packing builds its
    vocabulary from what it is handed and two separate calls would number the tokens differently --
    every distance would then be wrong in a way that still looks like a number.
    """
    reps = population_to_repr(*pop_arrays, collapse_subtypes=collapse)
    reps += population_to_repr(*(a[None] for a in ref_arrays), collapse_subtypes=collapse)
    codes = encode_population(reps)
    return float((codes[:-1] != codes[-1:]).sum(1).mean())


def _draw(net, n: int, beta: float, seed: int = EVAL_SEED):
    """A population at one spread. Seeded per draw so the ladder's schedule is reproducible and the
    bisected function is deterministic -- an unseeded probe makes the bisection chase sampling noise
    below its own tolerance and never converge."""
    torch.manual_seed(seed)
    return net.net.sample(n, beta=beta)


def beta_for(net, ref, target: float, n: int = DIST_N, iters: int = BISECT_ITERS,
             tol: float = BISECT_TOL) -> tuple[float, float]:
    """(beta, achieved mean distance) for a level at `target` modules.

    Mean distance falls monotonically in `beta`: `beta=0` is the grammar's uniform draw (furthest)
    and `beta=inf` is the committed body itself (zero). Bisection is in log-beta because the knob is
    a multiplier on logits, so the interesting range spans orders of magnitude.
    """
    lo, hi = LOG_BETA                                  # d(lo) is large, d(hi) ~ 0
    d = np.nan
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        d = mean_distance(_arrays(_draw(net, n, 10.0 ** mid)), ref)
        if abs(d - target) <= tol:
            break
        if d > target:
            lo = mid                                   # too far out -> sharpen
        else:
            hi = mid
    return 10.0 ** (0.5 * (lo + hi)), d


def schedule(net, ref) -> list[tuple[int, float, float]]:
    """[(k, beta, achieved distance)] for k = 0 .. floor(k_max), plus the raw uniform distance.

    Level 0 is `beta=inf` exactly rather than a bisected approximation of it: the greedy draw walks
    the limb visit order from a fixed seed, so every row is the committed body and the anchor is an
    identity rather than a limit.
    """
    k_max = mean_distance(_arrays(_draw(net, DIST_N, 0.0)), ref)
    levels = [(0, float("inf"), 0.0)]
    for k in range(1, int(np.floor(k_max)) + 1):
        beta, d = beta_for(net, ref, float(k))
        levels.append((k, beta, d))
    return levels, k_max


# ---- one (run x checkpoint) --------------------------------------------------------

def ladder_one(net, obs_norm, env, device, *, r_scale: float, episodes: int) -> dict:
    """Every level of one checkpoint's ladder: bisect the schedule, then roll out each level."""
    greedy = _draw(net, POP, float("inf"))
    idx, stats = modal_design(greedy)
    g = _arrays(greedy)
    ref = (g[0][idx], g[1][idx], g[2][idx])
    levels, k_max = schedule(net, ref)

    n = env.total_num_envs
    rows = []
    for k, beta, d_sched in levels:
        pop = _draw(net, n, beta)
        arrays = _arrays(pop)
        install(env, pop)
        r = rollout(net, obs_norm, env, device, episodes=episodes, label=f"k={k}")
        typed = mean_distance(arrays, ref)
        skel = mean_distance(arrays, ref, collapse=True)
        # GenCrit's value at the COMPLETED body -- the last prefix state, the same number
        # `eval.py`'s calibration uses -- divided back out of control's shaped units.
        pred = pop["v_states"][:, -1].cpu().numpy() / r_scale
        rows.append({"k": k, "beta": beta, "dist": typed,
                     "skel_share": skel / typed if typed > 0 else np.nan,
                     "G": float(r["return"].mean()), "G_sd": float(r["return"].std()),
                     "pred": float(pred.mean()), "pred_sd": float(pred.std()),
                     "fall_rate": float(r["fall_rate"].mean())})
        print(f"    k={k:<3} beta={beta:<9.3g} d={typed:5.2f} skel={rows[-1]['skel_share']:.2f} "
              f"G={rows[-1]['G']:8.1f} +-{rows[-1]['G_sd']:6.1f} pred={rows[-1]['pred']:8.1f}",
              flush=True)
    return {"levels": rows, "k_max": k_max, "modal_share": stats["modal_share"]}


# ---- the study ---------------------------------------------------------------------

KEYS = ("T", "beta", "dist", "skel_share", "G", "G_sd", "pred", "pred_sd", "fall_rate")


def run_study(study: str, *, arms=None, seeds=None, windows=None, envs: int = ENVS,
              episodes: int = EPISODES, device: str = "cuda:0") -> Path:
    spec = STUDIES[study]
    windows = ckpt_windows(study) if windows is None else windows
    arms = list(arms or spec.arms)
    seeds = list(seeds or spec.seeds)
    shape = (len(arms), len(seeds), len(windows), LEVEL_CAP)
    out = {k: np.full(shape, np.nan) for k in KEYS}
    out["k_max"] = np.full(shape[:3], np.nan)
    out["modal_share"] = np.full(shape[:3], np.nan)

    dev = torch.device(device)
    env = layout = None
    deepest = 0
    for run, w, epoch, ckpt in _targets(study, windows):
        if run.meta["arm"] not in arms or run.meta["seed"] not in seeds:
            continue
        if ckpt is None:
            print(f"[ladder] no boundary checkpoint for {run.name} @ w{w}", flush=True)
            continue
        cfg = yaml.safe_load((run.run_dir / "config.yaml").read_text())
        if env is None:
            env, _, layout = open_task(cfg, envs, device=dev)
        shaper = cfg["params"]["config"].get("reward_shaper", {})
        r_scale = float(shaper.get("scale_value", 1.0)) if isinstance(shaper, dict) else 1.0
        net, obs_norm = load_net(ckpt, cfg, layout, dev)
        print(f"[ladder] {run.name} @ w{w} (epoch {epoch})", flush=True)
        res = ladder_one(net, obs_norm, env, dev, r_scale=r_scale, episodes=episodes)

        i = (arms.index(run.meta["arm"]), seeds.index(run.meta["seed"]), list(windows).index(w))
        out["k_max"][i] = res["k_max"]
        out["modal_share"][i] = res["modal_share"]
        for row in res["levels"]:
            k = row["k"]
            if k >= LEVEL_CAP:
                break
            deepest = max(deepest, k)
            out["T"][(*i, k)] = 1.0 / row["beta"]      # the reported knob is the TEMPERATURE
            for key in KEYS[1:]:
                out[key][(*i, k)] = row[key]
        del net, obs_norm
        torch.cuda.empty_cache()

    for key in KEYS:                                   # trim the allocation to what was reached
        out[key] = out[key][..., :deepest + 1]
    path = sidecar(study)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out, arms=np.array(arms), seeds=np.array(seeds),
             windows=np.array(list(windows)), episodes=episodes, envs=envs)
    print(f"[ladder] {int(np.isfinite(out['G']).sum())} level(s) over "
          f"{int(np.isfinite(out['k_max']).sum())} checkpoint(s) -> {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description="Metrics 2 and 3: the spread ladder")
    ap.add_argument("study", choices=sorted(STUDIES))
    ap.add_argument("--arms", default=None, help="comma-separated; default every arm")
    ap.add_argument("--seeds", default=None, help="comma-separated; default the study's")
    ap.add_argument("--windows", default=None, help="ladder points; default from n_pretrain")
    ap.add_argument("--envs", type=int, default=ENVS, help="bodies per level")
    ap.add_argument("--episodes", type=int, default=EPISODES, help="K, episodes per body")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = ap.parse_args()

    windows = (tuple(int(w) for w in args.windows.split(","))
               if args.windows else ckpt_windows(args.study))
    arms = args.arms.split(",") if args.arms else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    if args.dry_run:
        for run, w, epoch, ckpt in _targets(args.study, windows):
            if arms and run.meta["arm"] not in arms:
                continue
            if seeds and run.meta["seed"] not in seeds:
                continue
            print(f"  w{w:<3} epoch {epoch:<5} {run.name:30} {ckpt.name if ckpt else 'MISSING'}")
        print(f"[ladder] {args.envs} bodies x {args.episodes} episodes per level")
        return
    run_study(args.study, arms=arms, seeds=seeds, windows=windows,
              envs=args.envs, episodes=args.episodes)


if __name__ == "__main__":
    main()
