"""Run dirs -> one per-experiment rollup npz, the single file every paper notebook loads.

    python -m experiments.harness.scrape clone           # -> data/paper/clone.npz
    python -m experiments.harness.scrape attention --dry-run

Reads three sources and reconciles them onto one (arm, seed, window) grid: TensorBoard scalars from
each run's `summaries/`, the per-window generator populations in `<run>/gen_pop/`, and the sidecar
npz files the GPU passes leave behind (`spec_<study>.npz` from `specialize.py`, `ladder_<study>.npz`
from `ladder.py`, `attn_<study>.npz` from `attnmap.py`). Nothing here launches or measures anything
-- it is the read half of the harness,
and the only place where a scalar's TB step becomes a window index.

Four properties the metrics depend on:

**The window index is computed from the step, never from the position in the series.** Scalar
families cover different spans of the run: `quality/*` and `clone/*` are written every window,
`build/n_modes` and `gencrit/value_ev` only during RL (so their first point is window 8, not 0),
`build/limbcount_base` only during pretrain, and `quality/by_limbcount/*` is ragged with interior
gaps because a limb count nobody sampled that window logs nothing. Reading the k-th value as window k
would shift metric 4 by eight windows and go unnoticed. Every point is placed by arithmetic on its
step and a point that does not land exactly on a boundary is an error, not a rounding.

**Two cadences, one converter.** rl_games indexes almost everything by FRAME, and the frame counter
lags the epoch by one: epoch e is written at frame (e-1)*num_actors*horizon_length. That single
conversion covers both the per-epoch families (`control/*`, `losses/*`) and the per-window ones
(`quality/*`, `build/*`, `clone/*`), which differ only in which epochs they write on. The exception
is the `*/iter` mirrors (`rewards/iter`), whose step IS the epoch -- that is the epoch-indexed axis
experiment 3's boundary fold needs, and it needs no second parser.

**Duplicate steps resolve to the last write.** A resumed run leaves two event files in one summaries
dir whose steps overlap (the crash came after the last save), and EventAccumulator does NOT
reconcile them -- verified: it returns the union with duplicates under both settings of
`purge_orphaned_data`, whose purge path needs a `SessionLog.START` record that torch's SummaryWriter
never writes. Dedup is by largest `wall_time` per step, which is "prefer the later file" without
depending on directory iteration order. It also collapses the *within*-epoch duplicates of
`losses/fd|fk` (write_stats runs twice per epoch, each flushing the minibatches accumulated since the
last flush) to the second of the pair; both are partial-batch means of the same epoch, so which one
survives is a wash, and one rule for both cases beats two.

**A run describes itself.** `num_actors`, `horizon_length` and `resample_interval` come from the
run's own stamped `config.yaml`, not from `launch.py`'s registry: the registry says what was launched
today, the stamp says what produced these numbers, and a later registry edit must not retro-corrupt
an old scrape. The registry is used only to enumerate which runs SHOULD exist, so a missing one is
reported rather than silently dropped from the mean.
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from experiments.harness.launch import STUDIES, Run, _window_epochs, study_runs   # noqa: E402

OUT_ROOT = _ROOT / "data" / "paper"
FOLD_OFFSETS = (-8, 24)          # epochs either side of a resample boundary, for the recovery trace


# ── what to pull ──────────────────────────────────────────────────────
# Data, not code. `window` keys are one point per resample window; `epoch` keys one per epoch. A tag
# absent from a run yields a NaN row rather than an error -- experiment 4 writes no `quality/*`,
# `build/*`, `gencrit/*` or `clone/*` at all, and that tolerance belongs here rather than in a
# per-study special case (ADR-0021).

WINDOW_TAGS = {                  # every study that has a generator
    "R_mean":     "quality/R_mean",
    "R_std":      "quality/R_std",
    "W_mean":     "quality/Window_Rew_Mean",
    "W_std":      "quality/Window_Rew_Std",
    "n_modes":    "build/n_modes",
    "div_struct": "build/div_struct",
    "gen_entropy": "gen/entropy",
}
# `r_step` is the per-epoch performance signal and the ONLY one the boundary fold uses. `rew_epoch`
# is rl_games' own series, kept because it is what the training logs and the tuner report -- but it
# is a ring buffer of the last 100 finished episodes, so it moves only when episodes end and its
# post-boundary shape is episode-completion phase rather than re-adaptation (see codesign_agent's
# note on control/r_step). Never fold it.
EPOCH_TAGS = {"r_step": "control/r_step", "rew_epoch": "rewards/iter"}

EXTRA_TAGS: dict[str, dict[str, dict[str, str]]] = {
    # The baseline is every ablation's comparison point, so it is scraped with the UNION of what
    # they read -- otherwise its npz lacks the very columns a panel plots the variant against.
    # `info/kl` is rl_games' own per-epoch policy KL, and it is here as a YARDSTICK rather than as a
    # measurement: `clone/actor_kl` of 0.02 nats is unreadable on its own, and against a median
    # `info/kl` of ~0.010 it says the resample displaces control more than a full epoch of learning
    # does. Experiment 3's falsifier turns on calling that drift "large" or "small", so the
    # reference travels with the arms it judges.
    "baseline": {"window": {"clone_kl": "clone/actor_kl", "clone_mse": "clone/critic_mse"},
                 "epoch":  {"fd": "losses/fd", "fk": "losses/fk", "ppo_kl": "info/kl"}},
    # Experiment 3: the two clone terms are the treatment's own readout -- `none` is the
    # counterfactual, so both are scraped for every arm, not just the ones that optimise them.
    "clone": {"window": {"clone_kl": "clone/actor_kl", "clone_mse": "clone/critic_mse"},
              "epoch":  {"ppo_kl": "info/kl"}},
    # Experiment 2: the hazard check. A flat `losses/fk` means the head never did any work, which
    # makes a null uninterpretable rather than informative. NaN for the `none` arm by construction.
    "aux": {"epoch": {"fd": "losses/fd", "fk": "losses/fk"}},
    # Experiment 4: no windows, so its whole panel is per-epoch.
    "attention": {"epoch": {"ep_len": "episode_lengths/iter", "sigma": "control/sigma_mean",
                            "adv_std": "control/adv_std"}},
}

SIDECARS = {                     # file stem -> the keys merged from it, reindexed by label
    "spec":   ("spec", "spec_sd", "fall_rate", "src_epoch"),
    "ladder": ("G", "G_sd", "T", "skel_share", "pred", "pred_sd", "dist", "k_max",
               "modal_share"),
    # Experiment 4's measurement E. Written for the `full` arm only by default, so the ablated arms'
    # cells stay NaN -- which is the right reading: there is no map to interpret where the mask
    # already fixed it. Merged by label like the rest, so running it on an ablated arm to verify the
    # mask lands in that arm's cell rather than shifting anyone else's.
    "attn":   ("attn_map", "attn_offdiag", "attn_epochs"),
}


# ── the reader ────────────────────────────────────────────────────────

def read_scalars(summaries: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{tag: (steps, values)} for one run, steps strictly increasing. See the module docstring on why
    duplicates resolve to the largest `wall_time` rather than to the first write or to a mean."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(summaries), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        ev = ea.Scalars(tag)
        steps = np.fromiter((e.step for e in ev), np.int64, len(ev))
        wall = np.fromiter((e.wall_time for e in ev), np.float64, len(ev))
        vals = np.fromiter((e.value for e in ev), np.float64, len(ev))
        order = np.lexsort((wall, steps))            # by step, then by write time within a step
        steps, vals = steps[order], vals[order]
        keep = np.empty(steps.size, bool)
        keep[:-1], keep[-1] = steps[:-1] != steps[1:], True   # last write of each step survives
        out[tag] = (steps[keep], vals[keep])
    return out


@dataclass(frozen=True)
class Axes:
    """A run's own x-axis arithmetic, read off its stamped config."""
    frames_per_epoch: int
    epochs_per_window: int       # 0 when the run never resamples (experiment 4, specialization)
    max_epochs: int
    n_pretrain: int

    @property
    def n_windows(self) -> int:
        return self.max_epochs // self.epochs_per_window if self.epochs_per_window else 0


def axes(run_dir: Path) -> Axes:
    cfg_path = run_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    ppo = cfg["params"]["config"]
    return Axes(frames_per_epoch=int(ppo["num_actors"]) * int(ppo["horizon_length"]),
                epochs_per_window=_window_epochs(cfg_path),
                max_epochs=int(ppo["max_epochs"]),
                n_pretrain=int(ppo.get("generator", {}).get("n_pretrain", 0)))


def epochs_of(tag: str, steps: np.ndarray, ax: Axes, first_epoch: int = 1) -> np.ndarray:
    """A tag's steps as 1-based epoch numbers.

    `first_epoch` un-shifts a WARM-STARTED run: a specialization pass restores `epoch_num` from its
    source checkpoint and counts on from there (ADR-0021), so its 250 epochs are numbered 2961..3211
    and only the caller knows which checkpoint it came from.
    """
    if tag.endswith("/iter"):                        # the epoch-indexed mirror: step IS the epoch
        ep = steps.copy()
    else:
        q, r = np.divmod(steps, ax.frames_per_epoch)
        bad = np.flatnonzero(r)
        if bad.size:
            raise ValueError(f"{tag}: step {steps[bad[0]]} is not a multiple of "
                             f"{ax.frames_per_epoch} frames/epoch")
        ep = q + 1
    return ep - (first_epoch - 1)


def window_series(sc: dict, tag: str, ax: Axes, n: int) -> np.ndarray:
    """One window-cadence tag as a length-`n` array indexed by window, NaN where unwritten.

    Window w's metrics are written by the resample that CLOSES it, at the end of epoch
    epochs_per_window*(w+1) -- so a tag point whose epoch is not a whole number of windows is a tag
    that is not on this cadence, and is refused rather than rounded onto a neighbour.
    """
    out = np.full(n, np.nan)
    if tag not in sc:
        return out
    steps, vals = sc[tag]
    ep = epochs_of(tag, steps, ax)
    w, r = np.divmod(ep, ax.epochs_per_window)
    bad = np.flatnonzero(r)
    if bad.size:
        raise ValueError(f"{tag}: epoch {ep[bad[0]]} is not a window boundary "
                         f"({ax.epochs_per_window} epochs/window)")
    w -= 1
    live = (w >= 0) & (w < n)
    out[w[live]] = vals[live]
    return out


def epoch_series(sc: dict, tag: str, ax: Axes, n: int, first_epoch: int = 1) -> np.ndarray:
    """One per-epoch tag as a length-`n` array indexed by epoch-1, NaN where unwritten. `rewards/iter`
    starts at epoch 2, so index 0 is legitimately NaN for it."""
    out = np.full(n, np.nan)
    if tag not in sc:
        return out
    steps, vals = sc[tag]
    ep = epochs_of(tag, steps, ax, first_epoch)
    live = (ep >= 1) & (ep <= n)
    out[ep[live] - 1] = vals[live]
    return out


def fold(rew: np.ndarray, ax: Axes, offsets: tuple[int, int] = FOLD_OFFSETS) -> np.ndarray:
    """`control/r_step` folded on the RL resample boundaries -- experiment 3's recovery trace.

    Offset 0 is the LAST epoch of the closing window, so the resample happens between offsets 0 and 1
    and the dip belongs to the new bodies. Averaged over the RL boundaries only (pretrain windows
    resample around a fixed base body and their transients are a different animal), which is
    `n_pretrain .. n_windows-1` -- 40 of the 48 at the shipped settings.
    """
    lo, hi = offsets
    n = ax.n_windows
    if not n or not np.isfinite(rew).any():
        return np.full(hi - lo + 1, np.nan)
    acc = np.full((n - ax.n_pretrain, hi - lo + 1), np.nan)
    for i, k in enumerate(range(ax.n_pretrain, n)):
        b = ax.epochs_per_window * k                 # epoch at whose end window k's bodies land
        idx = b + np.arange(lo, hi + 1) - 1          # -> 0-based index into `rew`
        live = (idx >= 0) & (idx < rew.size)
        acc[i, live] = rew[idx[live]]
    return np.nanmean(acc, axis=0) if np.isfinite(acc).any() else acc[0] * np.nan


# ── assembly ──────────────────────────────────────────────────────────

def _pop_series(run_dir: Path, n: int) -> dict[str, np.ndarray]:
    """Metric 4's travel/coverage, computed from the run's per-window population dumps.

    The math is `travel.py`'s (energy distance, its split-half null, cumulative mode coverage); this
    only locates the files and passes their window indices along. A run with no `gen_pop/` -- every
    run that predates the dump -- contributes nothing and is left NaN by the caller.
    """
    paths = {int(p.stem[1:]): p for p in sorted(run_dir.glob("gen_pop/w*.npz"))}
    if not paths:
        return {}
    from experiments.harness import travel
    return travel.window_series(paths, n)


def _tags(study: str) -> tuple[dict[str, str], dict[str, str]]:
    extra = EXTRA_TAGS.get(study, {})
    return ({**WINDOW_TAGS, **extra.get("window", {})},
            {**EPOCH_TAGS, **extra.get("epoch", {})})


def scrape(study: str, arms: list[str] | None = None,
           seeds: tuple[int, ...] | None = None) -> dict[str, np.ndarray]:
    """Every run of a study -> the rollup's arrays. Grid comes from the registry, numbers from disk."""
    spec = STUDIES[study]
    runs = study_runs(study, arms, seeds)
    arm_ix = {a: i for i, a in enumerate(arms or list(spec.arms))}
    seed_ix = {s: i for i, s in enumerate(seeds or spec.seeds)}
    shape = (len(arm_ix), len(seed_ix))

    # The axis lengths are the study's DECLARED budget, not the longest run seen: a short run must
    # read as trailing NaN on a full axis, so the notebook's shape assertion still means something
    # and a truncated arm cannot quietly rescale everyone else's x-axis.
    per_window = _window_epochs(_ROOT / spec.config, spec.common)
    n_win = spec.windows if per_window else 0
    n_ep = spec.epochs or (spec.windows * per_window)
    win_tags, ep_tags = _tags(study)
    if not n_win:
        win_tags = {}

    out = {k: np.full((*shape, n_win), np.nan) for k in win_tags}
    out |= {k: np.full((*shape, n_ep), np.nan) for k in ep_tags}
    if n_win:
        from experiments.harness import travel
        out |= {k: np.full((*shape, n_win), np.nan) for k in travel.SERIES}
    if n_win:
        out["rew_fold"] = np.full((*shape, FOLD_OFFSETS[1] - FOLD_OFFSETS[0] + 1), np.nan)
    status = np.full(shape, "missing", dtype=object)

    for run in runs:
        i = (arm_ix[run.meta["arm"]], seed_ix[run.meta["seed"]])
        if not (run.run_dir / "config.yaml").exists():
            print(f"[scrape] MISSING {run.name}", flush=True)
            continue
        ax = axes(run.run_dir)
        if ax.epochs_per_window != per_window:
            print(f"[scrape] WARN {run.name}: {ax.epochs_per_window} epochs/window, study says "
                  f"{per_window} -- its windows are not the others' windows", flush=True)
        sc = read_scalars(run.run_dir / "summaries")
        for key, tag in win_tags.items():
            out[key][i] = window_series(sc, tag, ax, n_win)
        for key, tag in ep_tags.items():
            out[key][i] = epoch_series(sc, tag, ax, n_ep)
        if n_win:
            out["rew_fold"][i] = fold(out["r_step"][i], ax)
            for key, arr in _pop_series(run.run_dir, n_win).items():
                out[key][i] = arr
        done = int(np.isfinite(out["rew_epoch"][i]).sum())
        status[i] = "ok" if done >= n_ep - 1 else f"short:{done}"
        print(f"[scrape] {run.name}: {status[i]}, {done}/{n_ep} epochs", flush=True)

    out["arms"] = np.array(list(arm_ix))
    out["seeds"] = np.array(list(seed_ix))
    out["status"] = status.astype(str)
    out["epochs_per_window"] = np.array(per_window)
    return out


def merge_sidecars(study: str, out: dict, arms: list[str], seeds: list[int]) -> dict:
    """Fold the GPU passes' artifacts in, reindexed BY LABEL onto this rollup's grid.

    The sidecars carry their own `arms`/`seeds`/`windows`, so a pass run over a subset of arms, or
    before an arm was added, lands in the right cells instead of being positionally smeared across
    the wrong ones. A sidecar that does not exist yet is simply not merged.
    """
    for stem, keys in SIDECARS.items():
        path = OUT_ROOT / f"{stem}_{study}.npz"
        if not path.exists():
            print(f"[scrape] no {path.name} yet", flush=True)
            continue
        z = np.load(path, allow_pickle=False)
        ai = [arms.index(a) for a in z["arms"].tolist()]
        si = [seeds.index(int(s)) for s in z["seeds"].tolist()]
        for key in keys:
            if key not in z:
                continue
            src = z[key]
            dst = np.full((len(arms), len(seeds), *src.shape[2:]),
                          np.nan if src.dtype.kind == "f" else 0, dtype=np.float64
                          if src.dtype.kind == "f" else src.dtype)
            dst[np.ix_(ai, si)] = src
            out[key] = dst
        if "windows" in z:
            out["ckpt_windows"] = z["windows"]
        print(f"[scrape] merged {path.name}: {', '.join(k for k in keys if k in z)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Scrape a paper experiment's runs into one npz")
    ap.add_argument("study", choices=sorted(STUDIES))
    ap.add_argument("--arms", help="comma-separated subset (default: all)")
    ap.add_argument("--seeds", help="comma-separated subset (default: the study's)")
    ap.add_argument("--out", type=Path, default=None, help="default: data/paper/<study>.npz")
    ap.add_argument("--dry-run", action="store_true", help="scrape and report; write nothing")
    args = ap.parse_args()

    arms = args.arms.split(",") if args.arms else list(STUDIES[args.study].arms)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(STUDIES[args.study].seeds))
    out = scrape(args.study, arms, tuple(seeds))
    out = merge_sidecars(args.study, out, arms, seeds)

    print(f"\n[scrape] {args.study}: " + ", ".join(
        f"{k}{tuple(v.shape)}" for k, v in out.items() if getattr(v, "ndim", 0) >= 2))
    st = out["status"]
    for lab in sorted(set(st.ravel().tolist())):
        print(f"  {lab:12} {int((st == lab).sum())}/{st.size}")
    if args.dry_run:
        return
    path = args.out or OUT_ROOT / f"{args.study}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"[scrape] -> {path}", flush=True)


if __name__ == "__main__":
    main()
