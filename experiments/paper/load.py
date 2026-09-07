"""One study's rollup, ready to plot: the arm grid joined, the units reconciled, the axes named.

    from experiments.paper import load
    r = load.study("aux")                 # baseline joined in as the gold arm
    r.R_mean.shape                        # (2, 8, 48) == (arm, seed, window)
    r.arms                                # ['aux', 'none']   -- doc names, not file names

`scrape.py` is the read half of the HARNESS and stops at "the numbers, as logged". This is the read
half of the NOTEBOOK, and it owns the four reconciliations that stand between a rollup and a figure.
Each one is a place a panel goes silently wrong, so each lives here once rather than in every cell.

**The comparison cell is in a different file.** Every ablation is a `--set` delta off the shared
`baseline` study, so `aux.npz` holds one arm (`none`) and its control lives in `baseline.npz` as
`tuned`. The join happens here, and it RENAMES: experiment 3's unmodified cell is `both` in
clone.md's 2x2 and experiment 2's is `aux` in aux.md's table, so the figures carry the names the
docs argue in rather than the name the registry stores. `attention` and `backbone` keep their own
in-study control arm and are never joined.

**Two window conventions, differing by one, and both are correct.** `_gen_window` counts generator
updates and is incremented BY the resample (`codesign_agent.py:908`), so:

  * metric-1's window index and the `gen_pop/w*.npz` dumps agree -- both name the window that just
    ENDED, which is `_gen_window - 1` (`codesign_agent.py:926`), and `scrape.window_series` places a
    tag logged at epoch `e` at index `e/epochs_per_window - 1`.
  * `specialize.ckpt_windows` and the `spec`/`ladder` sidecars name a CHECKPOINT by its
    `gen_window`, and `specialize._targets` reads it at `epoch = w * epochs_per_window`.

So the checkpoint labelled w=7 and the metric at window index 6 are the SAME INSTANT. `ckpt_gen` and
`ckpt_metric` below carry both, and anything aligning a sidecar against the window axis must use
`ckpt_metric` -- ADR-0021 calls this "off-by-one-prone in both directions" and it is.

**Two return units.** `quality/R_mean` is logged SHAPED: `codesign_agent.py:899` multiplies by
`reward_shaper.scale_value` (0.01 in the tuned config) before `_quality_log` sees it. Everything the
eval passes produce -- `spec` from `evalpass.rollout`, the ladder's `G`, and GenCrit's `pred` once
divided by the same scale -- is RAW. This module divides the shaped families up to raw so the
notebook has one unit, and it takes the scale from the study's own effective config rather than a
literal 100, because tuning can move it. `control/r_step` is documented raw at source and is NOT
rescaled; `rewards/iter` is a different population (rl_games' 100-episode ring buffer) and is left
alone as a continuity series that never shares an axis with the others.

**An absent producer is absent, not zero.** A pass that has not been run yet leaves its arrays
all-NaN. `synthetic()` fills them from a fixture and STAMPS the result, so `Rollup.synthetic` names
every key a figure must mark on its face. Its shapes follow `ladder.KEYS` and `attnmap`'s savez
exactly, so swapping a real pass in is a deleted call and nothing else.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from experiments.harness import stats                                        # noqa: E402
from experiments.harness.launch import STUDIES, _effective, _window_epochs   # noqa: E402
from experiments.harness.specialize import ckpt_windows                      # noqa: E402

DATA = _ROOT / "data" / "paper"

# The unmodified cell's name in each study's own table, and the studies that need the join at all.
# `attention` and `backbone` are absent deliberately: their control arm is in-study.
BASELINE_CELL = {"aux": "aux", "clone": "both"}
BASELINE_ARM = "tuned"                    # what it is called inside baseline.npz

# Shaped -> raw. `r_step` is raw at source (codesign_agent's own note) and `rew_epoch` is a
# different population entirely, so neither appears here. `spec`/`G`/`pred` arrive raw by contract.
SHAPED = ("R_mean", "R_std", "W_mean", "W_std")

# Everything a figure may treat as a return, after `_to_raw`. Used only to assert one unit.
RAW_RETURN = SHAPED + ("spec", "spec_sd", "G", "G_sd", "pred")


@dataclass
class Rollup:
    """One study's arrays on one (arm, seed, ...) grid. Attribute access reaches the arrays."""
    study: str
    arrays: dict[str, np.ndarray]
    arms: list[str]
    seeds: list[int]
    epochs_per_window: int
    n_windows: int
    n_pretrain: int
    r_scale: float
    ckpt_gen: tuple[int, ...]             # checkpoint labels, in gen_window
    ckpt_metric: tuple[int, ...]          # the same instants, on the metric-1 window axis
    n_slots: int = 0                      # limb slots; the token layout follows from it
    max_depth: int = 0
    is_fixture: bool = False              # built by `load.fixture`, not by `harness.scrape`
    synthetic: set[str] = field(default_factory=set)

    def __getattr__(self, k):
        try:
            return self.arrays[k]
        except KeyError as e:
            raise AttributeError(f"{self.study!r} has no array {k!r}; "
                                 f"have {sorted(self.arrays)}") from e

    def has(self, k: str) -> bool:
        """True when the key exists AND carries at least one finite number."""
        a = self.arrays.get(k)
        return a is not None and np.isfinite(a).any()

    def arm(self, name: str) -> int:
        return self.arms.index(name)

    @property
    def windows(self) -> np.ndarray:
        return np.arange(self.n_windows)

    @property
    def n_tokens(self) -> int:
        """`1 + n + n*max_len` (`architectures.py:265`) -- CLS, one start anchor per slot, then the
        module block. Derived from the run's library, never written down: attention.md stated a
        4-slot layout for an 8-slot library and every number in it was wrong."""
        return 1 + self.n_slots + self.n_slots * self.max_depth

    @property
    def content_start(self) -> int:
        return 1 + self.n_slots

    @property
    def rl_start(self) -> int:
        """First RL window on the metric-1 axis. `_in_pretrain` is `_gen_window < n_pretrain`, so
        windows 0..n_pretrain-1 closed under pretrain and the boundary rule goes here."""
        return self.n_pretrain


# ── loading ───────────────────────────────────────────────────────────

def _effective_cfg(study: str) -> dict:
    spec = STUDIES[study]
    _, _, cfg = _effective(_ROOT / spec.config, spec.common)
    return cfg


def _r_scale(study: str) -> float:
    shaper = _effective_cfg(study)["params"]["config"].get("reward_shaper", {}) or {}
    return float(shaper.get("scale_value", 1.0))


def _n_heads(study: str) -> int:
    net = _effective_cfg(study).get("params", {}).get("network", {}).get("transformer", {})
    return int(net.get("n_heads", 4))


def _library(study: str):
    """The study's ModuleLibrary -- the authority on the token layout, per the CoDesigner migration."""
    from codesigner.components.modular_libraries import REGISTRY
    cfg = _effective_cfg(study)
    return REGISTRY[(cfg.get("env") or {}).get("module_library", "simple")]()


def _npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _join(base: dict, sub: dict, base_label: str) -> tuple[dict, list[str]]:
    """Baseline's single arm in front of the study's arms, NaN-padding keys only one side carries.

    Padding rather than dropping: `EXTRA_TAGS` gives the baseline the UNION of what the ablations
    read, so it legitimately has columns `aux.npz` does not, and an ablation's panel plotting the
    baseline's `clone_kl` against nothing is a correct empty panel rather than a KeyError.
    """
    arms = [base_label] + [str(a) for a in sub["arms"]]
    out: dict[str, np.ndarray] = {}
    for key in sorted(set(base) | set(sub)):
        if key in ("arms", "seeds", "status", "epochs_per_window", "ckpt_windows", "is_fixture"):
            continue
        a, b = base.get(key), sub.get(key)
        ref = a if a is not None else b
        if getattr(ref, "ndim", 0) < 2:
            continue
        if a is None:
            a = np.full((1, *b.shape[1:]), np.nan)
        if b is None:
            b = np.full((len(sub["arms"]), *a.shape[1:]), np.nan)
        if a.shape[1:] != b.shape[1:]:
            # The LADDER's level axis is ragged across studies and legitimately so: `ladder.py`
            # trims to `deepest + 1`, and two studies whose generators sit at different distances
            # from the grammar's uniform draw reach different `k_max` (21.4 vs 20.7 here, so 26
            # levels vs 25). Pad the short one with NaN rather than refuse the join -- every reader
            # is already NaN-aware, and the alternative is truncating one arm's ladder to another's.
            # Any disagreement on an EARLIER axis is still an error: those are the study's declared
            # budget and must match.
            if a.shape[1:-1] != b.shape[1:-1]:
                raise ValueError(f"{key}: baseline {a.shape} vs study {b.shape} -- axes disagree")
            n = max(a.shape[-1], b.shape[-1])
            pad = lambda z: np.concatenate(
                [z.astype(float),
                 np.full((*z.shape[:-1], n - z.shape[-1]), np.nan)], axis=-1) \
                if z.shape[-1] < n else z.astype(float)
            a, b = pad(a), pad(b)
        out[key] = np.concatenate([a.astype(float), b.astype(float)], axis=0)
    return out, arms


def study(name: str, *, with_baseline: bool = True) -> Rollup:
    """`data/paper/<name>.npz` (+ baseline) -> one plot-ready grid, in raw return units."""
    path = DATA / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run `python experiments/harness/scrape.py {name}` "
                                f"or build a fixture with load.fixture()")
    sub = _npz(path)
    seeds = [int(s) for s in sub["seeds"]]

    if with_baseline and name in BASELINE_CELL:
        bpath = DATA / "baseline.npz"
        if not bpath.exists():
            raise FileNotFoundError(f"{name} is a delta off the baseline study, but {bpath} "
                                    f"is missing -- scrape `baseline` first")
        base = _npz(bpath)
        if [int(s) for s in base["seeds"]] != seeds:
            raise ValueError(f"{name} and baseline disagree on seeds; the join is by position")
        arrays, arms = _join(base, sub, BASELINE_CELL[name])
    else:
        arrays = {k: v.astype(float) for k, v in sub.items()
                  if getattr(v, "ndim", 0) >= 2
                  and k not in ("arms", "seeds", "status", "epochs_per_window", "ckpt_windows",
                                "is_fixture")}
        arms = [str(a) for a in sub["arms"]]

    scale = _r_scale(name)
    for k in SHAPED:                       # the one place shaped units stop existing
        if k in arrays:
            arrays[k] = arrays[k] / scale

    spec = STUDIES[name]
    per_window = int(sub["epochs_per_window"])
    n_win = spec.windows if per_window else 0
    cfg = _effective_cfg(name)["params"]["config"]
    n_pre = int(cfg.get("generator", {}).get("n_pretrain", 0)) if per_window else 0
    gen = ckpt_windows(name) if per_window else ()
    lib = _library(name)
    return Rollup(study=name, arrays=arrays, arms=arms, seeds=seeds,
                  is_fixture=bool(sub.get("is_fixture", np.array(0))),
                  epochs_per_window=per_window, n_windows=n_win, n_pretrain=n_pre,
                  r_scale=scale, ckpt_gen=gen, ckpt_metric=tuple(w - 1 for w in gen),
                  n_slots=getattr(lib, "n_slots", 0), max_depth=getattr(lib, "max_depth", 0))


# ── statistics ────────────────────────────────────────────────────────

# `mean_ci`, `seed_floor` and the rest live in `experiments.harness.stats` -- the harness owns every
# across-seed reduction, and a second copy here would be the first violation of its
# one-implementation-per-metric rule. Re-exported so a notebook cell has one import.
mean_ci = stats.mean_ci
paired_ci = stats.paired_ci
nested_bands = stats.nested_bands
noise_floor = stats.noise_floor
asymptote = stats.asymptote
crossing = stats.crossing


def retention_width(G: np.ndarray, k: np.ndarray, f: float = 0.8) -> np.ndarray:
    """Metric 3's valid width: the perturbation distance at which return first falls below `f*G(0)`.

    Undefined in every doc that pre-registers a falsifier against it (backbone.md, aux.md), so it is
    defined here and `f` is stated rather than tuned -- analysis.py reports 0.7/0.8/0.9 side by side
    so a conclusion that lives or dies on the threshold is visible instead of implicit.

    FIRST crossing, not last: `G` need not be monotone, and "the largest k that still clears the
    bar" would let a single noisy level far out extend the width past a collapse. Linearly
    interpolated between the bracketing levels, so the width is continuous in the data and two arms
    do not tie merely because the ladder is integer-spaced. NaN when level 0 is missing; `k[-1]`
    (censored) when the curve never crosses, which analysis.py annotates rather than plots as fact.
    """
    G = np.asarray(G, float)
    k = np.asarray(k, float)
    out = np.full(G.shape[:-1], np.nan)
    for idx in np.ndindex(*G.shape[:-1]):
        g = G[idx]
        ok = np.isfinite(g)
        if not ok.any() or not np.isfinite(g[0]):
            continue
        thresh = f * g[0]
        below = np.flatnonzero(ok & (g < thresh))
        if below.size == 0:
            out[idx] = k[ok][-1]                       # censored at the ladder's end
            continue
        j = below[0]
        if j == 0:
            out[idx] = k[0]
            continue
        g0, g1, k0, k1 = g[j - 1], g[j], k[j - 1], k[j]
        out[idx] = k1 if g0 == g1 else k0 + (g0 - thresh) * (k1 - k0) / (g0 - g1)
    return out


# ── fixtures ──────────────────────────────────────────────────────────
# Two kinds, and the difference matters. A REAL fixture re-labels finished past runs as this study's
# arms: the arms are fiction but every number in them was produced by a training run, so it exercises
# the readers, the axis arithmetic and the join against data that can surprise us. A SYNTHETIC
# fixture fabricates the arrays whose producer does not exist yet (`ladder.py`, the attention-map
# eval path, `control/r_step`); it exercises only the drawing, and every key it writes is stamped so
# a panel drawn from it says so on its face.

# Past runs standing in for arms. Three seeds throughout, because the baseline join is positional and
# the shortest usable family has three. These are NOT the conditions they are named after.
FIXTURE_RUNS: dict[str, dict[str, list[str]]] = {
    "baseline":  {"tuned":    ["phase3_s42", "phase3_s43", "phase3_s44"]},
    "aux":       {"none":     ["phase2_s42", "phase2_s43", "phase2_s44"]},
    "clone":     {"kl_only":  ["SingleTest_s42", "SingleTest_s43", "SingleTest_s44"],
                  "mse_only": ["SingleNewTest_s42", "SingleNewTest_s43", "SingleNewTest_s44"],
                  "none":     ["phase5_s42", "phase5_s43", "phase5_s44"]},
    "attention": {"full":     ["phase3_s45", "phase3_s46", "17-16-06-29"],
                  "self_cls": ["phase2_s45", "phase2_s46", "TimingTest"],
                  "self":     ["SingleTest_s42", "SingleNewTest_s42", "phase5_s42"]},
}
FIXTURE_ROOT = _ROOT / "runs" / "ant_codesign" / "codesign_single_transformer"


def _legacy_axes(run_dir: Path):
    """`scrape.axes` for a run stamped before `env.task` existed.

    `_window_epochs` resolves the Task class to read its `max_episode_length` default, and a legacy
    stamp has no `env.task` to resolve -- so the harness reader raises where `evalpass.open_task`
    would have fallen back ("runs stamped before `env.task` existed are ant runs by construction").
    Fixture-only: the real path never sees a legacy stamp, and widening `_resolve_task` to guess a
    task for every caller is the wrong trade for a test convenience.
    """
    import math

    import yaml

    from experiments.harness.scrape import Axes
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ppo = cfg["params"]["config"]
    interval = ppo.get("resample_interval", 0)
    ep_len = (cfg.get("env") or {}).get("max_episode_length", 1000)      # Ant's own default
    per = math.ceil(interval * ep_len / int(ppo["horizon_length"])) if interval else 0
    return Axes(frames_per_epoch=int(ppo["num_actors"]) * int(ppo["horizon_length"]),
                epochs_per_window=per, max_epochs=int(ppo["max_epochs"]),
                n_pretrain=int(ppo.get("generator", {}).get("n_pretrain", 0)))


def fixture(name: str, *, root: Path = FIXTURE_ROOT, out: Path | None = None) -> Path:
    """Past runs -> `data/paper/<name>.npz`, on the same grid and by the same readers as `scrape`.

    Reuses `scrape`'s functions rather than reimplementing them, so a fixture that loads proves the
    real path loads. Past runs are `horizon_length: 16` and therefore 63 epochs/window against the
    tuned config's 32 -- deliberately kept, because a fixture whose axis arithmetic differs from the
    study's is a far better test of the step->window conversion than one that matches.
    """
    from experiments.harness import scrape as S

    spec = STUDIES[name]
    per_window = _window_epochs(_ROOT / spec.config, spec.common)
    n_win = spec.windows if per_window else 0
    n_ep = spec.epochs or (spec.windows * per_window)
    win_tags, ep_tags = S._tags(name)
    if not n_win:
        win_tags = {}

    plan = FIXTURE_RUNS[name]
    arms, seeds = list(plan), list(range(len(next(iter(plan.values())))))
    shape = (len(arms), len(seeds))
    o = {k: np.full((*shape, n_win), np.nan) for k in win_tags}
    o |= {k: np.full((*shape, n_ep), np.nan) for k in ep_tags}
    if n_win:
        from experiments.harness import travel
        o |= {k: np.full((*shape, n_win), np.nan) for k in travel.SERIES}
        o["rew_fold"] = np.full((*shape, S.FOLD_OFFSETS[1] - S.FOLD_OFFSETS[0] + 1), np.nan)

    for ai, arm in enumerate(arms):
        for si, run_name in enumerate(plan[arm]):
            d = root / run_name
            if not (d / "config.yaml").exists():
                print(f"[fixture] MISSING {run_name}", flush=True)
                continue
            ax = _legacy_axes(d)
            sc = S.read_scalars(d / "summaries")
            for key, tag in win_tags.items():
                o[key][ai, si] = S.window_series(sc, tag, ax, n_win)
            for key, tag in ep_tags.items():
                o[key][ai, si] = S.epoch_series(sc, tag, ax, n_ep)
            if n_win:
                o["rew_fold"][ai, si] = S.fold(o["r_step"][ai, si], ax)
            print(f"[fixture] {name}/{arm}[{si}] <- {run_name} "
                  f"({ax.epochs_per_window} ep/window, {ax.max_epochs} epochs)", flush=True)

    o["arms"] = np.array(arms)
    o["seeds"] = np.array(seeds)
    o["epochs_per_window"] = np.array(per_window)
    # Fixtures and real scrapes write to the SAME path, so the file has to say which it is. Without
    # this a real `baseline` and a fixture `attention` sit side by side and nothing distinguishes
    # them at the point of reading.
    o["is_fixture"] = np.array(1)
    path = out or DATA / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **o)
    print(f"[fixture] -> {path}", flush=True)
    return path


def synthetic(r: Rollup, *, seed: int = 0) -> Rollup:
    """Fill in the arrays whose producer does not exist, and stamp every one of them.

    Shapes and semantics follow the documented sidecar contracts exactly (`scrape.SIDECARS`,
    Metrics.md), so swapping a real pass in later is a delete of this call and nothing else. The
    numbers are plausible, not predictive: arms are separated by a modest, arm-index-ordered effect
    so the panels have something to show, and nothing here should ever be read as a result.
    """
    rng = np.random.default_rng(seed)
    A, S_, W = len(r.arms), len(r.seeds), r.n_windows
    add: dict[str, np.ndarray] = {}
    eff = np.linspace(1.0, 0.75, A)[:, None]              # arm 0 (gold) best, monotone after

    if W and not r.has("spec"):                           # metric 5, (arm, seed, ckpt)
        C = len(r.ckpt_gen)
        base = np.nanmax(r.arrays["R_mean"]) if r.has("R_mean") else 5000.0
        grow = np.linspace(0.75, 1.0, C)[None, None, :]
        add["spec"] = base * 1.25 * eff[..., None] * grow * (1 + 0.09 * rng.standard_normal((A, S_, C)))
        add["spec_sd"] = base * 0.06 * (1 + 0.2 * rng.random((A, S_, C)))
        add["fall_rate"] = np.clip(0.25 * (2 - eff[..., None]) + 0.05 * rng.standard_normal((A, S_, C)), 0, 1)

    if W and not r.has("G"):                              # metrics 2 and 3 -- `ladder.KEYS` shapes
        C, K = len(r.ckpt_gen), 9
        k = np.arange(K, dtype=float)
        base = np.nanmax(r.arrays["R_mean"]) if r.has("R_mean") else 5000.0
        width = (3.0 + 3.5 * eff)[..., None, None]        # gold's control stays valid furthest out
        G0 = base * eff[..., None, None] * np.linspace(0.8, 1.0, C)[None, None, :, None]
        G = G0 * np.exp(-(k[None, None, None, :] / width) ** 1.6)
        add["G"] = G * (1 + 0.05 * rng.standard_normal((A, S_, C, K)))
        add["G_sd"] = 0.10 * G * (1 + k[None, None, None, :] / K)
        # `dist` is the MEASURED mean d_struct of each level, not the integer it was bisected onto --
        # the bisection lands close but not exactly, and it is `dist` the x-axis is drawn on.
        add["dist"] = np.broadcast_to(k, (A, S_, C, K)) * (1 + 0.02 * rng.standard_normal((A, S_, C, K)))
        add["T"] = np.broadcast_to(np.linspace(0.02, 6.0, K), (A, S_, C, K)).copy()
        add["skel_share"] = np.broadcast_to(np.clip(k / (K - 1), 0.05, 1.0), (A, S_, C, K)).copy()
        # GenCrit stays optimistic as the bodies get stranger -- the over-optimism of Metrics.md.
        add["pred"] = G0 * (1 - 0.10 * k[None, None, None, :] / K) * (
            1 + 0.04 * rng.standard_normal((A, S_, C, K)))
        add["pred_sd"] = 0.08 * np.abs(add["pred"])
        add["k_max"] = np.full((A, S_, C), float(K - 1))
        add["modal_share"] = np.clip(0.55 + 0.3 * eff[..., None] + 0.05 * rng.standard_normal((A, S_, C)), 0, 1)

    if W and not r.has("n_modes"):                        # metric 4's breadth (build/* predates most runs)
        w = np.arange(W, dtype=float)
        settle = np.exp(-(w - r.rl_start).clip(0) / 26)[None, None, :]
        add["n_modes"] = (1 + (7 * eff[..., None] - 1) * settle) * (1 + 0.15 * rng.standard_normal((A, S_, W)))
        add["div_struct"] = 3.2 * eff[..., None] * settle + 0.4
        for key in ("n_modes", "div_struct"):
            add[key][..., :r.rl_start] = np.nan

    if W and not r.has("energy"):                         # metric 4's travel, against its own null
        w = np.arange(W, dtype=float)
        decay = np.exp(-(w - r.rl_start).clip(0) / 22)[None, None, :]
        add["energy"] = (0.9 * eff[..., None] * decay * (1 + 0.25 * rng.standard_normal((A, S_, W)))).clip(-0.15)
        add["energy_null"] = 0.06 + 0.02 * rng.random((A, S_, W))
        add["coverage"] = np.cumsum(rng.poisson(3.0 * eff[..., None], (A, S_, W)), axis=-1).astype(float)
        for key in ("energy", "energy_null", "coverage"):
            add[key][..., :r.rl_start] = np.nan           # not logged during pretrain (ADR-0021)

    if not r.has("r_step"):                               # experiment 4's A/B/C and 3's fold
        n_ep = r.arrays["rew_epoch"].shape[-1]
        e = np.arange(n_ep, dtype=float)
        ceil = (3000 * eff)[..., None]
        add["r_step"] = (ceil * (1 - np.exp(-e / (500 * (2 - eff))[..., None]))
                         * (1 + 0.03 * rng.standard_normal((A, S_, n_ep))))
        if W:                                             # the boundary dip, per arm
            off = np.arange(-8, 25)
            e3 = eff[..., None]                        # (A,1,1) -> dip broadcasts to (A,1,offsets)
            dip = np.where(off <= 0, 0.0,
                           -(0.20 * (2 - e3)) * np.exp(-off.clip(0) / (4 + 6 * e3)))
            add["rew_fold"] = (1 + dip) * (1 + 0.01 * rng.standard_normal((A, S_, off.size)))

    if not r.has("attn_map"):                             # experiment 4's E -- `attnmap`'s savez
        n_tok, cs = r.n_tokens, r.content_start
        C = len(r.ckpt_gen) or 3
        heads = _n_heads(r.study)
        M = rng.random((A, S_, C, heads, n_tok, n_tok)) * 0.1
        M[..., np.arange(n_tok), np.arange(n_tok)] += 1.5           # self
        M[..., 0] += 0.8                                            # everyone looks at CLS
        M[0, ..., cs:, cs:] += 0.6                                  # arm 0 attends across modules
        # Apply each arm's own mask, so the fixture cannot show a masked arm attending where the
        # mask forbids it. `attn_offdiag` is then structurally 0 for self/self_cls, which is exactly
        # the floor the real panel reads them against.
        for j, arm in enumerate(r.arms):
            if arm in ("self", "self_cls"):
                keep = np.zeros((n_tok, n_tok), bool)
                np.fill_diagonal(keep, True)
                if arm == "self_cls":
                    keep[:, 0] = True
                M[j] *= keep
        M /= M.sum(-1, keepdims=True)
        add["attn_map"] = M
        # The decisive scalar: share of a module token's mass landing on OTHER module tokens.
        sub = M[..., cs:, :]
        idx = np.arange(n_tok - cs)
        off = sub[..., cs:].sum(-1) - sub[..., idx, cs + idx]
        add["attn_offdiag"] = np.nanmean(off, axis=(-1, -2))        # over heads and query tokens
        add["attn_epochs"] = np.broadcast_to(
            np.linspace(0, 1, C) * (r.arrays["rew_epoch"].shape[-1]), (A, S_, C)).copy()

    r.arrays.update(add)
    r.synthetic |= set(add)
    return r


# ── entry point ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build fixture rollups from finished past runs, so the notebook can be "
                    "developed before the real scrapes exist. Real numbers, FICTIONAL arms.")
    ap.add_argument("studies", nargs="*", default=list(FIXTURE_RUNS),
                    help=f"default: {' '.join(FIXTURE_RUNS)}")
    ap.add_argument("--root", type=Path, default=FIXTURE_ROOT)
    args = ap.parse_args()
    for name in args.studies:
        if name not in FIXTURE_RUNS:
            raise SystemExit(f"[fixture] no fixture mapping for {name!r}; "
                             f"have {', '.join(FIXTURE_RUNS)}")
        fixture(name, root=args.root)
    print(f"\n[fixture] {len(args.studies)} study rollup(s) in {DATA}. These are FIXTURES: the "
          f"numbers are real training runs, the arm labels are not.")


if __name__ == "__main__":
    main()
