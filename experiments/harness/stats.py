"""Across-seed reductions: the error bars, the noise floor, and the two curve summaries.

    from experiments.harness import stats
    m, lo, hi = stats.mean_ci(z["R_mean"], axis=1)          # (arm, window)
    d = stats.paired_ci(z["spec"][1], z["spec"][0])          # arm 1 - arm 0, seed-paired

Every paper array is on an `[arm, seed, ...]` grid with NaN for a run that is missing or short, so
every function here is nan-aware and reports the `n` it actually used. Nothing plots; the notebooks
do that.

Three decisions this module fixes once, because getting them wrong changes verdicts:

**A band across seeds is a t-interval, not 1.96 sigma.** Eight seeds is small enough that the normal
quantile understates the interval by ~19%, and the studies are read at exactly the width where that
matters. `n` is counted per cell after NaNs are dropped, so a study with one crashed run gets the
wider interval it has earned rather than the one its declared seed count implies.

**Arms are compared PAIRED.** Every study runs the same seeds 42-49 in every arm, and a seed fixes
the initialisation and the env stream -- so seed 44 being a bad seed is a shared term that cancels in
the difference and does not cancel in two independent means. ADR-0018 measured seed-only CV at 9-48%
against a hyperparameter spread of 14%, which is the whole reason the series runs 8 seeds; pairing is
what buys that back. Unpaired `mean_ci` on each arm is for *plotting* the two curves, never for
deciding between them.

**The noise floor is a seed SPREAD, not a CI.** ADR-0018's number is the across-seed CV of one fixed
config: how much outcome a study buys with no change at all. It is the scale a difference is read
against, and it does not shrink with more seeds -- the CI of the mean does. Reporting one in place of
the other is the mistake the ADR exists to prevent, so they are separate functions returning
separately-named things.
"""
import numpy as np
from scipy import stats as _st

CONF = 0.95


# ---- across-seed bands ------------------------------------------------------------

def mean_ci(a, axis: int = 1, conf: float = CONF):
    """(mean, lo, hi) across `axis`, NaN-aware, as a Student-t interval on the mean.

    The band every curve in the series is drawn with. Cells whose finite count is < 2 get the mean
    and a NaN band rather than a zero-width one, so a single surviving seed cannot look certain.
    """
    a = np.asarray(a, float)
    n = np.isfinite(a).sum(axis=axis)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(a, axis=axis)
        sd = np.nanstd(a, axis=axis, ddof=1)
    half = np.full(m.shape, np.nan)
    ok = n >= 2
    if ok.any():
        t = _st.t.ppf(0.5 + conf / 2, np.maximum(n - 1, 1))
        half[ok] = (t * sd / np.sqrt(np.maximum(n, 1)))[ok]
    return m, m - half, m + half


def paired_ci(a, b, axis: int = 0, conf: float = CONF):
    """The seed-paired difference `a - b`: (mean, lo, hi, n).

    Only positions finite in BOTH arms contribute, which is what makes it paired: dropping a seed
    from one arm and keeping it in the other would compare a 7-seed mean against an 8-seed one and
    call the leftover a difference. `axis` is the seed axis of the two arrays passed in -- callers
    hand it one arm each (`z["spec"][1], z["spec"][0]`), so the default is 0.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = np.isfinite(d).sum(axis=axis)
    m, lo, hi = mean_ci(d, axis=axis, conf=conf)
    return m, lo, hi, n


def pooled_sd(sd, axis: int = 1):
    """Per-seed spreads combined into one: the OUTER band of metric 2's nested pair.

    Root-mean-square, not a mean: these are standard deviations of the same quantity measured on
    different seeds, so it is their variances that average. Using the mean of the sds would report a
    band narrower than any honest pooling of the samples.
    """
    v = np.asarray(sd, float) ** 2
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(v, axis=axis))


def nested_bands(g, g_sd, axis: int = 1, conf: float = CONF):
    """Metric 2's two bands from `(G, G_sd)`: `(mean, ci_lo, ci_hi, out_lo, out_hi)`.

    Inner is the CI of the per-seed level mean across seeds -- "would another seed land here".
    Outer is the across-BODY spread at that level, seeds pooled -- "how different are the bodies a
    level draws". They answer different questions and the level-0 outer band is the metric's own
    noise floor (every body there is the identical committed body), which is why the pair is
    computed together and never collapsed into one.
    """
    m, lo, hi = mean_ci(g, axis=axis, conf=conf)
    s = pooled_sd(g_sd, axis=axis)
    return m, lo, hi, m - s, m + s


# ---- the noise floor --------------------------------------------------------------

def noise_floor(a, axis: int = -1):
    """ADR-0018's floor for one condition: `{mean, sd, cv, n}` of the across-seed spread.

    Hand it the study's control arm at the point being decided (e.g. `z["spec"][0, :, -1]`). `sd` is
    the band a difference is read against and `cv` is what compares to the ADR's 9-48% range; both
    describe the CONDITION and neither shrinks as seeds are added. For "is this arm above that one",
    use `paired_ci`.
    """
    a = np.asarray(a, float)
    n = int(np.isfinite(a).sum())
    m = float(np.nanmean(a)) if n else np.nan
    sd = float(np.nanstd(a, ddof=1)) if n >= 2 else np.nan
    return {"mean": m, "sd": sd, "cv": abs(sd / m) if n >= 2 and m else np.nan, "n": n}


# ---- curve summaries --------------------------------------------------------------

def asymptote(curve, tail: int = 200, axis: int = -1):
    """Mean over a curve's final `tail` points -- experiment 4's measurement B, and the endpoint
    summary of any per-epoch series. NaN-aware, so a short run averages what it wrote."""
    a = np.asarray(curve, float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(a[..., -tail:] if axis in (-1, a.ndim - 1)
                          else np.take(a, range(a.shape[axis] - tail, a.shape[axis]), axis=axis),
                          axis=axis)


def crossing(curve, threshold: float, smooth: int = 5, axis: int = -1):
    """First index where a `smooth`-point trailing mean of `curve` reaches `threshold`; NaN if it
    never does -- experiment 4's measurement C, sample efficiency.

    Trailing rather than centred so the answer never depends on the future, and a *threshold* rather
    than a fraction of each arm's own asymptote so the arms are compared at one bar. Experiment 4
    fixes that bar from `self_cls` before `full` is looked at; that ordering is the caller's job.
    """
    a = np.moveaxis(np.asarray(curve, float), axis, -1)
    k = np.ones(smooth) / smooth
    flat = a.reshape(-1, a.shape[-1])
    out = np.full(len(flat), np.nan)
    for i, row in enumerate(flat):
        if not np.isfinite(row).any():
            continue
        sm = np.convolve(np.nan_to_num(row, nan=-np.inf), k, mode="valid")
        hit = np.flatnonzero(sm >= threshold)
        if hit.size:
            out[i] = hit[0] + smooth - 1        # index of the last point in the crossing window
    return out.reshape(a.shape[:-1])


# ---- self-test --------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 5, size=8)                     # the seed effect both arms feel
    a = 10 + shared + rng.normal(0, 0.5, size=8)
    b = 9 + shared + rng.normal(0, 0.5, size=8)

    m, lo, hi = mean_ci(np.stack([a, b])[:, :, None], axis=1)
    assert m.shape == (2, 1) and np.all(lo < m) and np.all(m < hi)
    # the whole reason arms are compared paired: the shared seed term cancels
    _, plo, phi, n = paired_ci(a, b)
    assert n == 8 and plo > 0, "paired difference should clear zero"
    assert (hi - lo)[0, 0] > (phi - plo), "unpaired band should be far wider than the paired one"
    print(f"unpaired half-width {(hi - lo)[0, 0] / 2:.2f} vs paired {(phi - plo) / 2:.2f}")

    a_missing = a.copy(); a_missing[3] = np.nan
    assert paired_ci(a_missing, b)[3] == 7, "a NaN in one arm must drop the pair"

    nf = noise_floor(a)
    assert nf["n"] == 8 and 0 < nf["cv"] < 1
    print(f"noise floor: mean={nf['mean']:.2f} sd={nf['sd']:.2f} cv={nf['cv']:.1%}")

    curve = np.stack([np.linspace(0, 100, 500), np.linspace(0, 50, 500)])
    assert abs(asymptote(curve, tail=100)[0] - 90.1) < 1.0, asymptote(curve, tail=100)
    c = crossing(curve, 25.0)
    assert c[0] < c[1] and np.isfinite(c).all(), c
    assert np.isnan(crossing(curve, 1e9))[0]
    print(f"asymptote {asymptote(curve, tail=100)} crossing {c}")
    print("OK")
