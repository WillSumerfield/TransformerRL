"""Metric 4's travel and mode coverage, from a run's per-window generator population dumps.

`scrape.py` locates `<run>/gen_pop/w*.npz` and calls `window_series`; the statistics live here.
Breadth -- the third leg of metric 4 -- is a TensorBoard scalar (`build/n_modes`, `build/div_struct`)
and is not recomputed: this module measures the two things a snapshot cannot see.

**Travel is the energy distance between consecutive windows' populations**, on the same
subtype-collapsed skeleton the breadth scalars use:

    E(A,B) = 2*mean d_struct(A,B) - mean d_struct(A,A') - mean d_struct(B,B')

The two subtracted terms are what make it travel rather than breadth. Plain mean cross-window
`d_struct` was rejected in ADR-0021 for exactly this reason: for two IDENTICAL distributions it
equals the within-window mean pairwise distance, so a wide static generator scores as fast-moving.
Here identical distributions give 0 in expectation.

**Its null is measured, not assumed.** A finite sample of a stationary generator does not give
exactly 0, so each window's sample is split in half and the same statistic computed on the halves.
That is the same-distribution floor, in-band and per seed.

**Mode coverage is the cumulative companion**, and its slope is the discovery rate.

Two implementation notes, both load-bearing.

**The mean distances are computed from per-column marginals, not from a pairwise matrix.** Under
`encode_population`'s packing, `d_struct` IS Hamming (its docstring proves it at `W_OVERHANG == 1`),
and Hamming is a sum of independent per-column indicators, so linearity of expectation gives the
mean over all ordered pairs as `sum_c (1 - sum_v pA(c,v) * pB(c,v))` -- exact, and O(M*C) instead of
O(M^2*C). At 4096 draws x 48 windows x 64 runs the pairwise form is the difference between a
coffee break and an afternoon. The self-test checks it against brute-force `diversity.d_struct`.

**The within-population term is bias-corrected to match `build/div_struct` exactly.** The marginal
form averages over all M^2 ordered pairs including the zero diagonal; `modes_and_spread` averages
over the M(M-1)/2 unordered pairs excluding it. The ratio is M/(M-1), applied here, so travel is
readable against the breadth scalar without either being rescaled.

There is no shortcut past the encoder. A collapsed repr is not the count vector: a morphology cap
(foot/pad/ball) emits its own tip token while a bare cap emits none, so two limbs with equal
effector counts can still differ. `d_struct` on `population_to_repr(..., collapse_subtypes=True)` is
the definition, and this module reuses it rather than restating it.
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from experiments.harness.committance import encode_population, population_to_repr   # noqa: E402
from experiments.harness.diversity import TAU_MODE, _key                            # noqa: E402

NULL_REPEATS = 8          # random split-halves averaged per window, to quiet the null's own variance
NULL_SEED = 0             # the null is a measurement, so it is reproducible

# What `window_series` returns, and what `scrape.py` preallocates. `energy` is the headline, on the
# subtype-collapsed skeleton; `energy_typed` is the same statistic with the subtype axis kept. The
# free_entropy finding is what makes the pair worth carrying: the skeleton commits while the subtype
# axis stays free, so typed travel well above the null while collapsed travel sits AT it is a
# generator shuffling effector and cap subtypes on a body plan it has already stopped searching.
SERIES = ("energy", "energy_null", "energy_typed", "energy_typed_null", "coverage")


# ---- loading -------------------------------------------------------------------------

def _fold(reprs, mult):
    """Reprs + multiplicities -> distinct reprs + summed multiplicities."""
    folded = {}
    for body, m in zip(reprs, mult):
        k = _key(body)
        folded[k] = folded.get(k, 0) + int(m)
    return ([[list(limb) if limb else None for limb in k] for k in folded],
            np.array(list(folded.values()), dtype=np.float64))


def load_population(path, collapse=True):
    """One `gen_pop/w*.npz` -> (reps, weights): distinct designs and their multiplicities.

    Deduped on the RAW arrays first, in numpy, before any repr is built -- a collapsed generator
    draws 4096 copies of a handful of designs, and building 4096 python reprs to throw all but a
    few away is most of the cost of a scrape. Two raw rows can still fold to one repr (an absent
    slot's subtype ids are arbitrary; a collapsed view drops the subtype axis entirely), so the
    reprs are folded again afterwards.
    """
    return _views(path)[0 if collapse else 1]


def _views(path):
    """Both views of one dump, sharing the single raw dedup: (collapsed, typed)."""
    with np.load(path) as z:
        counts, eff, cap = z["counts"], z["eff_sub"], z["cap_sub"]
    flat = np.concatenate([counts.reshape(len(counts), -1),
                           eff.reshape(len(eff), -1),
                           cap.reshape(len(cap), -1)], axis=1)
    _, idx, mult = np.unique(flat, axis=0, return_index=True, return_counts=True)
    c, e, k = counts[idx], eff[idx], cap[idx]
    return (_fold(population_to_repr(c, e, k, collapse_subtypes=True), mult),
            _fold(population_to_repr(c, e, k, collapse_subtypes=False), mult))


# ---- mean distances via per-column marginals -----------------------------------------

def _encode(*populations):
    """Encode several populations under ONE vocabulary and one `max_len`.

    `encode_population` builds its vocabulary from whatever it is handed, so encoding two windows
    separately yields codes that are not comparable -- the same token could be 1 in one window and 2
    in the next, and every distance would be wrong in a way that still looks like a plausible
    number. Padding columns are PAD in both populations and contribute 0, so a shared `max_len`
    taken over the union changes nothing but the column count.
    """
    sizes = [len(p) for p in populations]
    codes = encode_population([b for p in populations for b in p])
    out, i = [], 0
    for m in sizes:
        out.append(codes[i:i + m])
        i += m
    return out, int(codes.max()) + 1


def _col_probs(codes, w, n_vals):
    """(C, n_vals) per-column value distribution of the weighted sample."""
    C = codes.shape[1]
    P = np.empty((C, n_vals))
    for c in range(C):
        P[c] = np.bincount(codes[:, c], weights=w, minlength=n_vals)
    return P / w.sum()


def _cross(codes_a, w_a, codes_b, w_b, n_vals):
    """Mean d_struct over all ordered cross pairs = sum_c P(a_c != b_c)."""
    pa, pb = _col_probs(codes_a, w_a, n_vals), _col_probs(codes_b, w_b, n_vals)
    return float(codes_a.shape[1] - (pa * pb).sum())


def _spread(codes, w, n_vals):
    """Mean pairwise d_struct within one population -- `build/div_struct`'s statistic."""
    M = w.sum()
    if M < 2:
        return 0.0
    p = _col_probs(codes, w, n_vals)
    return float((codes.shape[1] - (p * p).sum()) * M / (M - 1))


def energy(reps_a, w_a, reps_b, w_b):
    """Energy distance between two populations of collapsed skeletons."""
    (ca, cb), n_vals = _encode(reps_a, reps_b)
    return (2.0 * _cross(ca, w_a, cb, w_b, n_vals)
            - _spread(ca, w_a, n_vals) - _spread(cb, w_b, n_vals))


def split_half_null(reps, w, repeats=NULL_REPEATS, seed=NULL_SEED):
    """The same statistic on two halves of ONE window's sample: the same-distribution floor.

    Averaged over `repeats` random splits because a single split is itself noisy, and this number is
    the yardstick every travel value is read against.
    """
    M = int(w.sum())
    if M < 4:
        return np.nan
    codes, n_vals = _encode(reps)
    codes = codes[0]
    members = np.repeat(np.arange(len(reps)), w.astype(int))
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(repeats):
        perm = rng.permutation(M)
        w1 = np.bincount(members[perm[:M // 2]], minlength=len(reps)).astype(np.float64)
        w2 = np.bincount(members[perm[M // 2:]], minlength=len(reps)).astype(np.float64)
        acc.append(2.0 * _cross(codes, w1, codes, w2, n_vals)
                   - _spread(codes, w1, n_vals) - _spread(codes, w2, n_vals))
    return float(np.mean(acc))


# ---- cumulative mode coverage --------------------------------------------------------

def _hamming(codes_a, codes_b, chunk=512):
    """(Ma, Mb) d_struct matrix, chunked over the first axis."""
    out = np.empty((len(codes_a), len(codes_b)), dtype=np.int32)
    for i in range(0, len(codes_a), chunk):
        blk = codes_a[i:i + chunk]
        out[i:i + chunk] = (blk[:, None, :] != codes_b[None, :, :]).sum(-1)
    return out


class ModeCover:
    """Greedy cover at radius `tau`: a design opens a new mode if it is further than `tau` from
    every mode already discovered.

    Deliberately NOT `N_modes`' single-linkage clustering, which is the right answer for a snapshot
    and the wrong one here: single-linkage components MERGE as points accumulate, so a cumulative
    single-linkage count can go DOWN between windows and its slope is not a discovery rate. A greedy
    cover is monotone by construction, which is the property a cumulative curve has to have. The
    price is order-dependence -- windows are fed in time order, which is deterministic and is also
    the order the question is asked in.
    """

    def __init__(self, tau=TAU_MODE):
        self.tau = tau
        self.reps = []

    def update(self, reps):
        """Add one window's distinct designs; -> the cumulative mode count.

        Screened before it is walked: a design within `tau` of the cover cannot open a mode however
        the loop is ordered, so one batched distance pass rules out nearly everything and the
        sequential part -- which exists only to resolve candidates that are far from the cover but
        close to EACH OTHER -- runs over the handful that survive. Encoding is done once per window
        rather than once per design, which is the difference between seconds and hours on a window
        the generator has not yet collapsed.
        """
        if not self.reps:
            self.reps.append(reps[0])
        (cw, cc), _ = _encode(reps, self.reps)
        cover = cc
        for i in np.flatnonzero(_hamming(cw, cover).min(1) > self.tau):
            if _hamming(cw[i:i + 1], cover).min() > self.tau:
                self.reps.append(reps[i])
                cover = np.vstack([cover, cw[i:i + 1]])
        return len(self.reps)


# ---- the series scrape.py asks for ---------------------------------------------------

def window_series(paths, n):
    """{window index -> dump path} -> the three (n,) series, NaN where undefined.

    `energy[k]` is the move INTO window k -- the distance from window k-1's population to window k's
    -- so it is NaN unless both dumps are present, which also makes a crash-and-resume gap read as a
    gap rather than as a jump. `energy_null[k]` is the split-half floor for that same pair, averaged
    over its two windows. `coverage[k]` is cumulative over every window dumped so far.

    Window 0 never has a dump: it runs the seed body, before the generator has drawn anything. A
    panel that starts at the pretrain boundary should slice these, not re-zero them.
    """
    out = {k: np.full(n, np.nan) for k in SERIES}
    cover = ModeCover()
    prev = None
    for k in sorted(paths):
        if k >= n:
            break
        views = _views(paths[k])
        nulls = [split_half_null(r, w) for r, w in views]
        out["coverage"][k] = cover.update(views[0][0])
        if prev is not None and prev[0] == k - 1:
            for (key, nkey), (r, w), (pr, pw), pn, nn in zip(
                    (("energy", "energy_null"), ("energy_typed", "energy_typed_null")),
                    views, prev[1], prev[2], nulls):
                out[key][k] = energy(pr, pw, r, w)
                out[nkey][k] = np.nanmean([pn, nn])
        prev = (k, views, nulls)
    return out


# ---- self-test -----------------------------------------------------------------------

def _brute_mean(reps_a, w_a, reps_b, w_b):
    from experiments.harness.diversity import d_struct
    num = sum(w_a[i] * w_b[j] * d_struct(reps_a[i], reps_b[j])
              for i in range(len(reps_a)) for j in range(len(reps_b)))
    return num / (w_a.sum() * w_b.sum())


def _self_test():
    from experiments.harness.committance import modes_and_spread
    rng = np.random.default_rng(0)
    for trial in range(5):
        pops = []
        for _ in range(2):
            M = 40
            counts = rng.integers(0, 4, size=(M, 8))
            eff = rng.integers(0, 3, size=(M, 8, 3))
            cap = rng.integers(0, 4, size=(M, 8))
            counts[counts.sum(1) == 0, 0] = 1                  # no 0-module bodies
            bodies = population_to_repr(counts, eff, cap, collapse_subtypes=True)
            keyed = {}
            for b in bodies:
                keyed[_key(b)] = keyed.get(_key(b), 0) + 1
            pops.append(([[list(l) if l else None for l in k] for k in keyed],
                         np.array(list(keyed.values()), dtype=np.float64), bodies))
        (ra, wa, ba), (rb, wb, bb) = pops
        (ca, cb), nv = _encode(ra, rb)

        got, want = _cross(ca, wa, cb, wb, nv), _brute_mean(ra, wa, rb, wb)
        assert abs(got - want) < 1e-9, f"cross {got} != {want}"

        got = _spread(ca, wa, nv)
        want = modes_and_spread(ba)[1]
        assert abs(got - want) < 1e-9, f"spread {got} != {want}"

        # the whole estimator against brute-force d_struct, term by term
        brute = (2.0 * _brute_mean(ra, wa, rb, wb)
                 - modes_and_spread(ba)[1] - modes_and_spread(bb)[1])
        got_e = energy(ra, wa, rb, wb)
        assert abs(got_e - brute) < 1e-9, f"energy {got_e} != {brute}"
        print(f"trial {trial}: cross={got:.4f} energy={got_e:+.4f} (brute {brute:+.4f}) ok")

    # both populations are draws from the SAME process, so travel should sit at the split-half
    # floor rather than above it -- the property the null exists to make readable.
    ra, wa, _ = pops[0]
    rb, wb, _ = pops[1]
    print(f"same-distribution: energy={energy(ra, wa, rb, wb):+.4f} "
          f"null={split_half_null(ra, wa):+.4f}")

    # coverage is monotone
    cov = ModeCover()
    seq = [cov.update(p[0]) for p in (pops[0], pops[1], pops[0])]
    assert seq == sorted(seq), f"coverage not monotone: {seq}"
    print(f"coverage {seq} ok")


if __name__ == "__main__":
    _self_test()
