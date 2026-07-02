"""Morphology-diversity metrics for the phase-comparison harness (ADR-0015).

Shared representation (survives added module types/lengths across phases):
  morphology = list of 8 compass SLOTS. Slot = a limb as a DISTAL->PROXIMAL
  sequence of module tokens (index 0 = the tip), or None if the limb is absent.
  A module token is a type code str ('E' effector, 'L' link, 'C' cap, ...), or a
  tuple (type, *attrs) once attributes (length, angle) come online (P1/P5). P0
  limbs are ['E','E'] (distal + proximal effector), no attrs.

Three measures, each reported within-run and between-seed:
  d_comp    module-histogram L1 (bag-of-modules; position/limb-invariant)
  d_struct  slot-matched sum of per-limb TIP-ANCHORED edit distances
  N_modes   effective #distinct designs = Hill number (q=1) over d_struct clusters

All population stats dedup morphologies first (P0: <=256 distinct presence
patterns), so they cost O(distinct^2), not O(4096^2).

Run `python experiments/diversity.py` for the self-test.
"""
from collections import Counter

import numpy as np

# tunable defaults (ADR-0015 §Consequences; revisit after first runs)
W_OVERHANG = 1.0   # proximal-overhang charge per module in d_limb
LAM_ATTR = 0.0     # attribute weight in _sub (0 at P0: types only)
TAU_MODE = 1.0     # single-linkage cluster threshold for N_modes (= 1 module)
HILL_Q = 1         # Hill order (1 => perplexity)


# ---- module / limb / morphology distances ---------------------------------------

def _sub(x, y, lam=LAM_ATTR):
    """Substitution cost between two module tokens (str type or (type,*attrs))."""
    tx = x[0] if isinstance(x, tuple) else x
    ty = y[0] if isinstance(y, tuple) else y
    c = 0.0 if tx == ty else 1.0
    if lam and isinstance(x, tuple) and isinstance(y, tuple):
        c += lam * sum(abs(a - b) for a, b in zip(x[1:], y[1:]))
    return c


def d_limb(a, b, w=W_OVERHANG, lam=LAM_ATTR):
    """Tip-anchored edit distance between two distal->proximal limbs (None=absent).

    Aligned at index 0 (the tip); the extra proximal modules of the longer limb are
    charged `w` each. So E-C vs E-E-C -> tip-align [C,E] vs [C,E,E] -> 0+0+w."""
    a = a or []
    b = b or []
    n = min(len(a), len(b))
    core = sum(_sub(a[i], b[i], lam) for i in range(n))
    return core + w * abs(len(a) - len(b))


def d_struct(A, B, w=W_OVERHANG, lam=LAM_ATTR):
    """Slot-matched sum of tip-anchored limb distances over the 8 compass slots."""
    return sum(d_limb(sa, sb, w, lam) for sa, sb in zip(A, B))


def _histogram(M):
    c = Counter()
    for limb in M:
        if limb:
            c.update(m[0] if isinstance(m, tuple) else m for m in limb)
    return c


def d_comp(A, B, normalized=False):
    """L1 between the two robots' module-type histograms (bag-of-modules)."""
    ha, hb = _histogram(A), _histogram(B)
    keys = set(ha) | set(hb)
    if normalized:
        sa, sb = sum(ha.values()) or 1, sum(hb.values()) or 1
        return sum(abs(ha[k] / sa - hb[k] / sb) for k in keys)
    return sum(abs(ha[k] - hb[k]) for k in keys)


# ---- population helpers (dedup-based) --------------------------------------------

def _key(M):
    return tuple(tuple(limb) if limb else None for limb in M)


def _dedup(bodies):
    """-> (reps, counts): distinct morphologies and their multiplicities."""
    keyed = Counter(_key(b) for b in bodies)
    reps = [[list(limb) if limb else None for limb in k] for k in keyed]
    counts = np.array(list(keyed.values()), dtype=float)
    return reps, counts


def _mean_pairwise(bodies, dfun):
    """Mean pairwise distance over the sample (incl. duplicates -> distance 0)."""
    reps, counts = _dedup(bodies)
    M = counts.sum()
    if M < 2:
        return 0.0
    num = 0.0
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            num += counts[i] * counts[j] * dfun(reps[i], reps[j])
    return num / (M * (M - 1) / 2)


def _hill(masses, q=HILL_Q):
    p = masses[masses > 0] / masses.sum()
    if q == 1:
        return float(np.exp(-(p * np.log(p)).sum()))
    return float((p ** q).sum() ** (1.0 / (1.0 - q)))


def N_modes(bodies, tau=TAU_MODE, w=W_OVERHANG, lam=LAM_ATTR, q=HILL_Q):
    """Effective #distinct designs: Hill number over single-linkage d_struct clusters."""
    reps, counts = _dedup(bodies)
    n = len(reps)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if d_struct(reps[i], reps[j], w, lam) <= tau:
                parent[find(i)] = find(j)
    mass = {}
    for i in range(n):
        mass[find(i)] = mass.get(find(i), 0.0) + counts[i]
    return _hill(np.array(list(mass.values())), q)


# ---- metric bundles (what Module 3 dumps) ----------------------------------------

def within_run_metrics(bodies):
    """Spread over the M bodies one converged generator samples."""
    return {
        "div_comp": _mean_pairwise(bodies, d_comp),
        "div_comp_norm": _mean_pairwise(bodies, lambda a, b: d_comp(a, b, normalized=True)),
        "div_struct": _mean_pairwise(bodies, d_struct),
        "div_nmodes": N_modes(bodies),
    }


def _mode_overlap(dominant, tau=TAU_MODE, w=W_OVERHANG, lam=LAM_ATTR):
    n = len(dominant)
    pairs = shared = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            shared += d_struct(dominant[i], dominant[j], w, lam) <= tau
    return shared / pairs if pairs else 0.0


def between_seed_metrics(dominant_bodies):
    """Spread over each seed's dominant (argmax-likelihood) body."""
    return {
        "div_comp": _mean_pairwise(dominant_bodies, d_comp),
        "div_struct": _mean_pairwise(dominant_bodies, d_struct),
        "div_nmodes": N_modes(dominant_bodies),
        "mode_overlap": _mode_overlap(dominant_bodies),
    }


# ---- P0 adapter ------------------------------------------------------------------

def presence_to_repr(presence):
    """P0: 8-slot presence (bool/int) -> repr with each on-limb = ['E','E']."""
    return [["E", "E"] if p else None for p in presence]


# ---- self-test -------------------------------------------------------------------

if __name__ == "__main__":
    EC, EEC = ["C", "E"], ["C", "E", "E"]     # distal->proximal (tip first)
    assert d_limb(EC, EEC) == 1.0, "tip-aligned E-C vs E-E-C should differ by one overhang"
    assert d_limb(EC, EC) == 0.0
    assert d_limb(None, EEC) == 3.0           # absent vs 3-module limb
    assert d_limb(["C", "E"], ["C", "L"]) == 1.0   # one type mismatch

    a = presence_to_repr([1, 1, 0, 0, 0, 0, 0, 0])
    b = presence_to_repr([1, 0, 0, 0, 0, 0, 0, 0])
    assert d_struct(a, b) == 2.0              # one missing E-E limb -> 2 modules
    assert d_comp(a, b) == 2.0               # histograms: 4 E vs 2 E

    # attribute-aware limb (P1 forward-compat): (type, length)
    la = [("E", 1.0), ("E", 1.0)]
    lb = [("E", 3.0), ("E", 1.0)]
    assert d_limb(la, lb, lam=0.5) == 1.0     # 0.5*|3-1| on the tip module

    pop = [presence_to_repr([1, 1, 0, 0, 0, 0, 0, 0])] * 90 \
        + [presence_to_repr([1, 1, 1, 0, 0, 0, 0, 0])] * 10
    wm = within_run_metrics(pop)
    assert 1.0 < wm["div_nmodes"] < 1.5, wm["div_nmodes"]   # ~1 dominant + rare 2nd mode
    print("within_run:", {k: round(v, 4) for k, v in wm.items()})

    seeds = [presence_to_repr([1, 1, 0, 0, 0, 0, 0, 0]),
             presence_to_repr([1, 1, 1, 0, 0, 0, 0, 0]),
             presence_to_repr([1, 1, 0, 0, 0, 0, 0, 0])]
    bm = between_seed_metrics(seeds)
    print("between_seed:", {k: round(v, 4) for k, v in bm.items()})
    print("OK")
