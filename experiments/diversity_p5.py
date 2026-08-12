"""Phase-5 generator-commitment estimators, layered on experiments/diversity.py.

Kept OUT of diversity.py: that file is the ADR-0015 phase-comparison harness (d_comp / d_struct /
N_modes) and stays as specified. This module adds what the Phase-5 commitment investigation needs:

  arrays_to_repr        typed P5 body -> diversity.py's repr (subtype rides the type code)
  encode_population     pack reprs to int codes; d_struct == Hamming at W_OVERHANG=1
  pairwise_d_struct     vectorised (M,M) d_struct
  N_modes_curve         N_k modes for many tau at once (one dendrogram, not one pass per tau)
  modes_and_spread      (N_modes at one tau, mean pairwise d_struct) -- per-window training logging
  limb_entropies        within-limb randomness: N_limb(n) = exp H(L_n)
  pairwise_mi           8x8 I(L_m;L_n) -- where body-plan structure lives
  redundancy            cross-limb randomness: rho = exp(C), C = sum_n H(L_n) - H(B)
  rao_blackwell_h_body  H(B) from the generator's own analytic per-step entropies

Identity: H(B) = sum_n H(L_n) - C.  C == 0 exactly when limbs are independent.

Estimation is split by support size. H(L_n) and pairwise joints live over a few hundred limb
strings -> bias-corrected plug-in is fine. H(B) lives over ~1e20 bodies, where ANY plug-in is
hard-capped at ln(M) = 8.3 nats for M=4096 -- the rail that pinned build/n_distinct at 4096/4096.
So H(B) must come from rao_blackwell_h_body, never from counting the population.
"""
from collections import Counter

import numpy as np

from experiments.diversity import TAU_MODE, _dedup, _hill, d_struct  # noqa: F401  (d_struct: self-test)

EFF_NAMES = ("swing", "knee", "twist")             # vocab.py effector subtype order
CAP_NAMES = ("bare", "foot", "pad", "ball")        # vocab.py cap subtype order; 0 == bare


# ---- P5 adapter ------------------------------------------------------------------

def arrays_to_repr(counts, eff_sub, cap_sub, collapse_subtypes=False):
    """One body's (counts (8,), eff_sub (8,max_len), cap_sub (8,)) -> typed repr, tip-first.

    Subtype rides the TYPE CODE ('E:knee', 'C:foot'), not an attribute: subtypes are categorical, so
    diversity._sub's `lam*|a-b|` term would be meaningless (swing vs twist is not "2 units apart").
    Distinct codes => any subtype mismatch costs exactly 1.

    A BARE cap emits NO token (zero-morphology -- contact stays on the last effector), so an
    all-bare-cap P5 body has a repr byte-identical to the P1 `counts_to_repr` body. Morphology caps
    (foot/pad/ball) are real terminal bodies and DO emit a tip token. count 0 = absent.

    collapse_subtypes folds 'E:knee'->'E' / 'C:foot'->'C' for comparison against P0/P1 reprs, where
    the subtype axis does not exist -- the only view comparable across phases."""
    out = []
    for n, c in enumerate(counts):
        c = int(c)
        if c <= 0:
            out.append(None)
            continue
        cap = int(cap_sub[n])
        limb = []
        if cap > 0:                                        # morphology cap = a real tip module
            limb.append("C" if collapse_subtypes else f"C:{CAP_NAMES[cap]}")
        for d in range(c - 1, -1, -1):                     # distal -> proximal
            limb.append("E" if collapse_subtypes else f"E:{EFF_NAMES[int(eff_sub[n][d])]}")
        out.append(limb)
    return out


def population_to_repr(counts, eff_sub, cap_sub, collapse_subtypes=False):
    """(M,8), (M,8,max_len), (M,8) -> list of M reprs."""
    return [arrays_to_repr(counts[i], eff_sub[i], cap_sub[i], collapse_subtypes)
            for i in range(len(counts))]


# ---- fast pairwise d_struct (Hamming identity) -----------------------------------

def encode_population(bodies, max_len=None):
    """Pack reprs into (M, 8*max_len) int32 codes, tip-first, PAD(=0) past each limb's end.

    With W_OVERHANG == 1 and LAM_ATTR == 0, d_struct reduces EXACTLY to Hamming distance here.
    Proof (per limb, tip-anchored): summing [a_i != b_i] over ALL max_len slots counts the `core`
    mismatches over the shared prefix, plus one per position where exactly one limb still has a
    token and the other is PAD -- i.e. |len_a - len_b|, precisely the overhang charge at w=1. So
    sum-over-all = core + |dlen| = d_limb; summing the 8 slots gives d_struct."""
    if max_len is None:
        max_len = max((len(l) for b in bodies for l in b if l), default=1)
    vocab, codes = {}, np.zeros((len(bodies), len(bodies[0]) * max_len), dtype=np.int32)
    for i, body in enumerate(bodies):
        for n, limb in enumerate(body):
            for d, tok in enumerate(limb or ()):
                t = tok[0] if isinstance(tok, tuple) else tok
                codes[i, n * max_len + d] = vocab.setdefault(t, len(vocab) + 1)   # 0 reserved = PAD
    return codes


def pairwise_d_struct(codes, chunk=512):
    """Full (M,M) float32 d_struct matrix via chunked Hamming. Valid only for w=1, lam=0."""
    M = codes.shape[0]
    D = np.empty((M, M), dtype=np.float32)
    for i in range(0, M, chunk):
        blk = codes[i:i + chunk]
        D[i:i + chunk] = (blk[:, None, :] != codes[None, :, :]).sum(-1)
    return D


def _single_linkage_hill(D, counts, taus):
    """{tau: Hill number} over single-linkage clusters of the DEDUPED set at each cluster radius.

    Single-linkage at ALL thresholds is one dendrogram, so sort the edges once and union in
    increasing weight, snapshotting as each tau is passed -- O(n^2) distances total, not per tau."""
    n = len(counts)
    iu = np.triu_indices(n, 1)
    w = D[iu]
    order = np.argsort(w, kind="stable")
    ei, ej, ew = iu[0][order], iu[1][order], w[order]

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def hill_now():
        mass = {}
        for i in range(n):
            mass[find(i)] = mass.get(find(i), 0.0) + counts[i]
        return _hill(np.array(list(mass.values())))

    out, e = {}, 0
    for tau in sorted(taus):
        while e < len(ew) and ew[e] <= tau:
            parent[find(ei[e])] = find(ej[e])
            e += 1
        out[tau] = hill_now()
    return out


def N_modes_curve(bodies, taus, chunk=512):
    """N_k modes for every tau in `taus`: single-linkage Hill number at cluster radius k modules.

    taus=[0] reproduces plain n_distinct (as an effective count) -- the saturating statistic this
    curve exists to replace."""
    reps, counts = _dedup(bodies)
    return _single_linkage_hill(pairwise_d_struct(encode_population(reps), chunk), counts, taus)


def modes_and_spread(bodies, tau=TAU_MODE, chunk=512):
    """(N_modes at `tau`, mean pairwise d_struct) -- the two diversity headlines, from ONE pairwise
    matrix, because computing them separately would pay the O(n^2) distance twice.

    Intended for per-window logging during training, where the caller passes SUBTYPE-COLLAPSED
    skeletons: the free_entropy finding is that the skeleton commits while the subtype axis stays
    free, so a typed statistic stays near the sample size even under total skeleton collapse.

    N_modes is 1.0 exactly when every draw falls in one cluster (full collapse); the spread is the
    threshold-free companion, mean d_struct over all M(M-1)/2 sample pairs INCLUDING duplicate pairs
    (which contribute 0), matching diversity._mean_pairwise."""
    reps, counts = _dedup(bodies)
    D = pairwise_d_struct(encode_population(reps), chunk)
    n_modes = _single_linkage_hill(D, counts, [tau])[tau]
    M = counts.sum()
    # sum_{i<j} c_i c_j D_ij == 0.5 * (c @ D @ c), since D has a zero diagonal and is symmetric;
    # over M(M-1)/2 pairs the halves cancel, leaving c @ D @ c / (M(M-1)).
    spread = float(counts @ D @ counts / (M * (M - 1))) if M > 1 else 0.0
    return n_modes, spread


# ---- entropy decomposition: within-limb vs cross-limb ----------------------------

def _mm_entropy(counts):
    """Miller-Madow entropy (nats): plug-in + (K_observed - 1) / 2M bias correction."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    M = counts.sum()
    if M < 2:
        return 0.0
    p = counts / M
    return float(-(p * np.log(p)).sum() + (len(counts) - 1) / (2.0 * M))


def _limb_keys(bodies):
    """(M, n_slots) object array of hashable per-slot limb keys."""
    return np.array([[tuple(limb) if limb else None for limb in b] for b in bodies], dtype=object)


def limb_entropies(bodies):
    """Within-limb randomness: per-slot H(L_n) in nats and N_limb = exp(H), the EFFECTIVE NUMBER OF
    LIMB DESIGNS that slot chooses among. Perplexity units, not nats -- H_max grew from P1 to P5
    (more subtypes) so nats are not cross-phase comparable, while "effectively 12 limb shapes" is,
    and it shares units with N_modes."""
    K = _limb_keys(bodies)
    H = np.array([_mm_entropy(list(Counter(K[:, n]).values())) for n in range(K.shape[1])])
    return {"H_limb": H, "N_limb": np.exp(H), "H_within_sum": float(H.sum()),
            "N_limb_mean": float(np.exp(H).mean())}


def pairwise_mi(bodies):
    """(n_slots, n_slots) mutual information I(L_m; L_n) in nats, diagonal = H(L_n).

    The heatmap that SHOWS body-plan structure: a hot (front-left, front-right) cell means the
    generator learned to pair those limbs. MI is a difference of Miller-Madow-corrected terms, so
    residual bias is small but nonzero -- read off-diagonal values against the diagonal scale."""
    K = _limb_keys(bodies)
    n = K.shape[1]
    H = np.array([_mm_entropy(list(Counter(K[:, i]).values())) for i in range(n)])
    I = np.zeros((n, n))
    for i in range(n):
        I[i, i] = H[i]
        for j in range(i + 1, n):
            h_ij = _mm_entropy(list(Counter(zip(K[:, i], K[:, j])).values()))
            I[i, j] = I[j, i] = max(0.0, H[i] + H[j] - h_ij)
    return I


def redundancy(bodies, h_body):
    """Cross-limb randomness. rho = prod_n N_limb(n) / exp(H(B)) = exp(C), total correlation in
    perplexity units.

      rho ~= 1  -> eight INDEPENDENT limb-lotteries; no body plan (single-population-ES / common
                   core). Entropy coef buys jitter, not branching -- the lever is P8b structural
                   mutation.
      rho >> 1  -> limbs co-vary; real correlated body plans.

    CAVEAT: rho is confounded with TOTAL entropy -- a fully-committed generator has rho=1 by
    construction (no variance left to correlate). For cross-run comparison use the scale-free
    fraction C / sum_n H(L_n) instead.

    `h_body` MUST be the Rao-Blackwell estimate; a plug-in joint entropy silently caps rho from
    below (h_body <= ln M), the failure this measure exists to avoid."""
    H = limb_entropies(bodies)
    C = H["H_within_sum"] - h_body
    return {"rho": float(np.exp(C)), "C_nats": float(C), "H_body": float(h_body),
            "N_body": float(np.exp(h_body)), **H}


def rao_blackwell_h_body(step_entropy, active_step):
    """H(B | decode order) from the generator's OWN per-step entropies (nats):

        H(B) = E_{B~p} [ sum_t H(p_t(. | prefix_t)) ]

    Sample the PREFIXES (unavoidable -- the frontier is autoregressive) but use the ANALYTIC
    categorical entropy at each visited prefix instead of counting outcomes. No ln(M) ceiling, and
    low variance: each body contributes a smooth 32-term sum, not a one-hot count.

    step_entropy/active_step: (M, L) from GeneratorNet.sample(); inactive steps (no growable limb
    left) are no-ops contributing 0.

    CAVEAT to state when reporting: the frontier picks a random growable limb each step, so this is
    H(B | order), a lower bound on H(B). The gap is order-induced, not generator-induced."""
    e = np.asarray(step_entropy, dtype=np.float64)
    a = np.asarray(active_step, dtype=bool)
    return float((e * a).sum(axis=1).mean())


if __name__ == "__main__":
    from experiments.diversity import N_modes, TAU_MODE, counts_to_repr

    cnt = np.array([2, 0, 3, 0, 0, 0, 0, 0])
    eff = np.zeros((8, 4), dtype=int)
    cap = np.zeros(8, dtype=int)                       # 0 == bare == no tip token
    assert arrays_to_repr(cnt, eff, cap, collapse_subtypes=True) == counts_to_repr(cnt)
    cap[0] = 1
    assert arrays_to_repr(cnt, eff, cap)[0] == ["C:foot", "E:swing", "E:swing"]
    eff[2, 1] = 2
    assert d_struct(arrays_to_repr(cnt, eff, cap),
                    arrays_to_repr(cnt, np.zeros((8, 4), int), cap)) == 1.0

    rng = np.random.default_rng(0)
    pop = [arrays_to_repr(rng.integers(0, 4, 8), rng.integers(0, 3, (8, 4)),
                          rng.integers(0, 4, 8)) for _ in range(60)]
    D = pairwise_d_struct(encode_population(pop))
    for _ in range(200):
        i, j = rng.integers(0, len(pop), 2)
        assert abs(D[i, j] - d_struct(pop[i], pop[j])) < 1e-6

    curve = N_modes_curve(pop, taus=[0, 1, 2, 4, 8, 32])
    assert abs(curve[TAU_MODE] - N_modes(pop)) < 1e-6
    vals = [curve[t] for t in sorted(curve)]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))
    assert curve[32] == 1.0

    ind = [arrays_to_repr(rng.integers(0, 4, 8), np.zeros((8, 4), int), np.zeros(8, int))
           for _ in range(4000)]
    assert abs(redundancy(ind, sum(limb_entropies(ind)["H_limb"]))["rho"] - 1.0) < 1e-9
    corr = [arrays_to_repr(np.full(8, rng.integers(0, 4)), np.zeros((8, 4), int),
                           np.zeros(8, int)) for _ in range(4000)]
    le = limb_entropies(corr)
    assert redundancy(corr, le["H_limb"][0])["rho"] > 100
    assert pairwise_mi(corr)[0, 3] > 1.0 and pairwise_mi(ind)[0, 3] < 0.05

    se = np.array([[0.5, 0.25, 0.0], [0.5, 0.25, 99.0]])
    assert abs(rao_blackwell_h_body(se, np.array([[1, 1, 0], [1, 1, 0]], bool)) - 0.75) < 1e-12
    print("diversity_p5 OK")
