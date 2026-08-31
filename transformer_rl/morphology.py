"""Bodies, on the agent's side of the boundary.

`Morphology` is the package's type; how a body is *chosen* is ours. This holds the seed body every
codesign run starts from -- the generator replaces it at the first resample window, but window 0
runs on it, and the warmup teacher is defined relative to it.

Slots are **0-based**, matching the package. The pre-migration configs named limbs 1/4/6; the same
body is slots 0/3/5 here.
"""
import torch

from codesigner.interfaces import ModuleType, Morphology

# The canonical ant: three limbs, each a swing hip then a knee, left uncapped. Reproduces the
# pre-migration `AntCodesignEnv._BASE_MORPHOLOGY` (limbs 1, 4, 6 at count 2) slot for slot.
CANONICAL_SLOTS = (0, 3, 5)
CANONICAL_EFFECTORS = ("swing", "knee")
CANONICAL_CAP = "bare"


def seed_body(library, slots=CANONICAL_SLOTS, effectors=CANONICAL_EFFECTORS,
              cap=CANONICAL_CAP) -> Morphology:
    """The uniform seed body: the same chain in every named slot.

    Every config this project has ever run describes its seed this way -- a set of slots at one
    shared chain -- so that is what the `env.base_morphology` block takes. A body whose limbs differ
    from each other is built with `Morphology.from_names` directly.
    """
    return Morphology.from_names(library, {int(s): (tuple(effectors), cap) for s in slots})


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def designs_from_arrays(library, counts, eff_sub, cap_sub) -> list:
    """The generator's designed body grid -> the bodies the Task will build.

    This is the whole of the Algorithm->Task boundary on the morphology side: everything upstream is
    integer subtype ids in the generator's own tensor vocabulary, everything downstream is
    `Morphology`. The Task takes the list at `resample`.
      counts  (N, n_slots)            effectors per slot, 0 = slot empty
      eff_sub (N, n_slots, max_depth) effector subtype id per depth (only [:count] is read)
      cap_sub (N, n_slots)            cap subtype id per slot (-1 -> the canonical bare cap)
    Slots are 0-based on both sides, so column j is slot j and nothing is renumbered in transit.
    """
    counts, eff_sub, cap_sub = _np(counts), _np(eff_sub), _np(cap_sub)
    eff_names = library.names(ModuleType.EFFECTOR)
    cap_names = library.names(ModuleType.CAP)
    out = []
    for e in range(len(counts)):
        chains = {}
        for j in range(library.n_slots):
            k = int(counts[e][j])
            if k <= 0:                       # a slot with no effectors is absent, cap or no cap
                continue
            assert k <= library.max_effectors, \
                f"effector count {k} > max_effectors={library.max_effectors}"
            c = int(cap_sub[e][j])
            chains[j] = ([eff_names[int(eff_sub[e][j][d])] for d in range(k)],
                         CANONICAL_CAP if c < 0 else cap_names[c])
        assert chains, "0-module body; the generator must guarantee >=1 limb"
        out.append(Morphology.from_names(library, chains))
    return out


def stable_slot_sets(n_slots: int, min_limbs: int = 3, max_limbs: int = 8,
                     max_gap_deg: float = 135.0) -> list:
    """Slot subsets with no circular gap wider than `max_gap_deg` -- bodies that can stand up.

    Ported from the pre-migration full-ant sampler. Slots are evenly spaced around a ring, so the
    gap test is on 360/n_slots degree steps rather than the hardcoded 45.
    """
    step = 360.0 / n_slots
    out = []
    for mask in range(1, 1 << n_slots):
        active = frozenset(i for i in range(n_slots) if (mask >> i) & 1)
        if not min_limbs <= len(active) <= max_limbs:
            continue
        angles = sorted(i * step for i in active)
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append(360.0 - angles[-1] + angles[0])
        if max(gaps) <= max_gap_deg:
            out.append(active)
    return out


def sample_bodies(library, num: int, rng, effectors=CANONICAL_EFFECTORS,
                  cap=CANONICAL_CAP) -> list:
    """`num` random stable topologies: limb count uniform over the available counts, then a topology
    uniform within that count.

    The pre-migration sampler also drew each limb's hip and ankle LENGTH uniformly from a range.
    That has no successor: the ModuleLibrary port made module length uniform and library-owned, so
    the only axis left to randomise is which slots are occupied. Bodies are otherwise the canonical
    chain. Pass a persistent `rng` to make a run's resample stream reproducible.
    """
    by_count = {}
    for s in stable_slot_sets(library.n_slots):
        by_count.setdefault(len(s), []).append(s)
    counts = sorted(by_count)
    return [seed_body(library, slots=rng.choice(by_count[rng.choice(counts)]),
                      effectors=effectors, cap=cap)
            for _ in range(num)]


def canonical_chain(n_effectors: int) -> tuple:
    """The canonical limb of a given length: one hip, then knees. `CANONICAL_EFFECTORS` at n=2."""
    hip, joint = CANONICAL_EFFECTORS
    return (hip,) + (joint,) * (n_effectors - 1)


def body_from_counts(library, counts, cap=CANONICAL_CAP) -> Morphology:
    """{slot: n_effectors} -> the canonical chain of that length in each named slot.

    The successor to the pre-migration `Morphology.from_counts`, for the analysis scripts that
    describe a body by limb length alone. Slots are 0-based; a count of 0 leaves the slot empty.
    """
    return Morphology.from_names(
        library, {int(s): (canonical_chain(int(k)), cap) for s, k in counts.items() if int(k) > 0})


def arrays_from_designs(library, designs, max_depth=None, device=None):
    """The exact inverse of `designs_from_arrays`: bodies -> the generator's tensor grid.

    Needed wherever a body arrives from OUTSIDE the generator and has to be spoken about in the
    generator's own vocabulary -- a fixed-morphology phase pinning `_cur_*`, or GenCrit scoring a
    design it did not draw. `max_depth` sizes the effector axis and must be at least the network's
    `max_limb_length` for the arrays to drop straight into the agent's window state.
      -> counts (M, n_slots), eff_sub (M, n_slots, max_depth), cap_sub (M, n_slots)
    Empty slots read back as count 0 / cap -1, which is how the agent spells "no limb".

    NOT a total inverse, for one representational reason: an UNCAPPED occupied slot cannot be
    expressed. `cap_sub = -1` already means "still growable" in the frontier MDP, so the nearest
    thing is the canonical bare cap and that is what an uncapped limb becomes. The grammar caps
    every limb it finishes, so this is exact on any body the generator drew; it rounds only bodies
    from elsewhere -- `ModuleLibrary.random_morphology` in particular, which leaves limbs open.
    Fine where the arrays feed a PREDICTION about a body, wrong wherever they are read back as a
    statement of what was BUILT, since a Task builds the uncapped limb it was handed.
    """
    max_depth = library.max_effectors if max_depth is None else max_depth
    eff_id = {n: i for i, n in enumerate(library.names(ModuleType.EFFECTOR))}
    cap_id = {n: i for i, n in enumerate(library.names(ModuleType.CAP))}
    M, n = len(designs), library.n_slots
    counts = torch.zeros(M, n, dtype=torch.long, device=device)
    eff_sub = torch.full((M, n, max_depth), -1, dtype=torch.long, device=device)
    cap_sub = torch.full((M, n), -1, dtype=torch.long, device=device)
    for e, body in enumerate(designs):
        for j in body.occupied_slots:
            effectors = body.effectors(j)
            assert len(effectors) <= max_depth, \
                f"slot {j} has {len(effectors)} effectors, max_depth={max_depth}"
            counts[e, j] = len(effectors)
            for d, m in enumerate(effectors):
                eff_sub[e, j, d] = eff_id[m.name]
            cap = body.cap(j)
            cap_sub[e, j] = cap_id[CANONICAL_CAP] if cap is None else cap_id[cap.name]
    return counts, eff_sub, cap_sub
