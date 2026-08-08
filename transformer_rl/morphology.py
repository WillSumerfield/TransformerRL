"""Bodies, on the agent's side of the boundary.

`Morphology` is the package's type; how a body is *chosen* is ours. This holds the seed body every
codesign run starts from -- the generator replaces it at the first resample window, but window 0
runs on it, and the warmup teacher is defined relative to it.

Slots are **0-based**, matching the package. The pre-migration configs named limbs 1/4/6; the same
body is slots 0/3/5 here.
"""
from codesigner.interfaces import Morphology

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
