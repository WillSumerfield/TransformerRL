"""AntCodesignEnv: variable-length (multi-module) codesign env. The agent (owner of the morphology
generator) hands per-env module COUNTS (0..MAX_LIMB_LENGTH), not a flat presence bit; the env builds
REAL variable-limb bodies each resample window. count 0 = limb absent (presence emergent); a length-k
limb = 1 swing module + (k-1) knee modules at default lengths. Per-module continuous length design is
deferred (defaults only) — the generator designs COUNT in Phase 1. One body per env (EPM=1). The
32-DOF depth-major obs/DOF repack lives in the base AntMultiMorphEnv (Phase 1). See ADR-0010.
"""
import torch

from .ant_multimorph import AntMultiMorphEnv, _N_LIMBS
from .build_vsim import Morphology, bodies_from_counts


class AntCodesignEnv(AntMultiMorphEnv):
    def __init__(self, num_envs, device, *, base_legs=(1, 4, 6), **kwargs):
        self._base_legs = frozenset(base_legs)
        self._next_bodies = None                   # list[dict{limb->count}] len total_num_envs, agent-set
        kwargs.setdefault("sample_morphs", True)    # unlocks resample(); _draw_morphs is overridden
        super().__init__(num_envs, device, **kwargs)

    def set_next(self, counts):
        """Store per-env module counts (N, 8) in 0..MAX; column j -> limb j+1. Effective next rebuild.
        Replaces the old {0,1} presence: count is the limb's module chain length (0 = absent)."""
        c = counts.detach().cpu().numpy() if torch.is_tensor(counts) else counts
        self._next_bodies = bodies_from_counts(c, _N_LIMBS)

    def _draw_morphs(self, num):
        if self._next_bodies is None:               # initial build (pre-set_next): base @ length-2
            self._next_bodies = [{n: 2 for n in self._base_legs}] * num
        assert len(self._next_bodies) == num, \
            f"set_next gave {len(self._next_bodies)} bodies but env has {num} envs"
        return [Morphology.from_counts(c) for c in self._next_bodies]
