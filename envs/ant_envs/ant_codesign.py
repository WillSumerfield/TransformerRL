"""AntCodesignEnv: typed, variable-length codesign env. The agent (owner of the morphology
generator) hands per-env DESIGNS — effector counts, per-depth effector TYPES, and a terminal CAP
type per limb — and the env builds those bodies at each resample window. count 0 = limb absent
(presence emergent). One body per env (EPM=1). The 32-slot depth-major obs/DOF repack lives in the
base AntMultiMorphEnv. See ADR-0010; Phase-5 vocabulary in transformer_rl/vocab.py.
"""
import torch

from .ant_multimorph import AntMultiMorphEnv, _N_LIMBS
from .build_vsim import Morphology, designs_from_arrays


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


class AntCodesignEnv(AntMultiMorphEnv):
    def __init__(self, num_envs, device, *, base_legs=(1, 4, 6), **kwargs):
        self._base_legs = frozenset(base_legs)
        self._next_bodies = None                   # list[(eff_types, cap_types)], len total_num_envs
        kwargs.setdefault("sample_morphs", True)    # unlocks resample(); _draw_morphs is overridden
        super().__init__(num_envs, device, **kwargs)

    def set_next(self, counts, eff_sub, cap_sub):
        """Store the next window's per-env design. Effective at the next rebuild.
          counts  (N, 8)          effectors per limb (0 = limb absent), column j -> limb j+1
          eff_sub (N, 8, max_len) effector subtype per depth (only [:count] is read)
          cap_sub (N, 8)          cap subtype per limb (-1 -> the canonical bare cap)"""
        self._next_bodies = designs_from_arrays(_np(counts), _np(eff_sub), _np(cap_sub), _N_LIMBS)

    def _draw_morphs(self, num):
        if self._next_bodies is None:               # initial build (pre-set_next): canonical base body
            return [Morphology.from_counts({n: 2 for n in self._base_legs}) for _ in range(num)]
        assert len(self._next_bodies) == num, \
            f"set_next gave {len(self._next_bodies)} bodies but env has {num} envs"
        return [Morphology.from_design(eff, caps) for eff, caps in self._next_bodies]
