"""AntCodesignEnv: the agent (owner of the morphology generator) hands it the per-env presence to
build each resample window; the env builds REAL variable-limb bodies and reports.

Discrete presence: unlike the retired value-ascent AntCodesignEnv (ValueGradGenerator branch) there
is NO continuous obs_p signal and NO binary-limb stub -- a removed limb is genuinely absent, so the
dof-mask + lengths in obs reflect it for free (no allocate_buffers override). One body per env
(EPM=1, 4096 groups). See ADR-0010 / temp/ppg_phase2_plan.md.
"""
import torch

from .ant_multimorph import AntMultiMorphEnv, _N_LIMBS
from .build_vsim import Morphology


class AntCodesignEnv(AntMultiMorphEnv):
    def __init__(self, num_envs, device, *, base_legs=(1, 4, 6), **kwargs):
        self._base_legs = frozenset(base_legs)
        self._next_bodies = None                  # list[frozenset] len total_num_envs, agent-set
        kwargs.setdefault("sample_morphs", True)   # unlocks resample(); _draw_morphs is overridden
        super().__init__(num_envs, device, **kwargs)

    def set_next(self, presence):
        """Store per-env presence (N,8) {0,1}; column j -> limb j+1. Effective at next rebuild."""
        p = presence.detach().cpu().numpy() if torch.is_tensor(presence) else presence
        bodies = [frozenset(j + 1 for j in range(_N_LIMBS) if row[j]) for row in p]
        assert all(b for b in bodies), "0-limb body; generator must guarantee >=1 active limb"
        self._next_bodies = bodies

    def _draw_morphs(self, num):
        if self._next_bodies is None:              # initial build (pre-set_next): base everywhere
            self._next_bodies = [self._base_legs] * num
        assert len(self._next_bodies) == num, \
            f"set_next gave {len(self._next_bodies)} bodies but env has {num} envs"
        return [Morphology.from_legs(b) for b in self._next_bodies]
