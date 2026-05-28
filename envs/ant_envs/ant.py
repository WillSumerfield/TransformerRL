"""Standard vsim ant (no masking)."""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "vlearn-main" / "train"))

import vlearn as v
from envs.ant_environment_gpu import AntEnvironmentGpu


class AntEnv(AntEnvironmentGpu):
    """Standard 4-leg vsim ant, no limb masking."""

    _VALID_KWARGS = frozenset(inspect.signature(AntEnvironmentGpu.__init__).parameters) - {"self"}

    def __init__(self, num_envs: int, device, **kwargs):
        kwargs.setdefault("spacing", 3.0)
        super().__init__(num_envs, device, **{k: v for k, v in kwargs.items() if k in self._VALID_KWARGS})

    @property
    def unwrapped(self):
        return self

    # ---- Follow-camera interface ----------------------------------------------

    def follow_sets(self) -> list[list[int]]:
        return [list(range(self.num_envs))]

    def follow_world_pos(self, idx: int) -> v.Vec3:
        p = self.kine_handler.get_link_pose_buf[idx, 4:7]
        local = v.Vec3(float(p[0]), float(p[1]), float(p[2]))
        return self.env_transforms[idx].transform(local)
