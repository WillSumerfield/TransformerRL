"""Standard vsim ant (no masking)."""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "vlearn-main" / "train"))

from envs.ant_environment_gpu import AntEnvironmentGpu


class AntEnv(AntEnvironmentGpu):
    """Standard 4-leg vsim ant, no limb masking."""

    _VALID_KWARGS = frozenset(inspect.signature(AntEnvironmentGpu.__init__).parameters) - {"self"}

    def __init__(self, num_envs: int, device, **kwargs):
        super().__init__(num_envs, device, **{k: v for k, v in kwargs.items() if k in self._VALID_KWARGS})

    @property
    def unwrapped(self):
        return self
