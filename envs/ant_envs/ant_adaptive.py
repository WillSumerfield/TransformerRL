"""AntAdaptiveEnv: codesign env restricted to 3-4 leg stable morphologies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "vlearn-main" / "train"))

import torch

from .ant_codesign import AntCodesignEnv, _stable_morphologies

_MORPHOLOGIES = _stable_morphologies(min_legs=3, max_legs=4)


class AntAdaptiveEnv(AntCodesignEnv):
    """AntCodesignEnv with only 3-4 leg stable morphologies (46 total)."""

    def __init__(self, num_envs: int, device: torch.device, **kwargs):
        super().__init__(num_envs, device, morphologies=_MORPHOLOGIES, **kwargs)
