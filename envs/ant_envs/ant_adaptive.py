"""AntAdaptiveEnv: multi-morphology env over all stable 3-4 leg morphologies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "vlearn-main" / "train"))

import torch

from .ant_multimorph import AntMultiMorphEnv, _stable_morphologies

# All stable 3- and 4-leg morphologies (no circular gap > 135°). 46 total: 8 three-leg, 38 four-leg.
_MORPHOLOGIES = _stable_morphologies(min_legs=3, max_legs=4, max_gap_deg=135.0)


class AntAdaptiveEnv(AntMultiMorphEnv):
    """AntMultiMorphEnv over all stable 3-4 leg morphologies (circular gap <= 135 deg)."""

    def __init__(self, num_envs: int, device: torch.device, **kwargs):
        super().__init__(num_envs, device, morphologies=_MORPHOLOGIES, **kwargs)
