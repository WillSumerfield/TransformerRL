"""AntAdaptiveEnv: codesign env restricted to 3-4 leg stable morphologies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "vlearn-main" / "train"))

import torch

from .ant_codesign import AntCodesignEnv

_MORPHOLOGIES = [
    frozenset({1, 3, 6}),    # 3-leg: gaps 90°/135°/135°
    frozenset({2, 4, 6, 8}), # 4-leg: symmetric 90° spacing
]


class AntAdaptiveEnv(AntCodesignEnv):
    """AntCodesignEnv with one 3-leg and one 4-leg morphology."""

    def __init__(self, num_envs: int, device: torch.device, **kwargs):
        super().__init__(num_envs, device, morphologies=_MORPHOLOGIES, **kwargs)
