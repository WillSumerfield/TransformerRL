"""Uniform-action morphology control for the shared benchmark evaluator."""
from __future__ import annotations

from typing import Any

import torch

from .codesign import CodesignMethod, load_saved_controller


class UniformActionMethod(CodesignMethod):
    """Pair a trained controller with bodies drawn by a uniform grammar policy."""

    name = "uniform_action"
    sampling_mode = "uniform"


def load_uniform_action(
    config: dict[str, Any],
    device: torch.device,
) -> UniformActionMethod:
    """Load a uniform-action controller through the shared CoDesign loader."""
    return load_saved_controller(config, device, UniformActionMethod)
