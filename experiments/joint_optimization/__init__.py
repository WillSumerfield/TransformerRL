"""Joint-optimization toy: how a designer's search and a predictor's fidelity trade off.

  env.py             f, g, evaluate(), grid(), JointOptimizationEnv
  plot_landscape.py  3D surfaces of raw f, L1 (dead zone shaded) and L2
"""

from .env import (
    BOUNDS,
    DEFAULT_C,
    DEFAULT_SIGMA,
    JointOptimizationEnv,
    evaluate,
    f,
    g,
    grid,
)

__all__ = [
    "BOUNDS",
    "DEFAULT_C",
    "DEFAULT_SIGMA",
    "JointOptimizationEnv",
    "evaluate",
    "f",
    "g",
    "grid",
]
