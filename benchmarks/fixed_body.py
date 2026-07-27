"""Fixed canonical body baseline for the shared benchmark evaluator."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from transformer_rl.vocab import CAP_BARE, EFF_KNEE, EFF_SWING

from .codesign import CodesignMethod, load_saved_controller
from .data import EvaluationPairs


class FixedBodyMethod(CodesignMethod):
    """One trained controller paired with one unchanging canonical body."""

    name = "fixed_body"

    def _canonical_pairs(self, count: int) -> EvaluationPairs:
        base_legs = self.run_config.get("env", {}).get("base_legs", (1, 4, 6))
        max_length = int(self.network.net.max_limb_length)
        if max_length < 3:
            raise ValueError("the canonical body needs two effectors and one cap per limb")

        counts = np.zeros((count, 8), dtype=np.int64)
        effectors = np.full((count, 8, max_length), -1, dtype=np.int64)
        for limb in base_legs:
            counts[:, int(limb) - 1] = 2
            effectors[:, int(limb) - 1, 0] = EFF_SWING
            effectors[:, int(limb) - 1, 1] = EFF_KNEE

        return EvaluationPairs(
            counts=counts,
            eff_sub=effectors,
            cap_sub=np.full((count, 8), CAP_BARE, dtype=np.int64),
            controller_ids=np.full(count, self.checkpoint_path.name),
            weights=np.full(count, 1.0 / count),
        )

    def sample_pairs(self, count: int, seed: int) -> EvaluationPairs:
        """Repeat the single native pair for parallel rollout estimation."""
        del seed  # A fixed body has no morphology-sampling randomness.
        return self._canonical_pairs(count)

    def sample_designs(self, count: int, seed: int) -> EvaluationPairs:
        """Repeat the same body so shared diversity metrics correctly report one."""
        del seed
        return self._canonical_pairs(count)

    def install_pairs(self, environment: Any, pairs: EvaluationPairs) -> None:
        """The environment is already built as this body; guard against drift."""
        expected = self._canonical_pairs(pairs.size)
        if not (
            np.array_equal(pairs.counts, expected.counts)
            and np.array_equal(pairs.eff_sub, expected.eff_sub)
            and np.array_equal(pairs.cap_sub, expected.cap_sub)
        ):
            raise ValueError("fixed-body evaluation received a non-canonical body")


def load_fixed_body(
    config: dict[str, Any],
    device: torch.device,
) -> FixedBodyMethod:
    """Load a saved fixed-body controller through the shared CoDesign loader."""
    return load_saved_controller(config, device, FixedBodyMethod)
