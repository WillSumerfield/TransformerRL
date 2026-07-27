"""Small, explicit data containers shared by benchmark methods."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transformer_rl.vocab import N_CAP, N_EFF


@dataclass
class EvaluationPairs:
    """Morphology-controller pairs sampled from one trained method.

    ``counts`` is the number of effector modules in each of eight limbs.
    ``eff_sub`` stores their types and ``cap_sub`` stores each terminal cap.
    Repeated rows are intentional: they represent probability mass in a
    method's native output distribution.
    """

    counts: np.ndarray
    eff_sub: np.ndarray
    cap_sub: np.ndarray
    controller_ids: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        self.counts = np.asarray(self.counts, dtype=np.int64)
        self.eff_sub = np.asarray(self.eff_sub, dtype=np.int64)
        self.cap_sub = np.asarray(self.cap_sub, dtype=np.int64)
        self.controller_ids = np.asarray(self.controller_ids, dtype=np.str_)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self._validate()

    @property
    def size(self) -> int:
        return int(self.counts.shape[0])

    @property
    def max_limb_length(self) -> int:
        return int(self.eff_sub.shape[2])

    def _validate(self) -> None:
        if self.counts.ndim != 2 or self.counts.shape[1] != 8:
            raise ValueError(f"counts must have shape (pairs, 8), got {self.counts.shape}")
        pair_count = self.counts.shape[0]
        if pair_count < 1:
            raise ValueError("at least one evaluation pair is required")
        if self.eff_sub.ndim != 3 or self.eff_sub.shape[:2] != (pair_count, 8):
            raise ValueError(
                "eff_sub must have shape (pairs, 8, max_limb_length), "
                f"got {self.eff_sub.shape}"
            )
        if self.cap_sub.shape != (pair_count, 8):
            raise ValueError(f"cap_sub must have shape {(pair_count, 8)}")
        if self.controller_ids.shape != (pair_count,):
            raise ValueError("controller_ids must contain one entry per pair")
        if self.weights.shape != (pair_count,):
            raise ValueError("weights must contain one entry per pair")

        if np.any(self.counts < 0) or np.any(self.counts >= self.max_limb_length):
            raise ValueError("limb counts must leave one grammar slot for the cap")
        if np.any(self.counts.sum(axis=1) < 1):
            raise ValueError("every morphology needs at least one effector")
        if np.any((self.cap_sub < 0) | (self.cap_sub >= N_CAP)):
            raise ValueError(f"cap types must lie in [0, {N_CAP})")

        active = (
            np.arange(self.max_limb_length)[None, None, :]
            < self.counts[:, :, None]
        )
        active_effectors = self.eff_sub[active]
        if np.any((active_effectors < 0) | (active_effectors >= N_EFF)):
            raise ValueError(f"active effector types must lie in [0, {N_EFF})")

        if np.any(~np.isfinite(self.weights)) or np.any(self.weights < 0):
            raise ValueError("weights must be finite and non-negative")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("weights must sum to one")


@dataclass
class EpisodeResults:
    """Every completed episode outcome, arranged as ``[pair, episode]``."""

    returns: np.ndarray
    falls: np.ndarray
    lengths: np.ndarray
    start_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.returns = np.asarray(self.returns, dtype=np.float64)
        self.falls = np.asarray(self.falls, dtype=np.float64)
        self.lengths = np.asarray(self.lengths, dtype=np.float64)
        if self.start_values is not None:
            self.start_values = np.asarray(self.start_values, dtype=np.float64)

        if self.returns.ndim != 2:
            raise ValueError("episode returns must have shape (pairs, episodes)")
        if self.falls.shape != self.returns.shape or self.lengths.shape != self.returns.shape:
            raise ValueError("returns, falls and lengths must have the same shape")
        if self.start_values is not None and self.start_values.shape != self.returns.shape:
            raise ValueError("start values must have the same shape as returns")
        if not all(
            np.isfinite(values).all()
            for values in (self.returns, self.falls, self.lengths)
        ):
            raise ValueError("episode results contain non-finite values")

