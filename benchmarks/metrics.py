"""Cross-method metrics, kept separate so their definitions are easy to audit."""
from __future__ import annotations

from collections import Counter

import numpy as np

from transformer_rl.vocab import N_EFF

from .data import EpisodeResults, EvaluationPairs


def body_tokens(pairs: EvaluationPairs) -> np.ndarray:
    """Encode each body as fixed typed tokens; zero means absent/padding."""
    tokens = np.zeros(
        (pairs.size, 8, pairs.max_limb_length),
        dtype=np.int16,
    )
    for row in range(pairs.size):
        for limb in range(8):
            count = int(pairs.counts[row, limb])
            if count == 0:
                continue
            tokens[row, limb, :count] = 1 + pairs.eff_sub[row, limb, :count]
            tokens[row, limb, count] = 1 + N_EFF + pairs.cap_sub[row, limb]
    return tokens.reshape(pairs.size, -1)


def _body_keys(pairs: EvaluationPairs) -> list[bytes]:
    return [row.tobytes() for row in np.ascontiguousarray(body_tokens(pairs))]


def diversity_metrics(pairs: EvaluationPairs) -> dict[str, float]:
    """Measure support breadth, concentration and structural spread."""
    frequency = np.asarray(
        list(Counter(_body_keys(pairs)).values()),
        dtype=np.float64,
    )
    probability = frequency / frequency.sum()
    entropy = float(-(probability * np.log(probability)).sum())

    encoded = body_tokens(pairs)
    unique, counts = np.unique(encoded, axis=0, return_counts=True)
    weighted_distance = 0.0
    for index in range(len(unique) - 1):
        distances = (unique[index + 1 :] != unique[index]).mean(axis=1)
        weighted_distance += float(
            (distances * counts[index] * counts[index + 1 :]).sum()
        )
    pairings = pairs.size * (pairs.size - 1) / 2

    return {
        "benchmark/diversity/unique_fraction": float(len(unique) / pairs.size),
        "benchmark/diversity/entropy_nats": entropy,
        "benchmark/diversity/effective_body_count": float(np.exp(entropy)),
        "benchmark/diversity/typed_token_distance": (
            weighted_distance / pairings if pairings else 0.0
        ),
    }


def calculate_metrics(
    pairs: EvaluationPairs,
    episodes: EpisodeResults,
    diversity_pairs: EvaluationPairs,
    *,
    top_k: int,
) -> dict[str, float]:
    """Calculate the canonical performance, stability and diversity summary."""
    if episodes.returns.shape[0] != pairs.size:
        raise ValueError("episode rows must match evaluated pairs")
    if not 1 <= top_k <= pairs.size:
        raise ValueError("top_k must lie between one and the pair count")

    pair_returns = episodes.returns.mean(axis=1)
    pair_falls = episodes.falls.mean(axis=1)
    pair_lengths = episodes.lengths.mean(axis=1)

    returns_by_body: dict[bytes, list[float]] = {}
    for key, value in zip(_body_keys(pairs), pair_returns):
        returns_by_body.setdefault(key, []).append(float(value))
    unique_body_mean = np.mean(
        [np.mean(values) for values in returns_by_body.values()]
    )
    best_first = np.sort(pair_returns)[::-1]

    metrics = {
        "benchmark/return/expected": float(np.sum(pair_returns * pairs.weights)),
        "benchmark/return/unique_body_mean": float(unique_body_mean),
        "benchmark/return/pair_std": float(pair_returns.std()),
        "benchmark/selection/top1_of_m": float(best_first[0]),
        "benchmark/selection/topk_of_m": float(best_first[:top_k].mean()),
        "benchmark/stability/fall_rate": float(np.sum(pair_falls * pairs.weights)),
        "benchmark/stability/episode_length": float(
            np.sum(pair_lengths * pairs.weights)
        ),
    }
    metrics.update(diversity_metrics(diversity_pairs))
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("benchmark metrics contain a non-finite value")
    return metrics

