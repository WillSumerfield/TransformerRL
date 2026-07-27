from __future__ import annotations

import math
import unittest

import numpy as np

from benchmarks.data import EpisodeResults, EvaluationPairs
from benchmarks.metrics import body_tokens, calculate_metrics, diversity_metrics


def _pairs() -> EvaluationPairs:
    counts = np.zeros((4, 8), dtype=np.int64)
    counts[:, 0] = [1, 1, 2, 1]
    eff = np.full((4, 8, 4), -1, dtype=np.int64)
    eff[:, 0, 0] = [0, 0, 0, 2]
    eff[2, 0, 1] = 1
    cap = np.zeros((4, 8), dtype=np.int64)
    cap[2, 0] = 1
    return EvaluationPairs(
        counts=counts,
        eff_sub=eff,
        cap_sub=cap,
        controller_ids=np.asarray(["a", "a", "a", "a"]),
        weights=np.full(4, 0.25),
    )


class BenchmarkMetricTests(unittest.TestCase):
    def test_canonical_encoding_preserves_typed_structure(self) -> None:
        tokens = body_tokens(_pairs())
        self.assertEqual(tokens.shape, (4, 32))
        np.testing.assert_array_equal(tokens[0], tokens[1])
        self.assertFalse(np.array_equal(tokens[0], tokens[2]))
        self.assertFalse(np.array_equal(tokens[0], tokens[3]))

    def test_distribution_metrics_cover_breadth_concentration_and_distance(self) -> None:
        metrics = diversity_metrics(_pairs())
        self.assertAlmostEqual(metrics["benchmark/diversity/unique_fraction"], 0.75)
        expected_entropy = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))
        self.assertAlmostEqual(
            metrics["benchmark/diversity/entropy_nats"], expected_entropy
        )
        self.assertAlmostEqual(
            metrics["benchmark/diversity/effective_body_count"],
            math.exp(expected_entropy),
        )
        self.assertGreater(metrics["benchmark/diversity/typed_token_distance"], 0)
        self.assertLessEqual(metrics["benchmark/diversity/typed_token_distance"], 1)

    def test_native_pair_summary_keeps_sample_frequency_and_unique_mean_separate(self) -> None:
        episodes = EpisodeResults(
            returns=np.asarray([[1, 3], [3, 5], [9, 11], [5, 7]], dtype=float),
            falls=np.asarray([[0, 0], [0, 1], [1, 1], [0, 0]], dtype=float),
            lengths=np.asarray([[10, 10], [10, 8], [4, 4], [9, 9]], dtype=float),
        )
        summary = calculate_metrics(_pairs(), episodes, _pairs(), top_k=2)
        # Per-pair means are [2, 4, 10, 6]; the first two rows are one body.
        self.assertAlmostEqual(summary["benchmark/return/expected"], 5.5)
        self.assertAlmostEqual(
            summary["benchmark/return/unique_body_mean"],
            np.mean([3.0, 10.0, 6.0]),
        )
        self.assertAlmostEqual(summary["benchmark/selection/top1_of_m"], 10)
        self.assertAlmostEqual(summary["benchmark/selection/topk_of_m"], 8)

    def test_invalid_active_effector_subtype_is_rejected(self) -> None:
        pairs = _pairs()
        eff = pairs.eff_sub.copy()
        eff[0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "active effector"):
            EvaluationPairs(
                counts=pairs.counts,
                eff_sub=eff,
                cap_sub=pairs.cap_sub,
                controller_ids=pairs.controller_ids,
                weights=pairs.weights,
            )


if __name__ == "__main__":
    unittest.main()
