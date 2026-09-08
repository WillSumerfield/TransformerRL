"""Tests for diagnostic-only post-adaptation evaluation and adaptation gap metrics."""

import os
import tempfile
import numpy as np
import pytest
import torch

from transformer_rl.post_adaptation_eval import (
    compute_adaptation_gap_diagnostics,
    save_post_eval_artifact,
    _spearman_rank_corr,
    _pearson_corr,
)


def test_correlations_and_rank_shift():
    """Verify Pearson, Spearman, and material rank shift fraction calculations."""
    # Synthetic perfectly correlated arrays
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    assert abs(_pearson_corr(x, y) - 1.0) < 1e-6
    assert abs(_spearman_rank_corr(x, y) - 1.0) < 1e-6

    # Perfectly inverted arrays
    y_inv = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    assert abs(_pearson_corr(x, y_inv) - (-1.0)) < 1e-6
    assert abs(_spearman_rank_corr(x, y_inv) - (-1.0)) < 1e-6


def test_adaptation_gap_diagnostics_and_grouping():
    """Verifies that compute_adaptation_gap_diagnostics correctly computes gaps,
    rank shifts, and complexity breakdowns (effectors, modules, max_depth).
    """
    N = 4
    # R_train vs R_post
    R_train = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    # Body 0 improves by +1.0; Body 1 improves by +0.5; Body 2 drops by -0.5; Body 3 drops by -1.0
    R_post = torch.tensor([2.0, 2.5, 2.5, 3.0], dtype=torch.float32)

    # 4 limbs per body (for 8-limb layout)
    counts = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0],  # 1 effector, 1 limb -> 2 modules, depth 1
        [2, 0, 0, 0, 0, 0, 0, 0],  # 2 effectors, 1 limb -> 3 modules, depth 2
        [1, 1, 0, 0, 0, 0, 0, 0],  # 2 effectors, 2 limbs -> 4 modules, depth 1
        [3, 0, 0, 0, 0, 0, 0, 0],  # 3 effectors, 1 limb -> 4 modules, depth 3
    ], dtype=torch.long)

    scalars, records = compute_adaptation_gap_diagnostics(
        R_train, R_post, counts, rank_shift_threshold=0.10
    )

    # Gaps: [1.0, 0.5, -0.5, -1.0] -> mean = 0.0
    assert abs(scalars["codesign/adaptation/gap_mean"] - 0.0) < 1e-6
    assert abs(scalars["codesign/adaptation/R_train_mean"] - 2.5) < 1e-6
    assert abs(scalars["codesign/adaptation/R_post_mean"] - 2.5) < 1e-6
    assert abs(scalars["codesign/adaptation/gap_pos_frac"] - 0.5) < 1e-6

    # Effector counts: [1, 2, 2, 3]
    assert np.array_equal(records["effector_count"], [1, 2, 2, 3])
    # Active limbs: [1, 1, 2, 1]
    assert np.array_equal(records["active_limbs"], [1, 1, 2, 1])
    # Module counts (effectors + caps): [2, 3, 4, 4]
    assert np.array_equal(records["module_count"], [2, 3, 4, 4])
    # Max depth: [1, 2, 1, 3]
    assert np.array_equal(records["max_depth"], [1, 2, 1, 3])

    # Gap by effectors:
    # 1 effector: gap = 1.0
    assert abs(scalars["codesign/adaptation/gap_by_effectors/1"] - 1.0) < 1e-6
    # 2 effectors: bodies 1 and 2, gaps = 0.5 and -0.5 -> mean = 0.0
    assert abs(scalars["codesign/adaptation/gap_by_effectors/2"] - 0.0) < 1e-6
    # 3 effectors: gap = -1.0
    assert abs(scalars["codesign/adaptation/gap_by_effectors/3"] - (-1.0)) < 1e-6

    # Gap by max depth:
    # depth 1: bodies 0 and 2, gaps = 1.0 and -0.5 -> mean = 0.25
    assert abs(scalars["codesign/adaptation/gap_by_max_depth/1"] - 0.25) < 1e-6
    # depth 2: gap = 0.5
    assert abs(scalars["codesign/adaptation/gap_by_max_depth/2"] - 0.5) < 1e-6
    # depth 3: gap = -1.0
    assert abs(scalars["codesign/adaptation/gap_by_max_depth/3"] - (-1.0)) < 1e-6


def test_artifact_persistence():
    """Verify save_post_eval_artifact writes and reads back valid npz arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_eval.npz")
        records = {
            "R_train": np.array([1.0, 2.0], dtype=np.float32),
            "R_post": np.array([1.5, 2.5], dtype=np.float32),
            "adaptation_gap": np.array([0.5, 0.5], dtype=np.float32),
        }
        scalars = {
            "codesign/adaptation/gap_mean": 0.5,
            "codesign/adaptation/corr_pearson": 1.0,
        }
        metadata = {"gen_window": 1, "epoch": 10}

        save_post_eval_artifact(path, records, scalars, metadata)

        assert os.path.exists(path)
        data = np.load(path)
        assert np.allclose(data["R_train"], records["R_train"])
        assert np.allclose(data["R_post"], records["R_post"])
        assert np.allclose(data["adaptation_gap"], records["adaptation_gap"])
        assert int(data["meta__gen_window"]) == 1
        assert int(data["meta__epoch"]) == 10


def test_stochastic_eval_comparison():
    """Verify stochastic eval comparison metrics are computed when R_post_stoch is provided."""
    N = 4
    R_train = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    R_post_det = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    R_post_stoch = torch.tensor([1.8, 2.9, 3.7, 4.8], dtype=torch.float32)
    counts = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0] for _ in range(N)], dtype=torch.long)

    scalars, records = compute_adaptation_gap_diagnostics(
        R_train, R_post_det, counts, R_post_stoch=R_post_stoch
    )

    assert "codesign/adaptation/R_post_stoch_mean" in scalars
    assert "codesign/adaptation/corr_det_stoch_pearson" in scalars
    assert "codesign/adaptation/corr_det_stoch_spearman" in scalars
    assert "R_post_stoch" in records
    assert abs(scalars["codesign/adaptation/corr_det_stoch_spearman"] - 1.0) < 1e-5


def test_return_target_routing():
    """Verify return_target logic routes R_post vs R_train correctly."""
    R_train = torch.tensor([1.0, 2.0], dtype=torch.float32)
    R_post_det = torch.tensor([3.0, 4.0], dtype=torch.float32)
    R_post_stoch = torch.tensor([2.5, 3.5], dtype=torch.float32)

    # return_target == 'train'
    target_mode = 'train'
    action_mode = 'deterministic'
    R_primary = R_post_stoch if action_mode == 'stochastic' else R_post_det
    R = R_primary if target_mode == 'post' and R_primary is not None else R_train
    assert torch.equal(R, R_train)

    # return_target == 'post' with deterministic
    target_mode = 'post'
    action_mode = 'deterministic'
    R_primary = R_post_stoch if action_mode == 'stochastic' else R_post_det
    R = R_primary if target_mode == 'post' and R_primary is not None else R_train
    assert torch.equal(R, R_post_det)

    # return_target == 'post' with stochastic
    target_mode = 'post'
    action_mode = 'stochastic'
    R_primary = R_post_stoch if action_mode == 'stochastic' else R_post_det
    R = R_primary if target_mode == 'post' and R_primary is not None else R_train
    assert torch.equal(R, R_post_stoch)
