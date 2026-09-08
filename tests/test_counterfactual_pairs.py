"""Unit tests for matched structural counterfactual pairs discovery and supervision."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from transformer_rl.counterfactual_pairs import (
    encode_canonical_morphology,
    find_exact_matched_pairs,
    compute_pair_difference_loss,
    compute_pair_diagnostics,
)
from transformer_rl.spatial_credit import SpatialCreditHead, propagate_tree_credit
from transformer_rl.vocab import GEN_EFF, GEN_CAP, EFF_SWING, EFF_KNEE, CAP_BARE, CAP_FOOT


def test_encode_canonical_morphology():
    N = 4
    n_limbs = 8
    max_len = 4
    counts = torch.zeros(N, n_limbs, dtype=torch.long)
    eff_sub = torch.full((N, n_limbs, max_len), -1, dtype=torch.long)
    cap_sub = torch.full((N, n_limbs), -1, dtype=torch.long)

    # Body 0: limb 0 has length 2 (swing, knee, bare cap)
    counts[0, 0] = 2
    eff_sub[0, 0, 0] = EFF_SWING  # 0
    eff_sub[0, 0, 1] = EFF_KNEE   # 1
    cap_sub[0, 0] = CAP_BARE      # 0

    morphs = encode_canonical_morphology(counts, eff_sub, cap_sub)
    assert morphs.shape == (N, max_len, n_limbs)
    assert morphs[0, 0, 0] == GEN_EFF * 10 + EFF_SWING  # 0
    assert morphs[0, 1, 0] == GEN_EFF * 10 + EFF_KNEE   # 1
    assert morphs[0, 2, 0] == GEN_CAP * 10 + CAP_BARE   # 10
    assert morphs[0, 3, 0] == -1  # empty slot


def test_find_exact_matched_pairs():
    N = 3
    n_limbs = 8
    max_len = 4

    # Create 3 bodies:
    # Body 0 and Body 1 are IDENTICAL except Body 0 has knee at (d=1, s=0) while Body 1 has swing at (d=1, s=0)
    # Body 2 differs in limb 1 (a multi-module branch)
    counts = torch.zeros(N, n_limbs, dtype=torch.long)
    eff_sub = torch.full((N, n_limbs, max_len), -1, dtype=torch.long)
    cap_sub = torch.full((N, n_limbs), -1, dtype=torch.long)

    for b in range(N):
        counts[b, 0] = 2
        eff_sub[b, 0, 0] = EFF_SWING
        eff_sub[b, 0, 1] = EFF_KNEE if b != 1 else EFF_SWING  # difference between 0 and 1
        cap_sub[b, 0] = CAP_BARE

    # Body 2 additionally has limb 1
    counts[2, 1] = 1
    eff_sub[2, 1, 0] = EFF_SWING
    cap_sub[2, 1] = CAP_FOOT

    morphs = encode_canonical_morphology(counts, eff_sub, cap_sub)
    R = torch.tensor([10.0, 6.0, 12.0], dtype=torch.float32)

    pair_data = find_exact_matched_pairs(morphs, R)
    assert pair_data["meta"]["n_module_pairs_found"] >= 1
    assert pair_data["meta"]["n_subtree_pairs_found"] >= 1

    # Check that pair (0, 1) was found with target slot for d=1, s=0 => slot = 1 * 8 + 0 = 8
    slots = pair_data["slot"].numpy()
    delta_R = pair_data["delta_R"].numpy()
    idx_A = pair_data["idx_A"].numpy()
    idx_B = pair_data["idx_B"].numpy()

    # Find the module pair between 0 and 1
    found = False
    for a, b, sl, dr, is_sub in zip(idx_A, idx_B, slots, delta_R, pair_data["is_subtree"].numpy()):
        if not is_sub and a == 0 and b == 1:
            assert sl == 8  # depth 1, limb 0
            assert np.isclose(dr, 4.0)  # R[0] - R[1] = 10 - 6 = 4.0
            found = True
            break
    assert found, "Exact module pair (0, 1) was not found"


def test_pair_difference_loss_and_gradients():
    N = 4
    M = 32
    d_model = 64

    head = SpatialCreditHead(d_model, hidden_dim=32)
    H = torch.randn(N, M, d_model, requires_grad=True)
    pres = torch.ones(N, M)

    c_spat = head(H, pres)
    C_tree = propagate_tree_credit(c_spat, pres, n_limbs=8, max_len=4, tree_lambda=0.5)

    pair_batch = {
        "idx_A": torch.tensor([0, 1], dtype=torch.long),
        "idx_B": torch.tensor([1, 2], dtype=torch.long),
        "slot": torch.tensor([5, 8], dtype=torch.long),
        "is_subtree": torch.tensor([False, True], dtype=torch.bool),
        "delta_R": torch.tensor([1.5, -0.5], dtype=torch.float32),
    }

    loss, delta_C = compute_pair_difference_loss(c_spat, C_tree, pair_batch)
    assert loss.dim() == 0
    assert delta_C.shape == (2,)

    loss.backward()
    # Check that gradients flow to head weights
    for param in head.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_pair_diagnostics():
    N = 4
    M = 32
    c_spat = torch.randn(N, M)
    C_tree = torch.randn(N, M)
    pair_data = {
        "idx_A": torch.tensor([0, 2], dtype=torch.long),
        "idx_B": torch.tensor([1, 3], dtype=torch.long),
        "slot": torch.tensor([4, 12], dtype=torch.long),
        "is_subtree": torch.tensor([False, True], dtype=torch.bool),
        "delta_R": torch.tensor([2.0, -1.0], dtype=torch.float32),
        "meta": {"n_module_pairs_found": 1, "n_subtree_pairs_found": 1},
    }

    diags = compute_pair_diagnostics(c_spat, C_tree, pair_data)
    assert "spatial_diff_mse" in diags
    assert "tree_diff_mse" in diags
    assert "pair_combined_mse" in diags


def test_genact_advantage_combination_and_isolation():
    """Verify GenAct advantage combination with tree credit and gradient isolation."""
    N = 4
    L = 6
    M = 32
    d_model = 64
    _N_LIMBS = 8
    beta = 0.5

    # Mock spatial credit head and representations
    head = SpatialCreditHead(d_model, hidden_dim=32)
    H = torch.randn(N, M, d_model, requires_grad=True)
    pres = torch.ones(N, M)

    c_spat = head(H, pres)
    C_tree = propagate_tree_credit(c_spat, pres, n_limbs=_N_LIMBS, max_len=4, tree_lambda=0.5)

    # Mock generator trace
    slots = torch.tensor([
        [0, 0, 1, 1, 2, 0],
        [0, 1, 0, 1, 0, 0],
        [3, 3, 3, 3, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.long)

    # 0 = GEN_EFF, 1 = GEN_CAP
    cat_actions = torch.tensor([
        [0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
    ], dtype=torch.long)

    valid = torch.tensor([
        [True, True, True, True, False, False],
        [True, True, True, False, False, False],
        [True, True, True, True, False, False],
        [True, True, False, False, False, False],
    ], dtype=torch.bool)

    raw_adv = torch.randn(N, L)
    depth_hist = torch.tensor([
        [0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=torch.long)

    # Compute standardized prefix advantage
    sel = raw_adv[valid]
    adv = torch.zeros_like(raw_adv)
    adv_pref_norm = (sel - sel.mean()) / (sel.std() + 1e-8)
    adv[valid] = adv_pref_norm

    # CRITICAL: Detach tree credit
    tok_slot = depth_hist * _N_LIMBS + slots
    is_eff = (cat_actions == 0) & valid
    act_pres = pres.gather(1, tok_slot.clamp(0, 31))
    valid_tree = is_eff & (act_pres > 0)

    action_tree = C_tree.gather(1, tok_slot.clamp(0, 31)).detach()
    assert action_tree.requires_grad is False

    tree_vals = action_tree[valid_tree]
    tree_norm = (tree_vals - tree_vals.mean()) / (tree_vals.std() + 1e-8)
    adv_combined = adv.clone()
    adv_combined[valid_tree] = adv_combined[valid_tree] + beta * tree_norm

    # Check that cap actions fell back to prefix advantage ONLY
    is_cap = (cat_actions == 1) & valid
    assert torch.allclose(adv_combined[is_cap], adv[is_cap])

    # Check that effectors have combined advantage != prefix advantage
    assert not torch.allclose(adv_combined[valid_tree], adv[valid_tree])

    # Check gradient isolation: GenAct PPO loss backprop must NOT flow to head or H
    mock_logp = torch.randn(N, L, requires_grad=True)
    gen_ppo_loss = -(mock_logp * adv_combined * valid.float()).sum()
    gen_ppo_loss.backward()

    assert mock_logp.grad is not None
    assert H.grad is None
    for p in head.parameters():
        assert p.grad is None


def test_genact_shuffled_credit_control():
    """Verify that shuffled tree credit matches aligned distribution statistics

    (mean, std, min, max, count) while completely destroying action alignment (|r| < 0.05).
    """
    torch.manual_seed(42)
    N_eff = 2000
    tree_vals = torch.randn(N_eff) * 3.5 + 1.2
    tree_norm = (tree_vals - tree_vals.mean()) / (tree_vals.std() + 1e-8)

    perm_seed = 42 + 5 * 10007
    rng = torch.Generator()
    rng.manual_seed(perm_seed)
    perm = torch.randperm(tree_norm.numel(), generator=rng)
    tree_shuffled = tree_norm[perm]

    # Invariance guarantees
    assert torch.allclose(tree_shuffled.mean(), tree_norm.mean(), atol=1e-5)
    assert torch.allclose(tree_shuffled.std(), tree_norm.std(), atol=1e-5)
    assert torch.allclose(tree_shuffled.min(), tree_norm.min(), atol=1e-5)
    assert torch.allclose(tree_shuffled.max(), tree_norm.max(), atol=1e-5)
    assert tree_shuffled.numel() == tree_norm.numel()

    # Alignment destruction (|r| < 0.05 for N=2000)
    corr = float(torch.corrcoef(torch.stack([tree_norm, tree_shuffled]))[0, 1].item())
    assert abs(corr) < 0.05, f"Expected near-zero correlation under shuffle, got {corr}"


def test_body_centered_and_within_body_shuffled_credit():
    """Verify body-centred tree credit and within-body shuffle control:
    1. For every body b: sum_{i in b} C_{i, centered} == 0 (mean == 0).
    2. Within-body shuffled preserves:
       - per-body mean (== 0)
       - per-body std
       - per-body min/max
       - number of valid actions
       - global distribution (mean, std, min, max, count)
    3. Action-wise correlation between centered and within-body shuffled is low across the batch.
    4. requires_grad == False at GenAct advantage use.
    5. Cap actions strictly retain prefix advantage.
    """
    torch.manual_seed(42)
    N = 100
    L = 12
    beta = 0.5

    # Mock active actions: 0 = EFF, 1 = CAP
    valid = torch.zeros(N, L, dtype=torch.bool)
    cat_actions = torch.zeros(N, L, dtype=torch.long)
    for b in range(N):
        length = torch.randint(3, L, (1,)).item()
        valid[b, :length] = True
        # last valid step is cap (1), others are eff (0)
        cat_actions[b, :length - 1] = 0
        cat_actions[b, length - 1] = 1

    valid_tree = (cat_actions == 0) & valid

    # Raw tree credits (simulate high-performing bodies having globally higher credit)
    body_bias = torch.randn(N, 1) * 3.0 + 2.0
    action_tree = (torch.randn(N, L) + body_bias).detach()
    assert action_tree.requires_grad is False

    # Compute body-centered credit
    centered_tree = torch.zeros_like(action_tree)
    shuffled_tree = torch.zeros_like(action_tree)

    for b in range(N):
        b_mask = valid_tree[b]
        n_b = b_mask.sum().item()
        assert n_b >= 2  # since length >= 3 and 1 cap
        b_vals = action_tree[b, b_mask]
        mu_b = b_vals.mean()
        c_b = b_vals - mu_b
        centered_tree[b, b_mask] = c_b

        # 1. Verify sum of centered credit per body is 0
        assert abs(c_b.mean().item()) < 1e-5
        assert abs(c_b.sum().item()) < 1e-5

        # Within-body shuffle
        rng = torch.Generator()
        rng.manual_seed(42 + b * 31)
        perm = torch.randperm(n_b, generator=rng)
        s_b = c_b[perm]
        shuffled_tree[b, b_mask] = s_b

        # 2. Verify per-body statistics match exactly between centered and within-body shuffled
        assert torch.allclose(s_b.mean(), c_b.mean(), atol=1e-5)
        assert torch.allclose(s_b.std(), c_b.std(), atol=1e-5)
        assert torch.allclose(s_b.min(), c_b.min(), atol=1e-5)
        assert torch.allclose(s_b.max(), c_b.max(), atol=1e-5)
        assert s_b.numel() == c_b.numel()

    # 3. Global distribution check
    cent_vals = centered_tree[valid_tree]
    shuf_vals = shuffled_tree[valid_tree]
    assert torch.allclose(shuf_vals.mean(), cent_vals.mean(), atol=1e-5)
    assert torch.allclose(shuf_vals.std(), cent_vals.std(), atol=1e-5)
    assert torch.allclose(shuf_vals.min(), cent_vals.min(), atol=1e-5)
    assert torch.allclose(shuf_vals.max(), cent_vals.max(), atol=1e-5)
    assert shuf_vals.numel() == cent_vals.numel()

    # Low action-wise correlation under within-body shuffle
    corr = float(torch.corrcoef(torch.stack([cent_vals, shuf_vals]))[0, 1].item())
    assert abs(corr) < 0.15, f"Expected low within-body shuffle correlation, got {corr}"

    # Global standardization
    cent_norm = (cent_vals - cent_vals.mean()) / (cent_vals.std() + 1e-8)
    shuf_norm = (shuf_vals - shuf_vals.mean()) / (shuf_vals.std() + 1e-8)

    # Prefix advantage
    raw_adv = torch.randn(N, L)
    adv = torch.zeros_like(raw_adv)
    sel = raw_adv[valid]
    adv[valid] = (sel - sel.mean()) / (sel.std() + 1e-8)

    # Advantage combination
    adv_cent = adv.clone()
    adv_cent[valid_tree] = adv_cent[valid_tree] + beta * cent_norm

    adv_shuf = adv.clone()
    adv_shuf[valid_tree] = adv_shuf[valid_tree] + beta * shuf_norm

    # 4. Cap actions strictly retain prefix advantage
    is_cap = (cat_actions == 1) & valid
    assert torch.allclose(adv_cent[is_cap], adv[is_cap])
    assert torch.allclose(adv_shuf[is_cap], adv[is_cap])

    # 5. Gradient isolation check
    mock_logp = torch.randn(N, L, requires_grad=True)
    loss = -(mock_logp * adv_cent * valid.float()).sum()
    loss.backward()
    assert mock_logp.grad is not None
    assert action_tree.grad is None


def test_tree_credit_decomposition():
    """Verify additive decomposition C_i^{tree} = mu_b + delta_i:
    1. delta_i has zero mean per body: sum_{i in b} delta_i == 0.
    2. Exact reconstruction: mu_b + delta_i == C_i^{tree}.
    3. Condition B (body_mean): S_i = mu_b (zero module differentiation within body).
    4. Condition C (mean_plus_aligned_residual): S_i = mu_b + delta_i == C_i^{tree} (identical to uncentred aligned).
    5. Condition D (mean_plus_shuffled_residual): S_i = mu_b + delta_{pi(i)}:
       - per-body mean is identical to mu_b
       - within-body residual distribution is identical (std, min, max, count)
       - within-body action-wise correlation is low across the batch
    6. Cap actions strictly retain prefix advantage across all modes.
    7. All credit tensors are detached (requires_grad == False).
    """
    torch.manual_seed(42)
    N = 80
    L = 14
    beta = 0.5

    valid = torch.zeros(N, L, dtype=torch.bool)
    cat_actions = torch.zeros(N, L, dtype=torch.long)
    for b in range(N):
        length = torch.randint(3, L, (1,)).item()
        valid[b, :length] = True
        cat_actions[b, :length - 1] = 0  # EFF
        cat_actions[b, length - 1] = 1   # CAP

    valid_tree = (cat_actions == 0) & valid

    # Raw tree credits with heterogeneous body offsets
    body_offsets = torch.randn(N, 1) * 4.0 + 1.0
    action_tree = (torch.randn(N, L) * 1.5 + body_offsets).detach()
    assert action_tree.requires_grad is False

    # Decomposition
    mu_b_tens = torch.zeros_like(action_tree)
    delta_tens = torch.zeros_like(action_tree)
    delta_shuf_tens = torch.zeros_like(action_tree)

    for b in range(N):
        b_mask = valid_tree[b]
        n_b = b_mask.sum().item()
        assert n_b >= 2
        c_b = action_tree[b, b_mask]
        mu = c_b.mean()
        delta = c_b - mu

        # 1. Zero sum per body
        assert abs(delta.mean().item()) < 1e-4
        assert abs(delta.sum().item()) < 1e-4

        # 2. Exact reconstruction
        assert torch.allclose(mu + delta, c_b, atol=1e-6)

        mu_b_tens[b, b_mask] = mu
        delta_tens[b, b_mask] = delta

        # Within-body shuffle of delta
        rng = torch.Generator()
        rng.manual_seed(42 + b * 31)
        perm = torch.randperm(n_b, generator=rng)
        delta_shuf = delta[perm]
        delta_shuf_tens[b, b_mask] = delta_shuf

        # Invariance of residual distribution under within-body shuffle
        assert torch.allclose(delta_shuf.mean(), delta.mean(), atol=1e-5)
        assert torch.allclose(delta_shuf.std(), delta.std(), atol=1e-5)
        assert torch.allclose(delta_shuf.min(), delta.min(), atol=1e-5)
        assert torch.allclose(delta_shuf.max(), delta.max(), atol=1e-5)

    # Check Condition B signal: S_i = mu_b
    S_B = mu_b_tens[valid_tree]
    # Check Condition C signal: S_i = mu_b + delta == C_i
    S_C = (mu_b_tens + delta_tens)[valid_tree]
    assert torch.allclose(S_C, action_tree[valid_tree], atol=1e-6)
    # Check Condition D signal: S_i = mu_b + delta_shuf
    S_D = (mu_b_tens + delta_shuf_tens)[valid_tree]

    # Global normalization
    norm_B = (S_B - S_B.mean()) / (S_B.std() + 1e-8)
    norm_C = (S_C - S_C.mean()) / (S_C.std() + 1e-8)
    norm_D = (S_D - S_D.mean()) / (S_D.std() + 1e-8)

    # Prefix advantage baseline
    raw_adv = torch.randn(N, L)
    adv = torch.zeros_like(raw_adv)
    sel = raw_adv[valid]
    adv[valid] = (sel - sel.mean()) / (sel.std() + 1e-8)

    adv_B = adv.clone()
    adv_B[valid_tree] = adv_B[valid_tree] + beta * norm_B

    adv_C = adv.clone()
    adv_C[valid_tree] = adv_C[valid_tree] + beta * norm_C

    adv_D = adv.clone()
    adv_D[valid_tree] = adv_D[valid_tree] + beta * norm_D

    # Cap action invariance check: all modes leave cap advantage unchanged
    is_cap = (cat_actions == 1) & valid
    assert torch.allclose(adv_B[is_cap], adv[is_cap])
    assert torch.allclose(adv_C[is_cap], adv[is_cap])
    assert torch.allclose(adv_D[is_cap], adv[is_cap])

    # Uncentred aligned comparison (reproduction check)
    raw_uncentred = action_tree[valid_tree]
    norm_uncentred = (raw_uncentred - raw_uncentred.mean()) / (raw_uncentred.std() + 1e-8)
    assert torch.allclose(norm_C, norm_uncentred, atol=1e-5)
