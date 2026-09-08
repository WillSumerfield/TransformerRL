"""Unit tests for GPU-native contextual spatial credit head, additive value decomposition,
topology tree propagation, and diagnostic metrics.
"""

import numpy as np
import pytest
import torch

from transformer_rl.spatial_credit import (
    SpatialCreditHead,
    SpatialGlobalHead,
    compute_spatial_value,
    propagate_tree_credit,
    compute_spatial_credit_diagnostics,
)


def test_spatial_credit_head_shape_and_masking():
    """Verify SpatialCreditHead shapes, masking of inactive slots, and gradient flow."""
    B, M, D = 4, 32, 64
    head = SpatialCreditHead(d_model=D, hidden_dim=D)

    tokens = torch.randn(B, M, D, requires_grad=True)
    mask = torch.zeros(B, M)
    mask[:, :10] = 1.0  # first 10 modules active, rest padded

    c = head(tokens, mask=mask)
    assert c.shape == (B, M)

    # Inactive slots must be exactly zero
    assert torch.all(c[:, 10:] == 0.0)

    # Active slots should be non-zero
    assert torch.any(c[:, :10] != 0.0)

    # Gradients should flow
    loss = c.sum()
    loss.backward()
    assert tokens.grad is not None
    assert torch.any(tokens.grad[:, :10] != 0.0)
    assert torch.all(tokens.grad[:, 10:] == 0.0)


def test_spatial_global_head_shape():
    """Verify SpatialGlobalHead produces scalar baseline from CLS token."""
    B, D = 4, 64
    g_head = SpatialGlobalHead(d_model=D)
    cls_tok = torch.randn(B, D)
    v_glob = g_head(cls_tok)
    assert v_glob.shape == (B,)


def test_additive_value_decomposition():
    """Verify V^{spatial}(s) = v_{global}(h_{CLS}) + sum_i m_i c_i exact identity."""
    B, M, D = 4, 32, 64
    content_start = 9
    n_tokens = content_start + M

    c_head = SpatialCreditHead(d_model=D)
    g_head = SpatialGlobalHead(d_model=D)

    H = torch.randn(B, n_tokens, D)
    present_mask = torch.zeros(B, M)
    present_mask[0, [0, 1, 8]] = 1.0  # body 0 has 3 modules
    present_mask[1, [0, 8, 16]] = 1.0  # body 1 has 3 modules

    v_spatial, v_global, c = compute_spatial_value(
        c_head, g_head, H, present_mask, content_start=content_start
    )

    assert v_spatial.shape == (B,)
    assert v_global.shape == (B,)
    assert c.shape == (B, M)

    # Inactive slots must be 0
    assert torch.all(c[present_mask == 0] == 0.0)

    # Additive identity check: v_spatial == v_global + sum(c)
    sum_c = c.sum(dim=-1)
    reconstructed = v_global + sum_c
    assert torch.allclose(v_spatial, reconstructed, atol=1e-6)


def test_topology_tree_propagation_analytical():
    """Verify batched tree propagation against exact analytical closed-form formulas.
    For a limb chain of 3 modules:
        depth 0: c0
        depth 1: c1
        depth 2: c2 (terminal)
    with discount lambda=0.5:
        C_tree[2] = c2
        C_tree[1] = c1 + 0.5 * c2
        C_tree[0] = c0 + 0.5 * c1 + 0.25 * c2
    """
    B, n_limbs, max_len = 1, 8, 4
    M = n_limbs * max_len
    tree_lambda = 0.5

    c = torch.zeros(B, M)
    present = torch.zeros(B, M)

    # Set up limb 0 with 3 modules:
    # slot for (d, s) = d * n_limbs + s
    slot0 = 0 * 8 + 0  # depth 0
    slot1 = 1 * 8 + 0  # depth 1
    slot2 = 2 * 8 + 0  # depth 2

    c[0, slot0] = 1.0
    c[0, slot1] = 2.0
    c[0, slot2] = 4.0
    present[0, slot0] = 1.0
    present[0, slot1] = 1.0
    present[0, slot2] = 1.0

    # Limb 1 with 1 module only (terminal cap at depth 0)
    slot_l1_0 = 0 * 8 + 1
    c[0, slot_l1_0] = 3.0
    present[0, slot_l1_0] = 1.0

    C_tree = propagate_tree_credit(c, present, n_limbs=n_limbs, max_len=max_len, tree_lambda=tree_lambda)

    # Check limb 0:
    # depth 2: 4.0
    assert abs(C_tree[0, slot2].item() - 4.0) < 1e-6
    # depth 1: 2.0 + 0.5 * 4.0 = 4.0
    assert abs(C_tree[0, slot1].item() - 4.0) < 1e-6
    # depth 0: 1.0 + 0.5 * 4.0 = 3.0 (i.e. c0 + 0.5 * C_tree[1] = 1.0 + 0.5 * 4.0 = 3.0)
    assert abs(C_tree[0, slot0].item() - 3.0) < 1e-6

    # Check limb 1:
    assert abs(C_tree[0, slot_l1_0].item() - 3.0) < 1e-6

    # Inactive slots must be 0
    assert abs(C_tree[0, 3 * 8 + 0].item() - 0.0) < 1e-6  # depth 3 of limb 0
    assert abs(C_tree[0, 1 * 8 + 1].item() - 0.0) < 1e-6  # depth 1 of limb 1


def test_diagnostics_cancellation_and_max_fraction():
    """Verify cancellation ratio and max module fraction diagnostics."""
    B, M = 2, 8
    # Body 0: no cancellation (all positive credits: 1, 2, 3)
    # Body 1: severe cancellation (+5 and -5)
    c = torch.zeros(B, M)
    c[0, [0, 1, 2]] = torch.tensor([1.0, 2.0, 3.0])
    c[1, [0, 1]] = torch.tensor([5.0, -5.0])

    present = torch.zeros(B, M)
    present[0, [0, 1, 2]] = 1.0
    present[1, [0, 1]] = 1.0

    C_tree = c.clone()
    v_global = torch.tensor([0.5, 0.5])
    v_spatial = v_global + c.sum(dim=-1)
    R = torch.tensor([6.5, 0.5])

    scalars, records = compute_spatial_credit_diagnostics(
        c, C_tree, v_global, v_spatial, R, present
    )

    # Body 0 cancellation ratio: sum(|c|) / (|sum(c)|) = 6.0 / 6.0 = 1.0
    # Body 1 cancellation ratio: sum(|c|) / (|sum(c)|) = 10.0 / eps >> 1
    assert abs(records["cancellation_ratio"][0] - 1.0) < 1e-4
    assert records["cancellation_ratio"][1] > 1000.0

    # Body 0 max contribution fraction: 3.0 / 6.0 = 0.5
    # Body 1 max contribution fraction: 5.0 / 10.0 = 0.5
    assert abs(records["max_contrib_frac"][0] - 0.5) < 1e-4
    assert abs(records["max_contrib_frac"][1] - 0.5) < 1e-4

    assert "codesign/spatial/cancellation_ratio_mean" in scalars
    assert "codesign/spatial/max_contrib_frac_mean" in scalars
    assert "codesign/spatial/v_spatial_ev" in scalars
