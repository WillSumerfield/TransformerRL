"""Tests for generator credit assignment and action-module mapping."""

import numpy as np
import pytest
import torch

from transformer_rl.generator_credit import (
    build_action_module_mapping,
    compute_credit_diagnostics,
)
from transformer_rl.vocab import (
    GEN_EFF,
    GEN_CAP,
    EFF_SWING,
    EFF_KNEE,
    CAP_BARE,
    CAP_FOOT,
)


def test_credit_calculation_and_telescoping_synthetic():
    """Test with synthetic prefix values [1.0, 1.5, 1.2, 2.0]:
    deltas must be [+0.5, -0.3, +0.8] and telescope to 2.0 - 1.0 = 1.0.
    """
    v_states = torch.tensor([[1.0, 1.5, 1.2, 2.0]], dtype=torch.float32)  # (1, 4) -> 3 steps
    raw_adv = v_states[:, 1:] - v_states[:, :-1]
    expected_deltas = torch.tensor([[0.5, -0.3, 0.8]], dtype=torch.float32)
    assert torch.allclose(raw_adv, expected_deltas, atol=1e-6)

    total_sum = raw_adv.sum().item()
    telescoped = (v_states[:, -1] - v_states[:, 0]).item()
    assert abs(total_sum - 1.0) < 1e-6
    assert abs(telescoped - 1.0) < 1e-6
    assert abs(total_sum - telescoped) < 1e-6


def test_action_module_mapping_and_masking():
    """Tests the explicit 1-to-1 mapping from generator action to module token and controller DOF.
    Also verifies that inactive/padded steps produce zero records.
    """
    # 2 envs, 5 maximum steps
    N, L = 2, 5
    n_limbs = 8
    max_len = 4

    # Body 0:
    # Step 0: slot 0, EFF, sub EFF_SWING (depth 0)
    # Step 1: slot 0, EFF, sub EFF_KNEE (depth 1)
    # Step 2: slot 0, CAP, sub CAP_BARE (depth 2)
    # Steps 3, 4: Inactive

    # Body 1:
    # Step 0: slot 3, EFF, sub EFF_SWING (depth 0)
    # Step 1: slot 3, CAP, sub CAP_FOOT (depth 1)
    # Steps 2, 3, 4: Inactive

    slots = torch.zeros((N, L), dtype=torch.long)
    cat_a = torch.zeros((N, L), dtype=torch.long)
    sub_a = torch.zeros((N, L), dtype=torch.long)
    active = torch.zeros((N, L), dtype=torch.bool)

    # Body 0
    slots[0, 0] = 0; cat_a[0, 0] = GEN_EFF; sub_a[0, 0] = EFF_SWING; active[0, 0] = True
    slots[0, 1] = 0; cat_a[0, 1] = GEN_EFF; sub_a[0, 1] = EFF_KNEE;  active[0, 1] = True
    slots[0, 2] = 0; cat_a[0, 2] = GEN_CAP; sub_a[0, 2] = CAP_BARE;  active[0, 2] = True

    # Body 1
    slots[1, 0] = 3; cat_a[1, 0] = GEN_EFF; sub_a[1, 0] = EFF_SWING; active[1, 0] = True
    slots[1, 1] = 3; cat_a[1, 1] = GEN_CAP; sub_a[1, 1] = CAP_FOOT;  active[1, 1] = True

    trace = {
        "slots": slots,
        "cat_actions": cat_a,
        "sub_actions": sub_a,
        "active_step": active,
    }

    mapping_data = build_action_module_mapping(trace, max_len=max_len, n_limbs=n_limbs)
    records = mapping_data["records"]
    ctrl_to_act = mapping_data["controller_module_to_action"]

    # Exactly 3 + 2 = 5 active decisions across the batch
    assert len(records["body_id"]) == 5
    # Steps 3 and 4 of body 0, and steps 2, 3, 4 of body 1 must NEVER appear
    assert np.all(records["seq_idx"] <= 2)

    # Check Body 0 records
    b0_mask = (records["body_id"] == 0)
    assert np.sum(b0_mask) == 3
    assert np.array_equal(records["depth"][b0_mask], [0, 1, 2])
    assert np.array_equal(records["category"][b0_mask], [GEN_EFF, GEN_EFF, GEN_CAP])
    assert np.array_equal(records["subtype"][b0_mask], [EFF_SWING, EFF_KNEE, CAP_BARE])
    # Token slots: depth * 8 + limb_slot = 0*8+0=0, 1*8+0=8, 2*8+0=16
    assert np.array_equal(records["token_slot"][b0_mask], [0, 8, 16])
    # Parent token slot: root (-1), then token 0, then token 8
    assert np.array_equal(records["parent_module_id"][b0_mask], [-1, 0, 8])
    # Controller module IDs: effectors have DOFs 0 and 8; cap has -1
    assert np.array_equal(records["controller_module_id"][b0_mask], [0, 8, -1])

    # Check Reverse Query: Which generator action created controller module 0 and 8?
    # For body 0:
    assert ctrl_to_act[0, 0] == 0  # action at seq_idx 0
    assert ctrl_to_act[0, 8] == 1  # action at seq_idx 1
    assert ctrl_to_act[0, 16] == -1  # cap is not an actuated controller DOF
    assert ctrl_to_act[0, 1] == -1  # slot 1 was never grown

    # For body 1:
    b1_mask = (records["body_id"] == 1)
    assert np.sum(b1_mask) == 2
    assert np.array_equal(records["depth"][b1_mask], [0, 1])
    assert np.array_equal(records["category"][b1_mask], [GEN_EFF, GEN_CAP])
    # Token slots: 0*8+3=3, 1*8+3=11
    assert np.array_equal(records["token_slot"][b1_mask], [3, 11])
    # Reverse Query for body 1:
    assert ctrl_to_act[1, 3] == 0  # action at seq_idx 0 created DOF 3
    assert ctrl_to_act[1, 11] == -1  # cap at 11 is not an actuated DOF


def test_diagnostics_and_telescoping_with_values():
    """Verifies that compute_credit_diagnostics correctly computes deltas, telescoping residuals,
    and aggregate distributions on a controlled batch.
    """
    N, L = 2, 3
    slots = torch.zeros((N, L), dtype=torch.long)
    cat_a = torch.zeros((N, L), dtype=torch.long)
    sub_a = torch.zeros((N, L), dtype=torch.long)
    active = torch.ones((N, L), dtype=torch.bool)

    # v_states: (N, L+1)
    # Body 0: [1.0, 2.0, 1.5, 3.0] -> deltas: [+1.0, -0.5, +1.5], sum = 2.0 = 3.0 - 1.0
    # Body 1: [0.0, 0.5, 0.5, 1.0] -> deltas: [+0.5, 0.0, +0.5], sum = 1.0 = 1.0 - 0.0
    v_states = torch.tensor([
        [1.0, 2.0, 1.5, 3.0],
        [0.0, 0.5, 0.5, 1.0],
    ], dtype=torch.float32)

    trace = {
        "slots": slots,
        "cat_actions": cat_a,
        "sub_actions": sub_a,
        "active_step": active,
        "v_states": v_states,
    }
    R = torch.tensor([3.0, 1.0], dtype=torch.float32)

    mapping_data = build_action_module_mapping(trace)
    scalars, flat_records = compute_credit_diagnostics(mapping_data, trace, R)

    # Telescoping residual must be zero to machine precision
    assert scalars["codesign/credit/telescoping_residual_max"] < 1e-6
    assert scalars["codesign/credit/telescoping_residual_mean"] < 1e-6

    # Delta mean across all 6 decisions: (1.0 - 0.5 + 1.5 + 0.5 + 0.0 + 0.5) / 6 = 3.0 / 6 = 0.5
    assert abs(scalars["codesign/credit/gencrit_delta_mean"] - 0.5) < 1e-6

    # Flat records check
    expected_deltas = np.array([1.0, -0.5, 1.5, 0.5, 0.0, 0.5], dtype=np.float32)
    assert np.allclose(flat_records["delta"], expected_deltas, atol=1e-6)


def test_no_behaviour_change_regression():
    """Verifies that calling generator credit diagnostics does not mutate any input tensors,
    and produces no side-effects on advantages, log-probs, or losses.
    """
    from transformer_rl.architectures import MultiMorphLimbTransformer

    torch.manual_seed(1234)
    net = MultiMorphLimbTransformer(
        d_model=64,
        n_heads=4,
        n_layers=1,
        ffn=128,
        n_limbs=8,
        max_limb_length=4,
        codesign_tokens=True,
    )
    net.eval()

    # Generate a real, grammar-valid trace from net.sample
    N = 4
    trace = net.sample(N)
    L = trace["slots"].shape[1]
    v_states = trace["v_states"]
    active = trace["active_step"]
    R = torch.randn(N)

    # Compute raw_adv and adv exactly as CodesignAgent does
    valid = active
    raw_adv = v_states[:, 1:] - v_states[:, :-1]
    sel = raw_adv[valid]
    adv = torch.zeros_like(raw_adv)
    adv[valid] = (sel - sel.mean()) / (sel.std() + 1e-8)

    # Clone all tensors before credit calculation
    slots_clone = trace["slots"].clone()
    cat_a_clone = trace["cat_actions"].clone()
    sub_a_clone = trace["sub_actions"].clone()
    active_clone = active.clone()
    v_states_clone = v_states.clone()
    R_clone = R.clone()
    raw_adv_clone = raw_adv.clone()
    adv_clone = adv.clone()

    # Run mapping and credit diagnostics
    mapping_data = build_action_module_mapping(trace)
    scalars, flat_records = compute_credit_diagnostics(mapping_data, trace, R, adv=adv, raw_adv=raw_adv)

    # Assert exact bitwise identity of all original inputs
    assert torch.equal(trace["slots"], slots_clone)
    assert torch.equal(trace["cat_actions"], cat_a_clone)
    assert torch.equal(trace["sub_actions"], sub_a_clone)
    assert torch.equal(active, active_clone)
    assert torch.equal(v_states, v_states_clone)
    assert torch.equal(R, R_clone)
    assert torch.equal(raw_adv, raw_adv_clone)
    assert torch.equal(adv, adv_clone)
