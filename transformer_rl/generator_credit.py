"""Generator credit assignment instrumentation and diagnostics for TransformerRL CoDesign.

This module provides tools to:
1. Establish an explicit 1-to-1 mapping from each autoregressive generator construction decision
   to the resulting morphology token, depth, limb, and controller module (actuated DOF).
2. Compute marginal GenCrit credit (V(prefix_{t+1}) - V(prefix_t)), compare against normalized PPO
   advantages, and verify telescoping identities.
3. Compute aggregate distribution metrics, within-body credit variance, credit by structural
   properties, context dependence diagnostics, and correlations with completed-body returns.
4. Save lightweight per-window artifacts (.npz) containing per-action records for offline analysis.

IMPORTANT: This module is for instrumentation and analysis ONLY. It does not alter training
behaviour, generator objectives, advantage calculations, or optimizer steps.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from .vocab import (
    GEN_EFF,
    GEN_CAP,
)

EFF_NAMES = ("swing", "knee", "twist")
CAP_NAMES = ("bare", "foot", "pad", "ball")

_N_LIMBS = 8
_MAX_LEN = 4


def build_action_module_mapping(
    trace: Dict[str, torch.Tensor],
    max_len: int = _MAX_LEN,
    n_limbs: int = _N_LIMBS,
) -> Dict[str, Any]:
    """Reconstructs the step-by-step morphology state from a generator trace and establishes
    an explicit mapping from each active construction action to its corresponding morphology
    token slot, limb, depth, parent, and controller module (DOF).

    Args:
        trace: Dictionary from net.sample(N) containing:
            - 'slots': (N, L) limb index per step
            - 'cat_actions': (N, L) category (GEN_EFF or GEN_CAP)
            - 'sub_actions': (N, L) subtype
            - 'active_step': (N, L) bool mask of active steps
            - 'counts': (N, n_limbs) final effector counts
            - 'eff_sub': (N, n_limbs, max_len) final effector subtypes
            - 'cap_sub': (N, n_limbs) final cap subtypes
        max_len: Maximum modules per limb (default 4).
        n_limbs: Number of limbs (default 8).

    Returns:
        Dictionary containing:
            - 'records': Dict of 1D numpy arrays for all active actions across the batch:
                'body_id', 'seq_idx', 'gen_order', 'limb_slot', 'depth', 'token_slot',
                'category', 'subtype', 'parent_module_id', 'controller_module_id', 'is_terminal'
            - 'controller_module_to_action': (N, n_limbs * max_len) numpy array where entry
                [b, m] is the sequence index t that created controller module m (-1 if absent)
            - 'active_mask': (N, L) bool numpy array
            - 'depth_hist': (N, L) int numpy array of depth at each step
    """
    slots = trace["slots"]
    cat_a = trace["cat_actions"]
    sub_a = trace["sub_actions"]
    active = trace["active_step"]

    dev = slots.device
    N, L = slots.shape
    arange = torch.arange(N, device=dev)

    # Reconstruct the frontier state step-by-step, exactly matching net.sample / net._replay_states
    count = torch.zeros(N, n_limbs, dtype=torch.long, device=dev)
    cap_sub = torch.full((N, n_limbs), -1, dtype=torch.long, device=dev)

    depth_hist = torch.zeros(N, L, dtype=torch.long, device=dev)
    gen_order = torch.zeros(N, L, dtype=torch.long, device=dev)
    order_accum = torch.zeros(N, dtype=torch.long, device=dev)

    # Map: (N, n_limbs * max_len) -> seq_idx of generator action that created controller module
    # A controller module corresponds to an active effector at (limb_slot, depth).
    ctrl_mod_to_action = torch.full((N, n_limbs * max_len), -1, dtype=torch.long, device=dev)

    # Sanity tracking: ensure no (slot, depth) is visited more than once per body
    visited_slots = torch.zeros(N, n_limbs, max_len, dtype=torch.bool, device=dev)

    for t in range(L):
        act_t = active[:, t]
        s_t = slots[:, t]
        d_t = count[arange, s_t]
        depth_hist[:, t] = d_t

        # Order within this body
        gen_order[:, t] = torch.where(act_t, order_accum, torch.zeros_like(order_accum))
        order_accum += act_t.long()

        c_t = cat_a[:, t]
        sub_val = sub_a[:, t]

        # Sanity assertion: if active, slot must be valid and depth < max_len
        valid_indices = arange[act_t]
        if valid_indices.numel() > 0:
            s_valid = s_t[valid_indices]
            d_valid = d_t[valid_indices]
            already_visited = visited_slots[valid_indices, s_valid, d_valid]
            assert not already_visited.any(), (
                f"Duplicate module construction at step {t}: "
                f"slot={s_valid[already_visited]}, depth={d_valid[already_visited]}"
            )
            visited_slots[valid_indices, s_valid, d_valid] = True

            # If it's an effector, register in controller_module_to_action
            is_eff = (c_t[valid_indices] == GEN_EFF)
            eff_idx = valid_indices[is_eff]
            if eff_idx.numel() > 0:
                mod_id = d_t[eff_idx] * n_limbs + s_t[eff_idx]
                ctrl_mod_to_action[eff_idx, mod_id] = t

        # Update frontier
        is_eff_all = (c_t == GEN_EFF) & act_t
        is_cap_all = (c_t == GEN_CAP) & act_t
        count[arange, s_t] += is_eff_all.long()
        cap_sub[arange, s_t] = torch.where(is_cap_all, sub_val, cap_sub[arange, s_t])

    # Convert active entries to flat record arrays
    act_np = active.detach().cpu().numpy()
    slots_np = slots.detach().cpu().numpy()
    depth_np = depth_hist.detach().cpu().numpy()
    cat_np = cat_a.detach().cpu().numpy()
    sub_np = sub_a.detach().cpu().numpy()
    order_np = gen_order.detach().cpu().numpy()

    body_idx_grid, seq_idx_grid = np.meshgrid(np.arange(N), np.arange(L), indexing="ij")

    b_act = body_idx_grid[act_np]
    seq_act = seq_idx_grid[act_np]
    slot_act = slots_np[act_np]
    d_act = depth_np[act_np]
    c_act = cat_np[act_np]
    s_act = sub_np[act_np]
    ord_act = order_np[act_np]

    # Canonical module token slot in depth-major representation (0 .. n_limbs * max_len - 1)
    tok_slot = d_act * n_limbs + slot_act

    # Parent module token slot: if depth > 0, (depth - 1) * n_limbs + limb_slot; if depth == 0, -1 (torso)
    parent_id = np.where(d_act > 0, (d_act - 1) * n_limbs + slot_act, -1)

    # Controller module ID: only effectors actuate DOFs. Caps are passive terminal bodies.
    ctrl_mod_id = np.where(c_act == GEN_EFF, tok_slot, -1)
    is_term = (c_act == GEN_CAP)

    records = {
        "body_id": b_act.astype(np.int32),
        "seq_idx": seq_act.astype(np.int16),
        "gen_order": ord_act.astype(np.int16),
        "limb_slot": slot_act.astype(np.int8),
        "depth": d_act.astype(np.int8),
        "token_slot": tok_slot.astype(np.int8),
        "category": c_act.astype(np.int8),
        "subtype": s_act.astype(np.int8),
        "parent_module_id": parent_id.astype(np.int8),
        "controller_module_id": ctrl_mod_id.astype(np.int8),
        "is_terminal": is_term.astype(bool),
    }

    return {
        "records": records,
        "controller_module_to_action": ctrl_mod_to_action.detach().cpu().numpy().astype(np.int16),
        "active_mask": act_np,
        "depth_hist": depth_np,
    }


def compute_credit_diagnostics(
    mapping_data: Dict[str, Any],
    trace: Dict[str, torch.Tensor],
    R: torch.Tensor,
    adv: Optional[torch.Tensor] = None,
    raw_adv: Optional[torch.Tensor] = None,
    max_len: int = _MAX_LEN,
    n_limbs: int = _N_LIMBS,
    near_zero_thresh: float = 1e-4,
    uniform_std_thresh: float = 1e-4,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Calculates comprehensive credit assignment metrics and diagnostics across the resample window.

    Args:
        mapping_data: Output from build_action_module_mapping.
        trace: Generator trace dictionary.
        R: (N,) tensor of completed-body returns (scaled).
        adv: (N, L) tensor of standardized PPO advantages (if available).
        raw_adv: (N, L) tensor of raw GenCrit prefix differences (v_after - v_before).
        near_zero_thresh: Absolute threshold below which a GenCrit delta is deemed near-zero.
        uniform_std_thresh: Within-body std threshold below which credit is deemed uniform.

    Returns:
        scalars: Dict of aggregate scalar metrics for TensorBoard.
        flat_records: Dict of per-decision numpy arrays including credit values.
    """
    records = mapping_data["records"]
    act_mask = mapping_data["active_mask"]
    N, L = act_mask.shape

    v_states = trace["v_states"]
    if raw_adv is None:
        raw_adv = v_states[:, 1:] - v_states[:, :-1]

    # Pre-extract numpy arrays
    v_states_np = v_states.detach().cpu().numpy()
    raw_adv_np = raw_adv.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy()

    old_logp_np = trace.get("old_logp", torch.zeros((N, L), device=R.device)).detach().cpu().numpy()
    step_ent_np = trace.get("step_entropy", torch.zeros((N, L), device=R.device)).detach().cpu().numpy()

    if adv is not None:
        adv_np = adv.detach().cpu().numpy()
    else:
        # If adv wasn't computed (e.g. pretrain), standardize raw_adv[act_mask]
        sel = raw_adv_np[act_mask]
        s_std = sel.std() + 1e-8
        adv_np = np.zeros_like(raw_adv_np)
        adv_np[act_mask] = (sel - sel.mean()) / s_std

    # Prefix values before and after
    v_before = v_states_np[:, :L][act_mask]
    v_after = v_states_np[:, 1:][act_mask]
    deltas = raw_adv_np[act_mask]
    ppo_adv = adv_np[act_mask]
    logp_vals = old_logp_np[act_mask]
    ent_vals = step_ent_np[act_mask]
    body_r_vals = R_np[records["body_id"]]

    flat_records = dict(records)
    flat_records["v_before"] = v_before.astype(np.float32)
    flat_records["v_after"] = v_after.astype(np.float32)
    flat_records["delta"] = deltas.astype(np.float32)
    flat_records["advantage"] = ppo_adv.astype(np.float32)
    flat_records["logp"] = logp_vals.astype(np.float32)
    flat_records["entropy"] = ent_vals.astype(np.float32)
    flat_records["body_return"] = body_r_vals.astype(np.float32)

    # 1. Telescoping sanity check: for each body, sum(delta) == v_full - v_0
    v_0 = v_states_np[:, 0]
    v_full = v_states_np[:, L]
    expected_span = v_full - v_0

    # Sum of active deltas per body
    body_delta_sum = np.zeros(N, dtype=np.float64)
    body_abs_delta_sum = np.zeros(N, dtype=np.float64)
    body_action_count = np.zeros(N, dtype=np.int32)
    np.add.at(body_delta_sum, records["body_id"], deltas)
    np.add.at(body_abs_delta_sum, records["body_id"], np.abs(deltas))
    np.add.at(body_action_count, records["body_id"], 1)

    telescoping_residual = np.abs(body_delta_sum - expected_span)

    # 2. Within-body credit variance / uniformity
    # Compute sample std of deltas per body
    body_within_std = np.zeros(N, dtype=np.float32)
    body_ids = records["body_id"]
    for b in range(N):
        mask_b = (body_ids == b)
        if np.sum(mask_b) > 1:
            body_within_std[b] = np.std(deltas[mask_b], ddof=1)
        else:
            body_within_std[b] = 0.0

    # 3. Distribution metrics of GenCrit deltas
    n_decisions = len(deltas)
    if n_decisions > 0:
        delta_mean = float(np.mean(deltas))
        delta_std = float(np.std(deltas))
        delta_min = float(np.min(deltas))
        delta_max = float(np.max(deltas))
        delta_median = float(np.median(deltas))
        delta_mean_abs = float(np.mean(np.abs(deltas)))
        delta_pos_frac = float(np.mean(deltas > 0))
        delta_neg_frac = float(np.mean(deltas < 0))
        delta_near_zero_frac = float(np.mean(np.abs(deltas) < near_zero_thresh))
    else:
        delta_mean = delta_std = delta_min = delta_max = delta_median = delta_mean_abs = 0.0
        delta_pos_frac = delta_neg_frac = delta_near_zero_frac = 0.0

    # 4. Relationship to completed-body return R
    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return float("nan")
        c = np.corrcoef(x, y)[0, 1]
        return float(c) if np.isfinite(c) else float("nan")

    corr_sum_delta_R = _safe_corr(body_delta_sum, R_np)
    mean_abs_delta = body_abs_delta_sum / np.maximum(body_action_count, 1)
    corr_mean_abs_delta_R = _safe_corr(mean_abs_delta, R_np)
    corr_v_full_R = _safe_corr(v_full, R_np)

    v_full_mse = float(np.mean((v_full - R_np) ** 2))
    r_var = float(np.var(R_np))
    v_full_ev = float(1.0 - v_full_mse / r_var) if r_var > 1e-8 else float("nan")

    scalars: Dict[str, float] = {
        # Distribution of marginal credit
        "codesign/credit/gencrit_delta_mean": delta_mean,
        "codesign/credit/gencrit_delta_std": delta_std,
        "codesign/credit/gencrit_delta_min": delta_min,
        "codesign/credit/gencrit_delta_max": delta_max,
        "codesign/credit/gencrit_delta_median": delta_median,
        "codesign/credit/gencrit_delta_mean_abs": delta_mean_abs,
        "codesign/credit/gencrit_delta_pos_frac": delta_pos_frac,
        "codesign/credit/gencrit_delta_neg_frac": delta_neg_frac,
        "codesign/credit/gencrit_delta_near_zero_frac": delta_near_zero_frac,

        # Advantage stats
        "codesign/credit/gen_advantage_mean": float(np.mean(ppo_adv)) if len(ppo_adv) > 0 else 0.0,
        "codesign/credit/gen_advantage_std": float(np.std(ppo_adv)) if len(ppo_adv) > 0 else 0.0,

        # Within-body credit structure
        "codesign/credit/within_body_std_mean": float(np.mean(body_within_std)),
        "codesign/credit/within_body_std_median": float(np.median(body_within_std)),
        "codesign/credit/within_body_uniform_frac": float(np.mean(body_within_std < uniform_std_thresh)),

        # Telescoping sanity check
        "codesign/credit/telescoping_residual_mean": float(np.mean(telescoping_residual)),
        "codesign/credit/telescoping_residual_max": float(np.max(telescoping_residual)),

        # Return relationships
        "codesign/credit/corr_sum_delta_R": corr_sum_delta_R,
        "codesign/credit/corr_mean_abs_delta_R": corr_mean_abs_delta_R,
        "codesign/credit/corr_v_full_R": corr_v_full_R,
        "codesign/credit/v_full_mse": v_full_mse,
        "codesign/credit/v_full_ev": v_full_ev,
    }

    # Breakdown by category
    c_arr = records["category"]
    eff_m = (c_arr == GEN_EFF)
    cap_m = (c_arr == GEN_CAP)
    if np.any(eff_m):
        scalars["codesign/credit/effector_mean"] = float(np.mean(deltas[eff_m]))
        scalars["codesign/credit/effector_std"] = float(np.std(deltas[eff_m]))
    if np.any(cap_m):
        scalars["codesign/credit/cap_mean"] = float(np.mean(deltas[cap_m]))
        scalars["codesign/credit/cap_std"] = float(np.std(deltas[cap_m]))

    # Breakdown by subtype
    s_arr = records["subtype"]
    for t_idx, name in enumerate(EFF_NAMES):
        m = eff_m & (s_arr == t_idx)
        if np.any(m):
            scalars[f"codesign/credit/eff_{name}_mean"] = float(np.mean(deltas[m]))
    for t_idx, name in enumerate(CAP_NAMES):
        m = cap_m & (s_arr == t_idx)
        if np.any(m):
            scalars[f"codesign/credit/cap_{name}_mean"] = float(np.mean(deltas[m]))

    # Breakdown by depth
    d_arr = records["depth"]
    for d_val in range(max_len):
        m = (d_arr == d_val)
        if np.any(m):
            scalars[f"codesign/credit/depth_{d_val}_mean"] = float(np.mean(deltas[m]))

    # Breakdown by limb slot
    slot_arr = records["limb_slot"]
    for s_val in range(n_limbs):
        m = (slot_arr == s_val)
        if np.any(m):
            scalars[f"codesign/credit/limb_{s_val}_mean"] = float(np.mean(deltas[m]))

    # Breakdown by generation order (early vs middle vs late)
    ord_arr = records["gen_order"]
    for o_val in range(min(8, int(np.max(ord_arr, initial=0)) + 1)):
        m = (ord_arr == o_val)
        if np.any(m):
            scalars[f"codesign/credit/order_{o_val}_mean"] = float(np.mean(deltas[m]))

    return scalars, flat_records


def save_credit_artifact(
    filepath: str,
    flat_records: Dict[str, np.ndarray],
    controller_module_to_action: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves a lightweight compressed .npz artifact containing raw per-decision records
    and controller mapping for a resample window."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    save_dict = dict(flat_records)
    save_dict["controller_module_to_action"] = controller_module_to_action
    if metadata:
        for k, v in metadata.items():
            save_dict[f"meta__{k}"] = np.array(v)
    np.savez_compressed(filepath, **save_dict)
