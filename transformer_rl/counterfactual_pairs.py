"""Matched structural counterfactual supervision for CoDesign spatial credit.

Discovers pairs of morphologies from the parallel population that are identical
everywhere except at one module or one subtree, and uses their return differences
Delta R = R_A - R_B as counterfactual supervision targets for spatial and tree credit:
    c_i^A - c_i^B approx Delta R       (module matches)
    C_{i,tree}^A - C_{i,tree}^B approx Delta R (subtree matches)
"""

from __future__ import annotations

from collections import defaultdict
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .vocab import GEN_EFF, GEN_CAP

_N_LIMBS = 8
_MAX_LEN = 4
_TOTAL_SLOTS = _N_LIMBS * _MAX_LEN


def encode_canonical_morphology(
    counts: torch.Tensor | np.ndarray,
    eff_sub: torch.Tensor | np.ndarray,
    cap_sub: torch.Tensor | np.ndarray,
    max_len: int = _MAX_LEN,
    n_limbs: int = _N_LIMBS,
) -> np.ndarray:
    """Encodes morphology batch into canonical (N, max_len, n_limbs) integer matrix.

    Each slot (d, s) contains:
      -1: empty / padded slot
      GEN_EFF * 10 + sub (0..2): effector of subtype
      GEN_CAP * 10 + sub (10..13): cap of subtype

    Args:
        counts: (N, n_limbs) integer effector counts
        eff_sub: (N, n_limbs, max_len) effector subtypes (-1 if absent)
        cap_sub: (N, n_limbs) cap subtypes (-1 if absent)
        max_len: Maximum limb length (default 4)
        n_limbs: Number of limbs (default 8)

    Returns:
        (N, max_len, n_limbs) int16 array of canonical codes.
    """
    if isinstance(counts, torch.Tensor):
        counts = counts.detach().cpu().numpy()
    if isinstance(eff_sub, torch.Tensor):
        eff_sub = eff_sub.detach().cpu().numpy()
    if isinstance(cap_sub, torch.Tensor):
        cap_sub = cap_sub.detach().cpu().numpy()

    N = counts.shape[0]
    morphs = np.full((N, max_len, n_limbs), -1, dtype=np.int16)

    for b in range(N):
        for s in range(n_limbs):
            cnt = int(counts[b, s])
            for d in range(min(cnt, max_len)):
                sub = int(eff_sub[b, s, d])
                if sub >= 0:
                    morphs[b, d, s] = GEN_EFF * 10 + sub
            if 0 < cnt < max_len:
                c_sub = int(cap_sub[b, s])
                if c_sub >= 0:
                    morphs[b, cnt, s] = GEN_CAP * 10 + c_sub
            elif cnt == 0:
                c_sub = int(cap_sub[b, s])
                if c_sub >= 0:
                    morphs[b, 0, s] = GEN_CAP * 10 + c_sub

    return morphs


def find_exact_matched_pairs(
    morphs: np.ndarray,
    R: torch.Tensor | np.ndarray,
    max_module_pairs: int = 16384,
    max_subtree_pairs: int = 16384,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Finds exact single-module and exact subtree matched pairs across the population.

    Args:
        morphs: (N, max_len, n_limbs) canonical morphology array
        R: (N,) returns array or tensor
        max_module_pairs: Maximum number of module pairs to retain
        max_subtree_pairs: Maximum number of subtree pairs to retain
        device: Target torch device for output tensors

    Returns:
        Dictionary containing batched tensors:
            'idx_A': (P,) int64 tensor of environment indices for body A
            'idx_B': (P,) int64 tensor of environment indices for body B
            'slot': (P,) int64 tensor of token slot index (d * n_limbs + s)
            'depth': (P,) int64 tensor of depth
            'limb': (P,) int64 tensor of limb slot
            'is_subtree': (P,) bool tensor (True for subtree, False for single module)
            'delta_R': (P,) float32 tensor (R_A - R_B)
            'meta': dict with summary statistics
    """
    if isinstance(R, torch.Tensor):
        R_np = R.detach().cpu().numpy()
    else:
        R_np = np.asarray(R)

    N, max_len, n_limbs = morphs.shape

    # 1. Exact Single-Module Matches
    # Two bodies are identical in all slots except (d, s)
    module_pairs_list: List[Tuple[int, int, int, int, float]] = []
    participating_module_bodies = set()

    for d in range(max_len):
        for s in range(n_limbs):
            bucket: Dict[bytes, List[Tuple[int, int]]] = defaultdict(list)
            for b in range(N):
                key = morphs[b].copy()
                val = int(key[d, s])
                key[d, s] = -99
                bucket[key.tobytes()].append((b, val))

            for items in bucket.values():
                if len(items) > 1:
                    for i in range(len(items)):
                        b1, v1 = items[i]
                        for j in range(i + 1, len(items)):
                            b2, v2 = items[j]
                            if v1 != v2:
                                module_pairs_list.append((b1, b2, d, s, float(R_np[b1] - R_np[b2])))
                                participating_module_bodies.add(b1)
                                participating_module_bodies.add(b2)

    # 2. Exact Subtree Matches
    # Two bodies are identical outside the subtree rooted at (d_root, s)
    subtree_pairs_list: List[Tuple[int, int, int, int, float]] = []
    participating_subtree_bodies = set()

    for d_root in range(max_len):
        for s in range(n_limbs):
            bucket_sub: Dict[bytes, List[Tuple[int, Tuple[int, ...]]]] = defaultdict(list)
            for b in range(N):
                key = morphs[b].copy()
                subtr_val = tuple(key[d_root:, s].tolist())
                key[d_root:, s] = -99
                bucket_sub[key.tobytes()].append((b, subtr_val))

            for items in bucket_sub.values():
                if len(items) > 1:
                    for i in range(len(items)):
                        b1, v1 = items[i]
                        for j in range(i + 1, len(items)):
                            b2, v2 = items[j]
                            if v1 != v2:
                                subtree_pairs_list.append((b1, b2, d_root, s, float(R_np[b1] - R_np[b2])))
                                participating_subtree_bodies.add(b1)
                                participating_subtree_bodies.add(b2)

    total_module_found = len(module_pairs_list)
    total_subtree_found = len(subtree_pairs_list)

    # Subsample if exceeding budget to maintain balanced training
    rng = np.random.default_rng(42)
    if len(module_pairs_list) > max_module_pairs:
        perm = rng.permutation(len(module_pairs_list))[:max_module_pairs]
        module_pairs_list = [module_pairs_list[idx] for idx in perm]

    if len(subtree_pairs_list) > max_subtree_pairs:
        perm = rng.permutation(len(subtree_pairs_list))[:max_subtree_pairs]
        subtree_pairs_list = [subtree_pairs_list[idx] for idx in perm]

    # Combine into unified training batch
    all_A = []
    all_B = []
    all_slot = []
    all_depth = []
    all_limb = []
    all_is_sub = []
    all_dR = []

    for b1, b2, d, s, dr in module_pairs_list:
        all_A.append(b1)
        all_B.append(b2)
        all_slot.append(d * n_limbs + s)
        all_depth.append(d)
        all_limb.append(s)
        all_is_sub.append(False)
        all_dR.append(dr)

    for b1, b2, d, s, dr in subtree_pairs_list:
        all_A.append(b1)
        all_B.append(b2)
        all_slot.append(d * n_limbs + s)
        all_depth.append(d)
        all_limb.append(s)
        all_is_sub.append(True)
        all_dR.append(dr)

    all_participating = participating_module_bodies.union(participating_subtree_bodies)

    if device is None:
        device = torch.device("cpu")

    pair_data = {
        "idx_A": torch.tensor(all_A, dtype=torch.long, device=device),
        "idx_B": torch.tensor(all_B, dtype=torch.long, device=device),
        "slot": torch.tensor(all_slot, dtype=torch.long, device=device),
        "depth": torch.tensor(all_depth, dtype=torch.long, device=device),
        "limb": torch.tensor(all_limb, dtype=torch.long, device=device),
        "is_subtree": torch.tensor(all_is_sub, dtype=torch.bool, device=device),
        "delta_R": torch.tensor(all_dR, dtype=torch.float32, device=device),
        "meta": {
            "n_module_pairs_found": total_module_found,
            "n_subtree_pairs_found": total_subtree_found,
            "n_module_pairs_retained": len(module_pairs_list),
            "n_subtree_pairs_retained": len(subtree_pairs_list),
            "n_total_pairs": len(all_A),
            "n_bodies_participating": len(all_participating),
            "frac_bodies_participating": len(all_participating) / max(1, N),
            "delta_R_mean": float(np.mean(all_dR)) if all_dR else 0.0,
            "delta_R_std": float(np.std(all_dR)) if all_dR else 0.0,
        },
    }

    return pair_data


def compute_pair_difference_loss(
    c_spat: torch.Tensor,
    C_tree: torch.Tensor,
    pair_batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes pair difference MSE loss:
        Delta hat{C} = where(is_subtree, C_tree^A[slot] - C_tree^B[slot], c^A[slot] - c^B[slot])
        L_pair = MSE(Delta hat{C}, Delta R)

    Args:
        c_spat: (N, M) masked per-module spatial credits
        C_tree: (N, M) tree-propagated spatial credits
        pair_batch: dict with idx_A, idx_B, slot, is_subtree, delta_R

    Returns:
        loss: scalar MSE loss tensor
        delta_C: (P,) predicted credit difference tensor
    """
    idx_A = pair_batch["idx_A"]
    idx_B = pair_batch["idx_B"]
    slot = pair_batch["slot"]
    is_subtree = pair_batch["is_subtree"]
    delta_R = pair_batch["delta_R"]

    # Gather credit for body A and B at target slot
    cA_mod = c_spat[idx_A, slot]
    cB_mod = c_spat[idx_B, slot]
    diff_mod = cA_mod - cB_mod

    cA_tree = C_tree[idx_A, slot]
    cB_tree = C_tree[idx_B, slot]
    diff_tree = cA_tree - cB_tree

    delta_C = torch.where(is_subtree, diff_tree, diff_mod)
    loss = F.mse_loss(delta_C, delta_R)
    return loss, delta_C


def compute_pair_diagnostics(
    c_spat: torch.Tensor,
    C_tree: torch.Tensor,
    pair_data: Dict[str, torch.Tensor],
    prefix_delta: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Computes evaluation diagnostics comparing spatial, tree, and prefix differences against Delta R.

    Args:
        c_spat: (N, M) spatial credits
        C_tree: (N, M) tree credits
        pair_data: dict with idx_A, idx_B, slot, is_subtree, delta_R
        prefix_delta: (N, M) optional GenCrit prefix delta at each slot

    Returns:
        Dict of scalar diagnostics for TensorBoard.
    """
    with torch.no_grad():
        idx_A = pair_data["idx_A"]
        idx_B = pair_data["idx_B"]
        slot = pair_data["slot"]
        is_sub = pair_data["is_subtree"]
        delta_R = pair_data["delta_R"].cpu().numpy()

        if len(delta_R) == 0:
            return {}

        diff_spatial = (c_spat[idx_A, slot] - c_spat[idx_B, slot]).cpu().numpy()
        diff_tree = (C_tree[idx_A, slot] - C_tree[idx_B, slot]).cpu().numpy()

        def _metrics(pred: np.ndarray, target: np.ndarray, prefix: str) -> Dict[str, float]:
            mse = float(np.mean((pred - target) ** 2))
            var_t = float(np.var(target))
            ev = float(1.0 - mse / max(1e-8, var_t))
            p_corr = float(np.corrcoef(pred, target)[0, 1]) if np.std(pred) > 1e-8 and np.std(target) > 1e-8 else 0.0
            r_pred = np.argsort(np.argsort(pred))
            r_targ = np.argsort(np.argsort(target))
            s_corr = float(np.corrcoef(r_pred, r_targ)[0, 1]) if np.std(r_pred) > 1e-8 and np.std(r_targ) > 1e-8 else 0.0
            return {
                f"{prefix}_mse": mse,
                f"{prefix}_ev": ev,
                f"{prefix}_pearson": p_corr,
                f"{prefix}_spearman": s_corr,
            }

        out = {}
        out.update(_metrics(diff_spatial, delta_R, "spatial_diff"))
        out.update(_metrics(diff_tree, delta_R, "tree_diff"))

        is_sub_np = is_sub.cpu().numpy()
        delta_C = np.where(is_sub_np, diff_tree, diff_spatial)
        out.update(_metrics(delta_C, delta_R, "pair_combined"))

        if prefix_delta is not None:
            diff_pref = (prefix_delta[idx_A, slot] - prefix_delta[idx_B, slot]).cpu().numpy()
            out.update(_metrics(diff_pref, delta_R, "prefix_diff"))

        mod_mask = ~is_sub_np
        if mod_mask.sum() > 5:
            out.update(_metrics(diff_spatial[mod_mask], delta_R[mod_mask], "module_only_spatial"))
            out.update(_metrics(diff_tree[mod_mask], delta_R[mod_mask], "module_only_tree"))
            if prefix_delta is not None:
                out.update(_metrics(diff_pref[mod_mask], delta_R[mod_mask], "module_only_prefix"))

        if is_sub_np.sum() > 5:
            out.update(_metrics(diff_tree[is_sub_np], delta_R[is_sub_np], "subtree_only_tree"))
            out.update(_metrics(diff_spatial[is_sub_np], delta_R[is_sub_np], "subtree_only_spatial"))
            if prefix_delta is not None:
                out.update(_metrics(diff_pref[is_sub_np], delta_R[is_sub_np], "subtree_only_prefix"))

        meta = pair_data.get("meta", {})
        for k, v in meta.items():
            if isinstance(v, (int, float)):
                out[k] = float(v)

        return out
