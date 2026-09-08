#!/usr/bin/env python3
"""Analysis script for matched structural counterfactual supervision in CoDesign.

Evaluates:
1. Exact-match coverage across the population (module pairs, subtree pairs, body coverage).
2. Prediction accuracy of return differences Delta R = R_A - R_B by:
   - Raw spatial credit differences (Delta c)
   - Tree-propagated credit differences (Delta C_tree)
   - GenCrit prefix delta differences (Delta Delta V)
3. Conditionality validation: does credit vary across structural contexts (e.g. knee vs twist)?
4. Formats matched body pairs inspection tables.
5. Produces a multi-panel diagnostic figure.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transformer_rl.vocab import EFF_NAMES, CAP_NAMES, GEN_EFF, GEN_CAP
from transformer_rl.counterfactual_pairs import (
    encode_canonical_morphology,
    find_exact_matched_pairs,
)


def analyze_counterfactual_pairs(
    data: Dict[str, Any],
    out_dir: str,
    window_idx: int,
) -> Dict[str, Any]:
    """Analyzes counterfactual pairs in a single credit window artifact."""
    os.makedirs(out_dir, exist_ok=True)

    body_id = data["body_id"]
    slots = data["limb_slot"]
    depths = data["depth"]
    cats = data["category"]
    subs = data["subtype"]
    rets = data["body_return"]
    tok_slots = data["token_slot"]

    has_spatial = "spatial_credit" in data
    has_tree = "tree_credit" in data
    has_prefix = "delta" in data

    if not (has_spatial and has_tree):
        print(f"[warning] Window {window_idx} is missing spatial_credit or tree_credit. Skipping.")
        return {}

    c_spat_action = data["spatial_credit"]
    C_tree_action = data["tree_credit"]
    prefix_action = data["delta"] if has_prefix else None

    # Total environments
    N = int(np.max(body_id)) + 1
    M = 32

    # Build per-body dense matrices for (N, M):
    c_spat_grid = np.zeros((N, M), dtype=np.float32)
    C_tree_grid = np.zeros((N, M), dtype=np.float32)
    prefix_grid = np.zeros((N, M), dtype=np.float32)
    body_ret_grid = np.zeros(N, dtype=np.float32)

    # Also build morphology code matrix
    morphs = np.full((N, 4, 8), -1, dtype=np.int16)

    for b, s, dep, c, sub, ts, r, csp, ctr in zip(
        body_id, slots, depths, cats, subs, tok_slots, rets, c_spat_action, C_tree_action
    ):
        body_ret_grid[b] = r
        c_spat_grid[b, ts] = csp
        C_tree_grid[b, ts] = ctr
        morphs[b, dep, s] = c * 10 + sub

    if prefix_action is not None:
        for b, ts, p_del in zip(body_id, tok_slots, prefix_action):
            prefix_grid[b, ts] = p_del

    # Find or use precomputed pairs
    if "pair_idx_A" in data:
        print("Using stored pair data from artifact...")
        idx_A = data["pair_idx_A"]
        idx_B = data["pair_idx_B"]
        pair_slot = data["pair_slot"]
        pair_depth = data["pair_depth"]
        pair_limb = data["pair_limb"]
        is_sub = data["pair_is_subtree"]
        delta_R = data["pair_delta_R"]
        meta = {
            "n_module_pairs_retained": int((~is_sub).sum()),
            "n_subtree_pairs_retained": int(is_sub.sum()),
            "n_total_pairs": len(idx_A),
            "n_bodies_participating": len(set(idx_A).union(set(idx_B))),
            "frac_bodies_participating": len(set(idx_A).union(set(idx_B))) / max(1, N),
        }
    else:
        print("Discovering exact matched pairs from population morphology...")
        pair_dict = find_exact_matched_pairs(morphs, body_ret_grid)
        idx_A = pair_dict["idx_A"].numpy()
        idx_B = pair_dict["idx_B"].numpy()
        pair_slot = pair_dict["slot"].numpy()
        pair_depth = pair_dict["depth"].numpy()
        pair_limb = pair_dict["limb"].numpy()
        is_sub = pair_dict["is_subtree"].numpy()
        delta_R = pair_dict["delta_R"].numpy()
        meta = pair_dict["meta"]

    # Compute differences
    diff_spatial = c_spat_grid[idx_A, pair_slot] - c_spat_grid[idx_B, pair_slot]
    diff_tree = C_tree_grid[idx_A, pair_slot] - C_tree_grid[idx_B, pair_slot]
    diff_prefix = prefix_grid[idx_A, pair_slot] - prefix_grid[idx_B, pair_slot]
    diff_combined = np.where(is_sub, diff_tree, diff_spatial)

    def _eval_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        mse = float(np.mean((pred - target) ** 2))
        var_t = float(np.var(target))
        ev = float(1.0 - mse / max(1e-8, var_t))
        p_corr = float(np.corrcoef(pred, target)[0, 1]) if np.std(pred) > 1e-8 and np.std(target) > 1e-8 else 0.0
        r_pred = np.argsort(np.argsort(pred))
        r_targ = np.argsort(np.argsort(target))
        s_corr = float(np.corrcoef(r_pred, r_targ)[0, 1]) if np.std(r_pred) > 1e-8 and np.std(r_targ) > 1e-8 else 0.0
        return {"mse": mse, "ev": ev, "pearson": p_corr, "spearman": s_corr}

    metrics_spatial = _eval_metrics(diff_spatial, delta_R)
    metrics_tree = _eval_metrics(diff_tree, delta_R)
    metrics_combined = _eval_metrics(diff_combined, delta_R)
    metrics_prefix = _eval_metrics(diff_prefix, delta_R)

    print("\n" + "=" * 85)
    print(f"COUNTERFACTUAL SUPERVISION ANALYSIS: WINDOW {window_idx} (N_PAIRS = {len(delta_R)})")
    print("=" * 85)

    print("\n--- 1. Exact-Match Coverage in the 4096-Body Population ---")
    print(f"Total Bodies Simulated: {N}")
    print(f"Exact Single-Module Pairs: {meta.get('n_module_pairs_found', meta.get('n_module_pairs_retained'))}")
    print(f"Exact Subtree Pairs:       {meta.get('n_subtree_pairs_found', meta.get('n_subtree_pairs_retained'))}")
    print(f"Total Retained Pairs:      {len(delta_R)}")
    print(f"Bodies in >=1 Pair:        {meta.get('n_bodies_participating')} ({meta.get('frac_bodies_participating', 0):.1%})")

    print("\n--- 2. Counterfactual Difference (Delta R) Prediction Quality ---")
    print(f"{'Signal':<25} | {'MSE':<8} | {'EV':<8} | {'Pearson r':<10} | {'Spearman rho':<12}")
    print("-" * 75)
    print(f"{'Raw Spatial Credit (c_i)':<25} | {metrics_spatial['mse']:<8.4f} | {metrics_spatial['ev']:<8.4f} | {metrics_spatial['pearson']:<+10.4f} | {metrics_spatial['spearman']:<+12.4f}")
    print(f"{'Tree Credit (C_tree)':<25} | {metrics_tree['mse']:<8.4f} | {metrics_tree['ev']:<8.4f} | {metrics_tree['pearson']:<+10.4f} | {metrics_tree['spearman']:<+12.4f}")
    print(f"{'Combined (Mod:c, Sub:Tree)':<25} | {metrics_combined['mse']:<8.4f} | {metrics_combined['ev']:<8.4f} | {metrics_combined['pearson']:<+10.4f} | {metrics_combined['spearman']:<+12.4f}")
    print(f"{'GenCrit Prefix Delta':<25} | {metrics_prefix['mse']:<8.4f} | {metrics_prefix['ev']:<8.4f} | {metrics_prefix['pearson']:<+10.4f} | {metrics_prefix['spearman']:<+12.4f}")

    # Breakdown by match type
    mod_mask = ~is_sub
    sub_mask = is_sub

    if mod_mask.sum() > 0:
        print("\n--- 3. Single-Module Match Subset ---")
        m_spat = _eval_metrics(diff_spatial[mod_mask], delta_R[mod_mask])
        m_pref = _eval_metrics(diff_prefix[mod_mask], delta_R[mod_mask])
        print(f"Module Pairs Count: {mod_mask.sum()}")
        print(f"Spatial Credit Pearson: {m_spat['pearson']:+.4f}, Spearman: {m_spat['spearman']:+.4f}, MSE: {m_spat['mse']:.4f}")
        print(f"Prefix Delta   Pearson: {m_pref['pearson']:+.4f}, Spearman: {m_pref['spearman']:+.4f}, MSE: {m_pref['mse']:.4f}")

    if sub_mask.sum() > 0:
        print("\n--- 4. Subtree Match Subset ---")
        t_tree = _eval_metrics(diff_tree[sub_mask], delta_R[sub_mask])
        t_pref = _eval_metrics(diff_prefix[sub_mask], delta_R[sub_mask])
        print(f"Subtree Pairs Count: {sub_mask.sum()}")
        print(f"Tree Credit  Pearson: {t_tree['pearson']:+.4f}, Spearman: {t_tree['spearman']:+.4f}, MSE: {t_tree['mse']:.4f}")
        print(f"Prefix Delta Pearson: {t_pref['pearson']:+.4f}, Spearman: {t_pref['spearman']:+.4f}, MSE: {t_pref['mse']:.4f}")

    # Conditionality validation: knee vs twist across contexts
    print("\n--- 5. Conditionality Validation (Knee vs Twist across Structural Contexts) ---")
    knee_twist_contexts = []
    for i in range(len(delta_R)):
        if not is_sub[i]:  # module match
            b1 = idx_A[i]
            b2 = idx_B[i]
            d = pair_depth[i]
            s = pair_limb[i]
            v1 = morphs[b1, d, s]
            v2 = morphs[b2, d, s]
            # Check if one is knee (1) and other is twist (2) or swing (0)
            if (v1 == 1 and v2 == 2) or (v1 == 2 and v2 == 1):
                # Context: parent subtype, limb length
                p_sub = morphs[b1, d - 1, s] if d > 0 else -1
                knee_b = b1 if v1 == 1 else b2
                twist_b = b2 if v1 == 1 else b1
                dr = body_ret_grid[knee_b] - body_ret_grid[twist_b]
                dc = c_spat_grid[knee_b, pair_slot[i]] - c_spat_grid[twist_b, pair_slot[i]]
                knee_twist_contexts.append((d, s, p_sub, dr, dc))

    if knee_twist_contexts:
        print(f"Found {len(knee_twist_contexts)} matched pairs comparing Knee vs Twist.")
        by_depth = defaultdict(list)
        for d, s, p_sub, dr, dc in knee_twist_contexts:
            by_depth[d].append((dr, dc))
        for d in sorted(by_depth.keys()):
            d_drs = [x[0] for x in by_depth[d]]
            d_dcs = [x[1] for x in by_depth[d]]
            print(f"  Depth {d}: Mean Delta R(Knee - Twist) = {np.mean(d_drs):+.3f}, Mean Delta c(Knee - Twist) = {np.mean(d_dcs):+.3f} (N={len(d_drs)})")

    # Example matched body pairs table
    print("\n--- 6. Concrete Examples of Matched Body Pairs ---")
    examples = []
    for i in range(min(500, len(delta_R))):
        b1 = idx_A[i]
        b2 = idx_B[i]
        dr = delta_R[i]
        if abs(dr) > 1.0:  # significant return difference
            d = pair_depth[i]
            s = pair_limb[i]
            slot = pair_slot[i]
            subtr_A = [morphs[b1, dep, s] for dep in range(d, 4) if morphs[b1, dep, s] != -1]
            subtr_B = [morphs[b2, dep, s] for dep in range(d, 4) if morphs[b2, dep, s] != -1]
            examples.append({
                "idx_A": b1, "idx_B": b2, "depth": d, "limb": s,
                "is_subtree": is_sub[i], "subtr_A": subtr_A, "subtr_B": subtr_B,
                "R_A": body_ret_grid[b1], "R_B": body_ret_grid[b2],
                "delta_R": dr,
                "diff_spatial": diff_spatial[i],
                "diff_tree": diff_tree[i],
                "diff_prefix": diff_prefix[i],
            })
        if len(examples) >= 5:
            break

    def _code_to_str(c: int) -> str:
        if c < 0: return "empty"
        cat = c // 10
        sub = c % 10
        if cat == GEN_EFF:
            return EFF_NAMES[sub] if sub < len(EFF_NAMES) else f"eff_{sub}"
        else:
            return CAP_NAMES[sub] if sub < len(CAP_NAMES) else f"cap_{sub}"

    for ex in examples:
        st_A = " -> ".join(_code_to_str(c) for c in ex["subtr_A"])
        st_B = " -> ".join(_code_to_str(c) for c in ex["subtr_B"])
        kind = "Subtree" if ex["is_subtree"] else "Single-Module"
        print(f"\nMatch Type: {kind} at Limb {ex['limb']}, Depth {ex['depth']}")
        print(f"  Body A ({ex['idx_A']}): {st_A} (R = {ex['R_A']:.2f})")
        print(f"  Body B ({ex['idx_B']}): {st_B} (R = {ex['R_B']:.2f})")
        print(f"  True Delta R (A - B):     {ex['delta_R']:+.3f}")
        print(f"  Spatial Credit Delta c:   {ex['diff_spatial']:+.3f}")
        print(f"  Tree Credit Delta C_tree: {ex['diff_tree']:+.3f}")
        print(f"  GenCrit Prefix Delta:     {ex['diff_prefix']:+.3f}")

    # Generate multi-panel visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Tree Credit Difference vs Delta R Scatter
    ax1 = axes[0, 0]
    ax1.scatter(delta_R, diff_tree, alpha=0.3, color="#2563eb", s=15, label="Subtree Pairs")
    ax1.set_xlabel("True Return Difference $\\Delta R = R_A - R_B$", fontsize=11)
    ax1.set_ylabel("Predicted Tree Credit Difference $\\Delta C_{tree}$", fontsize=11)
    ax1.set_title(f"Tree Credit vs Return Difference (r = {metrics_tree['pearson']:+.3f})", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Reference diagonal
    lims = [min(np.percentile(delta_R, 1), np.percentile(diff_tree, 1)),
            max(np.percentile(delta_R, 99), np.percentile(diff_tree, 99))]
    ax1.plot(lims, lims, "r--", alpha=0.7, label="Ideal $\\Delta C = \\Delta R$")
    ax1.legend(loc="upper left")

    # Panel 2: Prefix Delta Difference vs Delta R Scatter
    ax2 = axes[0, 1]
    ax2.scatter(delta_R, diff_prefix, alpha=0.3, color="#dc2626", s=15, label="Prefix Deltas")
    ax2.set_xlabel("True Return Difference $\\Delta R = R_A - R_B$", fontsize=11)
    ax2.set_ylabel("GenCrit Prefix Difference $\\Delta (\\Delta V)$", fontsize=11)
    ax2.set_title(f"GenCrit Prefix vs Return Difference (r = {metrics_prefix['pearson']:+.3f})", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    # Panel 3: Correlation comparison by Depth
    ax3 = axes[1, 0]
    depth_cats = np.unique(pair_depth)
    corr_tree_by_d = []
    corr_pref_by_d = []
    for d_val in depth_cats:
        m = (pair_depth == d_val)
        if m.sum() > 10:
            ct = np.corrcoef(diff_tree[m], delta_R[m])[0, 1]
            cp = np.corrcoef(diff_prefix[m], delta_R[m])[0, 1]
            corr_tree_by_d.append(ct)
            corr_pref_by_d.append(cp)
        else:
            corr_tree_by_d.append(0.0)
            corr_pref_by_d.append(0.0)

    x_indices = np.arange(len(depth_cats))
    width = 0.35
    ax3.bar(x_indices - width/2, corr_tree_by_d, width, label="Tree Credit $\\Delta C_{tree}$", color="#2563eb")
    ax3.bar(x_indices + width/2, corr_pref_by_d, width, label="GenCrit Prefix Delta", color="#dc2626")
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([f"Depth {d}" for d in depth_cats])
    ax3.set_ylabel("Pearson Correlation with $\\Delta R$", fontsize=11)
    ax3.set_title("Counterfactual Correlation by Structural Depth", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend()

    # Panel 4: Distribution of Delta R across matched pairs
    ax4 = axes[1, 1]
    ax4.hist(delta_R, bins=50, color="#059669", alpha=0.7, edgecolor="black", label=f"N = {len(delta_R)}")
    ax4.set_xlabel("True Return Difference $\\Delta R$", fontsize=11)
    ax4.set_ylabel("Frequency", fontsize=11)
    ax4.set_title(f"Distribution of Matched Return Differences ($\sigma = {np.std(delta_R):.2f}$)", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"counterfactual_analysis_window_{window_idx:04d}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved to {fig_path}")
    print("=" * 85)

    return {
        "metrics_spatial": metrics_spatial,
        "metrics_tree": metrics_tree,
        "metrics_combined": metrics_combined,
        "metrics_prefix": metrics_prefix,
        "examples": examples,
        "fig_path": fig_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze matched structural counterfactual supervision.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to experiment directory.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for reports and figures.")
    args = parser.parse_args()

    credit_dir = os.path.join(args.run_dir, "credit")
    pattern = os.path.join(credit_dir, "credit_window_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No credit artifacts found in {credit_dir}")

    out_dir = args.out_dir or os.path.join(args.run_dir, "counterfactual_analysis")

    for f in files:
        data = np.load(f)
        w_idx = int(data["meta__gen_window"]) if "meta__gen_window" in data else 1
        analyze_counterfactual_pairs(dict(data), out_dir, w_idx)


if __name__ == "__main__":
    main()
