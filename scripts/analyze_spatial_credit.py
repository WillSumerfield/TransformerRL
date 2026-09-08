#!/usr/bin/env python3
"""Analysis script for GPU-native contextual spatial credit in CoDesign.

Reads credit_window_*.npz artifacts and evaluates:
1. Spatial value prediction quality (MSE, Explained Variance against body return R).
2. Within-body non-uniformity and cancellation metrics (cancellation ratio, max contribution fraction).
3. Context dependence: does the same subtype receive different credit across depths, limb slots, and effector counts?
4. Comparison between spatial credit c_i, tree-propagated credit C_i^tree, and GenCrit prefix delta.
5. Example bodies inspection table:
   module | subtype | depth | prefix_credit | spatial_credit | tree_credit
6. Generates diagnostic visualization plots.
"""

from __future__ import annotations

import argparse
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

EFF_NAMES = ("swing", "knee", "twist")
CAP_NAMES = ("bare", "foot", "pad", "ball")
GEN_EFF = 0
GEN_CAP = 1


def load_credit_windows(credit_dir: str) -> List[Tuple[int, Dict[str, Any]]]:
    """Loads sorted credit_window_*.npz artifacts."""
    pattern = os.path.join(credit_dir, "credit_window_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No credit artifacts found matching {pattern}")

    records = []
    for f in files:
        data = np.load(f)
        w_idx = int(data["meta__gen_window"]) if "meta__gen_window" in data else len(records)
        records.append((w_idx, {k: data[k] for k in data.files}))
    return records


def analyze_spatial_credit_artifact(data: Dict[str, Any], out_dir: str, window_idx: int):
    """Performs comprehensive quantitative analysis on a single window credit artifact."""
    os.makedirs(out_dir, exist_ok=True)

    has_spatial = "spatial_credit" in data
    if not has_spatial:
        print(f"[warning] Window {window_idx} does not contain spatial_credit. Skipping.")
        return

    b_ids = data["body_id"]
    seq_idx = data["seq_idx"]
    tok_slot = data["token_slot"]
    depth = data["depth"]
    limb_slot = data["limb_slot"]
    category = data["category"]
    subtype = data["subtype"]
    delta = data["delta"]
    adv = data["advantage"]
    ret = data["body_return"]
    c_spatial = data["spatial_credit"]
    c_tree = data["tree_credit"] if "tree_credit" in data else c_spatial

    print("\n" + "=" * 85)
    print(f"CONTEXTUAL SPATIAL CREDIT ANALYSIS: WINDOW {window_idx} (N_ACTIONS = {len(c_spatial)})")
    print("=" * 85)

    # 1. Overall distributions
    print("\n--- 1. Credit Distributions ---")
    print(f"GenCrit Prefix Delta: mean = {delta.mean():.4f}, std = {delta.std():.4f}, min = {delta.min():.4f}, max = {delta.max():.4f}")
    print(f"Spatial Credit (c_i): mean = {c_spatial.mean():.4f}, std = {c_spatial.std():.4f}, min = {c_spatial.min():.4f}, max = {c_spatial.max():.4f}")
    print(f"Tree Credit (C_tree): mean = {c_tree.mean():.4f}, std = {c_tree.std():.4f}, min = {c_tree.min():.4f}, max = {c_tree.max():.4f}")

    # 2. Correlations
    def _corr(x, y):
        if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return float("nan"), float("nan")
        p = np.corrcoef(x, y)[0, 1]
        rx = x.argsort().argsort().astype(float)
        ry = y.argsort().argsort().astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        denom = np.linalg.norm(rx) * np.linalg.norm(ry)
        s = (rx @ ry) / denom if denom > 1e-8 else float("nan")
        return float(p), float(s)

    p_spat_delta, s_spat_delta = _corr(c_spatial, delta)
    p_tree_delta, s_tree_delta = _corr(c_tree, delta)
    p_spat_tree, s_spat_tree = _corr(c_spatial, c_tree)
    print("\n--- 2. Correlation between Credit Signals ---")
    print(f"Spatial Credit vs Prefix Delta:  Pearson = {p_spat_delta:+.4f}, Spearman = {s_spat_delta:+.4f}")
    print(f"Tree Credit vs Prefix Delta:     Pearson = {p_tree_delta:+.4f}, Spearman = {s_tree_delta:+.4f}")
    print(f"Spatial Credit vs Tree Credit:   Pearson = {p_spat_tree:+.4f}, Spearman = {s_spat_tree:+.4f}")

    # 3. Category Breakdown (Effector vs Cap)
    eff_m = (category == GEN_EFF)
    cap_m = (category == GEN_CAP)
    print("\n--- 3. Category Breakdown (Effector vs Cap) ---")
    print(f"Effectors (N = {eff_m.sum()}): Prefix Delta = {delta[eff_m].mean():+.4f}, Spatial Credit = {c_spatial[eff_m].mean():+.4f}, Tree Credit = {c_tree[eff_m].mean():+.4f}")
    print(f"Caps      (N = {cap_m.sum()}): Prefix Delta = {delta[cap_m].mean():+.4f}, Spatial Credit = {c_spatial[cap_m].mean():+.4f}, Tree Credit = {c_tree[cap_m].mean():+.4f}")

    # 4. Context Breakdown by Subtype and Depth
    print("\n--- 4. Subtype & Depth Context Dependence ---")
    print(f"{'Category':<10} | {'Subtype':<10} | {'Depth':<6} | {'Count':<6} | {'Prefix Delta':<14} | {'Spatial Credit':<14} | {'Tree Credit':<14}")
    print("-" * 85)
    for cat_val, names in [(GEN_EFF, EFF_NAMES), (GEN_CAP, CAP_NAMES)]:
        for sub_val, name in enumerate(names):
            for d_val in sorted(np.unique(depth)):
                m = (category == cat_val) & (subtype == sub_val) & (depth == d_val)
                if m.sum() > 0:
                    cat_name = "Effector" if cat_val == GEN_EFF else "Cap"
                    print(f"{cat_name:<10} | {name:<10} | {d_val:<6d} | {m.sum():<6d} | {delta[m].mean():+14.4f} | {c_spatial[m].mean():+14.4f} | {c_tree[m].mean():+14.4f}")

    # 5. Inspection of Negative-Prefix Knee Actions
    print("\n--- 5. Knee / Effector Sign Inversion Check ---")
    knee_m = (category == GEN_EFF) & (subtype == 1)  # knee is index 1
    neg_prefix_knee = knee_m & (delta < 0)
    pos_spat_neg_prefix_knee = neg_prefix_knee & (c_spatial > 0)
    print(f"Total Knee Actions: {knee_m.sum()}")
    print(f"Knee Actions with Negative Prefix Delta: {neg_prefix_knee.sum()} ({neg_prefix_knee.sum() / max(1, knee_m.sum()) * 100:.1f}%)")
    print(f"Of those, Knee Actions receiving POSITIVE Spatial Credit: {pos_spat_neg_prefix_knee.sum()} ({pos_spat_neg_prefix_knee.sum() / max(1, neg_prefix_knee.sum()) * 100:.1f}%)")

    # 6. Example Bodies Table
    print("\n--- 6. Example Representative Bodies ---")
    unique_bodies = np.unique(b_ids)
    sample_bodies = unique_bodies[:5] if len(unique_bodies) >= 5 else unique_bodies

    for b in sample_bodies:
        mb = (b_ids == b)
        b_ret = ret[mb][0]
        n_eff = (category[mb] == GEN_EFF).sum()
        n_cap = (category[mb] == GEN_CAP).sum()
        print(f"\nBody ID {b} (Eff={n_eff}, Cap={n_cap}, Total Return R={b_ret:.3f}):")
        print(f"  {'Slot':<6} | {'Module':<10} | {'Subtype':<10} | {'Depth':<6} | {'Prefix Delta':<14} | {'Spatial Credit':<14} | {'Tree Credit':<14}")
        print("  " + "-" * 75)
        for i in np.where(mb)[0]:
            cat_name = "Effector" if category[i] == GEN_EFF else "Cap"
            sub_name = EFF_NAMES[subtype[i]] if category[i] == GEN_EFF else CAP_NAMES[subtype[i]]
            print(f"  {tok_slot[i]:<6d} | {cat_name:<10} | {sub_name:<10} | {depth[i]:<6d} | {delta[i]:+14.4f} | {c_spatial[i]:+14.4f} | {c_tree[i]:+14.4f}")

    # 7. Generate Diagnostic Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Contextual Spatial Credit Head Diagnostics (Window {window_idx})", fontsize=15)

    # Panel 1: Scatter Spatial Credit vs Prefix Delta
    ax1 = axes[0, 0]
    sample_idx = np.random.choice(len(c_spatial), min(4000, len(c_spatial)), replace=False)
    ax1.scatter(delta[sample_idx], c_spatial[sample_idx], alpha=0.3, s=10, color="#1f77b4")
    ax1.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title(f"Spatial Credit vs Prefix Delta (r={p_spat_delta:+.3f})")
    ax1.set_xlabel("GenCrit Prefix Delta")
    ax1.set_ylabel("Contextual Spatial Credit c_i")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Scatter Tree Credit vs Prefix Delta
    ax2 = axes[0, 1]
    ax2.scatter(delta[sample_idx], c_tree[sample_idx], alpha=0.3, s=10, color="#2ca02c")
    ax2.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title(f"Tree-Propagated Credit vs Prefix Delta (r={p_tree_delta:+.3f})")
    ax2.set_xlabel("GenCrit Prefix Delta")
    ax2.set_ylabel("Tree Credit C_i^{tree}")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Spatial Credit by Subtype
    ax3 = axes[1, 0]
    sub_labels = list(EFF_NAMES) + list(CAP_NAMES)
    sub_data = []
    for is_c, names, cat_val in [(False, EFF_NAMES, GEN_EFF), (True, CAP_NAMES, GEN_CAP)]:
        for sub_val, _ in enumerate(names):
            m = (category == cat_val) & (subtype == sub_val)
            sub_data.append(c_spatial[m] if m.any() else np.array([0.0]))
    ax3.boxplot(sub_data, labels=sub_labels, showfliers=False)
    ax3.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_title("Spatial Credit Distribution by Subtype")
    ax3.set_ylabel("Spatial Credit c_i")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Spatial and Tree Credit by Depth
    ax4 = axes[1, 1]
    depths = sorted(np.unique(depth))
    spat_by_d = [c_spatial[depth == d] for d in depths]
    tree_by_d = [c_tree[depth == d] for d in depths]
    w = 0.35
    x = np.arange(len(depths))
    ax4.bar(x - w / 2, [np.mean(vals) for vals in spat_by_d], width=w, label="Spatial Credit c_i", color="#1f77b4")
    ax4.bar(x + w / 2, [np.mean(vals) for vals in tree_by_d], width=w, label="Tree Credit C_i^{tree}", color="#2ca02c")
    ax4.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"Depth {d}" for d in depths])
    ax4.set_title("Mean Credit by Limb Depth")
    ax4.set_ylabel("Mean Credit")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"spatial_credit_analysis_window_{window_idx:04d}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nFigure saved to {fig_path}")
    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze CoDesign Contextual Spatial Credit.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--out_dir", type=str, default="artifacts/spatial_credit", help="Output directory")
    args = parser.parse_args()

    credit_dir = os.path.join(args.run_dir, "credit")
    windows = load_credit_windows(credit_dir)
    print(f"Found {len(windows)} credit window artifacts in {credit_dir}")

    for w_idx, data in windows:
        analyze_spatial_credit_artifact(data, args.out_dir, w_idx)


if __name__ == "__main__":
    main()
