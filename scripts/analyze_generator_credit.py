#!/usr/bin/env python3
"""Minimal visual and statistical analysis script for TransformerRL CoDesign generator credit.

Reads the logged credit artifacts (.npz) from an experiment run and produces:
1. Histogram of gencrit_delta (marginal credit distribution)
2. Credit vs generation depth (box/violin/mean plot)
3. Credit vs generation step / order
4. Within-body credit std over training
5. Generator entropy and diversity vs training
6. Body return and controller return vs training
7. GenCrit prediction vs actual body return
8. Heatmap / table of mean credit by module subtype x depth
9. Telescoping-residual diagnostic
10. Context-dependence analysis: credit variation of identical subtypes across contextual factors
    (depth, parent subtype, limb slot, total body module count)
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EFF_NAMES = ("swing", "knee", "twist")
CAP_NAMES = ("bare", "foot", "pad", "ball")


def load_credit_windows(credit_dir: str) -> List[Tuple[int, Dict[str, np.ndarray]]]:
    """Loads all credit_window_*.npz files in chronological window order."""
    pattern = os.path.join(credit_dir, "credit_window_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No credit artifacts found matching {pattern}")

    windows = []
    for f in files:
        data = np.load(f)
        w_idx = int(data["meta__gen_window"]) if "meta__gen_window" in data else -1
        windows.append((w_idx, {k: data[k] for k in data.files}))
    return windows


def analyze_context_dependence(all_records: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Evaluates whether marginal credit for each module subtype varies with morphology context:
    depth, parent module, limb slot, and total body module count.
    """
    subtypes = all_records["subtype"]
    categories = all_records["category"]
    deltas = all_records["delta"]
    depths = all_records["depth"]
    slots = all_records["limb_slot"]
    parents = all_records["parent_module_id"]
    body_ids = all_records["body_id"]

    # Calculate total module count per body
    unique_bodies, body_counts = np.unique(body_ids, return_counts=True)
    body_to_count = dict(zip(unique_bodies, body_counts))
    tot_counts = np.array([body_to_count[b] for b in body_ids])

    results = {}

    # Analyze for each effector subtype
    for cat_val, names in [(0, EFF_NAMES), (1, CAP_NAMES)]:
        cat_str = "eff" if cat_val == 0 else "cap"
        for sub_id, name in enumerate(names):
            mask = (categories == cat_val) & (subtypes == sub_id)
            if not np.any(mask):
                continue
            sub_deltas = deltas[mask]
            sub_depths = depths[mask]
            sub_slots = slots[mask]
            sub_parents = parents[mask]
            sub_tot = tot_counts[mask]

            total_var = float(np.var(sub_deltas))

            # Variance across depths
            depth_means = [np.mean(sub_deltas[sub_depths == d]) for d in np.unique(sub_depths)]
            depth_var = float(np.var(depth_means)) if len(depth_means) > 1 else 0.0

            # Variance across limb slots
            slot_means = [np.mean(sub_deltas[sub_slots == s]) for s in np.unique(sub_slots)]
            slot_var = float(np.var(slot_means)) if len(slot_means) > 1 else 0.0

            # Variance across total body size
            size_means = [np.mean(sub_deltas[sub_tot == sz]) for sz in np.unique(sub_tot)]
            size_var = float(np.var(size_means)) if len(size_means) > 1 else 0.0

            results[f"{cat_str}_{name}"] = {
                "count": int(len(sub_deltas)),
                "mean_credit": float(np.mean(sub_deltas)),
                "std_credit": float(np.std(sub_deltas)),
                "total_var": total_var,
                "var_across_depths": depth_var,
                "var_across_slots": slot_var,
                "var_across_body_sizes": size_var,
            }

    return results


def plot_diagnostics(windows: List[Tuple[int, Dict[str, np.ndarray]]], out_dir: str):
    """Generates and saves the required diagnostic figures."""
    os.makedirs(out_dir, exist_ok=True)

    # Aggregate records across all loaded windows
    all_deltas = []
    all_v_before = []
    all_v_after = []
    all_depths = []
    all_orders = []
    all_cats = []
    all_subs = []
    all_r = []

    # Window-level series over training
    w_indices = []
    within_body_std_means = []
    within_body_std_medians = []
    within_body_uniform_fracs = []
    delta_means = []
    delta_stds = []
    mean_abs_deltas = []
    entropies = []
    mean_body_returns = []
    v_full_means = []
    v_full_corrs = []
    telescoping_max_residuals = []

    for w_idx, data in windows:
        w_indices.append(w_idx)
        d = data["delta"]
        b_ids = data["body_id"]
        v_bef = data["v_before"]
        v_aft = data["v_after"]
        b_ret = data["body_return"]

        all_deltas.append(d)
        all_v_before.append(v_bef)
        all_v_after.append(v_aft)
        all_depths.append(data["depth"])
        all_orders.append(data["gen_order"])
        all_cats.append(data["category"])
        all_subs.append(data["subtype"])
        all_r.append(b_ret)

        delta_means.append(np.mean(d))
        delta_stds.append(np.std(d))
        mean_abs_deltas.append(np.mean(np.abs(d)))
        entropies.append(np.mean(data["entropy"]) if "entropy" in data else 0.0)

        # Body-level within-body std
        unique_b = np.unique(b_ids)
        body_stds = []
        body_sums = []
        body_returns = []
        for b in unique_b:
            b_mask = (b_ids == b)
            if np.sum(b_mask) > 1:
                body_stds.append(np.std(d[b_mask], ddof=1))
            body_sums.append(np.sum(d[b_mask]))
            body_returns.append(b_ret[b_mask][0])

        within_body_std_means.append(np.mean(body_stds) if body_stds else 0.0)
        within_body_std_medians.append(np.median(body_stds) if body_stds else 0.0)
        within_body_uniform_fracs.append(np.mean(np.array(body_stds) < 1e-4) if body_stds else 0.0)

        mean_body_returns.append(np.mean(body_returns) if body_returns else 0.0)

        # Telescoping check for this window
        # In each body, sum(delta) vs (v_after_last - v_before_first)
        # Using the saved flat records:
        residuals = []
        for b in unique_b:
            b_mask = (b_ids == b)
            sum_d = np.sum(d[b_mask])
            v_start = v_bef[b_mask][0]   # initial prefix of this body
            v_end = v_aft[b_mask][-1]    # final prefix of this body
            residuals.append(abs(sum_d - (v_end - v_start)))
        telescoping_max_residuals.append(np.max(residuals) if residuals else 0.0)

        # Calibration
        unique_returns = np.array(body_returns)
        # v_full estimate
        v_full_vals = np.array([v_aft[b_ids == b][-1] for b in unique_b])
        v_full_means.append(np.mean(v_full_vals) if len(v_full_vals) else 0.0)
        if len(unique_returns) > 2 and np.std(unique_returns) > 1e-6 and np.std(v_full_vals) > 1e-6:
            v_full_corrs.append(np.corrcoef(unique_returns, v_full_vals)[0, 1])
        else:
            v_full_corrs.append(float("nan"))

    # Concatenate all records
    cat_deltas = np.concatenate(all_deltas)
    cat_depths = np.concatenate(all_depths)
    cat_orders = np.concatenate(all_orders)
    cat_categories = np.concatenate(all_cats)
    cat_subtypes = np.concatenate(all_subs)
    cat_returns = np.concatenate(all_r)
    cat_v_after = np.concatenate(all_v_after)

    # -------------------------------------------------------------
    # Figure 1: 9-panel comprehensive dashboard
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("TransformerRL CoDesign Generator Credit Assignment Diagnostics", fontsize=16)

    # 1. Histogram of gencrit_delta
    ax = axes[0, 0]
    ax.hist(cat_deltas, bins=60, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2, label="Zero credit")
    ax.axvline(np.mean(cat_deltas), color="green", linestyle="-", linewidth=1.5,
               label=f"Mean: {np.mean(cat_deltas):.4f}")
    ax.set_title("1. Distribution of GenCrit Marginal Deltas")
    ax.set_xlabel(r"Marginal Credit $\Delta V_i = V(p_{i+1}) - V(p_i)$")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    # 2. Credit vs Generation Depth
    ax = axes[0, 1]
    depth_bins = sorted(np.unique(cat_depths))
    depth_data = [cat_deltas[cat_depths == d] for d in depth_bins]
    means = [np.mean(d) for d in depth_data]
    stds = [np.std(d) for d in depth_data]
    ax.errorbar(depth_bins, means, yerr=stds, fmt="-o", color="#2ca02c", capsize=5, lw=2)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_title("2. GenCrit Credit vs Generation Depth")
    ax.set_xlabel("Depth within Limb (0 = swing / proximal)")
    ax.set_ylabel("Mean Delta ± 1 Std")
    ax.set_xticks(depth_bins)

    # 3. Credit vs Generation Step (Order within body)
    ax = axes[0, 2]
    order_bins = sorted(np.unique(cat_orders))
    order_means = [np.mean(cat_deltas[cat_orders == o]) for o in order_bins]
    ax.plot(order_bins, order_means, "-s", color="#ff7f0e", lw=2)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_title("3. GenCrit Credit vs Generation Order")
    ax.set_xlabel("Construction Step within Body (0 .. K-1)")
    ax.set_ylabel("Mean Delta")

    # 4. Within-Body Credit Std over Training
    ax = axes[1, 0]
    ax.plot(w_indices, within_body_std_means, "-o", color="#d62728", lw=2, label="Mean within-body std")
    ax.plot(w_indices, within_body_std_medians, "--^", color="#9467bd", lw=1.5, label="Median within-body std")
    ax.set_title("4. Within-Body Credit Variation over Windows")
    ax.set_xlabel("Resample Window")
    ax.set_ylabel("Credit Std within Body")
    ax.legend(fontsize=9)

    # 5. Generator Entropy & Diversity vs Training
    ax = axes[1, 1]
    ax.plot(w_indices, entropies, "-o", color="#8c564b", lw=2, label="Generator Step Entropy")
    ax.set_title("5. Generator Policy Entropy over Training")
    ax.set_xlabel("Resample Window")
    ax.set_ylabel("Mean Step Entropy (nats)")
    ax.legend(fontsize=9)

    # 6. Body Return vs Window
    ax = axes[1, 2]
    ax.plot(w_indices, mean_body_returns, "-o", color="#17becf", lw=2, label="Body Return (R)")
    ax.set_title("6. Body Quality (Realized Return) over Training")
    ax.set_xlabel("Resample Window")
    ax.set_ylabel("Mean Body Return R")
    ax.legend(fontsize=9)

    # 7. GenCrit Prediction vs Actual Body Return
    ax = axes[2, 0]
    ax.scatter(cat_returns[:2000], cat_v_after[:2000], alpha=0.25, s=8, color="#7f7f7f")
    lims = [min(np.min(cat_returns), np.min(cat_v_after)), max(np.max(cat_returns), np.max(cat_v_after))]
    ax.plot(lims, lims, "k--", lw=1.2, label="Identity (y=x)")
    ax.set_title("7. GenCrit Value vs Body Return Target")
    ax.set_xlabel("Body Return Target (R)")
    ax.set_ylabel("GenCrit Value Prediction")
    ax.legend(fontsize=9)

    # 8. Heatmap of Mean Credit by Subtype x Depth
    ax = axes[2, 1]
    # Rows: 3 effectors + 4 caps = 7 types. Columns: depths 0..3
    type_labels = [f"eff_{n}" for n in EFF_NAMES] + [f"cap_{n}" for n in CAP_NAMES]
    grid = np.full((len(type_labels), 4), np.nan)
    for t_i, (c_v, s_v) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3)]):
        for d in range(4):
            m = (cat_categories == c_v) & (cat_subtypes == s_v) & (cat_depths == d)
            if np.any(m):
                grid[t_i, d] = np.mean(cat_deltas[m])

    im = ax.imshow(grid, cmap="coolwarm", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"d={d}" for d in range(4)])
    ax.set_yticks(range(len(type_labels)))
    ax.set_yticklabels(type_labels)
    ax.set_title("8. Mean Credit by Subtype × Depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Module Type")

    # 9. Telescoping-Residual Diagnostic
    ax = axes[2, 2]
    ax.plot(w_indices, telescoping_max_residuals, "-s", color="#bcbd22", lw=2)
    ax.set_title("9. Max Telescoping Residual |ΣΔV - (V_end - V_0)|")
    ax.set_xlabel("Resample Window")
    ax.set_ylabel("Max Residual (|error|)")
    ax.set_yscale("log" if np.max(telescoping_max_residuals) > 0 else "linear")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "generator_credit_analysis.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Diagnostic plots saved to {fig_path}")

    # -------------------------------------------------------------
    # Print Summary Report & Context Dependence
    # -------------------------------------------------------------
    flat_all = {
        "subtype": cat_subtypes,
        "category": cat_categories,
        "delta": cat_deltas,
        "depth": cat_depths,
        "limb_slot": np.concatenate([data["limb_slot"] for _, data in windows]),
        "parent_module_id": np.concatenate([data["parent_module_id"] for _, data in windows]),
        "body_id": np.concatenate([data["body_id"] for _, data in windows]),
    }
    context_res = analyze_context_dependence(flat_all)

    print("\n=======================================================")
    print("      GENERATOR CREDIT ASSIGNMENT AUDIT SUMMARY")
    print("=======================================================")
    print(f"Total resample windows analyzed: {len(windows)}")
    print(f"Total construction decisions:    {len(cat_deltas)}")
    print(f"Marginal Delta Mean:             {np.mean(cat_deltas):.5f}")
    print(f"Marginal Delta Std:              {np.std(cat_deltas):.5f}")
    print(f"Marginal Delta Min / Max:        [{np.min(cat_deltas):.5f}, {np.max(cat_deltas):.5f}]")
    print(f"Fraction Positive Credit:        {np.mean(cat_deltas > 0) * 100:.1f}%")
    print(f"Fraction Negative Credit:        {np.mean(cat_deltas < 0) * 100:.1f}%")
    print(f"Fraction Near-Zero (|Δ|<1e-4):   {np.mean(np.abs(cat_deltas) < 1e-4) * 100:.1f}%")
    print(f"Mean Within-Body Credit Std:     {np.mean(within_body_std_means):.5f}")
    print(f"Max Telescoping Residual:        {np.max(telescoping_max_residuals):.2e}")
    print("-------------------------------------------------------")
    print("Context Dependence Breakdown:")
    for k, v in context_res.items():
        print(f"  {k:12s} (N={v['count']:5d}): Mean={v['mean_credit']:+.4f}, "
              f"TotStd={v['std_credit']:.4f}, DepthVar={v['var_across_depths']:.4f}, "
              f"SlotVar={v['var_across_slots']:.4f}")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze CoDesign generator credit assignment.")
    parser.add_argument("--credit_dir", type=str, required=True,
                        help="Path to directory containing credit_window_*.npz files")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for plots (defaults to credit_dir)")
    args = parser.parse_args()

    out_dir = args.out_dir or args.credit_dir
    windows = load_credit_windows(args.credit_dir)
    plot_diagnostics(windows, out_dir)


if __name__ == "__main__":
    main()
