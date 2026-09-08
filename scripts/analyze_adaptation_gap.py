#!/usr/bin/env python3
"""Analysis script for CoDesign post-adaptation evaluation and adaptation gap.

Reads post_eval_window_*.npz artifacts and produces:
1. Scatter plot of R_train vs R_post with identity line and rank correlations.
2. Plots of adaptation_gap vs morphology complexity (effector count, module count, depth).
3. Histogram and CDF of normalized rank shifts.
4. Quantitative report addressing whether the current generator target R_train penalizes
   morphologies that are harder to learn initially but superior after controller adaptation.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_post_eval_windows(post_eval_dir: str) -> List[Tuple[int, Dict[str, np.ndarray]]]:
    """Loads all post_eval_window_*.npz files in window order."""
    pattern = os.path.join(post_eval_dir, "post_eval_window_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No post-adaptation evaluation artifacts found matching {pattern}")

    windows = []
    for f in files:
        data = np.load(f)
        w_idx = int(data["meta__gen_window"]) if "meta__gen_window" in data else -1
        windows.append((w_idx, {k: data[k] for k in data.files}))
    return windows


def plot_adaptation_analysis(windows: List[Tuple[int, Dict[str, np.ndarray]]], out_dir: str):
    """Generates comprehensive plots comparing R_train vs R_post across morphologies."""
    os.makedirs(out_dir, exist_ok=True)

    all_r_train = []
    all_r_post = []
    all_gaps = []
    all_eff = []
    all_mod = []
    all_depth = []
    all_rank_shifts = []

    for w_idx, data in windows:
        all_r_train.append(data["R_train"])
        all_r_post.append(data["R_post"])
        all_gaps.append(data["adaptation_gap"])
        all_eff.append(data["effector_count"])
        all_mod.append(data["module_count"])
        all_depth.append(data["max_depth"])
        all_rank_shifts.append(data["rank_shift"])

    r_train = np.concatenate(all_r_train)
    r_post = np.concatenate(all_r_post)
    gaps = np.concatenate(all_gaps)
    eff_counts = np.concatenate(all_eff)
    mod_counts = np.concatenate(all_mod)
    max_depths = np.concatenate(all_depth)
    rank_shifts = np.concatenate(all_rank_shifts)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Post-Adaptation Evaluation: R_train vs R_post & Adaptation Gap", fontsize=15)

    # 1. R_train vs R_post Scatter Plot
    ax = axes[0, 0]
    sample_idx = np.random.choice(len(r_train), min(3000, len(r_train)), replace=False)
    ax.scatter(r_train[sample_idx], r_post[sample_idx], alpha=0.3, s=10, color="#1f77b4", label="Bodies")

    lo = min(np.min(r_train), np.min(r_post))
    hi = max(np.max(r_train), np.max(r_post))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Identity (R_post = R_train)")

    # Correlation
    pearson_r = np.corrcoef(r_train, r_post)[0, 1] if np.std(r_train) > 1e-6 and np.std(r_post) > 1e-6 else float("nan")
    rx = r_train.argsort().argsort().astype(float); ry = r_post.argsort().argsort().astype(float)
    spearman_r = np.corrcoef(rx, ry)[0, 1] if np.std(rx) > 1e-6 and np.std(ry) > 1e-6 else float("nan")

    ax.set_title(f"1. R_train vs R_post (Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f})")
    ax.set_xlabel("R_train (Window-averaged Return Target)")
    ax.set_ylabel("R_post (Final Adapted Controller Return)")
    ax.legend(fontsize=9)

    # 2. Adaptation Gap vs Effector Count
    ax = axes[0, 1]
    unique_eff = sorted(np.unique(eff_counts))
    eff_means = [np.mean(gaps[eff_counts == k]) for k in unique_eff]
    eff_stds = [np.std(gaps[eff_counts == k]) for k in unique_eff]
    ax.errorbar(unique_eff, eff_means, yerr=eff_stds, fmt="-o", color="#2ca02c", capsize=5, lw=2)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_title("2. Adaptation Gap (R_post - R_train) vs Effector Count")
    ax.set_xlabel("Actuated Effector Count (DOFs)")
    ax.set_ylabel("Adaptation Gap ± 1 Std")
    ax.set_xticks(unique_eff)

    # 3. Adaptation Gap vs Max Limb Depth
    ax = axes[1, 0]
    unique_d = sorted(np.unique(max_depths))
    depth_means = [np.mean(gaps[max_depths == d]) for d in unique_d]
    depth_stds = [np.std(gaps[max_depths == d]) for d in unique_d]
    ax.errorbar(unique_d, depth_means, yerr=depth_stds, fmt="-s", color="#ff7f0e", capsize=5, lw=2)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_title("3. Adaptation Gap vs Max Limb Depth")
    ax.set_xlabel("Max Limb Depth (0 = swing only, 1..3 = knee chain)")
    ax.set_ylabel("Adaptation Gap ± 1 Std")
    ax.set_xticks(unique_d)

    # 4. Rank Shift Distribution
    ax = axes[1, 1]
    ax.hist(rank_shifts * 100, bins=40, color="#d62728", edgecolor="black", alpha=0.7, density=True)
    ax.axvline(10.0, color="black", linestyle="--", linewidth=1.2, label="Material Shift Threshold (10%)")
    shift_frac = np.mean(rank_shifts > 0.10) * 100
    ax.set_title(f"4. Rank Shift Distribution ({shift_frac:.1f}% shifted > 10%)")
    ax.set_xlabel("Rank Shift (|Rank_post - Rank_train| / N) in Percentile Points")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "adaptation_gap_analysis.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Adaptation gap diagnostic plot saved to {fig_path}")

    # Summary report
    print("\n=======================================================")
    print("       POST-ADAPTATION EVALUATION AUDIT SUMMARY")
    print("=======================================================")
    print(f"Total windows evaluated:         {len(windows)}")
    print(f"Total bodies evaluated:          {len(r_train)}")
    print(f"Mean R_train:                    {np.mean(r_train):.4f} ± {np.std(r_train):.4f}")
    print(f"Mean R_post:                     {np.mean(r_post):.4f} ± {np.std(r_post):.4f}")
    print(f"Mean Adaptation Gap:             {np.mean(gaps):+.4f} ± {np.std(gaps):.4f}")
    print(f"Positive Gap Fraction (R_post > R_train): {np.mean(gaps > 0) * 100:.1f}%")
    print(f"Pearson Correlation (R_tr, R_po):{pearson_r:.4f}")
    print(f"Spearman Rank Correlation:       {spearman_r:.4f}")
    print(f"Bodies with Rank Shift > 10%:    {shift_frac:.1f}%")
    print("-------------------------------------------------------")
    print("Adaptation Gap by Effector Count:")
    for k, mean_g, std_g in zip(unique_eff, eff_means, eff_stds):
        n_k = np.sum(eff_counts == k)
        print(f"  {k:2d} Effectors (N={n_k:5d}): Mean Gap = {mean_g:+.4f} ± {std_g:.4f}")
    print("-------------------------------------------------------")
    print("Adaptation Gap by Max Limb Depth:")
    for d, mean_d, std_d in zip(unique_d, depth_means, depth_stds):
        n_d = np.sum(max_depths == d)
        print(f"  Depth {d:1d} (N={n_d:5d}): Mean Gap = {mean_d:+.4f} ± {std_d:.4f}")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze CoDesign post-adaptation evaluation.")
    parser.add_argument("--post_eval_dir", type=str, required=True,
                        help="Path to directory containing post_eval_window_*.npz files")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    out_dir = args.out_dir or args.post_eval_dir
    windows = load_post_eval_windows(args.post_eval_dir)
    plot_adaptation_analysis(windows, out_dir)


if __name__ == "__main__":
    main()
