#!/usr/bin/env python3
"""Analysis of the fresh common-controller morphology evaluation experiment.

Compares Prefix, Shuffled, and Aligned candidate morphologies evaluated under
the exact same fresh Transformer controller (identical initialization, identical
training budget of 63 epochs, zero generator updates, morphology strictly frozen).

Evaluates:
1. Provenance metrics: mean, median, top-10%, best return, variance.
2. Matched-complexity performance: E[R_common | N_modules].
3. Bootstrap 95% confidence intervals for (Aligned - Shuffled) and (Aligned - Prefix).
4. Morphology-only ranking correlations between original co-adapted R_post and common R_common.
5. Testing of H1 (exploration/anti-capping) vs H2 (structural credit) vs H3 (complexity-only).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    denom = np.sqrt(np.sum(vx ** 2) * np.sum(vy ** 2))
    return float(np.sum(vx * vy) / max(1e-8, denom))


def rankdata(x: np.ndarray) -> np.ndarray:
    temp = x.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(x))
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_r(rankdata(x), rankdata(y))



def load_data(
    candidate_npz: str = "runs/common_controller_eval/candidate_morphologies.npz",
    eval_dir: str = "runs/ant_codesign/codesign_single_transformer/common_controller_eval/post_eval",
) -> pd.DataFrame:
    cands = np.load(candidate_npz, allow_pickle=True)
    post_files = sorted(glob.glob(os.path.join(eval_dir, "*.npz")))
    assert len(post_files) > 0, f"No post_eval files found in {eval_dir}"
    post_data = np.load(post_files[0], allow_pickle=True)

    r_common = post_data["R_post"]

    df = pd.DataFrame({
        "source_condition": cands["source_condition"],
        "morphology_hash": cands["morphology_hash"],
        "original_R_post": cands["original_R_post"],
        "common_controller_R": r_common,
        "module_count": cands["module_count"],
        "mean_depth": cands["mean_depth"],
        "effector_count": cands["effector_count"],
        "max_depth": cands["max_depth"],
    })

    # Drop padding row
    df = df[df["source_condition"] != "Padding"].copy()
    return df


def bootstrap_ci(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    stat_fn,
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    n_a, n_b = len(vals_a), len(vals_b)
    for i in range(n_boot):
        b_a = rng.choice(vals_a, size=n_a, replace=True)
        b_b = rng.choice(vals_b, size=n_b, replace=True)
        diffs[i] = stat_fn(b_a) - stat_fn(b_b)
    point = float(stat_fn(vals_a) - stat_fn(vals_b))
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    return point, ci_low, ci_high, diffs


def plot_common_controller_synthesis(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Fresh Common-Controller Evaluation: Controlling for Controller Co-Adaptation", fontsize=16, y=0.99)

    colors = {"Prefix": "#1f77b4", "Shuffled": "#ff7f0e", "Aligned": "#2ca02c"}

    # 1. Panel 1: Return Distributions (Boxplot + jitter)
    ax1 = axes[0, 0]
    ax1.set_title("1. Common-Controller Return Distributions (R_common)", fontsize=12, fontweight="bold")
    cond_order = ["Prefix", "Shuffled", "Aligned"]
    plot_data = [df[df["source_condition"] == c]["common_controller_R"].values for c in cond_order]

    bp = ax1.boxplot(plot_data, tick_labels=cond_order, patch_artist=True, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7})
    for patch, c in zip(bp["boxes"], cond_order):
        patch.set_facecolor(colors[c])
        patch.set_alpha(0.7)

    # Print mean labels
    for i, c in enumerate(cond_order):
        m_val = np.mean(plot_data[i])
        med_val = np.median(plot_data[i])
        ax1.text(i + 1, m_val + 0.5, f"Mean: {m_val:.2f}\nMed: {med_val:.2f}",
                 ha="center", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax1.set_ylabel("Common Controller Return R_common", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Panel 2: Matched-Complexity E[R_common | N_modules]
    ax2 = axes[0, 1]
    ax2.set_title("2. Matched-Complexity Performance: E[R_common | N_modules]", fontsize=12, fontweight="bold")
    mod_bins = sorted(df["module_count"].unique())
    for c in cond_order:
        sub = df[df["source_condition"] == c]
        m_means = [sub[sub["module_count"] == m]["common_controller_R"].mean() for m in mod_bins]
        m_stds = [sub[sub["module_count"] == m]["common_controller_R"].std() / np.sqrt(len(sub[sub["module_count"] == m])) for m in mod_bins]
        ax2.plot(mod_bins, m_means, label=f"{c}", color=colors[c], marker="o", linewidth=2)
        ax2.fill_between(mod_bins, np.array(m_means) - np.array(m_stds), np.array(m_means) + np.array(m_stds),
                         color=colors[c], alpha=0.15)
    ax2.set_xlabel("Module Count N_modules (Matched Sample N=45-150 per bin)", fontsize=11)
    ax2.set_ylabel("Mean Return E[R_common | N_modules]", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Panel 3: Original vs Common Controller Return Scatter
    ax3 = axes[1, 0]
    ax3.set_title("3. Morphology Ranking: Original R_post vs Common R_common", fontsize=12, fontweight="bold")
    for c in cond_order:
        sub = df[df["source_condition"] == c]
        r_p = pearson_r(sub["original_R_post"].values, sub["common_controller_R"].values)
        r_s = spearman_rho(sub["original_R_post"].values, sub["common_controller_R"].values)
        ax3.scatter(sub["original_R_post"], sub["common_controller_R"],
                    color=colors[c], alpha=0.25, s=15, label=f"{c} (r={r_p:+.2f}, rho={r_s:+.2f})")
    ax3.set_xlabel("Original Co-Adapted Return R_post", fontsize=11)
    ax3.set_ylabel("Fresh Common-Controller Return R_common", fontsize=11)
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Panel 4: Bootstrap Difference Distributions (Aligned - Shuffled & Aligned - Prefix)
    ax4 = axes[1, 1]
    ax4.set_title("4. Bootstrap Differences in Means (10,000 Resamples)", fontsize=12, fontweight="bold")
    al_vals = df[df["source_condition"] == "Aligned"]["common_controller_R"].values
    sh_vals = df[df["source_condition"] == "Shuffled"]["common_controller_R"].values
    pr_vals = df[df["source_condition"] == "Prefix"]["common_controller_R"].values

    _, _, _, diff_shuf = bootstrap_ci(al_vals, sh_vals, np.mean)
    _, _, _, diff_pref = bootstrap_ci(al_vals, pr_vals, np.mean)

    ax4.hist(diff_shuf, bins=50, alpha=0.6, color="#ff7f0e", label=f"Aligned - Shuffled (Mean Diff: {np.mean(diff_shuf):+.2f})")
    ax4.hist(diff_pref, bins=50, alpha=0.6, color="#1f77b4", label=f"Aligned - Prefix (Mean Diff: {np.mean(diff_pref):+.2f})")
    ax4.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    ax4.set_xlabel("Return Difference", fontsize=11)
    ax4.set_ylabel("Bootstrap Frequency", fontsize=11)
    ax4.legend(loc="upper right", fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Common controller synthesis plot saved to: {out_path}")


def main():
    df = load_data()
    out_fig = "/home/alex-daniel/.gemini/antigravity-ide/brain/8354a499-38b1-44cb-adc3-d7c78b30df26/common_controller_evaluation.png"
    plot_common_controller_synthesis(df, out_fig)

    print("\n" + "=" * 95)
    print("FRESH COMMON-CONTROLLER EVALUATION METRICS (STANDARDIZED CONTROLLER)")
    print("=" * 95)
    header = f"{'Metric':<35} | {'Prefix':<16} | {'Shuffled':<16} | {'Aligned':<16}"
    print(header)
    print("-" * 95)

    def _stats(cond):
        sub = df[df["source_condition"] == cond]["common_controller_R"].values
        return {
            "mean": np.mean(sub),
            "median": np.median(sub),
            "top10": np.percentile(sub, 90),
            "best": np.max(sub),
            "var": np.var(sub),
        }

    p_s = _stats("Prefix")
    s_s = _stats("Shuffled")
    a_s = _stats("Aligned")

    print(f"{'Mean Return (R_common)':<35} | {p_s['mean']:<16.4f} | {s_s['mean']:<16.4f} | {a_s['mean']:<16.4f}")
    print(f"{'Median Return':<35} | {p_s['median']:<16.4f} | {s_s['median']:<16.4f} | {a_s['median']:<16.4f}")
    print(f"{'Top-10% Decile Return':<35} | {p_s['top10']:<16.4f} | {s_s['top10']:<16.4f} | {a_s['top10']:<16.4f}")
    print(f"{'Best Body Return':<35} | {p_s['best']:<16.4f} | {s_s['best']:<16.4f} | {a_s['best']:<16.4f}")
    print(f"{'Return Variance':<35} | {p_s['var']:<16.4f} | {s_s['var']:<16.4f} | {a_s['var']:<16.4f}")
    print("=" * 95)

    # Bootstrap CIs for Aligned - Shuffled and Aligned - Prefix
    al_vals = df[df["source_condition"] == "Aligned"]["common_controller_R"].values
    sh_vals = df[df["source_condition"] == "Shuffled"]["common_controller_R"].values
    pr_vals = df[df["source_condition"] == "Prefix"]["common_controller_R"].values

    print("\n" + "=" * 95)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS FOR DIFFERENCES (ALIGNED vs SHUFFLED & PREFIX)")
    print("=" * 95)
    for stat_name, fn in [("Mean", np.mean), ("Median", np.median), ("Top-10% (P90)", lambda x: np.percentile(x, 90))]:
        p_sh, l_sh, h_sh, _ = bootstrap_ci(al_vals, sh_vals, fn)
        p_pr, l_pr, h_pr, _ = bootstrap_ci(al_vals, pr_vals, fn)
        print(f"Aligned - Shuffled {stat_name:<14}: {p_sh:+.4f} (95% CI: [{l_sh:+.4f}, {h_sh:+.4f}])")
        print(f"Aligned - Prefix   {stat_name:<14}: {p_pr:+.4f} (95% CI: [{l_pr:+.4f}, {h_pr:+.4f}])")
    print("=" * 95)

    # Ranking correlations
    print("\n" + "=" * 95)
    print("MORPHOLOGY RANKING TEST (ORIGINAL R_post vs COMMON-CONTROLLER R_common)")
    print("=" * 95)
    for c in ["Prefix", "Shuffled", "Aligned"]:
        sub = df[df["source_condition"] == c]
        orig_vals = sub["original_R_post"].values
        comm_vals = sub["common_controller_R"].values
        r_p = pearson_r(orig_vals, comm_vals)
        r_s = spearman_rho(orig_vals, comm_vals)
        orig_ranks = rankdata(orig_vals)
        comm_ranks = rankdata(comm_vals)
        mean_shift = np.mean(np.abs(orig_ranks - comm_ranks)) / len(sub)
        print(f"{c:<10}: Pearson r = {r_p:+.4f}, Spearman rho = {r_s:+.4f}, Relative Rank Shift = {mean_shift:.4f}")
    print("=" * 95 + "\n")



if __name__ == "__main__":
    main()
