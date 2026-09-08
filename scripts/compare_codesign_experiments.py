#!/usr/bin/env python3
"""Comparative analysis script for CoDesign experiments: Baseline (return_target=train) vs Proposed (return_target=post).

Loads TensorBoard scalars and .npz artifacts from both runs and produces:
1. 6-panel comparison figure:
   - Panel 1: R_train and R_post over training windows.
   - Panel 2: Mean effector count over training.
   - Panel 3: Morphology complexity (module count and limb depth).
   - Panel 4: Generator entropy and diversity.
   - Panel 5: Cap vs effector marginal credit (A_cap - A_eff).
   - Panel 6: Final adapted performance (R_post distribution / trajectory).
2. Summary comparison table and answers to core scientific questions:
   - Does training GenCrit on R_post alter or reverse the negative effector credit bias?
   - Does the generator retain more complex, higher-effector morphologies?
   - How does final adapted performance compare?
   - What is the wall-clock overhead of post-adaptation evaluation?
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tb_scalars(run_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Loads all scalar series from TensorBoard event files in run_dir.

    Returns:
        dict mapping scalar tag to (steps, values) numpy arrays.
    """
    event_files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not event_files:
        # Check subdirectories
        event_files = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    if not event_files:
        return {}

    acc = EventAccumulator(event_files[-1], size_guidance={"scalars": 0})
    acc.Reload()

    tags = acc.Tags().get("scalars", [])
    series = {}
    for tag in tags:
        events = acc.Scalars(tag)
        steps = np.array([e.step for e in events])
        vals = np.array([e.value for e in events])
        series[tag] = (steps, vals)
    return series


def load_npz_artifacts(run_dir: str, subfolder: str) -> List[Tuple[int, Dict[str, Any]]]:
    """Loads sorted .npz artifacts from run_dir/subfolder."""
    pattern = os.path.join(run_dir, subfolder, "*.npz")
    files = sorted(glob.glob(pattern))
    records = []
    for f in files:
        data = np.load(f)
        w_idx = int(data["meta__gen_window"]) if "meta__gen_window" in data else len(records)
        records.append((w_idx, {k: data[k] for k in data.files}))
    return records


def plot_experiment_comparison(
    base_tb: Dict[str, Tuple[np.ndarray, np.ndarray]],
    prop_tb: Dict[str, Tuple[np.ndarray, np.ndarray]],
    base_post: List[Tuple[int, Dict[str, Any]]],
    prop_post: List[Tuple[int, Dict[str, Any]]],
    base_credit: List[Tuple[int, Dict[str, Any]]],
    prop_credit: List[Tuple[int, Dict[str, Any]]],
    out_path: str,
    base_label: str = "Baseline (target=train)",
    prop_label: str = "Proposed (target=post)",
):
    """Generates the 6-panel comparison figure."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle(f"TransformerRL CoDesign: {base_label} vs {prop_label}", fontsize=16, y=0.99)

    color_base = "#1f77b4"  # blue
    color_prop = "#d62728"  # red

    # Helper to plot scalar if present
    def _plot_series(ax, tb_dict, tag, label, color, linestyle="-", marker="o"):
        if tag in tb_dict:
            steps, vals = tb_dict[tag]
            ax.plot(steps, vals, label=label, color=color, linestyle=linestyle, marker=marker, markersize=3, alpha=0.85)

    # 1. Panel 1: R_train and R_post over training
    ax1 = axes[0, 0]
    ax1.set_title("1. Adaptation Return: R_train vs R_post", fontsize=12)
    _plot_series(ax1, base_tb, "codesign/adaptation/R_train_mean", f"{base_label} R_train", color_base, linestyle="--", marker="s")
    _plot_series(ax1, base_tb, "codesign/adaptation/R_post_mean", f"{base_label} R_post", color_base, linestyle="-", marker="o")
    _plot_series(ax1, prop_tb, "codesign/adaptation/R_train_mean", f"{prop_label} R_train", color_prop, linestyle="--", marker="s")
    _plot_series(ax1, prop_tb, "codesign/adaptation/R_post_mean", f"{prop_label} R_post", color_prop, linestyle="-", marker="o")
    ax1.set_xlabel("Environment Frames / Steps")
    ax1.set_ylabel("Mean Return (Scaled)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Panel 2: Effector count over training
    ax2 = axes[0, 1]
    ax2.set_title("2. Morphology Complexity: Mean Effector / Module Count", fontsize=12)
    def _extract_window_means(post_list, key):
        windows, means = [], []
        for w, d in post_list:
            if key in d:
                windows.append(w)
                means.append(float(np.mean(d[key])))
        return np.array(windows), np.array(means)

    b_w, b_eff = _extract_window_means(base_post, "effector_count")
    p_w, p_eff = _extract_window_means(prop_post, "effector_count")
    if len(b_w) > 0:
        ax2.plot(b_w, b_eff, label=f"{base_label} Effectors", color=color_base, marker="o", linewidth=2)
    if len(p_w) > 0:
        ax2.plot(p_w, p_eff, label=f"{prop_label} Effectors", color=color_prop, marker="s", linewidth=2)
    _plot_series(ax2, base_tb, "build/modulecount", f"{base_label} Modules", color_base, linestyle=":")
    _plot_series(ax2, prop_tb, "build/modulecount", f"{prop_label} Modules", color_prop, linestyle=":")
    ax2.set_xlabel("Generator Window Index / Steps")
    ax2.set_ylabel("Count per Body")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Panel 3: Morphology complexity (module count and limb depth)
    ax3 = axes[1, 0]
    ax3.set_title("3. Morphology Complexity: Max Limb Depth & Active Limbs", fontsize=12)
    b_w, b_depth = _extract_window_means(base_post, "max_depth")
    p_w, p_depth = _extract_window_means(prop_post, "max_depth")
    b_w_l, b_limbs = _extract_window_means(base_post, "active_limbs")
    p_w_l, p_limbs = _extract_window_means(prop_post, "active_limbs")

    if len(b_w) > 0:
        ax3.plot(b_w, b_depth, label=f"{base_label} Max Depth", color=color_base, linestyle="-", marker="o")
    if len(b_w_l) > 0:
        ax3.plot(b_w_l, b_limbs, label=f"{base_label} Active Limbs", color=color_base, linestyle="--", marker="^")
    if len(p_w) > 0:
        ax3.plot(p_w, p_depth, label=f"{prop_label} Max Depth", color=color_prop, linestyle="-", marker="s")
    if len(p_w_l) > 0:
        ax3.plot(p_w_l, p_limbs, label=f"{prop_label} Active Limbs", color=color_prop, linestyle="--", marker="v")
    ax3.set_xlabel("Generator Window Index")
    ax3.set_ylabel("Count")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Panel 4: Generator entropy and body diversity
    ax4 = axes[1, 1]
    ax4.set_title("4. Generator Entropy & Body Diversity", fontsize=12)
    _plot_series(ax4, base_tb, "gen/entropy", f"{base_label} Gen Entropy", color_base, linestyle="-")
    _plot_series(ax4, prop_tb, "gen/entropy", f"{prop_label} Gen Entropy", color_prop, linestyle="-")
    _plot_series(ax4, base_tb, "build/body_diversity", f"{base_label} Diversity", color_base, linestyle="--")
    _plot_series(ax4, prop_tb, "build/body_diversity", f"{prop_label} Diversity", color_prop, linestyle="--")
    ax4.set_xlabel("Environment Frames / Steps")
    ax4.set_ylabel("Entropy / Diversity")
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Panel 5: Cap vs effector marginal credit
    ax5 = axes[2, 0]
    ax5.set_title("5. Marginal Credit: Cap vs Effector Mean Delta", fontsize=12)
    _plot_series(ax5, base_tb, "codesign/credit/effector_mean", f"{base_label} Effector Delta", color_base, linestyle="-", marker="o")
    _plot_series(ax5, prop_tb, "codesign/credit/effector_mean", f"{prop_label} Effector Delta", color_prop, linestyle="-", marker="s")
    _plot_series(ax5, base_tb, "codesign/credit/cap_mean", f"{base_label} Cap Delta", color_base, linestyle=":", marker=None)
    _plot_series(ax5, prop_tb, "codesign/credit/cap_mean", f"{prop_label} Cap Delta", color_prop, linestyle=":", marker=None)
    ax5.axhline(0.0, color="gray", linestyle="--", alpha=0.6)
    ax5.set_xlabel("Environment Frames / Steps")
    ax5.set_ylabel("Mean Marginal Delta (Prefix Difference)")
    ax5.legend(loc="best", fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. Panel 6: Final adapted performance (R_post trajectory and distribution)
    ax6 = axes[2, 1]
    ax6.set_title("6. Final Adapted Performance (R_post vs Quality R)", fontsize=12)
    b_w, b_rpost = _extract_window_means(base_post, "R_post")
    p_w, p_rpost = _extract_window_means(prop_post, "R_post")
    if len(b_w) > 0:
        ax6.plot(b_w, b_rpost, label=f"{base_label} R_post", color=color_base, marker="o", linewidth=2)
    if len(p_w) > 0:
        ax6.plot(p_w, p_rpost, label=f"{prop_label} R_post", color=color_prop, marker="s", linewidth=2)
    _plot_series(ax6, base_tb, "quality/R_mean", f"{base_label} Quality R", color_base, linestyle=":")
    _plot_series(ax6, prop_tb, "quality/R_mean", f"{prop_label} Quality R", color_prop, linestyle=":")
    ax6.set_xlabel("Generator Window Index / Steps")
    ax6.set_ylabel("Return (Scaled)")
    ax6.legend(loc="best", fontsize=9)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Comparison figure saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs Proposed CoDesign Experiments.")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline run directory (return_target=train)")
    parser.add_argument("--proposed", type=str, required=True, help="Path to proposed run directory (return_target=post)")
    parser.add_argument("--out_dir", type=str, default="artifacts/codesign_comparison", help="Output directory")
    args = parser.parse_args()

    print(f"Loading Baseline: {args.baseline}")
    base_tb = load_tb_scalars(args.baseline)
    base_post = load_npz_artifacts(args.baseline, "post_eval")
    base_credit = load_npz_artifacts(args.baseline, "credit")

    print(f"Loading Proposed: {args.proposed}")
    prop_tb = load_tb_scalars(args.proposed)
    prop_post = load_npz_artifacts(args.proposed, "post_eval")
    prop_credit = load_npz_artifacts(args.proposed, "credit")

    out_fig = os.path.join(args.out_dir, "codesign_causal_pilot_comparison.png")
    plot_experiment_comparison(
        base_tb, prop_tb, base_post, prop_post, base_credit, prop_credit, out_fig
    )

    # Print summary metrics table
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON SUMMARY: BASELINE vs PROPOSED")
    print("=" * 80)
    print(f"{'Metric':<40} | {'Baseline (train)':<16} | {'Proposed (post)':<16}")
    print("-" * 80)

    def _latest_val(tb_dict, tag):
        if tag in tb_dict and len(tb_dict[tag][1]) > 0:
            return float(tb_dict[tag][1][-1])
        return float("nan")

    def _npz_latest_mean(post_list, key):
        if post_list and key in post_list[-1][1]:
            return float(np.mean(post_list[-1][1][key]))
        return float("nan")

    metrics = [
        ("R_train (Mean)", "codesign/adaptation/R_train_mean", "R_train"),
        ("R_post (Mean)", "codesign/adaptation/R_post_mean", "R_post"),
        ("Adaptation Gap (R_post - R_train)", "codesign/adaptation/gap_mean", "adaptation_gap"),
        ("Effector Count (Mean)", None, "effector_count"),
        ("Active Limbs (Mean)", None, "active_limbs"),
        ("Max Depth (Mean)", None, "max_depth"),
        ("Cap Delta (Prefix Diff Mean)", "codesign/credit/cap_mean", None),
        ("Effector Delta (Prefix Diff Mean)", "codesign/credit/effector_mean", None),
        ("Rank Correlation (R_train vs R_post)", "codesign/adaptation/corr_spearman", None),
        ("Material Rank Shift Fraction", "codesign/adaptation/material_rank_shift_frac", None),
    ]

    for label, tb_key, npz_key in metrics:
        b_val = float("nan")
        p_val = float("nan")
        if tb_key and tb_key in base_tb:
            b_val = _latest_val(base_tb, tb_key)
        elif npz_key:
            b_val = _npz_latest_mean(base_post, npz_key)

        if tb_key and tb_key in prop_tb:
            p_val = _latest_val(prop_tb, tb_key)
        elif npz_key:
            p_val = _npz_latest_mean(prop_post, npz_key)

        print(f"{label:<40} | {b_val:<16.4f} | {p_val:<16.4f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
