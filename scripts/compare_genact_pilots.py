#!/usr/bin/env python3
"""Comparative analysis script for GenAct causal pilot:
Baseline (use_for_genact=false) vs Spatial-Tree (use_for_genact=true, beta=0.5).

Produces:
1. Multi-panel comparison figure:
   - Panel 1: Out-of-sample pair prediction correlation (Tree vs Prefix diffs vs Delta R).
   - Panel 2: GenAct advantage variance ratio std(beta*C_tree) / std(A_prefix).
   - Panel 3: Morphology complexity (module count and limb count evolution).
   - Panel 4: Morphology diversity and generator entropy.
   - Panel 5: Cap vs effector action probabilities.
   - Panel 6: Performance outcomes (mean, max, top-decile post-adaptation return).
2. Comprehensive summary metrics answering the 6 core research questions.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tb_scalars(run_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Loads all scalar series from TensorBoard event files in run_dir."""
    event_files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not event_files:
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


def plot_genact_comparison(
    base_tb: Dict[str, Tuple[np.ndarray, np.ndarray]],
    tree_tb: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: str,
    base_label: str = "Baseline (Prefix Adv Only)",
    tree_label: str = "Spatial-Tree (Prefix + 0.5 Tree Credit)",
):
    """Generates 6-panel causal comparison figure."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle(f"GenAct Causal Pilot: {base_label} vs {tree_label}", fontsize=16, y=0.99)

    color_base = "#1f77b4"  # Blue
    color_tree = "#2ca02c"  # Green
    color_pref = "#ff7f0e"  # Orange

    def _plot_series(ax, tb_dict, tag, label, color, linestyle="-", marker="o", alpha=1.0):
        if tag in tb_dict:
            s, v = tb_dict[tag]
            valid = ~np.isnan(v) & ~np.isinf(v)
            if np.any(valid):
                ax.plot(s[valid], v[valid], linestyle=linestyle, marker=marker,
                        color=color, label=label, alpha=alpha, linewidth=2)
                return True
        return False

    # Panel 1: Out-of-Sample Pair Correlation
    ax1 = axes[0, 0]
    ax1.set_title("Panel 1: Out-of-Sample Pair Correlation (Unseen Morphs)")
    _plot_series(ax1, tree_tb, "codesign/spatial/pair/oos_tree_diff_pearson", "Tree Credit Pearson r", color_tree, marker="s")
    _plot_series(ax1, tree_tb, "codesign/spatial/pair/oos_tree_diff_spearman", "Tree Credit Spearman rho", color_tree, linestyle="--", marker="^")
    _plot_series(ax1, tree_tb, "codesign/spatial/pair/oos_prefix_diff_pearson", "Prefix Delta Pearson r", color_pref, marker="o")
    _plot_series(ax1, tree_tb, "codesign/spatial/pair/oos_prefix_diff_spearman", "Prefix Delta Spearman rho", color_pref, linestyle="--", marker="x")
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Correlation against Delta R")
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: GenAct Advantage Ratio & Std
    ax2 = axes[0, 1]
    ax2.set_title("Panel 2: Advantage Magnitude Ratio std(beta*C_tree) / std(A_prefix)")
    _plot_series(ax2, tree_tb, "codesign/genact/tree_to_prefix_std_ratio_raw", "Raw Ratio (beta=0.5)", color_tree, marker="s")
    _plot_series(ax2, tree_tb, "codesign/genact/tree_to_prefix_std_ratio_norm", "Standardized Nominal (0.5)", color="black", linestyle="--")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Advantage Standard Deviation Ratio")
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Morphology Complexity
    ax3 = axes[1, 0]
    ax3.set_title("Panel 3: Realized Module & Limb Count")
    _plot_series(ax3, base_tb, "build/modulecount_realized", f"{base_label} - Modules", color_base, marker="o")
    _plot_series(ax3, tree_tb, "build/modulecount_realized", f"{tree_label} - Modules", color_tree, marker="s")
    _plot_series(ax3, base_tb, "build/limbcount_realized", f"{base_label} - Limbs", color_base, linestyle="--", marker="x", alpha=0.7)
    _plot_series(ax3, tree_tb, "build/limbcount_realized", f"{tree_label} - Limbs", color_tree, linestyle="--", marker="^", alpha=0.7)
    ax3.set_xlabel("Environment Steps")
    ax3.set_ylabel("Average Count per Body")
    ax3.legend(loc="best", framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Morphology Diversity & Entropy
    ax4 = axes[1, 1]
    ax4.set_title("Panel 4: Generator Diversity & Entropy")
    _plot_series(ax4, base_tb, "gen/entropy", f"{base_label} - Entropy", color_base, marker="o")
    _plot_series(ax4, tree_tb, "gen/entropy", f"{tree_label} - Entropy", color_tree, marker="s")
    _plot_series(ax4, base_tb, "build/modulecount_var", f"{base_label} - ModCount Var", color_base, linestyle="--", alpha=0.6)
    _plot_series(ax4, tree_tb, "build/modulecount_var", f"{tree_label} - ModCount Var", color_tree, linestyle="--", alpha=0.6)
    ax4.set_xlabel("Environment Steps")
    ax4.set_ylabel("Entropy / Variance")
    ax4.legend(loc="best", framealpha=0.9)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Cap vs Effector Action Probabilities
    ax5 = axes[2, 0]
    ax5.set_title("Panel 5: Action Probabilities (Effector vs Cap Emission)")
    _plot_series(ax5, base_tb, "gen/action_prob/eff", f"{base_label} - P(Effector)", color_base, marker="o")
    _plot_series(ax5, tree_tb, "gen/action_prob/eff", f"{tree_label} - P(Effector)", color_tree, marker="s")
    _plot_series(ax5, base_tb, "gen/action_prob/cap", f"{base_label} - P(Cap)", color_base, linestyle=":", marker="x")
    _plot_series(ax5, tree_tb, "gen/action_prob/cap", f"{tree_label} - P(Cap)", color_tree, linestyle=":", marker="+")
    ax5.set_xlabel("Environment Steps")
    ax5.set_ylabel("Empirical Probability")
    ax5.legend(loc="best", framealpha=0.9)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Performance Outcomes
    ax6 = axes[2, 1]
    ax6.set_title("Panel 6: Post-Adaptation Return (Mean & Top-Decile)")
    _plot_series(ax6, base_tb, "codesign/adaptation/R_post_mean", f"{base_label} - Mean R_post", color_base, marker="o")
    _plot_series(ax6, tree_tb, "codesign/adaptation/R_post_mean", f"{tree_label} - Mean R_post", color_tree, marker="s")
    _plot_series(ax6, base_tb, "quality/R_top10_mean", f"{base_label} - Top 10% Decile", color_base, linestyle="--", marker="x", alpha=0.7)
    _plot_series(ax6, tree_tb, "quality/R_top10_mean", f"{tree_label} - Top 10% Decile", color_tree, linestyle="--", marker="^", alpha=0.7)
    _plot_series(ax6, base_tb, "quality/R_max", f"{base_label} - Best Body", color_base, linestyle=":", alpha=0.5)
    _plot_series(ax6, tree_tb, "quality/R_max", f"{tree_label} - Best Body", color_tree, linestyle=":", alpha=0.5)
    ax6.set_xlabel("Environment Steps")
    ax6.set_ylabel("Return Target")
    ax6.legend(loc="best", framealpha=0.9)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved causal comparison plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Baseline run directory")
    parser.add_argument("--tree_dir", type=str, required=True, help="Spatial-tree run directory")
    parser.add_argument("--out_fig", type=str, default="genact_causal_pilot_comparison.png", help="Output figure path")
    args = parser.parse_args()

    base_tb = load_tb_scalars(args.base_dir)
    tree_tb = load_tb_scalars(args.tree_dir)

    print(f"Loaded {len(base_tb)} scalar series for Baseline")
    print(f"Loaded {len(tree_tb)} scalar series for Spatial-Tree")

    plot_genact_comparison(base_tb, tree_tb, args.out_fig)


if __name__ == "__main__":
    main()
