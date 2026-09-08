#!/usr/bin/env python3
"""Tri-condition comparative analysis script:
Prefix Only (none) vs Shuffled Tree Credit (shuffled) vs Aligned Tree Credit (aligned).

Validates the falsification experiment to determine whether spatial-tree generator credit
provides genuine module-aligned structural guidance or merely unaligned advantage magnitude/noise.
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
        data = np.load(f, allow_pickle=True)
        w_idx = int(data["meta__gen_window"]) if "meta__gen_window" in data else len(records)
        records.append((w_idx, {k: data[k] for k in data.files}))
    return records


def plot_tri_condition_comparison(
    cond_data: Dict[str, Dict[str, Any]],
    out_path: str,
):
    """Generates an 8-panel comprehensive comparison figure across the 3 conditions."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    fig.suptitle("Falsification Experiment: Prefix Only vs Shuffled Tree Credit vs Aligned Tree Credit", fontsize=16, y=0.995)

    colors = {
        "Prefix Only (none)": "#1f77b4",       # blue
        "Shuffled Tree (control)": "#ff7f0e",  # orange
        "Aligned Tree (spatial)": "#2ca02c",   # green
    }
    linestyles = {
        "Prefix Only (none)": "--",
        "Shuffled Tree (control)": "-.",
        "Aligned Tree (spatial)": "-",
    }

    def _plot_series(ax, tb_dict, tag, label, color, linestyle="-", marker="o", alpha=0.85):
        if tag in tb_dict:
            steps, vals = tb_dict[tag]
            ax.plot(steps, vals, label=label, color=color, linestyle=linestyle, marker=marker, markersize=4, alpha=alpha)

    # 1. Panel 1: Return Trajectory
    ax1 = axes[0, 0]
    ax1.set_title("1. Final Adapted Return Trajectory (quality/R_mean)", fontsize=12, fontweight="bold")
    for name, d in cond_data.items():
        tb = d["tb"]
        _plot_series(ax1, tb, "quality/R_mean", f"{name}", colors[name], linestyle=linestyles[name])
    ax1.set_xlabel("Environment Steps / Epochs")
    ax1.set_ylabel("Mean Return")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Panel 2: Morphology Module Count
    ax2 = axes[0, 1]
    ax2.set_title("2. Realized Morphology Complexity: Total Module Count", fontsize=12, fontweight="bold")
    for name, d in cond_data.items():
        tb = d["tb"]
        _plot_series(ax2, tb, "build/modulecount", f"{name}", colors[name], linestyle=linestyles[name])
    ax2.set_xlabel("Environment Steps / Epochs")
    ax2.set_ylabel("Mean Modules per Body")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Panel 3: Effector Count
    ax3 = axes[1, 0]
    ax3.set_title("3. Morphology Structure: Effector Count", fontsize=12, fontweight="bold")
    for name, d in cond_data.items():
        post_list = d["post"]
        if post_list:
            windows = [w for w, rec in post_list if "effector_count" in rec]
            effs = [float(np.mean(rec["effector_count"])) for w, rec in post_list if "effector_count" in rec]
            if windows:
                ax3.plot(windows, effs, label=f"{name}", color=colors[name], linestyle=linestyles[name], marker="o", markersize=4)
    ax3.set_xlabel("Generator Window Index")
    ax3.set_ylabel("Mean Effector Count")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Panel 4: Max Depth & Mean Depth
    ax4 = axes[1, 1]
    ax4.set_title("4. Tree Depth: Max Depth (Solid) & Mean Depth (Dotted)", fontsize=12, fontweight="bold")
    for name, d in cond_data.items():
        post_list = d["post"]
        if post_list:
            windows = [w for w, rec in post_list if "max_depth" in rec]
            max_depths = [float(np.mean(rec["max_depth"])) for w, rec in post_list if "max_depth" in rec]
            mean_depths = [float(np.mean(rec["mean_depth"])) for w, rec in post_list if "mean_depth" in rec]
            if windows:
                ax4.plot(windows, max_depths, label=f"{name} Max Depth", color=colors[name], linestyle=linestyles[name], marker="s", markersize=4)
                ax4.plot(windows, mean_depths, label=f"{name} Mean Depth", color=colors[name], linestyle=":", marker=None, alpha=0.7)
    ax4.set_xlabel("Generator Window Index")
    ax4.set_ylabel("Limb Depth")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Panel 5: Advantage & Permutation Correlation
    ax5 = axes[2, 0]
    ax5.set_title("5. Credit Fidelity: Effector Marginal Credit Delta", fontsize=12, fontweight="bold")
    for name, d in cond_data.items():
        tb = d["tb"]
        _plot_series(ax5, tb, "codesign/credit/effector_mean", f"{name} Effector Delta", colors[name], linestyle="-", marker="o")
        _plot_series(ax5, tb, "codesign/credit/cap_mean", f"{name} Cap Delta", colors[name], linestyle=":", marker=None)
    ax5.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Generator Window / Epochs")
    ax5.set_ylabel("Marginal Delta")
    ax5.legend(loc="best", fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Panel 6: Diversity and Entropy
    ax6 = axes[2, 1]
    ax6.set_title("6. Generator Entropy & Morphology Diversity", fontsize=12, fontweight="bold")
    for name, d in cond_data.items():
        tb = d["tb"]
        _plot_series(ax6, tb, "gen/entropy", f"{name} Entropy", colors[name], linestyle="-")
    ax6.set_xlabel("Environment Steps / Epochs")
    ax6.set_ylabel("Entropy")
    ax6.legend(loc="best", fontsize=8)
    ax6.grid(True, alpha=0.3)

    # 7. Panel 7: Out-of-Sample Pair Generalization (Pearson r)
    ax7 = axes[3, 0]
    ax7.set_title("7. Out-of-Sample Pair Generalization: Tree Diff Pearson r", fontsize=12, fontweight="bold")
    for name, d in cond_data.items():
        tb = d["tb"]
        _plot_series(ax7, tb, "codesign/spatial/pair/oos_tree_diff_pearson", f"{name} Tree r", colors[name], linestyle="-", marker="o")
        _plot_series(ax7, tb, "codesign/spatial/pair/oos_subtree_only_tree_pearson", f"{name} Subtree r", colors[name], linestyle="--", marker="x")
    ax7.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax7.set_xlabel("Generator Window Index")
    ax7.set_ylabel("Out-of-Sample Pearson r")
    ax7.legend(loc="best", fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 8. Panel 8: Return vs Module Count Scaling (Window 2)
    ax8 = axes[3, 1]
    ax8.set_title("8. Final Window: E[R | N_modules] (Complexity Scaling)", fontsize=12, fontweight="bold")
    for name, d in cond_data.items():
        post_list = d["post"]
        if post_list:
            last_w, last_rec = post_list[-1]
            r = last_rec["R_post"]
            mods = last_rec["module_count"]
            u_mods = np.unique(mods)
            # Only plot bins with >= 10 bodies
            valid_mods = [m for m in u_mods if np.sum(mods == m) >= 10]
            if valid_mods:
                mean_r = [float(np.mean(r[mods == m])) for m in valid_mods]
                ax8.plot(valid_mods, mean_r, label=f"{name}", color=colors[name], linestyle=linestyles[name], marker="o", markersize=4)
    ax8.set_xlabel("Module Count N_modules")
    ax8.set_ylabel("Mean Return E[R | N_modules]")
    ax8.legend(loc="best", fontsize=8)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[compare_three_conditions] Comparison figure saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Tri-condition comparison of CoDesign experiments.")
    parser.add_argument("--base-dir", type=str, default="runs/ant_codesign/codesign_single_transformer/pilot_genact_baseline_matched",
                        help="Path to Condition A (Prefix only, none)")
    parser.add_argument("--shuffled-dir", type=str, default="runs/ant_codesign/codesign_single_transformer/pilot_genact_shuffled_tree",
                        help="Path to Condition B (Shuffled tree credit, control)")
    parser.add_argument("--aligned-dir", type=str, default="runs/ant_codesign/codesign_single_transformer/pilot_genact_spatial_tree",
                        help="Path to Condition C (Aligned tree credit, spatial)")
    parser.add_argument("--output", type=str, default="/home/alex-daniel/.gemini/antigravity-ide/brain/8354a499-38b1-44cb-adc3-d7c78b30df26/falsification_tri_condition_comparison.png",
                        help="Path to output figure.")
    args = parser.parse_args()

    conditions = {
        "Prefix Only (none)": {
            "dir": args.base_dir,
            "tb": load_tb_scalars(args.base_dir),
            "post": load_npz_artifacts(args.base_dir, "post_eval"),
        },
        "Shuffled Tree (control)": {
            "dir": args.shuffled_dir,
            "tb": load_tb_scalars(args.shuffled_dir),
            "post": load_npz_artifacts(args.shuffled_dir, "post_eval"),
        },
        "Aligned Tree (spatial)": {
            "dir": args.aligned_dir,
            "tb": load_tb_scalars(args.aligned_dir),
            "post": load_npz_artifacts(args.aligned_dir, "post_eval"),
        },
    }

    plot_tri_condition_comparison(conditions, args.output)

    # Print summary metrics
    print("\n" + "=" * 90)
    print("FALSIFICATION EXPERIMENT SUMMARY METRICS (FINAL EPOCH / WINDOW)")
    print("=" * 90)
    header = f"{'Metric':<38} | {'Prefix Only (none)':<16} | {'Shuffled Tree':<16} | {'Aligned Tree':<16}"
    print(header)
    print("-" * 90)

    for cname in ["Prefix Only (none)", "Shuffled Tree (control)", "Aligned Tree (spatial)"]:
        pass

    # Extract final window values
    def get_final(post_list, key):
        if not post_list: return "N/A"
        rec = post_list[-1][1]
        return f"{float(np.mean(rec[key])):10.4f}" if key in rec else "N/A"

    def get_tb_final(tb_dict, tag):
        if tag in tb_dict and len(tb_dict[tag][1]) > 0:
            return f"{float(tb_dict[tag][1][-1]):10.4f}"
        return "       N/A"

    metrics = [
        ("Return R_mean (quality/R_mean)", lambda c: get_tb_final(c["tb"], "quality/R_mean")),
        ("Post Return (R_post)", lambda c: get_final(c["post"], "R_post")),
        ("Module Count (build/modulecount)", lambda c: get_tb_final(c["tb"], "build/modulecount")),
        ("Effector Count", lambda c: get_final(c["post"], "effector_count")),
        ("Mean Depth", lambda c: get_final(c["post"], "mean_depth")),
        ("Max Depth", lambda c: get_final(c["post"], "max_depth")),
        ("Generator Entropy (gen/entropy)", lambda c: get_tb_final(c["tb"], "gen/entropy")),
        ("Body Diversity", lambda c: get_tb_final(c["tb"], "build/body_diversity")),
        ("Effector Marginal Delta", lambda c: get_tb_final(c["tb"], "codesign/credit/effector_mean")),
        ("Cap Marginal Delta", lambda c: get_tb_final(c["tb"], "codesign/credit/cap_mean")),
        ("OOS Tree Pearson r", lambda c: get_tb_final(c["tb"], "codesign/spatial/pair/oos_tree_diff_pearson")),
        ("OOS Subtree Tree Pearson r", lambda c: get_tb_final(c["tb"], "codesign/spatial/pair/oos_subtree_only_tree_pearson")),
    ]

    for label, fn in metrics:
        v0 = fn(conditions["Prefix Only (none)"])
        v1 = fn(conditions["Shuffled Tree (control)"])
        v2 = fn(conditions["Aligned Tree (spatial)"])
        print(f"{label:<38} | {v0:<16} | {v1:<16} | {v2:<16}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
