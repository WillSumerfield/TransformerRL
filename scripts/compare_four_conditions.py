#!/usr/bin/env python3
"""Four-condition comparative analysis script:
A: Prefix Only (none)
B: Existing Aligned Tree (aligned)
C: Body-Centred Aligned Tree (centered_aligned)
D: Body-Centred Within-Body Shuffled Tree (centered_within_body_shuffled)

Refinement test to determine whether body-centering removes global morphology elongation
bias while retaining module-specific structural guidance.
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
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_rl.counterfactual_pairs import encode_canonical_morphology


def load_tb_scalars(run_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
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


def load_post_eval_data(run_dir: str) -> Dict[str, np.ndarray]:
    files = sorted(glob.glob(os.path.join(run_dir, "post_eval", "*.npz")))
    if not files:
        files = sorted(glob.glob(os.path.join(run_dir, "**", "post_eval", "*.npz"), recursive=True))
    assert len(files) > 0, f"No post_eval files in {run_dir}"
    d = np.load(files[-1], allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_credit_artifact(run_dir: str) -> Optional[Dict[str, np.ndarray]]:
    files = sorted(glob.glob(os.path.join(run_dir, "credit", "*.npz")))
    if not files:
        files = sorted(glob.glob(os.path.join(run_dir, "**", "credit", "*.npz"), recursive=True))
    if not files:
        return None
    d = np.load(files[-1], allow_pickle=True)
    return {k: d[k] for k in d.files}


def compute_condition_metrics(name: str, tb: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
    R = post["R_post"] if "R_post" in post else post["R"]
    counts = post["counts"]
    eff_sub = post["eff_sub"]
    cap_sub = post["cap_sub"]
    N_mod = counts.sum(axis=1)
    N_limb = (counts > 0).sum(axis=1)

    # Depths
    active_mask = (counts > 0)
    mean_depth = np.zeros(counts.shape[0])
    max_depth = counts.max(axis=1)
    for b in range(counts.shape[0]):
        if N_limb[b] > 0:
            mean_depth[b] = N_mod[b] / N_limb[b]

    frac_limbs_max_depth = float((counts == 4).sum() / max(1, (counts > 0).sum()))
    frac_bodies_max_modules = float((N_mod == 32).mean())
    top_threshold = np.percentile(N_mod, 90)
    frac_top10 = float((N_mod >= top_threshold).mean())

    # Unique bodies
    morphs = encode_canonical_morphology(counts, eff_sub, cap_sub)
    morphs_flat = morphs.reshape(morphs.shape[0], -1)
    unique_rows = np.unique(morphs_flat, axis=0)
    n_unique_bodies = len(unique_rows)

    # TB final scalars
    def get_final(tag: str, default: float = np.nan) -> float:
        if tag in tb:
            return float(tb[tag][1][-1])
        return default

    return {
        "name": name,
        "R_mean": float(np.mean(R)),
        "R_median": float(np.median(R)),
        "R_p90": float(np.percentile(R, 90)),
        "R_top10_mean": float(np.mean(R[R >= np.percentile(R, 90)])),
        "R_max": float(np.max(R)),
        "R_std": float(np.std(R)),
        "N_mod_mean": float(np.mean(N_mod)),
        "N_mod_var": float(np.var(N_mod)),
        "N_mod_max": int(np.max(N_mod)),
        "N_limb_mean": float(np.mean(N_limb)),
        "mean_depth": float(np.mean(mean_depth)),
        "max_depth_mean": float(np.mean(max_depth)),
        "frac_limbs_max_depth": frac_limbs_max_depth,
        "frac_bodies_max_modules": frac_bodies_max_modules,
        "frac_top10_complexity": frac_top10,
        "n_literal_unique": n_unique_bodies,
        "policy_support_diversity": get_final("build/body_diversity"),
        "P_eff": get_final("gen/action_prob/eff"),
        "P_cap": get_final("gen/action_prob/cap"),
        "entropy": get_final("gen/entropy"),
        "tb_modcount_intent": get_final("build/modulecount"),
        "shuffle_corr": get_final("codesign/genact/shuffle_corr"),
        "adv_valid_corr": get_final("codesign/genact/adv_valid_corr"),
        "body_raw_mean": get_final("codesign/genact/body_raw_mean"),
        "body_centered_mean": get_final("codesign/genact/body_centered_mean"),
        "body_centered_std": get_final("codesign/genact/body_centered_std"),
        "tree_to_prefix_ratio": get_final("codesign/genact/tree_to_prefix_std_ratio_raw"),
        "tree_valid_fraction": get_final("codesign/genact/tree_valid_fraction"),
    }


def plot_four_conditions(
    all_metrics: Dict[str, Dict[str, Any]],
    all_posts: Dict[str, Dict[str, Any]],
    all_tbs: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    out_fig: str,
):
    os.makedirs(os.path.dirname(os.path.abspath(out_fig)), exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle("Refined Spatial-Tree Credit: 4-Condition Evaluation\\n"
                 "Prefix vs Aligned vs Centered Aligned vs Centered Within-Body Shuffled",
                 fontsize=16, y=0.995)

    colors = {
        "Prefix (none)": "#1f77b4",
        "Aligned (uncentered)": "#2ca02c",
        "Centered Aligned": "#9467bd",
        "Centered Shuffled": "#d62728",
    }
    styles = {
        "Prefix (none)": "--",
        "Aligned (uncentered)": "-.",
        "Centered Aligned": "-",
        "Centered Shuffled": ":",
    }

    # 1. Panel 1: Return Trajectory
    ax1 = axes[0, 0]
    ax1.set_title("1. Quality Trajectory: quality/R_mean", fontsize=12, fontweight="bold")
    for name, tb in all_tbs.items():
        if "quality/R_mean" in tb:
            steps, vals = tb["quality/R_mean"]
            ax1.plot(steps, vals, label=name, color=colors[name], linestyle=styles[name], marker="o", markersize=3, alpha=0.85)
    ax1.set_xlabel("Environment Steps / Epochs")
    ax1.set_ylabel("Mean Return")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Panel 2: Module Count Intent Trajectory
    ax2 = axes[0, 1]
    ax2.set_title("2. Generator Policy Intent: build/modulecount", fontsize=12, fontweight="bold")
    for name, tb in all_tbs.items():
        if "build/modulecount" in tb:
            steps, vals = tb["build/modulecount"]
            ax2.plot(steps, vals, label=name, color=colors[name], linestyle=styles[name], marker="s", markersize=3, alpha=0.85)
    ax2.set_xlabel("Environment Steps / Epochs")
    ax2.set_ylabel("Generated Module Count Intent")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Panel 3: Matched-Complexity E[R_post | N_modules]
    ax3 = axes[1, 0]
    ax3.set_title("3. Matched-Complexity: E[R_post | N_modules]", fontsize=12, fontweight="bold")
    mod_bins = list(range(6, 21))
    for name, post in all_posts.items():
        R = post["R_post"] if "R_post" in post else post["R"]
        N_mod = post["counts"].sum(axis=1)
        means = []
        valid_b = []
        for m in mod_bins:
            mask = (N_mod == m)
            if mask.sum() >= 15:
                means.append(float(R[mask].mean()))
                valid_b.append(m)
        ax3.plot(valid_b, means, label=name, color=colors[name], linestyle=styles[name], marker="o", linewidth=2)
    ax3.set_xlabel("Total Realized Modules")
    ax3.set_ylabel("Expected Post-Adapt Return E[R_post]")
    ax3.set_xticks(mod_bins)
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Panel 4: Matched-Depth E[R_post | mean_depth]
    ax4 = axes[1, 1]
    ax4.set_title("4. Matched-Depth: E[R_post | mean_depth]", fontsize=12, fontweight="bold")
    depth_bins = np.linspace(1.0, 3.5, 11)
    for name, post in all_posts.items():
        R = post["R_post"] if "R_post" in post else post["R"]
        counts = post["counts"]
        N_mod = counts.sum(axis=1)
        N_limb = (counts > 0).sum(axis=1)
        md = np.where(N_limb > 0, N_mod / np.maximum(1, N_limb), 0.0)
        bin_idx = np.digitize(md, depth_bins) - 1
        d_means = []
        d_centers = []
        for i in range(len(depth_bins) - 1):
            mask = (bin_idx == i)
            if mask.sum() >= 15:
                d_means.append(float(R[mask].mean()))
                d_centers.append(0.5 * (depth_bins[i] + depth_bins[i+1]))
        ax4.plot(d_centers, d_means, label=name, color=colors[name], linestyle=styles[name], marker="^", linewidth=2)
    ax4.set_xlabel("Mean Limb Depth (Modules / Active Limb)")
    ax4.set_ylabel("Expected Post-Adapt Return E[R_post]")
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Panel 5: Realized Module Count Distributions (KDE / Histogram)
    ax5 = axes[2, 0]
    ax5.set_title("5. Evaluated Population Module Count Distribution", fontsize=12, fontweight="bold")
    for name, post in all_posts.items():
        N_mod = post["counts"].sum(axis=1)
        counts_b, edges = np.histogram(N_mod, bins=np.arange(4, 33), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax5.plot(centers, counts_b, label=name, color=colors[name], linestyle=styles[name], linewidth=2)
    ax5.set_xlabel("Module Count")
    ax5.set_ylabel("Density")
    ax5.legend(loc="best", fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. Panel 6: Return Boxplots across conditions
    ax6 = axes[2, 1]
    ax6.set_title("6. Evaluated Return Distributions (Window 2)", fontsize=12, fontweight="bold")
    data_boxes = [(all_posts[name]["R_post"] if "R_post" in all_posts[name] else all_posts[name]["R"]) for name in colors.keys() if name in all_posts]
    labels_boxes = [name for name in colors.keys() if name in all_posts]
    bp = ax6.boxplot(data_boxes, labels=labels_boxes, patch_artist=True, showmeans=True, showfliers=False)
    for patch, label in zip(bp['boxes'], labels_boxes):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.6)
    ax6.set_ylabel("Post-Adaptation Return")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[saved figure] {out_fig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_fig", type=str, default="/home/alex-daniel/.gemini/antigravity-ide/brain/8354a499-38b1-44cb-adc3-d7c78b30df26/four_condition_comparison.png")
    args = parser.parse_args()

    run_dirs = {
        "Prefix (none)": "runs/ant_codesign/codesign_single_transformer/pilot_genact_baseline_matched",
        "Aligned (uncentered)": "runs/ant_codesign/codesign_single_transformer/pilot_genact_spatial_tree",
        "Centered Aligned": "runs/ant_codesign/codesign_single_transformer/pilot_genact_centered_aligned",
        "Centered Shuffled": "runs/ant_codesign/codesign_single_transformer/pilot_genact_centered_within_body_shuffled",
    }

    all_tbs = {}
    all_posts = {}
    all_metrics = {}

    for name, path in run_dirs.items():
        if os.path.exists(path):
            print(f"Loading {name} from {path}...")
            tb = load_tb_scalars(path)
            post = load_post_eval_data(path)
            all_tbs[name] = tb
            all_posts[name] = post
            all_metrics[name] = compute_condition_metrics(name, tb, post)
        else:
            print(f"Warning: {name} path {path} does not exist yet.")

    # Print summary table
    df = pd.DataFrame(all_metrics).T
    print("\\n" + "=" * 120)
    print("FOUR-CONDITION EXPERIMENTAL RESULTS SUMMARY TABLE")
    print("=" * 120)
    cols = ["R_mean", "R_median", "R_p90", "R_max", "N_mod_mean", "N_mod_var", "N_mod_max", "mean_depth", "n_literal_unique", "policy_support_diversity", "P_eff", "P_cap", "shuffle_corr"]
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].to_string())
    print("=" * 120 + "\\n")

    # Matched complexity table
    print("\\n" + "=" * 120)
    print("MATCHED-COMPLEXITY PERFORMANCE: E[R_post | N_modules] (Sample Counts in Parentheses)")
    print("=" * 120)
    mod_bins = list(range(6, 21))
    header = f"{'Modules':<8} | " + " | ".join([f"{name:<25}" for name in all_posts.keys()])
    print(header)
    print("-" * len(header))
    for m in mod_bins:
        row_str = f"{m:<8} | "
        for name, post in all_posts.items():
            R = post["R_post"] if "R_post" in post else post["R"]
            N_mod = post["counts"].sum(axis=1)
            mask = (N_mod == m)
            count = mask.sum()
            if count > 0:
                mean_r = R[mask].mean()
                row_str += f"{mean_r:6.2f} (N={count:<4})       | "
            else:
                row_str += f"{'N/A':<25} | "
        print(row_str)
    print("=" * 120 + "\\n")

    if len(all_posts) >= 2:
        plot_four_conditions(all_metrics, all_posts, all_tbs, args.out_fig)


if __name__ == "__main__":
    main()
