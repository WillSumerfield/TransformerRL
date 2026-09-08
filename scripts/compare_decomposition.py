#!/usr/bin/env python3
"""Additive Decomposition Comparative Analysis Script:
A: none (prefix baseline)
B: body_mean (mu_b only, zero spatial differentiation)
C: mean_plus_aligned_residual (mu_b + delta_i, exact uncentred tree credit)
D: mean_plus_shuffled_residual (mu_b + delta_{pi(i)}, permuted within body)

Determines whether the benefit of tree credit comes from:
1. Body-level completed-morphology signal (mu_b), or
2. Genuinely useful module-specific spatial alignment (delta_i), or
3. Residual exploration / noise (delta_{pi(i)}).
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


def compute_condition_metrics(name: str, tb: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
    R = post["R_post"] if "R_post" in post else post["R"]
    counts = post["counts"]
    eff_sub = post["eff_sub"]
    cap_sub = post["cap_sub"]
    N_mod = counts.sum(axis=1)
    N_limb = (counts > 0).sum(axis=1)

    mean_depth = np.zeros(counts.shape[0])
    max_depth = counts.max(axis=1)
    for b in range(counts.shape[0]):
        if N_limb[b] > 0:
            mean_depth[b] = N_mod[b] / N_limb[b]

    frac_limbs_max_depth = float((counts == 4).sum() / max(1, (counts > 0).sum()))
    frac_bodies_max_modules = float((N_mod == 32).mean())
    top_threshold = np.percentile(N_mod, 90)
    frac_top10 = float((N_mod >= top_threshold).mean())

    # Literal unique morphologies
    morphs = encode_canonical_morphology(counts, eff_sub, cap_sub)
    morphs_flat = morphs.reshape(morphs.shape[0], -1)
    unique_rows = np.unique(morphs_flat, axis=0)
    n_unique_bodies = len(unique_rows)

    def get_final(tag: str, default: float = np.nan) -> float:
        if tag in tb:
            return float(tb[tag][1][-1])
        return default

    def get_mean_last_n(tag: str, n: int = 5, default: float = np.nan) -> float:
        if tag in tb and len(tb[tag][1]) > 0:
            return float(np.mean(tb[tag][1][-n:]))
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
        "body_mean_std": get_mean_last_n("codesign/genact/body_raw_std"),
        "body_residual_std": get_mean_last_n("codesign/genact/body_centered_std"),
        "body_mean_to_residual_ratio": get_mean_last_n("codesign/genact/body_mu_to_delta_ratio"),
        "shuffle_corr": get_mean_last_n("codesign/genact/shuffle_corr"),
        "adv_valid_corr": get_mean_last_n("codesign/genact/adv_valid_corr"),
    }


def plot_decomposition_comparison(
    all_metrics: Dict[str, Dict[str, Any]],
    all_posts: Dict[str, Dict[str, Any]],
    all_tbs: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    out_fig: str,
):
    os.makedirs(os.path.dirname(os.path.abspath(out_fig)), exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle("Spatial-Tree Credit Decomposition: 4-Condition Evaluation\n"
                 "Prefix Baseline vs Body Mean vs Mean + Aligned Residual vs Mean + Shuffled Residual",
                 fontsize=16, y=0.995)

    colors = {
        "A: Prefix (none)": "#1f77b4",
        "B: Body Mean (mu_b)": "#ff7f0e",
        "C: Mean + Aligned Residual (mu_b + delta_i)": "#2ca02c",
        "D: Mean + Shuffled Residual (mu_b + delta_pi)": "#d62728",
    }
    styles = {
        "A: Prefix (none)": "--",
        "B: Body Mean (mu_b)": "-.",
        "C: Mean + Aligned Residual (mu_b + delta_i)": "-",
        "D: Mean + Shuffled Residual (mu_b + delta_pi)": ":",
    }

    # 1. Panel 1: Return Trajectory
    ax1 = axes[0, 0]
    ax1.set_title("1. Quality Trajectory: quality/R_mean", fontsize=12, fontweight="bold")
    for name, tb in all_tbs.items():
        if "quality/R_mean" in tb:
            steps, vals = tb["quality/R_mean"]
            c = colors.get(name, "#333333")
            s = styles.get(name, "-")
            ax1.plot(steps, vals, label=name, color=c, linestyle=s, marker="o", markersize=3, alpha=0.85)
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
            c = colors.get(name, "#333333")
            s = styles.get(name, "-")
            ax2.plot(steps, vals, label=name, color=c, linestyle=s, marker="s", markersize=3, alpha=0.85)
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
        c = colors.get(name, "#333333")
        s = styles.get(name, "-")
        ax3.plot(valid_b, means, label=name, color=c, linestyle=s, marker="o", linewidth=2)
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
        c = colors.get(name, "#333333")
        s = styles.get(name, "-")
        ax4.plot(d_centers, d_means, label=name, color=c, linestyle=s, marker="^", linewidth=2)
    ax4.set_xlabel("Mean Limb Depth (Modules / Active Limb)")
    ax4.set_ylabel("Expected Post-Adapt Return E[R_post]")
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Panel 5: Realized Module Count Distributions
    ax5 = axes[2, 0]
    ax5.set_title("5. Evaluated Population Module Count Distribution", fontsize=12, fontweight="bold")
    for name, post in all_posts.items():
        N_mod = post["counts"].sum(axis=1)
        counts_b, edges = np.histogram(N_mod, bins=np.arange(4, 33), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        c = colors.get(name, "#333333")
        s = styles.get(name, "-")
        ax5.plot(centers, counts_b, label=name, color=c, linestyle=s, linewidth=2)
    ax5.set_xlabel("Module Count")
    ax5.set_ylabel("Density")
    ax5.legend(loc="best", fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. Panel 6: Return Boxplots across conditions
    ax6 = axes[2, 1]
    ax6.set_title("6. Evaluated Return Distributions (Window 2)", fontsize=12, fontweight="bold")
    labels_boxes = [name for name in colors.keys() if name in all_posts]
    data_boxes = [(all_posts[name]["R_post"] if "R_post" in all_posts[name] else all_posts[name]["R"]) for name in labels_boxes]
    if data_boxes:
        bp = ax6.boxplot(data_boxes, tick_labels=[l.split(":")[0] for l in labels_boxes], patch_artist=True, showmeans=True, showfliers=False)
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
    parser.add_argument("--out_fig", type=str, default="/home/alex-daniel/.gemini/antigravity-ide/brain/8354a499-38b1-44cb-adc3-d7c78b30df26/decomposition_comparison.png")
    args = parser.parse_args()

    run_dirs = {
        "A: Prefix (none)": "runs/ant_codesign/codesign_single_transformer/pilot_genact_baseline_matched",
        "B: Body Mean (mu_b)": "runs/ant_codesign/codesign_single_transformer/pilot_genact_body_mean",
        "C: Mean + Aligned Residual (mu_b + delta_i)": "runs/ant_codesign/codesign_single_transformer/pilot_genact_spatial_tree",
        "D: Mean + Shuffled Residual (mu_b + delta_pi)": "runs/ant_codesign/codesign_single_transformer/pilot_genact_mean_plus_shuffled_residual",
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

    df = pd.DataFrame(all_metrics).T
    print("\n" + "=" * 130)
    print("DECOMPOSITION EXPERIMENT SUMMARY TABLE")
    print("=" * 130)
    cols = ["R_mean", "R_median", "R_p90", "R_max", "N_mod_mean", "N_mod_var", "mean_depth", "n_literal_unique", "policy_support_diversity", "P_eff", "P_cap", "body_mean_std", "body_residual_std", "body_mean_to_residual_ratio", "within_body_shuffle_corr"]
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].to_string())
    print("=" * 130 + "\n")

    # Matched complexity table
    print("\n" + "=" * 130)
    print("MATCHED-COMPLEXITY PERFORMANCE: E[R_post | N_modules] (Sample Counts in Parentheses)")
    print("=" * 130)
    mod_bins = list(range(6, 21))
    header = f"{'Modules':<8} | " + " | ".join([f"{name:<28}" for name in all_posts.keys()])
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
                row_str += f"{mean_r:6.2f} (N={count:<4})          | "
            else:
                row_str += f"{'N/A':<28} | "
        print(row_str)
    print("=" * 130 + "\n")

    if len(all_posts) >= 2:
        plot_decomposition_comparison(all_metrics, all_posts, all_tbs, args.out_fig)


if __name__ == "__main__":
    main()
