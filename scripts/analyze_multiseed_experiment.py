#!/usr/bin/env python3
"""Multi-seed comparative analysis script:
Evaluates Condition B (body_mean) vs Condition C (mean_plus_aligned_residual)
across 5 seeds (42, 43, 44, 45, 46).

Computes:
1. Aggregate statistics (mean +/- std across seeds)
2. Paired seed differences (aligned - bodymean) with bootstrap 95% CI
3. Training learning curves with seed spread and AUC_R
4. Pooled matched-complexity analysis E[R_post | N_modules]
5. Three comprehensive publication-quality figures
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
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

SEEDS = [42, 43, 44, 45, 46]
BASE_DIR = Path("runs/ant_codesign/codesign_single_transformer")


def load_tb_scalars(run_dir: str | Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    event_files = sorted(glob.glob(os.path.join(str(run_dir), "**", "events.out.tfevents.*"), recursive=True))
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


def load_post_eval_data(run_dir: str | Path) -> Dict[str, np.ndarray]:
    files = sorted(glob.glob(os.path.join(str(run_dir), "**", "post_eval", "*.npz"), recursive=True))
    assert len(files) > 0, f"No post_eval files in {run_dir}"
    d = np.load(files[-1], allow_pickle=True)
    return {k: d[k] for k in d.files}


def compute_single_run_metrics(run_dir: Path, tb: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
    R = post["R_post"] if "R_post" in post else post["R"]
    counts = post["counts"]
    eff_sub = post["eff_sub"]
    cap_sub = post["cap_sub"]
    N_mod = counts.sum(axis=1)
    N_limb = (counts > 0).sum(axis=1)

    mean_depth = np.zeros(counts.shape[0])
    for b in range(counts.shape[0]):
        if N_limb[b] > 0:
            mean_depth[b] = N_mod[b] / N_limb[b]

    # Literal unique morphologies
    morphs = encode_canonical_morphology(counts, eff_sub, cap_sub)
    morphs_flat = morphs.reshape(morphs.shape[0], -1)
    unique_rows = np.unique(morphs_flat, axis=0)
    n_unique_bodies = len(unique_rows)

    def get_final(tag: str, default: float = np.nan) -> float:
        if tag in tb and len(tb[tag][1]) > 0:
            return float(tb[tag][1][-1])
        return default

    def get_mean_last_n(tag: str, n: int = 5, default: float = np.nan) -> float:
        if tag in tb and len(tb[tag][1]) > 0:
            return float(np.mean(tb[tag][1][-n:]))
        return default

    # Compute training return AUC (trapezoidal integration over steps)
    auc_r = np.nan
    if "quality/R_mean" in tb:
        steps, vals = tb["quality/R_mean"]
        if len(steps) >= 2:
            norm_steps = (steps - steps[0]) / (steps[-1] - steps[0] + 1e-8)
            auc_r = float(np.trapezoid(vals, norm_steps))

    return {
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
        "n_literal_unique": n_unique_bodies,
        "unique_fraction": float(n_unique_bodies / len(N_mod)),
        "policy_support_diversity": get_final("build/body_diversity"),
        "P_eff": get_final("gen/action_prob/eff"),
        "P_cap": get_final("gen/action_prob/cap"),
        "entropy": get_final("gen/entropy"),
        "body_mean_std": get_mean_last_n("codesign/genact/body_raw_std"),
        "body_residual_std": get_mean_last_n("codesign/genact/body_centered_std"),
        "tree_valid_fraction": get_mean_last_n("codesign/genact/tree_valid_fraction"),
        "auc_r": auc_r,
    }


def bootstrap_ci(diffs: np.ndarray, num_boot: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(num_boot)])
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


def plot_learning_curves(
    all_tbs: Dict[str, Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]],
    all_metrics: Dict[str, Dict[int, Dict[str, Any]]],
    out_fig: str,
):
    os.makedirs(os.path.dirname(os.path.abspath(out_fig)), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Multi-Seed Training Dynamics: Body Mean vs Aligned Spatial Credit (5 Seeds)\n"
                 "Solid lines: Mean trajectory; Shaded bands: ±1 Standard Deviation across seeds",
                 fontsize=15, y=0.98)

    tags_to_plot = [
        ("quality/R_mean", "Mean Return (quality/R_mean)", axes[0, 0]),
        ("quality/R_top10_mean", "Top-10% Return (quality/R_top10_mean)", axes[0, 1]),
        ("build/modulecount", "Module Count Intent (build/modulecount)", axes[0, 2]),
        ("gen/action_prob/eff", "P(Effector Action) (gen/action_prob/eff)", axes[1, 0]),
        ("gen/entropy", "Generator Policy Entropy (gen/entropy)", axes[1, 1]),
    ]

    colors = {"bodymean": "#ff7f0e", "aligned": "#2ca02c"}
    labels = {"bodymean": "Body Mean (mu_b)", "aligned": "Aligned Spatial (mu_b + delta_i)"}

    for tag, title, ax in tags_to_plot:
        ax.set_title(title, fontsize=12, fontweight="bold")
        for method in ["bodymean", "aligned"]:
            runs = all_tbs.get(method, {})
            all_steps = []
            all_vals = []
            for seed, tb in runs.items():
                if tag in tb:
                    steps, vals = tb[tag]
                    if len(steps) > 0:
                        all_steps.append(steps)
                        all_vals.append(vals)

            if len(all_vals) >= 2:
                # Interpolate to common step grid
                min_step = max(s[0] for s in all_steps)
                max_step = min(s[-1] for s in all_steps)
                common_steps = np.linspace(min_step, max_step, 50)
                interp_vals = np.array([np.interp(common_steps, s, v) for s, v in zip(all_steps, all_vals)])
                mean_v = interp_vals.mean(axis=0)
                std_v = interp_vals.std(axis=0)

                ax.plot(common_steps, mean_v, label=labels[method], color=colors[method], linewidth=2.2)
                ax.fill_between(common_steps, mean_v - std_v, mean_v + std_v, color=colors[method], alpha=0.2)
            elif len(all_vals) == 1:
                ax.plot(all_steps[0], all_vals[0], label=labels[method], color=colors[method], linewidth=2)

        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    # 6th panel: AUC_R comparison
    ax6 = axes[1, 2]
    ax6.set_title("Training Return Efficiency: AUC_R by Seed", fontsize=12, fontweight="bold")
    x = np.arange(len(SEEDS))
    width = 0.35
    bm_aucs = [all_metrics["bodymean"][s]["auc_r"] for s in SEEDS if s in all_metrics["bodymean"]]
    al_aucs = [all_metrics["aligned"][s]["auc_r"] for s in SEEDS if s in all_metrics["aligned"]]
    if bm_aucs and al_aucs:
        ax6.bar(x - width/2, bm_aucs, width, label="Body Mean (mu_b)", color=colors["bodymean"], alpha=0.85)
        ax6.bar(x + width/2, al_aucs, width, label="Aligned (mu_b + delta)", color=colors["aligned"], alpha=0.85)
        ax6.set_xticks(x)
        ax6.set_xticklabels([str(s) for s in SEEDS])
        ax6.set_xlabel("Seed")
        ax6.set_ylabel("AUC (Normalized Return * Steps)")
        ax6.legend(loc="best", fontsize=9)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[saved figure] {out_fig}")


def plot_matched_complexity(
    pooled_posts: Dict[str, Dict[str, np.ndarray]],
    out_fig: str,
):
    os.makedirs(os.path.dirname(os.path.abspath(out_fig)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Pooled Matched-Complexity Analysis across 5 Seeds (N=20,480 bodies per method)\n"
                 "Evaluating E[R_post | N_modules] to test whether Aligned Spatial Credit selects superior structures",
                 fontsize=14, y=0.98)

    colors = {"bodymean": "#ff7f0e", "aligned": "#2ca02c"}
    labels = {"bodymean": "Body Mean (mu_b)", "aligned": "Aligned Spatial (mu_b + delta_i)"}

    mod_bins = list(range(5, 22))

    for method in ["bodymean", "aligned"]:
        data = pooled_posts[method]
        R = data["R"]
        counts = data["counts"]
        N_mod = counts.sum(axis=1)

        valid_m = []
        means = []
        ci_lows = []
        ci_highs = []
        sample_counts = []

        for m in mod_bins:
            mask = (N_mod == m)
            n_samples = mask.sum()
            if n_samples >= 30:
                sub_r = R[mask]
                mean_r = float(np.mean(sub_r))
                # 95% bootstrap CI
                ci_l, ci_h = bootstrap_ci(sub_r, num_boot=1000)
                valid_m.append(m)
                means.append(mean_r)
                ci_lows.append(ci_l)
                ci_highs.append(ci_h)
                sample_counts.append(n_samples)

        valid_m = np.array(valid_m)
        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        ax1.plot(valid_m, means, label=labels[method], color=colors[method], marker="o", linewidth=2.2)
        ax1.fill_between(valid_m, ci_lows, ci_highs, color=colors[method], alpha=0.2)

        # Plot sample counts on ax2
        ax2.plot(valid_m, sample_counts, label=labels[method], color=colors[method], marker="s", linewidth=2)

    ax1.set_title("Expected Return E[R_post | N_modules] (95% Bootstrap CI)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Total Realized Modules (N_modules)")
    ax1.set_ylabel("Expected Post-Adaptation Return")
    ax1.set_xticks(mod_bins)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=10)

    ax2.set_title("Evaluated Morphology Sample Count per Complexity Bin", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Total Realized Modules (N_modules)")
    ax2.set_ylabel("Number of Evaluated Morphologies")
    ax2.set_xticks(mod_bins)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[saved figure] {out_fig}")


def plot_paired_differences(
    paired_df: pd.DataFrame,
    out_fig: str,
):
    os.makedirs(os.path.dirname(os.path.abspath(out_fig)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Paired Seed Analysis (Aligned Spatial vs Body Mean across 5 Seeds)\n"
                 "Delta_s = Aligned_s - BodyMean_s",
                 fontsize=14, y=0.98)

    # 1. Delta Return
    ax1 = axes[0]
    seeds = paired_df["seed"].values
    d_r = paired_df["delta_R_mean"].values
    bars1 = ax1.bar([str(s) for s in seeds], d_r, color=["#2ca02c" if v >= 0 else "#d62728" for v in d_r], alpha=0.8)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1)
    ax1.axhline(d_r.mean(), color="blue", linestyle="-", linewidth=1.5, label=f"Mean Delta = {d_r.mean():+.2f}")
    ax1.set_title("Paired Delta Mean Return (Delta R)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Delta Mean Return")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # 2. Delta P90
    ax2 = axes[1]
    d_p90 = paired_df["delta_R_p90"].values
    bars2 = ax2.bar([str(s) for s in seeds], d_p90, color=["#2ca02c" if v >= 0 else "#d62728" for v in d_p90], alpha=0.8)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.axhline(d_p90.mean(), color="blue", linestyle="-", linewidth=1.5, label=f"Mean Delta = {d_p90.mean():+.2f}")
    ax2.set_title("Paired Delta P90 Return", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("Delta P90 Return")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # 3. Delta Module Count
    ax3 = axes[2]
    d_mod = paired_df["delta_N_mod"].values
    bars3 = ax3.bar([str(s) for s in seeds], d_mod, color=["#2ca02c" if v >= 0 else "#d62728" for v in d_mod], alpha=0.8)
    ax3.axhline(0, color="black", linestyle="--", linewidth=1)
    ax3.axhline(d_mod.mean(), color="blue", linestyle="-", linewidth=1.5, label=f"Mean Delta = {d_mod.mean():+.2f}")
    ax3.set_title("Paired Delta Module Count", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Seed")
    ax3.set_ylabel("Delta Module Count")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[saved figure] {out_fig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/home/alex-daniel/.gemini/antigravity-ide/brain/8354a499-38b1-44cb-adc3-d7c78b30df26")
    args = parser.parse_args()

    methods = ["bodymean", "aligned"]
    all_tbs = {m: {} for m in methods}
    all_posts = {m: {} for m in methods}
    all_metrics = {m: {} for m in methods}

    for m in methods:
        for seed in SEEDS:
            run_name = f"multiseed_{m}_seed{seed}"
            run_path = BASE_DIR / run_name
            if (run_path / "post_eval" / "post_eval_window_0002.npz").exists():
                print(f"Loading {run_name}...")
                tb = load_tb_scalars(run_path)
                post = load_post_eval_data(run_path)
                all_tbs[m][seed] = tb
                all_posts[m][seed] = post
                all_metrics[m][seed] = compute_single_run_metrics(run_path, tb, post)
            else:
                print(f"Pending/missing: {run_name}")

    # Check if all seeds are present
    completed_seeds = [s for s in SEEDS if s in all_metrics["bodymean"] and s in all_metrics["aligned"]]
    print(f"\nCompleted matched seeds: {completed_seeds} ({len(completed_seeds)}/5)\n")

    if not completed_seeds:
        print("No completed matched seeds found yet.")
        return

    # Build per-seed tables
    df_bm = pd.DataFrame(all_metrics["bodymean"]).T
    df_al = pd.DataFrame(all_metrics["aligned"]).T

    # 1. Aggregate Statistics across seeds
    print("=" * 100)
    print("5-SEED AGGREGATE SUMMARY: BODY MEAN vs ALIGNED SPATIAL CREDIT")
    print("=" * 100)
    summary_rows = []
    main_cols = [
        ("R_mean", "Mean Return"),
        ("R_median", "Median Return"),
        ("R_p90", "P90 Return"),
        ("R_top10_mean", "Top-10% Mean Return"),
        ("R_max", "Best Return"),
        ("N_mod_mean", "Mean Modules"),
        ("N_mod_var", "Module Variance"),
        ("mean_depth", "Mean Limb Depth"),
        ("unique_fraction", "Literal Unique Fraction"),
        ("P_eff", "P(Effector Action)"),
        ("P_cap", "P(Cap Action)"),
        ("entropy", "Generator Policy Entropy"),
        ("auc_r", "Training Return AUC"),
    ]

    for col_key, col_label in main_cols:
        bm_vals = df_bm[col_key].dropna().values
        al_vals = df_al[col_key].dropna().values
        summary_rows.append({
            "Metric": col_label,
            "Body Mean (mean +/- std)": f"{bm_vals.mean():.3f} +/- {bm_vals.std():.3f}",
            "Aligned (mean +/- std)": f"{al_vals.mean():.3f} +/- {al_vals.std():.3f}",
            "Raw Diff": f"{al_vals.mean() - bm_vals.mean():+.3f}",
        })

    sum_df = pd.DataFrame(summary_rows)
    print(sum_df.to_string(index=False))
    print("=" * 100 + "\n")

    # 2. Paired Seed Differences
    print("=" * 100)
    print("PAIRED SEED DIFFERENCES (Delta_s = Aligned_s - BodyMean_s)")
    print("=" * 100)
    paired_rows = []
    for s in completed_seeds:
        bm = all_metrics["bodymean"][s]
        al = all_metrics["aligned"][s]
        paired_rows.append({
            "seed": s,
            "bodymean_R": bm["R_mean"],
            "aligned_R": al["R_mean"],
            "delta_R_mean": al["R_mean"] - bm["R_mean"],
            "bodymean_p90": bm["R_p90"],
            "aligned_p90": al["R_p90"],
            "delta_R_p90": al["R_p90"] - bm["R_p90"],
            "bodymean_top10": bm["R_top10_mean"],
            "aligned_top10": al["R_top10_mean"],
            "delta_R_top10": al["R_top10_mean"] - bm["R_top10_mean"],
            "bodymean_mod": bm["N_mod_mean"],
            "aligned_mod": al["N_mod_mean"],
            "delta_N_mod": al["N_mod_mean"] - bm["N_mod_mean"],
            "bodymean_auc": bm["auc_r"],
            "aligned_auc": al["auc_r"],
            "delta_auc": al["auc_r"] - bm["auc_r"],
        })

    paired_df = pd.DataFrame(paired_rows)
    print(paired_df[["seed", "bodymean_R", "aligned_R", "delta_R_mean", "delta_R_p90", "delta_R_top10", "delta_N_mod", "delta_auc"]].to_string(index=False))
    print("-" * 100)

    # Paired Statistics & Bootstrap CIs
    diff_r = paired_df["delta_R_mean"].values
    diff_p90 = paired_df["delta_R_p90"].values
    diff_top10 = paired_df["delta_R_top10"].values
    diff_mod = paired_df["delta_N_mod"].values
    diff_auc = paired_df["delta_auc"].values

    ci_r = bootstrap_ci(diff_r)
    ci_p90 = bootstrap_ci(diff_p90)
    ci_top10 = bootstrap_ci(diff_top10)
    ci_mod = bootstrap_ci(diff_mod)
    ci_auc = bootstrap_ci(diff_auc)

    wins_r = int((diff_r > 0).sum())
    wins_p90 = int((diff_p90 > 0).sum())
    wins_top10 = int((diff_top10 > 0).sum())
    wins_auc = int((diff_auc > 0).sum())

    print(f"Mean Paired Delta R_mean:   {diff_r.mean():+.3f}  [95% Bootstrap CI: {ci_r[0]:+.3f}, {ci_r[1]:+.3f}]  Aligned Wins: {wins_r}/{len(completed_seeds)}")
    print(f"Mean Paired Delta R_p90:    {diff_p90.mean():+.3f}  [95% Bootstrap CI: {ci_p90[0]:+.3f}, {ci_p90[1]:+.3f}]  Aligned Wins: {wins_p90}/{len(completed_seeds)}")
    print(f"Mean Paired Delta R_top10:  {diff_top10.mean():+.3f}  [95% Bootstrap CI: {ci_top10[0]:+.3f}, {ci_top10[1]:+.3f}]  Aligned Wins: {wins_top10}/{len(completed_seeds)}")
    print(f"Mean Paired Delta Modules:  {diff_mod.mean():+.3f}  [95% Bootstrap CI: {ci_mod[0]:+.3f}, {ci_mod[1]:+.3f}]")
    print(f"Mean Paired Delta AUC_R:    {diff_auc.mean():+.3f}  [95% Bootstrap CI: {ci_auc[0]:+.3f}, {ci_auc[1]:+.3f}]  Aligned Wins: {wins_auc}/{len(completed_seeds)}")
    print("=" * 100 + "\n")

    # 3. Pooled Matched-Complexity Analysis
    pooled_posts = {}
    for m in methods:
        pooled_r = []
        pooled_counts = []
        for s in completed_seeds:
            post = all_posts[m][s]
            r = post["R_post"] if "R_post" in post else post["R"]
            pooled_r.append(r)
            pooled_counts.append(post["counts"])
        pooled_posts[m] = {
            "R": np.concatenate(pooled_r, axis=0),
            "counts": np.concatenate(pooled_counts, axis=0),
        }

    print("=" * 100)
    print(f"POOLED MATCHED-COMPLEXITY PERFORMANCE: E[R_post | N_modules] (Total N={len(pooled_posts['bodymean']['R'])} bodies/method)")
    print("=" * 100)
    mod_bins = list(range(5, 21))
    header = f"{'Modules':<8} | {'Body Mean (mu_b)':<30} | {'Aligned Spatial (mu_b + delta)':<30} | {'Delta':<10}"
    print(header)
    print("-" * len(header))
    for m in mod_bins:
        mask_bm = (pooled_posts["bodymean"]["counts"].sum(axis=1) == m)
        mask_al = (pooled_posts["aligned"]["counts"].sum(axis=1) == m)
        n_bm = mask_bm.sum()
        n_al = mask_al.sum()
        if n_bm >= 30 and n_al >= 30:
            r_bm = pooled_posts["bodymean"]["R"][mask_bm].mean()
            r_al = pooled_posts["aligned"]["R"][mask_al].mean()
            delta = r_al - r_bm
            print(f"{m:<8} | {r_bm:6.2f} (N={n_bm:<5})               | {r_al:6.2f} (N={n_al:<5})               | {delta:+6.2f}")
        elif n_bm >= 30:
            r_bm = pooled_posts["bodymean"]["R"][mask_bm].mean()
            print(f"{m:<8} | {r_bm:6.2f} (N={n_bm:<5})               | {'N < 30':<30} | {'N/A':<10}")
        elif n_al >= 30:
            r_al = pooled_posts["aligned"]["R"][mask_al].mean()
            print(f"{m:<8} | {'N < 30':<30} | {r_al:6.2f} (N={n_al:<5})               | {'N/A':<10}")
    print("=" * 100 + "\n")

    # Render Visual Figures
    plot_learning_curves(all_tbs, all_metrics, os.path.join(args.out_dir, "multiseed_learning_curves.png"))
    plot_matched_complexity(pooled_posts, os.path.join(args.out_dir, "multiseed_matched_complexity.png"))
    plot_paired_differences(paired_df, os.path.join(args.out_dir, "multiseed_paired_analysis.png"))


if __name__ == "__main__":
    main()
