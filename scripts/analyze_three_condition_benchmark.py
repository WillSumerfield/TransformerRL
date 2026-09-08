#!/usr/bin/env python3
"""Multi-seed comparative analysis script for the Three-Condition Benchmark:
Condition A: Prefix Only (none)
Condition B: Body Mean (mu_b) [Reference middle condition]
Condition C: Direct Body Feedback from R_post

Evaluates across 5 seeds (42, 43, 44, 45, 46).
Computes:
1. Performance, morphology, efficiency, and diagnostic metrics per seed & condition.
2. Aggregate 3-condition summary table (mean +/- std across seeds).
3. Paired seed differences: (B - A), (C - A), (C - B) with bootstrap 95% CIs and win counts.
4. Matched-complexity analysis E[R_post | N_modules].
5. Publication figures: learning curves, matched complexity, and paired difference bars.
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


def bootstrap_ci(diffs: np.ndarray, num_boot: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(num_boot)])
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


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

    # Elapsed wall-clock and total frames
    wall_clock = np.nan
    fps = np.nan
    total_frames = np.nan
    if "quality/R_mean" in tb:
        event_files = sorted(glob.glob(os.path.join(str(run_dir), "**", "events.out.tfevents.*"), recursive=True))
        if event_files:
            acc = EventAccumulator(event_files[-1])
            acc.Reload()
            sc = acc.Scalars("quality/R_mean")
            if len(sc) >= 2:
                wall_clock = sc[-1].wall_time - sc[0].wall_time
                total_frames = float(sc[-1].step)
                if wall_clock > 0:
                    fps = total_frames / wall_clock

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
        "auc_r": auc_r,
        "wall_clock": wall_clock,
        "fps": fps,
        "total_frames": total_frames,
        # Direct body diagnostics
        "raw_r_post_mean": get_mean_last_n("codesign/genact/raw_r_post_mean"),
        "raw_r_post_std": get_mean_last_n("codesign/genact/raw_r_post_std"),
        "body_adv_mean": get_mean_last_n("codesign/genact/body_adv_mean"),
        "body_adv_std": get_mean_last_n("codesign/genact/body_adv_std"),
        "corr_modcount_body_mass": get_mean_last_n("codesign/genact/corr_modcount_body_mass"),
    }


def plot_learning_curves(
    all_tbs: Dict[str, Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]],
    all_metrics: Dict[str, Dict[int, Dict[str, Any]]],
    out_fig: str,
):
    os.makedirs(os.path.dirname(os.path.abspath(out_fig)), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Three-Condition Multi-Seed Training Dynamics (5 Seeds: 42-46)\n"
                 "Solid lines: Mean trajectory; Shaded bands: ±1 Standard Deviation across seeds",
                 fontsize=15, y=0.98)

    tags_to_plot = [
        ("quality/R_mean", "Mean Return (quality/R_mean)", axes[0, 0]),
        ("quality/R_top10_mean", "Top-10% Return (quality/R_top10_mean)", axes[0, 1]),
        ("build/modulecount", "Module Count Intent (build/modulecount)", axes[0, 2]),
        ("gen/action_prob/eff", "P(Effector Action) (gen/action_prob/eff)", axes[1, 0]),
        ("gen/entropy", "Generator Policy Entropy (gen/entropy)", axes[1, 1]),
    ]

    colors = {"prefix": "#1f77b4", "bodymean": "#ff7f0e", "directbody": "#2ca02c"}
    labels = {
        "prefix": "Condition A: Prefix Only (none)",
        "bodymean": "Condition B: Body Mean (mu_b)",
        "directbody": "Condition C: Direct Body (R_post)",
    }

    for tag, title, ax in tags_to_plot:
        ax.set_title(title, fontsize=12, fontweight="bold")
        for method in ["prefix", "bodymean", "directbody"]:
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
                min_step = max(s[0] for s in all_steps)
                max_step = min(s[-1] for s in all_steps)
                common_steps = np.linspace(min_step, max_step, 50)
                interp_vals = np.array([np.interp(common_steps, s, v) for s, v in zip(all_steps, all_vals)])
                mean_v = interp_vals.mean(axis=0)
                std_v = interp_vals.std(axis=0)

                ax.plot(common_steps, mean_v, label=labels[method], color=colors[method], linewidth=2.2)
                ax.fill_between(common_steps, mean_v - std_v, mean_v + std_v, color=colors[method], alpha=0.18)
            elif len(all_vals) == 1:
                ax.plot(all_steps[0], all_vals[0], label=labels[method], color=colors[method], linewidth=2)

        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    # 6th panel: AUC_R comparison
    ax6 = axes[1, 2]
    ax6.set_title("Training Return Efficiency: AUC_R by Seed", fontsize=12, fontweight="bold")
    x = np.arange(len(SEEDS))
    width = 0.25
    p_aucs = [all_metrics["prefix"][s]["auc_r"] for s in SEEDS if s in all_metrics["prefix"]]
    bm_aucs = [all_metrics["bodymean"][s]["auc_r"] for s in SEEDS if s in all_metrics["bodymean"]]
    db_aucs = [all_metrics["directbody"][s]["auc_r"] for s in SEEDS if s in all_metrics["directbody"]]

    if p_aucs and bm_aucs and db_aucs:
        ax6.bar(x - width, p_aucs, width, label="Prefix Only", color=colors["prefix"], alpha=0.85)
        ax6.bar(x, bm_aucs, width, label="Body Mean (mu_b)", color=colors["bodymean"], alpha=0.85)
        ax6.bar(x + width, db_aucs, width, label="Direct Body (R_post)", color=colors["directbody"], alpha=0.85)
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
                 "Evaluating E[R_post | N_modules] to test whether Direct Body Feedback matches Body Mean",
                 fontsize=14, y=0.98)

    colors = {"prefix": "#1f77b4", "bodymean": "#ff7f0e", "directbody": "#2ca02c"}
    labels = {
        "prefix": "Prefix Only (none)",
        "bodymean": "Body Mean (mu_b)",
        "directbody": "Direct Body (R_post)",
    }

    mod_bins = list(range(5, 22))

    for method in ["prefix", "bodymean", "directbody"]:
        if method not in pooled_posts:
            continue
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

        ax1.plot(valid_m, means, "o-", label=labels[method], color=colors[method], linewidth=2.2, markersize=5)
        ax1.fill_between(valid_m, ci_lows, ci_highs, color=colors[method], alpha=0.18)
        ax2.plot(valid_m, sample_counts, "s--", label=labels[method], color=colors[method], linewidth=1.8, markersize=4)

    ax1.set_xlabel("Realized Module Count ($N_{\\mathrm{modules}}$)", fontsize=11)
    ax1.set_ylabel("Expected Post-Adaptation Return $\\mathbb{E}[R_{\\mathrm{post}} \\mid N]$", fontsize=11)
    ax1.set_title("Matched Return by Module Count (95% Bootstrap CI)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=10)

    ax2.set_xlabel("Realized Module Count ($N_{\\mathrm{modules}}$)", fontsize=11)
    ax2.set_ylabel("Number of Evaluated Morphologies", fontsize=11)
    ax2.set_title("Morphology Population Distribution by Module Count", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[saved figure] {out_fig}")


def plot_paired_differences(
    all_metrics: Dict[str, Dict[int, Dict[str, Any]]],
    out_fig: str,
):
    os.makedirs(os.path.dirname(os.path.abspath(out_fig)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Seed-by-Seed Paired Return Differences (5 Seeds: 42-46)", fontsize=14, y=0.98)

    comparisons = [
        ("B - A: Body Mean vs Prefix Only", "bodymean", "prefix", axes[0], "#ff7f0e"),
        ("C - A: Direct Body vs Prefix Only", "directbody", "prefix", axes[1], "#2ca02c"),
        ("C - B: Direct Body vs Body Mean", "directbody", "bodymean", axes[2], "#9467bd"),
    ]

    for title, m1, m2, ax, col in comparisons:
        diffs = []
        seeds_present = []
        for s in SEEDS:
            if s in all_metrics[m1] and s in all_metrics[m2]:
                d = all_metrics[m1][s]["R_mean"] - all_metrics[m2][s]["R_mean"]
                diffs.append(d)
                seeds_present.append(s)

        diffs = np.array(diffs)
        x = np.arange(len(seeds_present))
        bars = ax.bar(x, diffs, color=[col if v >= 0 else "#d62728" for v in diffs], alpha=0.85, width=0.55)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0)

        mean_d = float(np.mean(diffs))
        ci_l, ci_h = bootstrap_ci(diffs, num_boot=10000)
        ax.axhline(mean_d, color=col, linestyle="-", linewidth=2.0, label=f"Mean Diff: {mean_d:+.3f}")
        ax.axhspan(ci_l, ci_h, color=col, alpha=0.15, label=f"95% CI: [{ci_l:+.2f}, {ci_h:+.2f}]")

        for bar, val in zip(bars, diffs):
            y_pos = val + (0.2 if val >= 0 else -0.5)
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{val:+.2f}", ha="center", fontsize=9, fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in seeds_present])
        ax.set_ylabel("$\\Delta R_{\\mathrm{mean}}$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[saved figure] {out_fig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="/home/alex-daniel/.gemini/antigravity-ide/brain/46c09224-ba5d-4ed0-9ab2-41ac56ab6102")
    args = parser.parse_args()

    methods = ["prefix", "bodymean", "directbody"]
    prefixes = {
        "prefix": "multiseed_prefix",
        "bodymean": "multiseed_bodymean",
        "directbody": "multiseed_directbody",
    }

    all_metrics = {m: {} for m in methods}
    all_tbs = {m: {} for m in methods}
    all_posts = {m: {} for m in methods}

    for method in methods:
        for seed in SEEDS:
            run_name = f"{prefixes[method]}_seed{seed}"
            run_dir = BASE_DIR / run_name
            if not run_dir.exists():
                print(f"[WARN] Missing {run_dir}")
                continue
            tb = load_tb_scalars(run_dir)
            post = load_post_eval_data(run_dir)
            all_tbs[method][seed] = tb
            all_posts[method][seed] = post
            all_metrics[method][seed] = compute_single_run_metrics(run_dir, tb, post)

    # Aggregate summaries
    print("\n" + "=" * 120)
    print("INDIVIDUAL SEED METRICS")
    print("=" * 120)
    for method in methods:
        print(f"\n--- Method: {method} ---")
        rows = []
        for seed in SEEDS:
            if seed in all_metrics[method]:
                m = all_metrics[method][seed]
                rows.append({
                    "Seed": seed,
                    "R_mean": m["R_mean"],
                    "R_median": m["R_median"],
                    "R_p90": m["R_p90"],
                    "R_top10": m["R_top10_mean"],
                    "R_max": m["R_max"],
                    "N_mod": m["N_mod_mean"],
                    "Depth": m["mean_depth"],
                    "Unique%": m["unique_fraction"],
                    "P_eff": m["P_eff"],
                    "Entropy": m["entropy"],
                    "AUC_R": m["auc_r"],
                })
        print(pd.DataFrame(rows).to_string(index=False))

    # Pooled posts for matched complexity
    pooled_posts = {}
    for method in methods:
        r_all = []
        counts_all = []
        for seed in SEEDS:
            if seed in all_posts[method]:
                p = all_posts[method][seed]
                r = p["R_post"] if "R_post" in p else p["R"]
                r_all.append(r)
                counts_all.append(p["counts"])
        if r_all:
            pooled_posts[method] = {
                "R": np.concatenate(r_all, axis=0),
                "counts": np.concatenate(counts_all, axis=0),
            }

    # Aggregate 3-condition summary table
    print("\n" + "=" * 120)
    print("THREE-CONDITION AGGREGATE SUMMARY TABLE (MEAN +/- SD ACROSS 5 SEEDS)")
    print("=" * 120)
    keys_to_report = [
        ("R_mean", "Mean Post-Adapt Return (R_mean)"),
        ("R_median", "Median Return (R_median)"),
        ("R_p90", "P90 Return (R_p90)"),
        ("R_top10_mean", "Top-10% Mean Return"),
        ("R_max", "Best Return (R_max)"),
        ("N_mod_mean", "Mean Realized Modules (N_mod)"),
        ("N_mod_var", "Module Count Variance"),
        ("mean_depth", "Mean Limb Depth"),
        ("unique_fraction", "Literal Unique Morphology Frac"),
        ("P_eff", "P(Effector Action)"),
        ("P_cap", "P(Cap Action)"),
        ("entropy", "Generator Policy Entropy"),
        ("auc_r", "Training Return AUC (AUC_R)"),
        ("wall_clock", "Wall-Clock Time per Run (s)"),
        ("fps", "Throughput (FPS)"),
        ("corr_modcount_body_mass", "Corr(N_mod, body_mass) [C only]"),
    ]

    table_data = []
    for key, label in keys_to_report:
        row = {"Metric": label}
        for method in methods:
            vals = [all_metrics[method][s][key] for s in SEEDS if s in all_metrics[method] and not np.isnan(all_metrics[method][s][key])]
            if vals:
                row[method] = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
            else:
                row[method] = "N/A"
        table_data.append(row)
    df_summary = pd.DataFrame(table_data)
    df_summary.rename(columns={
        "prefix": "Condition A: Prefix Only",
        "bodymean": "Condition B: Body Mean (mu_b)",
        "directbody": "Condition C: Direct Body (R_post)",
    }, inplace=True)
    print(df_summary.to_string(index=False))

    # Paired differences
    print("\n" + "=" * 120)
    print("PAIRED SEED DIFFERENCES ACROSS CONDITIONS")
    print("=" * 120)
    pair_comps = [
        ("B - A (Body Mean - Prefix)", "bodymean", "prefix"),
        ("C - A (Direct Body - Prefix)", "directbody", "prefix"),
        ("C - B (Direct Body - Body Mean)", "directbody", "bodymean"),
    ]
    pair_rows = []
    for comp_name, m1, m2 in pair_comps:
        d_mean = np.array([all_metrics[m1][s]["R_mean"] - all_metrics[m2][s]["R_mean"] for s in SEEDS if s in all_metrics[m1] and s in all_metrics[m2]])
        d_top10 = np.array([all_metrics[m1][s]["R_top10_mean"] - all_metrics[m2][s]["R_top10_mean"] for s in SEEDS if s in all_metrics[m1] and s in all_metrics[m2]])
        d_auc = np.array([all_metrics[m1][s]["auc_r"] - all_metrics[m2][s]["auc_r"] for s in SEEDS if s in all_metrics[m1] and s in all_metrics[m2]])
        d_mod = np.array([all_metrics[m1][s]["N_mod_mean"] - all_metrics[m2][s]["N_mod_mean"] for s in SEEDS if s in all_metrics[m1] and s in all_metrics[m2]])

        ci_mean = bootstrap_ci(d_mean)
        ci_top10 = bootstrap_ci(d_top10)
        ci_auc = bootstrap_ci(d_auc)
        win_count = int(np.sum(d_mean > 0))

        pair_rows.append({
            "Comparison": comp_name,
            "Delta R_mean": f"{np.mean(d_mean):+.4f} +/- {np.std(d_mean):.4f}",
            "95% CI (R_mean)": f"[{ci_mean[0]:+.3f}, {ci_mean[1]:+.3f}]",
            "Win Count (R_mean)": f"{win_count} / {len(d_mean)}",
            "Delta R_top10": f"{np.mean(d_top10):+.4f}",
            "95% CI (R_top10)": f"[{ci_top10[0]:+.3f}, {ci_top10[1]:+.3f}]",
            "Delta AUC_R": f"{np.mean(d_auc):+.4f}",
            "Delta N_mod": f"{np.mean(d_mod):+.4f}",
        })
    print(pd.DataFrame(pair_rows).to_string(index=False))

    # Matched complexity breakdown table
    print("\n" + "=" * 120)
    print("POOLED MATCHED COMPLEXITY BREAKDOWN: E[R_post | N_modules]")
    print("=" * 120)
    mod_rows = []
    for m in range(5, 20):
        row = {"N_mod": m}
        for method in methods:
            if method in pooled_posts:
                data = pooled_posts[method]
                mask = (data["counts"].sum(axis=1) == m)
                cnt = mask.sum()
                if cnt >= 20:
                    sub_r = data["R"][mask]
                    row[method] = f"{np.mean(sub_r):.2f} (N={cnt})"
                else:
                    row[method] = f"- (N={cnt})"
        mod_rows.append(row)
    print(pd.DataFrame(mod_rows).to_string(index=False))

    # Save figures
    out_dir = Path(args.out_dir)
    plot_learning_curves(all_tbs, all_metrics, str(out_dir / "three_condition_learning_curves.png"))
    plot_matched_complexity(pooled_posts, str(out_dir / "three_condition_matched_complexity.png"))
    plot_paired_differences(all_metrics, str(out_dir / "three_condition_paired_analysis.png"))


if __name__ == "__main__":
    main()
