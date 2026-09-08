#!/usr/bin/env python3
"""Sequential runner for Three-Condition CoDesign multi-seed benchmark.

Conditions:
A: Prefix Only (none) -> multiseed_prefix_seed{42..46}
B: Body Mean (mu_b)   -> multiseed_bodymean_seed{42..46} (already completed; reused)
C: Direct Body R_post -> multiseed_directbody_seed{42..46}

Checks completion via post_eval_window_0002.npz.
Reuses pilot_genact_baseline_matched for Condition A seed 42 if available.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42, 43, 44, 45, 46]
BASE_DIR = Path("runs/ant_codesign/codesign_single_transformer")


def is_run_completed(run_name: str) -> bool:
    post_eval = BASE_DIR / run_name / "post_eval" / "post_eval_window_0002.npz"
    return post_eval.exists()


def check_and_reuse_seed42_baseline():
    target_dir = BASE_DIR / "multiseed_prefix_seed42"
    source_dir = BASE_DIR / "pilot_genact_baseline_matched"
    if is_run_completed("multiseed_prefix_seed42"):
        print("[INFO] multiseed_prefix_seed42 already completed.")
        return

    source_eval = source_dir / "post_eval" / "post_eval_window_0002.npz"
    if source_eval.exists():
        print(f"[REUSE] Reusing bit-for-bit identical seed 42 baseline from {source_dir} -> {target_dir}...")
        os.makedirs(target_dir, exist_ok=True)
        for item in source_dir.iterdir():
            dest = target_dir / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        print("[REUSE] Seed 42 baseline successfully staged for multiseed_prefix_seed42.")


def print_config_diff():
    print("=" * 100)
    print("THREE-CONDITION CONFIGURATION COMPARISON & VERIFICATION")
    print("=" * 100)
    print("""
Common Hyperparameters (Bit-for-bit identical across all conditions):
  - Environment: ant-codesign-env (num_actors=4096, horizon_length=16)
  - Training duration: max_epochs=190 (3 resample windows, 12.3M control steps)
  - Pretrain schedule: n_pretrain=1, resample_interval=1
  - Optimizer: minibatch_size=8192, lr=0.001, kl_threshold=0.0083
  - GenCrit: max_prefixes=4000, return_target=post, gencrit_coef=0.5
  - GenAct: entropy_coef=0.05, clip=0.2, mini_epochs=4

Condition A: Prefix Only (multiseed_prefix_seed*)
  - generator advantage: Pure prefix telescoping delta V_G (hat A^prefix)
  - spatial_credit.use_for_genact: false
  - spatial_credit.genact_credit_mode: none
  - spatial_credit.enabled: true (passive logging/trunk matched to pilot)

Condition B: Body Mean mu_b (multiseed_bodymean_seed*) [EXISTING / REUSED]
  - generator advantage: hat A^prefix + 0.5 * hat mu_b
  - spatial_credit.use_for_genact: true
  - spatial_credit.genact_credit_mode: body_mean
  - spatial_credit.genact_beta: 0.5
  - spatial_credit.enabled: true (loss_coef=0.1, tree_lambda=0.5, pair_supervision=true)

Condition C: Direct Body R_post (multiseed_directbody_seed*) [NEW]
  - generator advantage: hat A^prefix + 0.5 * A^body (standardized R_post)
  - spatial_credit.use_for_genact: true
  - spatial_credit.genact_credit_mode: direct_body_rpost
  - spatial_credit.genact_beta: 0.5
  - spatial_credit.enabled: false (no spatial head, tree propagation, or pair supervision)
""")
    print("=" * 100 + "\n")


def run_job(run_name: str, condition: str, seed: int):
    if is_run_completed(run_name):
        print(f"[SKIP] {run_name} already completed.")
        return

    print(f"\n{'='*80}\n[START] {run_name} (condition={condition}, seed={seed})\n{'='*80}\n")

    cmd = [
        sys.executable,
        "scripts/train_ant_codesign_single.py",
        "train",
        "--headless", "True",
        "--name", run_name,
        "--seed", str(seed),
        "--max_epochs", "190",
        "--set", "params.config.minibatch_size=8192",
        "--set", "params.config.generator.max_prefixes=4000",
        "--set", "params.config.resample_interval=1",
        "--set", "params.config.generator.n_pretrain=1",
        "--set", "params.config.generator.return_target=post",
    ]

    if condition == "prefix_only":
        cmd += [
            "--set", "params.config.generator.spatial_credit.enabled=true",
            "--set", "params.config.generator.spatial_credit.loss_coef=0.1",
            "--set", "params.config.generator.spatial_credit.tree_lambda=0.5",
            "--set", "params.config.generator.spatial_credit.pair_supervision=true",
            "--set", "params.config.generator.spatial_credit.pair_loss_coef=0.1",
            "--set", "params.config.generator.spatial_credit.use_for_genact=false",
            "--set", "params.config.generator.spatial_credit.genact_credit_mode=none",
            "--set", "params.config.generator.spatial_credit.genact_beta=0.5",
        ]
    elif condition == "direct_body_rpost":
        cmd += [
            "--set", "params.config.generator.spatial_credit.enabled=false",
            "--set", "params.config.generator.spatial_credit.pair_supervision=false",
            "--set", "params.config.generator.spatial_credit.use_for_genact=true",
            "--set", "params.config.generator.spatial_credit.genact_credit_mode=direct_body_rpost",
            "--set", "params.config.generator.spatial_credit.genact_beta=0.5",
        ]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    env = os.environ.copy()
    env["VL_WORKING_DIRECTORY"] = str(Path(".").resolve())

    t0 = time.time()
    ret = subprocess.run(cmd, env=env)
    t1 = time.time()
    elapsed = t1 - t0

    if ret.returncode != 0:
        print(f"[ERROR] {run_name} failed with exit code {ret.returncode} after {elapsed:.1f}s")
        sys.exit(ret.returncode)
    else:
        print(f"[SUCCESS] {run_name} completed in {elapsed:.1f}s ({elapsed/60:.2f} min)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-seed42-prefix", action="store_true", help="Force rerunning seed 42 for prefix_only instead of reusing.")
    args = parser.parse_args()

    print_config_diff()

    # Stage or verify Condition A seed 42
    if not args.rerun_seed42_prefix:
        check_and_reuse_seed42_baseline()

    # Sequentially execute missing seeds
    for seed in SEEDS:
        # Condition A: Prefix Only
        run_name_prefix = f"multiseed_prefix_seed{seed}"
        run_job(run_name_prefix, "prefix_only", seed)

        # Condition C: Direct Body R_post
        run_name_direct = f"multiseed_directbody_seed{seed}"
        run_job(run_name_direct, "direct_body_rpost", seed)

    print("\nAll benchmark jobs finished successfully!")


if __name__ == "__main__":
    main()
