#!/usr/bin/env python3
"""Sequential runner for multi-seed CoDesign experiment.
Runs Condition B (body_mean) and Condition C (mean_plus_aligned_residual) across seeds 42, 43, 44, 45, 46.
Skips runs that are already completed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [42, 43, 44, 45, 46]
METHODS = [
    ("bodymean", "body_mean"),
    ("aligned", "mean_plus_aligned_residual"),
]

BASE_DIR = Path("runs/ant_codesign/codesign_single_transformer")


def is_run_completed(run_name: str) -> bool:
    post_eval = BASE_DIR / run_name / "post_eval" / "post_eval_window_0002.npz"
    return post_eval.exists()


def run_job(run_name: str, mode: str, seed: int):
    if is_run_completed(run_name):
        print(f"[SKIP] {run_name} already completed.")
        return

    print(f"\n{'='*80}\n[START] {run_name} (mode={mode}, seed={seed})\n{'='*80}\n")
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
        "--set", "params.config.generator.spatial_credit.enabled=true",
        "--set", "params.config.generator.spatial_credit.loss_coef=0.1",
        "--set", "params.config.generator.spatial_credit.tree_lambda=0.5",
        "--set", "params.config.generator.spatial_credit.pair_supervision=true",
        "--set", "params.config.generator.spatial_credit.pair_loss_coef=0.1",
        "--set", "params.config.generator.spatial_credit.use_for_genact=true",
        f"--set", f"params.config.generator.spatial_credit.genact_credit_mode={mode}",
        "--set", "params.config.generator.spatial_credit.genact_beta=0.5",
    ]

    t0 = time.time()
    ret = subprocess.run(cmd)
    t1 = time.time()
    elapsed = t1 - t0

    if ret.returncode != 0:
        print(f"[ERROR] {run_name} failed with exit code {ret.returncode} after {elapsed:.1f}s")
        sys.exit(ret.returncode)
    else:
        print(f"[SUCCESS] {run_name} completed in {elapsed:.1f}s ({elapsed/60:.2f} min)\n")


def main():
    # Run pairwise by seed so paired differences can be evaluated incrementally
    for seed in SEEDS:
        for prefix, mode in METHODS:
            run_name = f"multiseed_{prefix}_seed{seed}"
            run_job(run_name, mode, seed)

    print("\nAll 10 multi-seed jobs completed successfully!")


if __name__ == "__main__":
    main()
