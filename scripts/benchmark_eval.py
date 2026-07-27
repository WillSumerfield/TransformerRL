#!/usr/bin/env python3
"""Headless final-checkpoint evaluation for CoDesign and its baselines.

Like ``scripts/eval.py``, this accepts one or more already-trained run
directories, evaluates each requested epoch, prints a side-by-side table, and
writes one comparison result. The benchmark adds final-budget/seed checks and
retains every sampled body and episode.

Usage:
  python scripts/benchmark_eval.py RUN [RUN ...]
      [--method codesign|fixed_body|uniform_action]
      [--epochs final|N,N,...] [--preset paper|smoke]
      [--episodes N] [--top-k K] [--num-envs N] [--out DIRECTORY]

An unprefixed RUN uses ``--method`` (or the YAML's ``method``). To compare
different methods in one result, prefix each run, for example:

  python scripts/benchmark_eval.py \
      codesign=RUN_A fixed_body=RUN_B uniform_action=RUN_C

``final`` is the paper default. Numeric epochs are intended for development and
normally require ``--allow-incomplete-training``.
"""
import os
import sys
import sysconfig
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parent.parent


def _ensure_vlearn_native_libraries() -> None:
    """Restart once so Linux resolves native libraries from this virtualenv.

    Updating ``LD_LIBRARY_PATH`` after Python starts is too late for ``dlopen``.
    The benchmark is commonly invoked as ``.venv/bin/python ...``, so make that
    direct command as reliable as sourcing ``scripts/activate_uv.sh`` first.
    """
    site_packages = Path(sysconfig.get_path("purelib"))
    required = [
        site_packages / "nvidia/cu13/lib",
        site_packages / "vlearn/lib",
    ]
    required = [str(path) for path in required if path.is_dir()]
    current = [
        path
        for path in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if path
    ]

    vlearn_root = (_ROOT.parent / "vlearn-main").resolve()
    os.environ.setdefault("VL_WORKING_DIRECTORY", str(vlearn_root))
    if all(path in current for path in required):
        return

    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        [*required, *[path for path in current if path not in required]]
    )
    original_arguments = getattr(sys, "orig_argv", [sys.executable, *sys.argv])
    os.execve(
        sys.executable,
        [sys.executable, *original_arguments[1:]],
        environment,
    )


_ensure_vlearn_native_libraries()

import argparse
from typing import Any

import torch
import yaml

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from benchmarks.codesign import (  # noqa: E402
    checkpoints_for_run,
    load_codesign,
)
from benchmarks.evaluate import evaluate_runs, load_config, validate_config  # noqa: E402
from benchmarks.fixed_body import load_fixed_body  # noqa: E402
from benchmarks.uniform_action import load_uniform_action  # noqa: E402

_METHOD_LOADERS = {
    "codesign": load_codesign,
    "fixed_body": load_fixed_body,
    "uniform_action": load_uniform_action,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help=(
            "run directory, optionally prefixed with codesign=, fixed_body= "
            "or uniform_action="
        ),
    )
    parser.add_argument(
        "--method",
        choices=tuple(_METHOD_LOADERS),
        help="method for unprefixed RUNs (default comes from YAML)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/benchmarks/benchmark.yaml",
        help="benchmark YAML (default: configs/benchmarks/benchmark.yaml)",
    )
    parser.add_argument(
        "--epochs",
        default="final",
        help="'final' (default) or comma-separated epoch numbers",
    )
    parser.add_argument(
        "--preset",
        choices=("paper", "smoke"),
        help="evaluation size preset (default comes from YAML)",
    )
    parser.add_argument("--episodes", type=int, help="episodes per sampled pair")
    parser.add_argument("--top-k", type=int, dest="top_k")
    parser.add_argument(
        "--num-envs",
        type=int,
        help="sampled pair count, with one fixed body per VSim environment",
    )
    parser.add_argument("--design-samples", type=int, help="diversity sample count")
    parser.add_argument(
        "--seed",
        type=int,
        help="use this exact integer for morphology, rollout and diversity",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="comparison directory (default: evals/benchmarks/<timestamp>)",
    )
    parser.add_argument("--device", help="override runtime.device (default cuda:0)")
    parser.add_argument(
        "--allow-incomplete-training",
        action="store_true",
        help="allow a non-final training budget (development only)",
    )
    parser.add_argument(
        "--allow-seed-mismatch",
        action="store_true",
        help="allow a run outside the required paper training seeds (development only)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the fully resolved config and exit without loading the method",
    )
    return parser


def _jobs(
    config: dict[str, Any],
    runs: list[str],
    epochs: str,
    device: torch.device,
):
    """Load one policy at a time, matching the job loop in ``scripts/eval.py``."""
    default_method = config["method"]
    for specification in runs:
        prefix, separator, value = specification.partition("=")
        if separator and prefix in _METHOD_LOADERS:
            method_name, run = prefix, value
        else:
            method_name, run = default_method, specification
        method_config = config[method_name]
        for label, checkpoint in checkpoints_for_run(
            method_config,
            run,
            epochs,
        ):
            print(
                f"[benchmark] {method_name}: {Path(run).name} @ {label}: "
                f"{checkpoint.name}",
                flush=True,
            )
            checkpoint_config = {
                **method_config,
                "run_dir": run,
                "checkpoint": str(checkpoint),
            }
            yield _METHOD_LOADERS[method_name](checkpoint_config, device)


def _print_table(rows: list[dict[str, Any]], top_k: int) -> None:
    """Print the headline columns while ``summary.csv`` retains every metric."""
    header = (
        f"{'method':<10} {'run':<24} {'epoch':>12} "
        f"{'expected':>10} {f'top{top_k}':>10} {'fall':>8} {'unique':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method']:<10} "
            f"{row['run'][:24]:<24} "
            f"{row['epoch']:>12} "
            f"{row['benchmark/return/expected']:>10.2f} "
            f"{row['benchmark/selection/topk_of_m']:>10.2f} "
            f"{row['benchmark/stability/fall_rate']:>8.3f} "
            f"{row['benchmark/diversity/unique_fraction']:>8.3f}"
        )
    print()


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    config = load_config(args.config, preset=args.preset)
    evaluation = config["evaluation"]
    if args.episodes is not None:
        evaluation["episodes_per_pair"] = args.episodes
    if args.top_k is not None:
        evaluation["top_k"] = args.top_k
    if args.num_envs is not None:
        evaluation["pairs"] = args.num_envs
    if args.design_samples is not None:
        evaluation["design_samples"] = args.design_samples
    if args.seed is not None:
        evaluation["seeds"] = {
            "morphology": args.seed,
            "rollout": args.seed,
            "diversity": args.seed,
        }
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.method is not None:
        config["method"] = args.method
    if args.allow_incomplete_training:
        config["checks"]["enforce_training_budget"] = False
    if args.allow_seed_mismatch:
        config["checks"]["enforce_training_seed"] = False
    validate_config(config)

    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
        return
    if not args.runs:
        parser.error("at least one RUN is required unless --print-config is used")

    print(
        f"[benchmark] {len(args.runs)} run(s); "
        f"{evaluation['pairs']} bodies x "
        f"{evaluation['episodes_per_pair']} episodes/body; "
        f"preset={evaluation['preset']}",
        flush=True,
    )
    device = torch.device(config["runtime"]["device"])
    jobs = _jobs(config, args.runs, args.epochs, device)
    destination, rows = evaluate_runs(
        config,
        jobs,
        project_root=_ROOT,
        destination=args.out,
    )
    _print_table(rows, int(evaluation["top_k"]))
    print(f"[benchmark] results -> {destination} ({len(rows)} row(s))", flush=True)


if __name__ == "__main__":
    main()
