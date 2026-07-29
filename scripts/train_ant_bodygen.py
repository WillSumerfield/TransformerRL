#!/usr/bin/env python3
"""Train faithful BodyGen on the shared VSim ant task.

Examples:
  python scripts/train_ant_bodygen.py train --seed 42 --name s42
  python scripts/train_ant_bodygen.py train --seed 42 --name smoke_s42 --smoke
  python scripts/train_ant_bodygen.py train \
      --resume runs/benchmarks/bodygen/bodygen_mosat/s42/checkpoints/update_0005.pth

``--smoke`` and ``--max-environment-steps`` mark the saved run as
development-only. Such checkpoints remain useful for integration testing but
are rejected by the paper benchmark's fidelity checks.
"""
from __future__ import annotations

import argparse
import os
import sys
import sysconfig
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))


def _ensure_vlearn_native_libraries() -> None:
    """Restart once so direct ``.venv/bin/python`` calls can load VSim."""
    site_packages = Path(sysconfig.get_path("purelib"))
    required = [
        site_packages / "nvidia/cu13/lib",
        site_packages / "vlearn/lib",
    ]
    required = [str(path) for path in required if path.is_dir()]
    current = [
        value
        for value in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if value
    ]
    os.environ.setdefault(
        "VL_WORKING_DIRECTORY",
        str((_ROOT.parent / "vlearn-main").resolve()),
    )
    if all(path in current for path in required):
        return
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        [*required, *[path for path in current if path not in required]]
    )
    original = getattr(sys, "orig_argv", [sys.executable, *sys.argv])
    os.execve(
        sys.executable,
        [sys.executable, *original[1:]],
        environment,
    )


_ensure_vlearn_native_libraries()

from benchmarks.bodygen.training import (  # noqa: E402
    BodyGenTrainer,
    load_bodygen_config,
    validate_bodygen_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train",
        help="train or resume one BodyGen seed",
    )
    train.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/benchmarks/bodygen.yaml",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--name",
        help=(
            "run name beneath runs/benchmarks/bodygen/bodygen_mosat "
            "(default s<seed>)"
        ),
    )
    train.add_argument("--device", help="override runtime.device")
    train.add_argument(
        "--smoke",
        action="store_true",
        help="tiny development run (not benchmark-compliant)",
    )
    train.add_argument(
        "--max-environment-steps",
        type=int,
        help="shorter exact physics budget for development",
    )
    train.add_argument(
        "--resume",
        type=Path,
        help="between-update checkpoint; config is read from its run",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.resume is not None:
        checkpoint = args.resume.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {checkpoint}"
            )
        if checkpoint.parent.name != "checkpoints":
            raise ValueError(
                "resume checkpoint must be inside a run's checkpoints/"
            )
        run_dir = checkpoint.parent.parent
        config = load_bodygen_config(run_dir / "config.yaml")
        resume = checkpoint
    else:
        config = load_bodygen_config(args.config)
        config["seed"] = int(args.seed)
        name = args.name or f"s{args.seed}"
        run_dir = (
            _ROOT
            / "runs/benchmarks/bodygen/bodygen_mosat"
            / name
        )
        resume = None

    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.smoke:
        if resume is not None:
            raise ValueError("--smoke cannot change a resumed run")
        config["runtime"]["development"] = True
        config["budget"].update(
            {
                "environment_steps": 100,
                "parallel_envs": 20,
            }
        )
        config["environment"]["max_episode_length"] = 4
        config["collection"].update(
            {
                "minimum_batch_transitions": 120,
                "minimum_transitions_per_stream": 6,
            }
        )
        config["training"].update(
            {
                "optimization_epochs": 1,
                "minibatch_size": 64,
            }
        )
        config["checkpoint"]["every_updates"] = 1
        config["native_evaluation"]["enabled"] = False
        config["training_evaluation"]["enabled"] = False
    if args.max_environment_steps is not None:
        if resume is not None:
            raise ValueError(
                "--max-environment-steps cannot change a resumed run"
            )
        if args.max_environment_steps <= 0:
            raise ValueError("--max-environment-steps must be positive")
        config["runtime"]["development"] = True
        config["budget"]["environment_steps"] = int(
            args.max_environment_steps
        )

    validate_bodygen_config(config)
    trainer = BodyGenTrainer(config, run_dir, resume=resume)
    trainer.train()


if __name__ == "__main__":
    main()
