#!/usr/bin/env python3
"""Train faithful Neural Graph Evolution on the shared VSim ant task.

Examples:
  python scripts/train_ant_nge.py train --seed 42 --name s42
  python scripts/train_ant_nge.py train --seed 42 --name smoke_s42 --smoke
  python scripts/train_ant_nge.py train \
      --resume runs/benchmarks/nge/nge_nervenetpp/s42/checkpoints/generation_0005.pth

``--max-environment-steps`` is a development-only override.  It must still end
at a complete NGE generation boundary.
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
    """Restart once so direct ``.venv/bin/python`` invocations can load VSim."""
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

from benchmarks.nge.training import (  # noqa: E402
    NGETrainer,
    load_nge_config,
    validate_nge_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train or resume one NGE seed")
    train.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/benchmarks/nge.yaml",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--name",
        help="run name beneath runs/benchmarks/nge/nge_nervenetpp (default s<seed>)",
    )
    train.add_argument("--device", help="override runtime.device")
    train.add_argument(
        "--smoke",
        action="store_true",
        help="tiny one-generation development run (not benchmark-compliant)",
    )
    train.add_argument(
        "--max-environment-steps",
        type=int,
        help="shorter exact generation-boundary budget for development",
    )
    train.add_argument(
        "--resume",
        type=Path,
        help="complete generation checkpoint; config is read from its run",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.resume is not None:
        checkpoint = args.resume.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
        if checkpoint.parent.name != "checkpoints":
            raise ValueError("resume checkpoint must be inside a run's checkpoints/")
        run_dir = checkpoint.parent.parent
        config = load_nge_config(run_dir / "config.yaml")
        resume = checkpoint
    else:
        config = load_nge_config(args.config)
        config["seed"] = int(args.seed)
        name = args.name or f"s{args.seed}"
        run_dir = (
            _ROOT
            / "runs/benchmarks/nge/nge_nervenetpp"
            / name
        )
        resume = None

    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.smoke:
        if resume is not None:
            raise ValueError("--smoke cannot change a resumed run")
        config["budget"].update(
            {
                "parallel_envs": 8,
                # One tiny PPO batch (8 * 4) plus one complete selection pass
                # (4 species * 1 env * 5 ticks).
                "environment_steps": 52,
            }
        )
        config["environment"]["max_episode_length"] = 4
        config["population"].update(
            {
                "size": 4,
                "candidate_pool_size": 8,
                "elimination_rate": 0.25,
            }
        )
        config["training"].update(
            {
                "rollout_steps": 4,
                "updates_per_generation": 1,
                "sequence_length": 2,
                "optimization_epochs": 1,
                "minibatch_sequences": 4,
            }
        )
        config["selection_evaluation"].update(
            {
                "environments_per_species": 1,
                "rollout_steps": 5,
            }
        )
        config["fidelity_constraints"][
            "minimum_transitions_per_species_batch"
        ] = 5
        config["checkpoint"]["every_generations"] = 1
    if args.max_environment_steps is not None:
        config["budget"]["environment_steps"] = int(
            args.max_environment_steps
        )
    validate_nge_config(config)
    trainer = NGETrainer(config, run_dir, resume=resume)
    trainer.train()


if __name__ == "__main__":
    main()
