"""Faithful loading, sampling and control for the existing CoDesign method."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from transformer_rl.models import _raw_tail

from .data import EvaluationPairs

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RUN_ROOT = _ROOT / "runs/ant_codesign/codesign_single_transformer"
_VLEARN_TRAIN = _ROOT.parent / "vlearn-main" / "train"


def _resolve_run_dir(
    value: str | Path | None,
    default_run_root: str | Path = _DEFAULT_RUN_ROOT,
) -> Path:
    if value is None:
        raise ValueError("method run_dir is required")
    supplied = Path(value).expanduser()
    default_root = Path(default_run_root).expanduser()
    if not default_root.is_absolute():
        default_root = _ROOT / default_root
    candidates = (
        [supplied]
        if supplied.is_absolute()
        else [_ROOT / supplied, default_root / supplied]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"run directory does not exist: {supplied}")


def _resolve_relative_path(value: str | Path, run_dir: Path) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_absolute():
        return supplied.resolve()
    for candidate in (run_dir / supplied, _ROOT / supplied):
        if candidate.exists():
            return candidate.resolve()
    return (run_dir / supplied).resolve()


def _load_run_config(
    run_dir: Path,
    configured_path: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    path = (
        _resolve_relative_path(configured_path, run_dir)
        if configured_path is not None
        else run_dir / "config.yaml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"CoDesign run config does not exist: {path}")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"CoDesign run config must be a YAML mapping: {path}")
    return path, config


def resolve_checkpoint(
    run_dir: Path,
    run_config: dict[str, Any],
    requested: str | Path | None,
) -> tuple[str, Path]:
    """Select an explicit epoch checkpoint, never rl_games' bare best file."""
    if requested not in (None, "final"):
        path = _resolve_relative_path(requested, run_dir)
        if not path.is_file():
            path = (run_dir / "nn" / Path(requested)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CoDesign checkpoint does not exist: {path}")
        epoch = re.search(r"_ep_(\d+)", path.name)
        if epoch is None:
            raise ValueError(
                "an explicit checkpoint must contain '_ep_<N>'; "
                "the rl_games bare best checkpoint is not benchmark-eligible"
            )
        return f"ep_{epoch.group(1)}", path

    try:
        final_epoch = int(run_config["params"]["config"]["max_epochs"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("run config has no valid params.config.max_epochs") from error

    matches = []
    for path in (run_dir / "nn").glob("*_ep_*.pth"):
        epoch = re.search(r"_ep_(\d+)", path.name)
        if epoch and int(epoch.group(1)) == final_epoch:
            matches.append(path.resolve())
    if not matches:
        raise FileNotFoundError(
            f"no epoch-{final_epoch} checkpoint exists under {run_dir / 'nn'}"
        )
    if len(matches) > 1:
        raise ValueError(
            "multiple final checkpoints match; select one explicitly: "
            + ", ".join(path.name for path in sorted(matches))
        )
    return f"final_ep_{final_epoch}", matches[0]


def checkpoints_for_run(
    config: dict[str, Any],
    run: str | Path,
    epochs: str,
) -> list[tuple[str, Path]]:
    """Resolve ``final`` or a comma-separated epoch list, like ``scripts/eval.py``."""
    run_dir = _resolve_run_dir(run, config.get("run_root", _DEFAULT_RUN_ROOT))
    _, run_config = _load_run_config(run_dir, config.get("run_config"))
    if epochs == "final":
        return [resolve_checkpoint(run_dir, run_config, "final")]
    if epochs == "best":
        raise ValueError(
            "the benchmark cannot account for the bare best checkpoint; "
            "use final or explicit epoch numbers"
        )
    try:
        requested = {int(value) for value in epochs.split(",")}
    except ValueError as error:
        raise ValueError("--epochs must be 'final' or comma-separated numbers") from error
    if not requested:
        raise ValueError("--epochs must select at least one checkpoint")

    matches: dict[int, Path] = {}
    for path in (run_dir / "nn").glob("*_ep_*.pth"):
        match = re.search(r"_ep_(\d+)", path.name)
        if match and int(match.group(1)) in requested:
            epoch = int(match.group(1))
            if epoch in matches:
                raise ValueError(f"multiple checkpoints match epoch {epoch} in {run_dir}")
            matches[epoch] = path.resolve()
    missing = requested - set(matches)
    if missing:
        raise FileNotFoundError(
            f"no checkpoint for epoch(s) {sorted(missing)} under {run_dir / 'nn'}"
        )
    return [(f"ep_{epoch}", matches[epoch]) for epoch in sorted(matches)]


def training_budget(
    run_config: dict[str, Any],
    checkpoint_label: str,
) -> tuple[int, int]:
    """Return physics steps consumed and the run's parallel environment count."""
    epoch = re.search(r"(?:final_)?ep_(\d+)", checkpoint_label)
    if epoch is None:
        raise ValueError(f"checkpoint label has no epoch: {checkpoint_label}")
    try:
        settings = run_config["params"]["config"]
        parallel_envs = int(settings["num_actors"])
        horizon = int(settings["horizon_length"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("run config has no valid actor/horizon settings") from error
    return int(epoch.group(1)) * parallel_envs * horizon, parallel_envs


class CodesignMethod:
    """The small method-specific surface consumed by ``benchmarks.evaluate``."""

    name = "codesign"

    def __init__(
        self,
        *,
        network: Any,
        observation_normalizer: Any,
        device: torch.device,
        run_dir: Path,
        run_config_path: Path,
        run_config: dict[str, Any],
        checkpoint_path: Path,
        checkpoint_label: str,
        training_seed: int,
    ) -> None:
        self.network = network
        self.observation_normalizer = observation_normalizer
        self.device = device
        self.run_dir = run_dir
        self.run_config_path = run_config_path
        self.run_config = run_config
        self.checkpoint_path = checkpoint_path
        self.checkpoint_label = checkpoint_label
        self.training_seed = training_seed
        self.training_steps, self.parallel_envs = training_budget(
            run_config,
            checkpoint_label,
        )
        self.trainable_parameters = sum(
            parameter.numel() for parameter in network.parameters()
        )

    def _sample(self, count: int, seed: int) -> EvaluationPairs:
        """Sample with the trained generator's native stochastic distribution."""
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            sample = self.network.net.sample(count, mode="stochastic")
        return EvaluationPairs(
            counts=sample["counts"].detach().cpu().numpy(),
            eff_sub=sample["eff_sub"].detach().cpu().numpy(),
            cap_sub=sample["cap_sub"].detach().cpu().numpy(),
            controller_ids=np.full(count, self.checkpoint_path.name),
            weights=np.full(count, 1.0 / count),
        )

    def sample_pairs(self, count: int, seed: int) -> EvaluationPairs:
        return self._sample(count, seed)

    def sample_designs(self, count: int, seed: int) -> EvaluationPairs:
        return self._sample(count, seed)

    def create_environment(self, pair_count: int, rollout_seed: int) -> Any:
        from envs.ant_envs.ant_codesign import AntCodesignEnv

        saved = self.run_config.get("env", {})
        arguments: dict[str, Any] = {
            "sample_morphs": True,
            "rendering": False,
            "raise_exception": False,
            "seed": int(rollout_seed),
            "with_window": False,
            "base_legs": tuple(saved.get("base_legs", (1, 4, 6))),
            "value_size": int(saved.get("value_size", 1)),
        }
        for key in (
            "max_episode_length",
            "ctrl_cost_weight",
            "healthy_reward",
            "healthy_y_range",
            "reset_noise_scale",
            "timestep",
            "frame_skip",
            "spacing",
            "max_contact_pairs_per_env",
        ):
            if key in saved:
                arguments[key] = saved[key]
        return AntCodesignEnv(pair_count, self.device, **arguments)

    def install_pairs(self, environment: Any, pairs: EvaluationPairs) -> None:
        environment.set_next(pairs.counts, pairs.eff_sub, pairs.cap_sub)
        environment.resample()

    @torch.no_grad()
    def deterministic_action(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self.observation_normalizer(observation).clone()
        tail_start, tail_length = _raw_tail(self.network.net)
        normalized[..., tail_start : tail_start + tail_length] = observation[
            ..., tail_start : tail_start + tail_length
        ]
        action, _, value, _ = self.network({"obs": normalized})
        return action.clamp(-1.0, 1.0), None if value is None else value.squeeze(-1)


def load_saved_controller(
    config: dict[str, Any],
    device: torch.device,
    method_class: type[CodesignMethod] = CodesignMethod,
) -> CodesignMethod:
    """Load a saved controller shared by CoDesign and its control baselines."""
    if str(_VLEARN_TRAIN) not in sys.path:
        sys.path.insert(0, str(_VLEARN_TRAIN))
    from envs.ant_envs.ant_multimorph import _N_DOFS_FULL, _OBS_TOTAL
    from experiments.ppg_parity import _load_policy

    run_dir = _resolve_run_dir(
        config.get("run_dir"),
        config.get("run_root", _DEFAULT_RUN_ROOT),
    )
    config_path, run_config = _load_run_config(run_dir, config.get("run_config"))
    checkpoint_label, checkpoint_path = resolve_checkpoint(
        run_dir,
        run_config,
        config.get("checkpoint"),
    )
    params = run_config["params"]
    network, normalizer = _load_policy(
        checkpoint_path,
        params["network"],
        device,
        value_size=int(run_config.get("env", {}).get("value_size", 1)),
        obs_base=_OBS_TOTAL,
        n_act=_N_DOFS_FULL,
    )
    training_seed = config.get("training_seed")
    if training_seed is None:
        training_seed = params.get("seed")
    if training_seed is None:
        raise ValueError("training seed is absent from the run and benchmark config")
    return method_class(
        network=network,
        observation_normalizer=normalizer,
        device=device,
        run_dir=run_dir,
        run_config_path=config_path,
        run_config=run_config,
        checkpoint_path=checkpoint_path,
        checkpoint_label=checkpoint_label,
        training_seed=int(training_seed),
    )


def load_codesign(config: dict[str, Any], device: torch.device) -> CodesignMethod:
    """Load one saved CoDesign run using the same policy loader as legacy eval."""
    return load_saved_controller(config, device)
