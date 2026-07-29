"""Adapter from a frozen BodyGen policy to the shared benchmark evaluator."""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from benchmarks.data import EvaluationPairs

from .credit import ReturnNormalizer
from .design import BodyGenDesign
from .mosat import ACTION_SIZE, OBSERVATION_SIZE, BodyGenNetworks

_ROOT = Path(__file__).resolve().parents[2]


def _checkpoint_config_matches(
    checkpoint_config: dict[str, Any],
    run_config: dict[str, Any],
) -> bool:
    """Ignore only the CUDA placement selected when a run was resumed."""

    checkpoint_algorithm = copy.deepcopy(checkpoint_config)
    saved_algorithm = copy.deepcopy(run_config)
    checkpoint_algorithm.get("runtime", {}).pop("device", None)
    saved_algorithm.get("runtime", {}).pop("device", None)
    return checkpoint_algorithm == saved_algorithm


def _run_directory(
    config: dict[str, Any],
    selected: str | Path | None = None,
) -> Path:
    """Resolve an absolute path or a run name below BodyGen's run root."""

    value = selected if selected is not None else config.get("run_dir")
    if value is None:
        raise ValueError("a BodyGen run directory is required")
    path = Path(value).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        root = Path(
            config.get(
                "run_root",
                "runs/benchmarks/bodygen/bodygen_mosat",
            )
        ).expanduser()
        if not root.is_absolute():
            root = _ROOT / root
        candidates = [_ROOT / path, root / path]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"BodyGen run directory does not exist: {path}")


def checkpoints_for_bodygen_run(
    config: dict[str, Any],
    selected_run: str | Path,
    requested: str,
) -> list[tuple[str, Path]]:
    """Resolve ``final`` or numeric native PPO-update checkpoints."""

    run_dir = _run_directory(config, selected_run)
    checkpoint_dir = run_dir / "checkpoints"
    if requested == "final":
        final = checkpoint_dir / "final.pth"
        if not final.is_file():
            raise FileNotFoundError(f"no final BodyGen checkpoint: {final}")
        return [("final", final.resolve())]
    try:
        updates = {int(value.strip()) for value in requested.split(",")}
    except ValueError as error:
        raise ValueError(
            "BodyGen --epochs accepts 'final' or comma-separated PPO updates"
        ) from error
    if not updates:
        raise ValueError("at least one BodyGen update must be selected")
    if any(update < 0 for update in updates):
        raise ValueError("BodyGen update numbers cannot be negative")

    result = []
    for update in sorted(updates):
        path = checkpoint_dir / f"update_{update:04d}.pth"
        if not path.is_file():
            raise FileNotFoundError(
                f"no BodyGen update {update} checkpoint under {checkpoint_dir}"
            )
        result.append((f"update_{update}", path.resolve()))
    return result


class BodyGenMethod:
    """Sample the frozen native design policy and use its shared controller."""

    name = "bodygen"

    def __init__(
        self,
        *,
        state: dict[str, Any],
        device: torch.device,
        run_dir: Path,
        run_config_path: Path,
        run_config: dict[str, Any],
        checkpoint_path: Path,
        checkpoint_label: str,
    ) -> None:
        if state.get("method") != "bodygen" or state.get("format_version") != 1:
            raise ValueError("checkpoint is not a supported BodyGen checkpoint")
        if not _checkpoint_config_matches(
            state.get("config", {}),
            run_config,
        ):
            raise ValueError(
                "BodyGen checkpoint config does not match the saved run config"
            )
        self.device = device
        self.run_dir = run_dir
        self.run_config_path = run_config_path
        self.run_config = run_config
        self.checkpoint_path = checkpoint_path
        self.checkpoint_label = checkpoint_label
        package_dir = Path(__file__).resolve().parent
        self.provenance_paths = (
            package_dir / "upstream.yaml",
            package_dir / "ADAPTATIONS.md",
            package_dir / "LICENSE.upstream",
        )

        self.training_seed = int(state["training_seed"])
        self.training_steps = int(state["environment_steps"])
        # Development mode relaxes native batch/horizon constraints for smoke
        # runs. Even an accidentally full-sized development budget must not be
        # reported as a paper-compliant result.
        self.paper_eligible = not bool(
            run_config.get("runtime", {}).get("development", False)
        )
        # This is the largest synchronous VSim wave actually used, not the
        # shared configuration cap.
        self.parallel_envs = int(state["peak_parallel_envs"])

        network = run_config["network"]
        self.networks = BodyGenNetworks(
            observation_size=OBSERVATION_SIZE,
            action_size=ACTION_SIZE,
            hidden_size=int(network["hidden_size"]),
            num_blocks=int(network["blocks"]),
            dtype=torch.float64,
            layer_norm=str(network["layer_norm"]),
            topology_embeddings=int(network["topology_embeddings"]),
            feed_forward_ratio=int(network["feed_forward_ratio"]),
            critic_hidden=tuple(int(width) for width in network["critic_hidden"]),
            initial_control_log_std=float(network["initial_control_log_std"]),
        ).to(device)
        self.networks.load_state_dict(state["networks"])
        self.networks.eval()
        self.control_return_normalizer = ReturnNormalizer(
            demean=False,
            dtype=torch.float64,
        ).to(device)
        self.control_return_normalizer.load_state_dict(
            state["control_return_normalizer"]
        )
        self.normalize_returns = bool(
            run_config.get("training", {}).get("normalize_returns", True)
        )
        self.trainable_parameters = sum(
            parameter.numel() for parameter in self.networks.parameters()
        )
        self._installed_designs: list[BodyGenDesign] = []

    def _sample(
        self,
        count: int,
        seed: int,
        *,
        deterministic: bool,
    ) -> EvaluationPairs:
        if count < 1:
            raise ValueError("BodyGen sample count must be positive")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        with torch.no_grad():
            designs, _ = self.networks.sample_designs(
                count,
                generator,
                deterministic=deterministic,
            )
        arrays = [design.to_arrays() for design in designs]
        return EvaluationPairs(
            counts=np.stack([item[0] for item in arrays]),
            eff_sub=np.stack([item[1] for item in arrays]),
            cap_sub=np.stack([item[2] for item in arrays]),
            controller_ids=np.full(count, "shared"),
            # Repeated bodies deliberately keep their sampled probability mass.
            weights=np.full(count, 1.0 / count),
        )

    def sample_pairs(self, count: int, seed: int) -> EvaluationPairs:
        """Draw fresh stochastic native designs at the literal given seed."""

        return self._sample(count, seed, deterministic=False)

    def sample_designs(self, count: int, seed: int) -> EvaluationPairs:
        return self._sample(count, seed, deterministic=False)

    def sample_pairs_deterministic(
        self,
        count: int,
        seed: int,
    ) -> EvaluationPairs:
        """Return upstream-style greedy/mean designs for diagnostics only."""

        return self._sample(count, seed, deterministic=True)

    def create_environment(self, pair_count: int, rollout_seed: int):
        from envs.ant_envs.ant_codesign import AntCodesignEnv

        saved = self.run_config["environment"]
        arguments: dict[str, Any] = {
            "sample_morphs": True,
            "rendering": False,
            "raise_exception": False,
            "seed": int(rollout_seed),
            "with_window": False,
            "base_legs": tuple(saved["base_legs"]),
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

    def install_pairs(
        self,
        environment: Any,
        pairs: EvaluationPairs,
    ) -> None:
        if np.any(pairs.controller_ids != "shared"):
            raise ValueError("BodyGen evaluation pairs must use its shared controller")
        designs = [
            BodyGenDesign.from_arrays(counts, effectors, caps)
            for counts, effectors, caps in zip(
                pairs.counts,
                pairs.eff_sub,
                pairs.cap_sub,
                strict=True,
            )
        ]
        environment.set_next(pairs.counts, pairs.eff_sub, pairs.cap_sub)
        environment.resample()
        self._installed_designs = designs

    def begin_rollout(self, pairs: EvaluationPairs) -> None:
        if len(self._installed_designs) != pairs.size:
            raise RuntimeError("BodyGen pairs must be installed before rollout")

    @torch.no_grad()
    def deterministic_action(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self._installed_designs) != observation.shape[0]:
            raise RuntimeError("BodyGen rollout has no matching installed designs")
        output = self.networks.control(
            observation.to(device=self.device, dtype=torch.float64),
            self._installed_designs,
            deterministic=True,
        )
        value = output.value
        if self.normalize_returns:
            value = self.control_return_normalizer.unscale(value)
        return output.mean.clamp(-1.0, 1.0), value


def load_bodygen(
    config: dict[str, Any],
    device: torch.device,
) -> BodyGenMethod:
    """Load one complete BodyGen checkpoint for shared evaluation."""

    run_dir = _run_directory(config)
    run_config_path = (
        Path(config["run_config"]).expanduser().resolve()
        if config.get("run_config")
        else run_dir / "config.yaml"
    )
    if not run_config_path.is_file():
        raise FileNotFoundError(
            f"BodyGen run config does not exist: {run_config_path}"
        )
    run_config = yaml.safe_load(run_config_path.read_text())
    checkpoint_value = config.get("checkpoint", "final")
    if checkpoint_value == "final":
        checkpoint_path = run_dir / "checkpoints" / "final.pth"
        checkpoint_label = "final"
    else:
        checkpoint_path = Path(checkpoint_value).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = run_dir / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        match = re.search(r"update_(\d+)", checkpoint_path.stem)
        checkpoint_label = (
            f"update_{int(match.group(1))}" if match else checkpoint_path.stem
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"BodyGen checkpoint does not exist: {checkpoint_path}"
        )
    state = torch.load(
        checkpoint_path,
        # Avoid materialising the checkpoint's unused optimizer/RNG tensors on
        # the evaluator GPU. BodyGenMethod copies only its networks and return
        # normalizer to ``device``.
        map_location="cpu",
        weights_only=False,
    )
    return BodyGenMethod(
        state=state,
        device=device,
        run_dir=run_dir,
        run_config_path=run_config_path,
        run_config=run_config,
        checkpoint_path=checkpoint_path,
        checkpoint_label=checkpoint_label,
    )
