"""Adapter from a frozen NGE population to the shared benchmark evaluator."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from benchmarks.data import EvaluationPairs

from .graph import NGEGraph
from .nervenet import ControllerState, N_ACTIONS, normalize_observation
from .population import Population

_ROOT = Path(__file__).resolve().parents[2]


def _run_directory(config: dict[str, Any], selected: str | Path | None = None) -> Path:
    value = selected if selected is not None else config.get("run_dir")
    if value is None:
        raise ValueError("an NGE run directory is required")
    path = Path(value).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        root = Path(config.get("run_root", "runs/benchmarks/nge")).expanduser()
        if not root.is_absolute():
            root = _ROOT / root
        candidates = [_ROOT / path, root / path]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"NGE run directory does not exist: {path}")


def checkpoints_for_nge_run(
    config: dict[str, Any],
    selected_run: str | Path,
    requested: str,
) -> list[tuple[str, Path]]:
    """Resolve ``final`` or numeric NGE generation checkpoints."""
    run_dir = _run_directory(config, selected_run)
    checkpoint_dir = run_dir / "checkpoints"
    if requested == "final":
        final = checkpoint_dir / "final.pth"
        if not final.is_file():
            raise FileNotFoundError(f"no final NGE checkpoint: {final}")
        return [("final", final.resolve())]
    try:
        generations = {int(value.strip()) for value in requested.split(",")}
    except ValueError as error:
        raise ValueError(
            "NGE --epochs accepts 'final' or comma-separated generation numbers"
        ) from error
    if not generations:
        raise ValueError("at least one NGE generation must be selected")
    result = []
    for generation in sorted(generations):
        path = checkpoint_dir / f"generation_{generation:04d}.pth"
        if not path.is_file():
            raise FileNotFoundError(
                f"no NGE generation {generation} checkpoint under {checkpoint_dir}"
            )
        result.append((f"generation_{generation}", path.resolve()))
    return result


class NGEMethod:
    """Uniformly samples morphology-controller pairs from final survivors."""

    name = "nge"

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
        self.parallel_envs = int(state["parallel_envs"])
        self.population = Population.from_state_dict(state["population"])
        hidden_size = int(run_config["network"]["hidden_size"])
        self.controllers: dict[int, ControllerState] = {}
        for species in self.population.species:
            controller = ControllerState.create(device, hidden_size=hidden_size)
            learner_state = state["learners"][species.species_id]
            controller.load_state_dict(learner_state["controller"])
            controller.policy.eval()
            controller.value.eval()
            self.controllers[species.species_id] = controller

        controller_parameters = sum(
            parameter.numel()
            for controller in self.controllers.values()
            for network in (controller.policy, controller.value)
            for parameter in network.parameters()
        )
        gm_uc_parameters = sum(
            tensor.numel()
            for tensor in state["gm_uc"]["model"].values()
            if torch.is_tensor(tensor)
        )
        self.trainable_parameters = controller_parameters + gm_uc_parameters
        self._species_by_id = self.population.by_id()
        self._routes: dict[int, torch.Tensor] = {}
        self._hidden: dict[int, torch.Tensor] = {}

    def _sample(self, count: int, seed: int) -> EvaluationPairs:
        rng = np.random.default_rng(int(seed))
        selected = self.population.sample(count, rng)
        arrays = [species.graph.to_arrays() for species in selected]
        return EvaluationPairs(
            counts=np.stack([item[0] for item in arrays]),
            eff_sub=np.stack([item[1] for item in arrays]),
            cap_sub=np.stack([item[2] for item in arrays]),
            controller_ids=np.asarray(
                [f"species_{species.species_id}" for species in selected]
            ),
            weights=np.full(count, 1.0 / count),
        )

    def sample_pairs(self, count: int, seed: int) -> EvaluationPairs:
        return self._sample(count, seed)

    def sample_designs(self, count: int, seed: int) -> EvaluationPairs:
        return self._sample(count, seed)

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

    def install_pairs(self, environment: Any, pairs: EvaluationPairs) -> None:
        environment.set_next(pairs.counts, pairs.eff_sub, pairs.cap_sub)
        environment.resample()
        ids = []
        for controller_id in pairs.controller_ids:
            match = re.fullmatch(r"species_(\d+)", str(controller_id))
            if match is None:
                raise ValueError(f"invalid NGE controller ID: {controller_id}")
            species_id = int(match.group(1))
            if species_id not in self.controllers:
                raise ValueError(f"NGE checkpoint has no species {species_id}")
            ids.append(species_id)
        self._installed_ids = np.asarray(ids, dtype=np.int64)

    def begin_rollout(self, pairs: EvaluationPairs) -> None:
        if not hasattr(self, "_installed_ids"):
            raise RuntimeError("NGE pairs must be installed before rollout")
        self._routes = {}
        self._hidden = {}
        for species_id in np.unique(self._installed_ids):
            rows = np.flatnonzero(self._installed_ids == species_id)
            route = torch.as_tensor(rows, dtype=torch.long, device=self.device)
            graph = self._species_by_id[int(species_id)].graph
            self._routes[int(species_id)] = route
            self._hidden[int(species_id)] = self.controllers[
                int(species_id)
            ].policy.initial_hidden(len(rows), graph, self.device)

    @torch.no_grad()
    def deterministic_action(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._routes:
            raise RuntimeError("begin_rollout must be called before NGE inference")
        action = torch.zeros(
            observation.shape[0], N_ACTIONS, device=observation.device
        )
        value = torch.zeros(observation.shape[0], device=observation.device)
        for species_id, rows in self._routes.items():
            controller = self.controllers[species_id]
            graph = self._species_by_id[species_id].graph
            normalized = normalize_observation(
                observation[rows], controller.normalizer
            )
            mean, _, next_hidden = controller.policy.forward_step(
                normalized,
                graph,
                self._hidden[species_id],
            )
            action[rows] = mean.clamp(-1.0, 1.0)
            value[rows] = controller.value(normalized)
            self._hidden[species_id] = next_hidden
        return action, value

    def reset_controllers(self, done: torch.Tensor) -> None:
        """Clear recurrent state for rows whose VSim episode just reset."""
        for species_id, rows in self._routes.items():
            keep = (~done[rows]).to(self._hidden[species_id].dtype)
            self._hidden[species_id] *= keep.view(-1, 1, 1)


def load_nge(config: dict[str, Any], device: torch.device) -> NGEMethod:
    run_dir = _run_directory(config)
    run_config_path = (
        Path(config["run_config"]).expanduser().resolve()
        if config.get("run_config")
        else run_dir / "config.yaml"
    )
    if not run_config_path.is_file():
        raise FileNotFoundError(f"NGE run config does not exist: {run_config_path}")
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
        match = re.search(r"generation_(\d+)", checkpoint_path.stem)
        checkpoint_label = (
            f"generation_{int(match.group(1))}" if match else checkpoint_path.stem
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"NGE checkpoint does not exist: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state.get("method") != "nge" or state.get("format_version") != 1:
        raise ValueError("checkpoint is not a supported NGE population checkpoint")
    return NGEMethod(
        state=state,
        device=device,
        run_dir=run_dir,
        run_config_path=run_config_path,
        run_config=run_config,
        checkpoint_path=checkpoint_path,
        checkpoint_label=checkpoint_label,
    )
