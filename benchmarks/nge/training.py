"""Measured VSim training loop for the faithful NGE baseline.

One population member owns one NerveNet++ policy, value network, observation
normaliser, and pair of PPO optimizers.  VSim lays the population out as
contiguous morphology groups, so all species collect physics concurrently while
their policy updates remain independent, as in the upstream multi-process code.
"""
from __future__ import annotations

import copy
import os
import random
import re
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rl_games.algos_torch.torch_ext import AverageMeter
from torch.nn import functional as F

from .gm_uc import GraphMutationWithUncertainty
from .graph import MUTATION_NAMES, NGEGraph
from .nervenet import (
    ControllerState,
    N_ACTIONS,
    OBSERVATION_SIZE,
    PHYSICAL_OBSERVATION_SIZE,
    action_mask,
    gaussian_entropy,
    gaussian_log_prob,
    normalize_observation,
)
from .population import Population


FORMAT_VERSION = 1
RL_GAMES_REWARD_WINDOW = 100
DEFAULT_TRAINING_EVALUATION = {
    "enabled": False,
    "every_generations": 1,
    "evaluate_final": False,
    "pairs": 16,
    "episodes_per_pair": 2,
    "seeds": {
        "morphology": 10001,
        "rollout": 20001,
    },
}


def nge_generation_environment_steps(config: dict[str, Any]) -> int:
    """Physics transitions consumed by one complete NGE generation."""
    population_size = int(config["population"]["size"])
    training_envs = int(config["budget"]["parallel_envs"])
    training = config["training"]
    selection = config["selection_evaluation"]
    training_steps = (
        training_envs
        * int(training["rollout_steps"])
        * int(training["updates_per_generation"])
    )
    selection_steps = (
        population_size
        * int(selection["environments_per_species"])
        * int(selection["rollout_steps"])
    )
    return training_steps + selection_steps


def load_nge_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("NGE config must be a YAML mapping")
    validate_nge_config(config)
    return config


def resume_configs_match(
    saved: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    """Allow monitoring changes without changing the resumed algorithm."""
    saved_algorithm = copy.deepcopy(saved)
    current_algorithm = copy.deepcopy(current)
    saved_algorithm.pop("training_evaluation", None)
    current_algorithm.pop("training_evaluation", None)
    return saved_algorithm == current_algorithm


def validate_nge_config(config: dict[str, Any]) -> None:
    """Reject settings that break fidelity or exact step accounting."""
    if config.get("method") != "nge":
        raise ValueError("NGE training config must set method: nge")
    missing_protocol_blocks = [
        key
        for key in ("selection_evaluation", "fidelity_constraints")
        if key not in config
    ]
    if missing_protocol_blocks:
        raise ValueError(
            "NGE config predates the counted complete-episode selection "
            "protocol and cannot be resumed as the same benchmark run; missing "
            + ", ".join(missing_protocol_blocks)
        )
    budget = config["budget"]
    environment_steps = int(budget["environment_steps"])
    parallel_envs = int(budget["parallel_envs"])
    population_size = int(config["population"]["size"])
    if environment_steps <= 0 or parallel_envs <= 0 or population_size <= 0:
        raise ValueError("NGE environment budget values must be positive")
    if parallel_envs % population_size:
        raise ValueError("parallel_envs must divide evenly across the population")

    training = config["training"]
    rollout_steps = int(training["rollout_steps"])
    updates_per_generation = int(training["updates_per_generation"])
    sequence_length = int(training["sequence_length"])
    if min(rollout_steps, updates_per_generation, sequence_length) <= 0:
        raise ValueError("NGE rollout, update, and sequence lengths must be positive")
    if rollout_steps % sequence_length:
        raise ValueError("rollout_steps must be divisible by sequence_length")

    environments_per_species = parallel_envs // population_size
    minimum_batch = int(
        config["fidelity_constraints"][
            "minimum_transitions_per_species_batch"
        ]
    )
    if minimum_batch <= 0:
        raise ValueError(
            "minimum_transitions_per_species_batch must be positive"
        )
    training_batch = environments_per_species * rollout_steps
    if training_batch < minimum_batch:
        raise ValueError(
            "each NGE PPO batch must meet the configured per-species sample "
            "minimum: "
            f"({parallel_envs} / {population_size}) * {rollout_steps} = "
            f"{training_batch} < {minimum_batch}"
        )

    selection = config["selection_evaluation"]
    selection_environments_per_species = int(
        selection["environments_per_species"]
    )
    selection_rollout_steps = int(selection["rollout_steps"])
    if selection_environments_per_species <= 0 or selection_rollout_steps <= 0:
        raise ValueError("selection-evaluation sizes must be positive")
    selection_parallel_envs = (
        population_size * selection_environments_per_species
    )
    if selection_parallel_envs > parallel_envs:
        raise ValueError(
            "selection evaluation cannot exceed budget.parallel_envs: "
            f"{population_size} * {selection_environments_per_species} = "
            f"{selection_parallel_envs} > {parallel_envs}"
        )
    max_episode_length = int(config["environment"]["max_episode_length"])
    if max_episode_length <= 0:
        raise ValueError("max_episode_length must be positive")
    # VSim reports a horizon truncation on the following returned step, so a
    # rollout equal to the nominal horizon is one tick too short to observe it.
    if selection_rollout_steps <= max_episode_length:
        raise ValueError(
            "selection_evaluation.rollout_steps must be greater than "
            "environment.max_episode_length so every initial episode is "
            "observed to finish"
        )
    selection_batch = (
        selection_environments_per_species * selection_rollout_steps
    )
    if selection_batch < minimum_batch:
        raise ValueError(
            "the complete-episode selection batch must meet the configured "
            "per-species sample minimum: "
            f"{selection_environments_per_species} * "
            f"{selection_rollout_steps} = {selection_batch} < {minimum_batch}"
        )
    if selection["action_mode"] not in {"stochastic", "deterministic"}:
        raise ValueError(
            "selection_evaluation.action_mode must be stochastic or deterministic"
        )

    generation_environment_steps = nge_generation_environment_steps(config)
    if environment_steps % generation_environment_steps:
        raise ValueError(
            "the budget must end at a generation boundary: "
            f"{environment_steps} is not divisible by the complete generation "
            f"cost {generation_environment_steps} (PPO rollouts plus selection "
            "evaluation)"
        )

    mutation = config["population"]["mutation_probabilities"]
    if set(mutation) != set(MUTATION_NAMES):
        raise ValueError(f"mutation probabilities must name {MUTATION_NAMES}")
    if not np.isclose(sum(float(value) for value in mutation.values()), 1.0):
        raise ValueError("mutation probabilities must sum to one")
    if int(config["population"]["candidate_pool_size"]) < population_size:
        raise ValueError("candidate_pool_size cannot be smaller than population")
    if not 0.0 < float(config["population"]["elimination_rate"]) < 1.0:
        raise ValueError("elimination_rate must lie in (0, 1)")

    logging = config["logging"]
    if logging.get("tensorboard") is not True:
        raise ValueError("TensorBoard must remain enabled for benchmark training")
    wandb = logging["wandb"]
    if wandb["enabled"] and not wandb.get("project"):
        raise ValueError("a W&B project is required when W&B is enabled")
    if wandb["mode"] not in {"online", "offline"}:
        raise ValueError("W&B mode must be online or offline")

    training_evaluation = config.get(
        "training_evaluation",
        DEFAULT_TRAINING_EVALUATION,
    )
    if not isinstance(training_evaluation, dict):
        raise ValueError("training_evaluation must be a YAML mapping")
    if not isinstance(training_evaluation.get("enabled"), bool):
        raise ValueError("training_evaluation.enabled must be true or false")
    for key in ("every_generations", "pairs", "episodes_per_pair"):
        value = training_evaluation.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training_evaluation.{key} must be a positive integer")
    if not isinstance(training_evaluation.get("evaluate_final"), bool):
        raise ValueError(
            "training_evaluation.evaluate_final must be true or false"
        )
    evaluation_seeds = training_evaluation.get("seeds", {})
    for key in ("morphology", "rollout"):
        value = evaluation_seeds.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"training_evaluation.seeds.{key} must be an integer"
            )


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    episode_starts: torch.Tensor
    hidden: torch.Tensor
    rollout_return_estimate: float
    raw_step_reward_mean: float
    completed_returns: tuple[float, ...]


@dataclass(frozen=True)
class SelectionEvaluation:
    """Complete episode data used to rank the current species."""

    fitness: dict[int, float]
    completed_returns: tuple[float, ...]
    completed_lengths: tuple[int, ...]
    episodes_per_species: dict[int, int]
    environment_steps: int


def summarize_selection_episodes(
    returns_by_species: dict[int, list[float]],
    lengths_by_species: dict[int, list[int]],
    *,
    environment_steps: int,
) -> SelectionEvaluation:
    """Turn completed selection episodes into species fitness values."""
    if set(returns_by_species) != set(lengths_by_species):
        raise ValueError("selection return and length species do not match")

    fitness: dict[int, float] = {}
    episodes_per_species: dict[int, int] = {}
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    for species_id, returns in returns_by_species.items():
        lengths = lengths_by_species[species_id]
        if not returns:
            raise RuntimeError(
                f"selection evaluation completed no episode for species {species_id}"
            )
        if len(returns) != len(lengths):
            raise ValueError(
                f"selection return/length count differs for species {species_id}"
            )
        fitness[species_id] = float(np.mean(returns))
        episodes_per_species[species_id] = len(returns)
        completed_returns.extend(float(value) for value in returns)
        completed_lengths.extend(int(value) for value in lengths)

    return SelectionEvaluation(
        fitness=fitness,
        completed_returns=tuple(completed_returns),
        completed_lengths=tuple(completed_lengths),
        episodes_per_species=episodes_per_species,
        environment_steps=int(environment_steps),
    )


def summarize_training_update(
    rollouts: dict[int, Rollout],
    ppo_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Canonical scalar summary written after every NGE PPO update.

    The continuously available short-rollout return estimate has its own tag,
    so a partial trajectory is never mislabeled as a completed episode or as
    the complete-return species fitness.
    The three rl_games-compatible completed-return views are written separately
    because each one needs a different TensorBoard x-axis.
    """
    return_estimate = np.asarray(
        [rollout.rollout_return_estimate for rollout in rollouts.values()],
        dtype=np.float64,
    )
    completed = [
        value
        for rollout in rollouts.values()
        for value in rollout.completed_returns
    ]
    metrics = {
        "rewards/rollout_return_estimate": float(return_estimate.mean()),
        "rewards/raw_step_mean": float(
            np.mean(
                [
                    rollout.raw_step_reward_mean
                    for rollout in rollouts.values()
                ]
            )
        ),
        "rewards/completed_episodes": float(len(completed)),
        "nge/training/return_estimate_mean": float(return_estimate.mean()),
        "nge/training/return_estimate_std": float(return_estimate.std()),
        "nge/training/return_estimate_min": float(return_estimate.min()),
        "nge/training/return_estimate_max": float(return_estimate.max()),
        "nge/ppo/policy_loss_update": float(
            np.mean([item["policy"] for item in ppo_metrics])
        ),
        "nge/ppo/value_loss_update": float(
            np.mean([item["value"] for item in ppo_metrics])
        ),
        "nge/ppo/entropy_update": float(
            np.mean([item["entropy"] for item in ppo_metrics])
        ),
        "nge/ppo/kl_update": float(
            np.mean([item["kl"] for item in ppo_metrics])
        ),
        "nge/ppo/learning_rate_mean": float(
            np.mean([item["learning_rate"] for item in ppo_metrics])
        ),
    }
    return metrics


class SpeciesLearner:
    """Controller state plus the optimizers that train one species."""

    def __init__(
        self,
        controller: ControllerState,
        *,
        policy_learning_rate: float,
        value_learning_rate: float,
    ) -> None:
        self.controller = controller
        self.initial_policy_lr = float(policy_learning_rate)
        self.value_lr = float(value_learning_rate)
        self.current_policy_lr = self.initial_policy_lr
        self.reset_optimizers()

    @classmethod
    def create(
        cls,
        device: torch.device,
        training: dict[str, Any],
        *,
        hidden_size: int,
    ) -> "SpeciesLearner":
        return cls(
            ControllerState.create(device, hidden_size=hidden_size),
            policy_learning_rate=float(training["policy_learning_rate"]),
            value_learning_rate=float(training["value_learning_rate"]),
        )

    def reset_optimizers(self) -> None:
        """Rebuild optimizer state without discarding the adaptive policy LR.

        Upstream creates fresh optimizers at a generation boundary, but the
        species record carries its current learning rate into the new worker.
        """
        self.policy_optimizer = torch.optim.Adam(
            self.controller.policy.parameters(),
            lr=self.current_policy_lr,
        )
        self.value_optimizer = torch.optim.Adam(
            self.controller.value.parameters(),
            lr=self.value_lr,
            eps=1.0e-5,
        )

    def inherited_copy(self) -> "SpeciesLearner":
        child = SpeciesLearner(
            self.controller.inherited_copy(),
            policy_learning_rate=self.initial_policy_lr,
            value_learning_rate=self.value_lr,
        )
        child.current_policy_lr = self.current_policy_lr
        child.reset_optimizers()
        return child

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller": self.controller.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "initial_policy_lr": self.initial_policy_lr,
            "current_policy_lr": self.current_policy_lr,
            "value_lr": self.value_lr,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        device: torch.device,
        *,
        hidden_size: int,
    ) -> "SpeciesLearner":
        controller = ControllerState.create(device, hidden_size=hidden_size)
        controller.load_state_dict(state["controller"])
        learner = cls(
            controller,
            policy_learning_rate=float(state["initial_policy_lr"]),
            value_learning_rate=float(state["value_lr"]),
        )
        learner.policy_optimizer.load_state_dict(state["policy_optimizer"])
        learner.value_optimizer.load_state_dict(state["value_optimizer"])
        learner.current_policy_lr = float(state["current_policy_lr"])
        for group in learner.policy_optimizer.param_groups:
            group["lr"] = learner.current_policy_lr
        return learner


class TrainingLogger:
    """TensorBoard first, with an explicitly enabled W&B mirror."""

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        run_identity: str,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(run_dir / "tensorboard"))
        self.wandb_run = None
        wandb_config = config["logging"]["wandb"]
        if wandb_config["enabled"]:
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError(
                    "W&B is enabled in the NGE YAML but wandb is unavailable"
                ) from error
            self.wandb_run = wandb.init(
                project=wandb_config["project"],
                entity=wandb_config.get("entity"),
                group=wandb_config.get("group"),
                name=run_dir.name,
                id=run_identity,
                resume="allow",
                mode=wandb_config["mode"],
                config=config,
                reinit=True,
            )
            self.wandb_run.define_metric(
                "benchmark/training/environment_steps"
            )
            self.wandb_run.define_metric(
                "*",
                step_metric="benchmark/training/environment_steps",
            )
            self.wandb_run.define_metric(
                "rewards/step",
                step_metric="benchmark/training/environment_steps",
            )
            self.wandb_run.define_metric(
                "rewards/iter",
                step_metric="benchmark/training/iteration",
            )
            self.wandb_run.define_metric(
                "rewards/time",
                step_metric="benchmark/training/wall_seconds",
            )
            self.wandb_run.define_metric(
                "rewards/step_eval",
                step_metric="benchmark/training/environment_steps",
            )

    def log(self, metrics: dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            if np.isfinite(value):
                self.writer.add_scalar(name, value, step)
        self.writer.flush()
        if self.wandb_run is not None:
            payload = {
                name: value
                for name, value in metrics.items()
                if np.isfinite(value)
            }
            payload.setdefault(
                "benchmark/training/environment_steps",
                float(step),
            )
            self.wandb_run.log(payload)

    def log_training_update(
        self,
        metrics: dict[str, float],
        *,
        environment_steps: int,
        iteration: int,
        wall_seconds: float,
        episode_reward: float | None,
    ) -> None:
        """Write one update, including rl_games' three reward-axis views."""
        for name, value in metrics.items():
            if np.isfinite(value):
                self.writer.add_scalar(name, value, environment_steps)
        if episode_reward is not None:
            self.writer.add_scalar(
                "rewards/step",
                episode_reward,
                environment_steps,
            )
            self.writer.add_scalar(
                "rewards/iter",
                episode_reward,
                iteration,
            )
            self.writer.add_scalar(
                "rewards/time",
                episode_reward,
                wall_seconds,
            )
        self.writer.flush()

        if self.wandb_run is not None:
            payload = {
                name: value
                for name, value in metrics.items()
                if np.isfinite(value)
            }
            payload.update(
                {
                    "benchmark/training/environment_steps": float(
                        environment_steps
                    ),
                    "benchmark/training/iteration": float(iteration),
                    "benchmark/training/wall_seconds": wall_seconds,
                }
            )
            if episode_reward is not None:
                payload.update(
                    {
                        "rewards/step": episode_reward,
                        "rewards/iter": episode_reward,
                        "rewards/time": episode_reward,
                    }
                )
            self.wandb_run.log(payload)

    def log_training_evaluation(
        self,
        *,
        expected_return: float,
        environment_steps: int,
        completed_generation: int,
        evaluated_episodes: int,
    ) -> None:
        """Log a complete-episode evaluation on the training-step x-axis."""
        metrics = {
            "rewards/step_eval": float(expected_return),
            "benchmark/training_eval/completed_generation": float(
                completed_generation
            ),
            "benchmark/training_eval/completed_episodes": float(
                evaluated_episodes
            ),
        }
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, environment_steps)
        self.writer.flush()
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    **metrics,
                    "benchmark/training/environment_steps": float(
                        environment_steps
                    ),
                }
            )

    def close(self) -> None:
        self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


class NGETrainer:
    """Owns all state needed to train, checkpoint, and resume NGE."""

    def __init__(
        self,
        config: dict[str, Any],
        run_dir: Path,
        *,
        resume: Path | None = None,
    ) -> None:
        validate_nge_config(config)
        self.config = copy.deepcopy(config)
        self.run_dir = run_dir.resolve()
        self.device = torch.device(config["runtime"]["device"])
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("NGE VSim training requires an available CUDA device")

        self.seed = int(config["seed"])
        self.parallel_envs = int(config["budget"]["parallel_envs"])
        self.target_environment_steps = int(config["budget"]["environment_steps"])
        self.environment_steps = 0
        self.controller_environment_steps = 0
        self.selection_environment_steps = 0
        self.training = config["training"]
        self.selection_config = config["selection_evaluation"]
        self.training_evaluation = config.get(
            "training_evaluation",
            DEFAULT_TRAINING_EVALUATION,
        )
        self.population_config = config["population"]
        self.hidden_size = int(config["network"]["hidden_size"])
        self.rng = np.random.default_rng(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        gm = config["gm_uc"]
        self.gm_uc = GraphMutationWithUncertainty(
            self.device,
            hidden_size=int(gm["hidden_size"]),
            dropout=float(gm["dropout"]),
            learning_rate=float(gm["learning_rate"]),
            batch_size=int(gm["batch_size"]),
            gradient_steps=int(gm["gradient_steps"]),
            temperature=float(gm["temperature"]),
        )
        if resume is None:
            self.run_identity = re.sub(
                r"[^A-Za-z0-9_-]+",
                "-",
                f"nge-{self.run_dir.name}-s{self.seed}",
            )
            graph = NGEGraph.canonical(config["environment"]["base_legs"])
            self.population = Population.initial(
                int(self.population_config["size"]), graph
            )
            self.learners: dict[int, SpeciesLearner] = {}
            # The pinned implementation seeds each population worker
            # independently as base_seed + worker/species ID.
            for species in self.population.species:
                torch.manual_seed(self.seed + species.species_id)
                self.learners[species.species_id] = SpeciesLearner.create(
                    self.device,
                    self.training,
                    hidden_size=self.hidden_size,
                )
        else:
            self._load_checkpoint(resume)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        config_path = self.run_dir / "config.yaml"
        if resume is None:
            if config_path.exists():
                raise FileExistsError(f"NGE run already exists: {self.run_dir}")
            config_path.write_text(yaml.safe_dump(self.config, sort_keys=False))
        elif not config_path.is_file():
            raise FileNotFoundError(f"resume run has no config.yaml: {self.run_dir}")
        self.logger = TrainingLogger(
            self.run_dir,
            self.config,
            self.run_identity,
        )
        # This is the same AverageMeter and default 100-episode window used by
        # rl_games for rewards/{step,iter,time}.
        self.completed_reward_meter = AverageMeter(
            1,
            RL_GAMES_REWARD_WINDOW,
        ).to(self.device)
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)

    def _run_training_evaluation(
        self,
        *,
        completed_generation: int,
        final: bool,
    ) -> None:
        """Evaluate this exact population without changing training state."""
        schedule = self.training_evaluation
        if not schedule["enabled"]:
            return
        interval_due = (
            completed_generation % int(schedule["every_generations"]) == 0
        )
        if not interval_due and not (final and schedule["evaluate_final"]):
            return

        # The shared evaluator creates environments and may touch process-wide
        # random generators. Restore all training RNGs afterwards so enabling
        # monitoring cannot change the optimisation trajectory.
        rng_state = self._rng_state()
        try:
            from benchmarks.evaluate import evaluate_return

            from .method import NGEMethod

            method = NGEMethod(
                state=self.checkpoint_state(),
                device=self.device,
                run_dir=self.run_dir,
                run_config_path=self.run_dir / "config.yaml",
                run_config=self.config,
                checkpoint_path=self.run_dir / "checkpoints" / (
                    "final.pth"
                    if final
                    else f"generation_{self.population.generation:04d}.pth"
                ),
                checkpoint_label=(
                    "final"
                    if final
                    else f"generation_{self.population.generation}"
                ),
            )
            expected_return, _, episodes = evaluate_return(
                method,
                pairs=int(schedule["pairs"]),
                episodes_per_pair=int(schedule["episodes_per_pair"]),
                morphology_seed=int(schedule["seeds"]["morphology"]),
                rollout_seed=int(schedule["seeds"]["rollout"]),
            )
        finally:
            self._restore_rng_state(rng_state)

        evaluated_episodes = int(np.prod(episodes.returns.shape))
        self.logger.log_training_evaluation(
            expected_return=expected_return,
            environment_steps=self.environment_steps,
            completed_generation=completed_generation,
            evaluated_episodes=evaluated_episodes,
        )
        print(
            "[nge/eval] "
            f"generation={completed_generation} "
            f"steps={self.environment_steps:,} "
            f"rewards/step_eval={expected_return:.3f} "
            f"episodes={evaluated_episodes}",
            flush=True,
        )

    @property
    def environments_per_species(self) -> int:
        return self.parallel_envs // self.population.size

    def _environment_arguments(self) -> dict[str, Any]:
        saved = self.config["environment"]
        arguments = {
            "rendering": False,
            "raise_exception": False,
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
        return arguments

    def _create_environment(self, num_envs: int | None = None):
        from envs.ant_envs.ant_nge import AntNGEEnv

        num_envs = self.parallel_envs if num_envs is None else int(num_envs)
        environment_seed = int(self.rng.integers(0, 2**31 - 1))
        return AntNGEEnv(
            num_envs,
            self.device,
            graphs=[species.graph for species in self.population.species],
            seed=environment_seed,
            **self._environment_arguments(),
        )

    def _collect_rollout(
        self,
        environment: Any,
        observation: torch.Tensor,
        hidden: dict[int, torch.Tensor],
        episode_start: torch.Tensor,
        episode_return: torch.Tensor,
    ) -> tuple[
        dict[int, Rollout],
        torch.Tensor,
        dict[int, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        steps = int(self.training["rollout_steps"])
        epm = self.environments_per_species
        species_data: dict[int, dict[str, list[torch.Tensor]]] = {
            species.species_id: {
                key: []
                for key in (
                    "raw_observations",
                    "observations",
                    "actions",
                    "log_prob",
                    "values",
                    "rewards",
                    "dones",
                    "episode_starts",
                    "hidden",
                )
            }
            for species in self.population.species
        }
        completed_returns: dict[int, list[float]] = {
            species.species_id: [] for species in self.population.species
        }

        for _ in range(steps):
            environment_action = torch.zeros(
                self.parallel_envs,
                N_ACTIONS,
                device=self.device,
            )
            for group, species in enumerate(self.population.species):
                species_id = species.species_id
                rows = slice(group * epm, (group + 1) * epm)
                learner = self.learners[species_id]
                learner.controller.policy.eval()
                learner.controller.value.eval()
                normalized = normalize_observation(
                    observation[rows],
                    learner.controller.normalizer,
                )
                start = episode_start[rows]
                current_hidden = hidden[species_id] * (
                    ~start
                ).to(observation.dtype).view(-1, 1, 1)
                with torch.no_grad():
                    mean, log_std, next_hidden = (
                        learner.controller.policy.forward_step(
                            normalized,
                            species.graph,
                            current_hidden,
                        )
                    )
                    raw_action = mean + torch.exp(log_std) * torch.randn_like(mean)
                    mask = action_mask(species.graph, device=self.device)
                    log_prob = gaussian_log_prob(
                        raw_action, mean, log_std, mask
                    )
                    value = learner.controller.value(normalized)
                environment_action[rows] = raw_action.clamp(-1.0, 1.0)
                data = species_data[species_id]
                data["raw_observations"].append(observation[rows])
                data["observations"].append(normalized)
                data["actions"].append(raw_action)
                data["log_prob"].append(log_prob)
                data["values"].append(value)
                data["episode_starts"].append(start)
                data["hidden"].append(current_hidden)
                hidden[species_id] = next_hidden

            next_observation, reward, terminated, truncated, _ = environment.step(
                environment_action
            )
            reward = reward.squeeze(-1) if reward.ndim > 1 else reward
            done = terminated | truncated
            episode_return += reward
            for group, species in enumerate(self.population.species):
                species_id = species.species_id
                rows = slice(group * epm, (group + 1) * epm)
                data = species_data[species_id]
                data["rewards"].append(reward[rows])
                data["dones"].append(done[rows])
                if bool(done[rows].any()):
                    completed_returns[species_id].extend(
                        float(value)
                        for value in episode_return[rows][done[rows]].detach().cpu()
                    )
                hidden[species_id] = hidden[species_id] * (
                    ~done[rows]
                ).to(observation.dtype).view(-1, 1, 1)
            episode_return = torch.where(done, 0.0, episode_return)
            episode_start = done
            observation = next_observation
            self.environment_steps += self.parallel_envs
            self.controller_environment_steps += self.parallel_envs

        rollouts: dict[int, Rollout] = {}
        gamma = float(self.training["gamma"])
        gae_lambda = float(self.training["gae_lambda"])
        max_episode_length = int(self.config["environment"]["max_episode_length"])
        for group, species in enumerate(self.population.species):
            species_id = species.species_id
            rows = slice(group * epm, (group + 1) * epm)
            learner = self.learners[species_id]
            data = species_data[species_id]
            observations = torch.stack(data["observations"])
            raw_observations = torch.stack(data["raw_observations"])
            actions = torch.stack(data["actions"])
            old_log_prob = torch.stack(data["log_prob"])
            values = torch.stack(data["values"])
            rewards = torch.stack(data["rewards"])
            dones = torch.stack(data["dones"])
            starts = torch.stack(data["episode_starts"])
            stored_hidden = torch.stack(data["hidden"])

            with torch.no_grad():
                normalized_next = normalize_observation(
                    observation[rows],
                    learner.controller.normalizer,
                )
                next_value = learner.controller.value(normalized_next)
            advantages = torch.zeros_like(rewards)
            last_advantage = torch.zeros(epm, device=self.device)
            for time_index in range(steps - 1, -1, -1):
                following_value = (
                    next_value if time_index == steps - 1 else values[time_index + 1]
                )
                non_terminal = (~dones[time_index]).to(rewards.dtype)
                delta = (
                    rewards[time_index]
                    + gamma * following_value * non_terminal
                    - values[time_index]
                )
                last_advantage = (
                    delta
                    + gamma * gae_lambda * non_terminal * last_advantage
                )
                advantages[time_index] = last_advantage

            finished = completed_returns[species_id]
            if finished:
                rollout_return_estimate = float(np.mean(finished))
            else:
                # A wide 2,048-transition PPO batch is only 32 ticks deep.
                # Until one of those trajectories finishes, expose a rough
                # horizon-scaled return diagnostic. Species selection never
                # uses this value; it has a separate complete-episode pass.
                rollout_return_estimate = (
                    float(rewards.mean()) * max_episode_length
                )

            rollouts[species_id] = Rollout(
                observations=observations,
                actions=actions,
                old_log_prob=old_log_prob,
                values=values,
                advantages=advantages,
                returns=advantages + values,
                episode_starts=starts,
                hidden=stored_hidden,
                rollout_return_estimate=rollout_return_estimate,
                raw_step_reward_mean=float(rewards.mean()),
                completed_returns=tuple(finished),
            )
            learner.controller.normalizer.update(
                raw_observations[..., :PHYSICAL_OBSERVATION_SIZE]
            )

        return (
            rollouts,
            observation,
            hidden,
            episode_start,
            episode_return,
        )

    def _evaluate_population_for_selection(self) -> SelectionEvaluation:
        """Run a paid, fixed-length complete-episode pass for species ranking.

        The controllers and observation normalisers are frozen. The rollout is
        temporally longer than the task horizon, so every initial environment
        contributes exactly one completed ranking episode. Auto-reset episodes
        still consume counted physics steps but never receive extra ranking
        weight.
        """
        epm = int(self.selection_config["environments_per_species"])
        rollout_steps = int(self.selection_config["rollout_steps"])
        num_envs = self.population.size * epm
        stochastic = self.selection_config["action_mode"] == "stochastic"
        environment = self._create_environment(num_envs)
        returns_by_species = {
            species.species_id: [] for species in self.population.species
        }
        lengths_by_species = {
            species.species_id: [] for species in self.population.species
        }
        starting_environment_steps = self.environment_steps

        try:
            observation, _ = environment.reset()
            hidden = {
                species.species_id: self.learners[
                    species.species_id
                ].controller.policy.initial_hidden(
                    epm,
                    species.graph,
                    self.device,
                )
                for species in self.population.species
            }
            episode_return = torch.zeros(num_envs, device=self.device)
            episode_length = torch.zeros(
                num_envs,
                dtype=torch.long,
                device=self.device,
            )
            ranking_complete = torch.zeros(
                num_envs,
                dtype=torch.bool,
                device=self.device,
            )

            for _ in range(rollout_steps):
                environment_action = torch.zeros(
                    num_envs,
                    N_ACTIONS,
                    device=self.device,
                )
                for group, species in enumerate(self.population.species):
                    species_id = species.species_id
                    rows = slice(group * epm, (group + 1) * epm)
                    controller = self.learners[species_id].controller
                    controller.policy.eval()
                    normalized = normalize_observation(
                        observation[rows],
                        controller.normalizer,
                    )
                    with torch.no_grad():
                        mean, log_std, next_hidden = (
                            controller.policy.forward_step(
                                normalized,
                                species.graph,
                                hidden[species_id],
                            )
                        )
                        action = mean
                        if stochastic:
                            action = mean + torch.exp(log_std) * torch.randn_like(
                                mean
                            )
                    environment_action[rows] = action.clamp(-1.0, 1.0)
                    hidden[species_id] = next_hidden

                observation, reward, terminated, truncated, _ = environment.step(
                    environment_action
                )
                reward = reward.squeeze(-1) if reward.ndim > 1 else reward
                done = terminated | truncated
                episode_return += reward
                episode_length += 1

                for group, species in enumerate(self.population.species):
                    species_id = species.species_id
                    rows = slice(group * epm, (group + 1) * epm)
                    group_done = done[rows]
                    first_completion = (
                        group_done & ~ranking_complete[rows]
                    )
                    if bool(first_completion.any()):
                        returns_by_species[species_id].extend(
                            float(value)
                            for value in episode_return[rows][
                                first_completion
                            ].detach().cpu()
                        )
                        lengths_by_species[species_id].extend(
                            int(value)
                            for value in episode_length[rows][
                                first_completion
                            ].detach().cpu()
                        )
                    ranking_complete[rows] |= group_done
                    hidden[species_id] *= (
                        ~group_done
                    ).to(observation.dtype).view(-1, 1, 1)

                episode_return = torch.where(done, 0.0, episode_return)
                episode_length = torch.where(done, 0, episode_length)
                self.environment_steps += num_envs
                self.selection_environment_steps += num_envs
        finally:
            environment.close()

        if not bool(ranking_complete.all()):
            missing = torch.nonzero(
                ~ranking_complete,
                as_tuple=False,
            ).flatten().detach().cpu().tolist()
            raise RuntimeError(
                "selection evaluation did not finish the initial ranking "
                f"episode for environment rows {missing}"
            )
        return summarize_selection_episodes(
            returns_by_species,
            lengths_by_species,
            environment_steps=self.environment_steps - starting_environment_steps,
        )

    def _ppo_update(
        self,
        species_graph: NGEGraph,
        learner: SpeciesLearner,
        rollout: Rollout,
    ) -> dict[str, float]:
        policy = learner.controller.policy
        value_network = learner.controller.value
        policy.train()
        value_network.train()
        sequence_length = int(self.training["sequence_length"])
        minibatch_sequences = int(self.training["minibatch_sequences"])
        epochs = int(self.training["optimization_epochs"])
        clip = float(self.training["ppo_clip"])
        gradient_clip = float(self.training["gradient_clip"])
        target_kl = float(self.training["target_kl"])
        high = float(self.training["target_kl_high"])
        low = float(self.training["target_kl_low"])

        time_steps, environments = rollout.values.shape
        sequence_positions = [
            (start, environment)
            for start in range(0, time_steps, sequence_length)
            for environment in range(environments)
        ]
        advantages = rollout.advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1.0e-8
        )
        mask = action_mask(species_graph, device=self.device)
        losses: dict[str, list[float]] = {
            "policy": [],
            "value": [],
            "entropy": [],
            "kl": [],
        }

        for _ in range(epochs):
            order = self.rng.permutation(len(sequence_positions))
            for offset in range(0, len(order), minibatch_sequences):
                chosen = order[offset : offset + minibatch_sequences]
                if len(chosen) == 0:
                    continue
                positions = [sequence_positions[int(index)] for index in chosen]

                def gather(values: torch.Tensor) -> torch.Tensor:
                    return torch.stack(
                        [
                            values[
                                start : start + sequence_length,
                                environment,
                            ]
                            for start, environment in positions
                        ],
                        dim=1,
                    )

                observation = gather(rollout.observations)
                actions = gather(rollout.actions)
                old_log_prob = gather(rollout.old_log_prob)
                batch_advantage = gather(advantages)
                target_return = gather(rollout.returns)
                starts = gather(rollout.episode_starts)
                initial_hidden = torch.stack(
                    [
                        rollout.hidden[start, environment]
                        for start, environment in positions
                    ]
                )

                mean, log_std, _ = policy.forward_sequence(
                    observation,
                    species_graph,
                    initial_hidden,
                    starts,
                )
                log_prob = gaussian_log_prob(actions, mean, log_std, mask)
                ratio = torch.exp(log_prob - old_log_prob)
                unclipped = ratio * batch_advantage
                clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * batch_advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                entropy = gaussian_entropy(log_std, mask).mean()

                learner.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip)
                learner.policy_optimizer.step()

                predicted_value = value_network(
                    observation.reshape(-1, OBSERVATION_SIZE)
                ).view_as(target_return)
                value_loss = F.mse_loss(predicted_value, target_return)
                learner.value_optimizer.zero_grad()
                value_loss.backward()
                learner.value_optimizer.step()

                approximate_kl = (old_log_prob - log_prob).mean()
                losses["policy"].append(float(policy_loss.detach()))
                losses["value"].append(float(value_loss.detach()))
                losses["entropy"].append(float(entropy.detach()))
                losses["kl"].append(float(approximate_kl.detach()))

        mean_kl = float(np.mean(losses["kl"]))
        if mean_kl > high * target_kl:
            learner.current_policy_lr /= 2.0
        elif mean_kl < low * target_kl:
            learner.current_policy_lr *= 1.5
        learner.current_policy_lr = float(
            np.clip(learner.current_policy_lr, 3.0e-10, 1.0e-2)
        )
        for group in learner.policy_optimizer.param_groups:
            group["lr"] = learner.current_policy_lr
        return {
            name: float(np.mean(values)) for name, values in losses.items()
        } | {"learning_rate": learner.current_policy_lr}

    def _inherit_after_selection(
        self,
        child_parent_ids: dict[int, int],
    ) -> None:
        old = self.learners
        new: dict[int, SpeciesLearner] = {}
        for species in self.population.species:
            if species.species_id in old:
                learner = old[species.species_id]
                learner.reset_optimizers()
            else:
                learner = old[child_parent_ids[species.species_id]].inherited_copy()
            new[species.species_id] = learner
        self.learners = new

    def _rng_state(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy_legacy": np.random.get_state(),
            "numpy_generator": self.rng.bit_generator.state,
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        }

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy_legacy"])
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["numpy_generator"]
        torch.set_rng_state(state["torch_cpu"])
        torch.cuda.set_rng_state_all(state["torch_cuda"])

    def checkpoint_state(self) -> dict[str, Any]:
        if (
            self.controller_environment_steps
            + self.selection_environment_steps
            != self.environment_steps
        ):
            raise RuntimeError("NGE environment-step audit counters do not sum")
        return {
            "format_version": FORMAT_VERSION,
            "method": "nge",
            "config": self.config,
            "training_seed": self.seed,
            "run_identity": self.run_identity,
            "environment_steps": self.environment_steps,
            "controller_environment_steps": self.controller_environment_steps,
            "selection_environment_steps": self.selection_environment_steps,
            "parallel_envs": self.parallel_envs,
            "population": self.population.state_dict(),
            "learners": {
                species_id: learner.state_dict()
                for species_id, learner in self.learners.items()
            },
            "gm_uc": self.gm_uc.state_dict(),
            "rng": self._rng_state(),
            "resume_boundary": "between_generations",
        }

    def save_checkpoint(self, *, final: bool = False) -> Path:
        name = (
            "final.pth"
            if final
            else f"generation_{self.population.generation:04d}.pth"
        )
        path = self.run_dir / "checkpoints" / name
        temporary = path.with_suffix(".tmp")
        torch.save(self.checkpoint_state(), temporary)
        os.replace(temporary, path)
        return path

    def _load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("format_version") != FORMAT_VERSION or state.get("method") != "nge":
            raise ValueError("unsupported or non-NGE checkpoint")
        if state.get("resume_boundary") != "between_generations":
            raise ValueError("NGE can resume only complete generation-boundary checkpoints")
        if not resume_configs_match(state["config"], self.config):
            raise ValueError(
                "resume algorithm config differs from the checkpoint config"
            )
        self.seed = int(state["training_seed"])
        self.run_identity = str(state["run_identity"])
        self.environment_steps = int(state["environment_steps"])
        self.controller_environment_steps = int(
            state.get("controller_environment_steps", self.environment_steps)
        )
        self.selection_environment_steps = int(
            state.get("selection_environment_steps", 0)
        )
        if (
            self.controller_environment_steps
            + self.selection_environment_steps
            != self.environment_steps
        ):
            raise ValueError("checkpoint environment-step audit counters do not sum")
        self.parallel_envs = int(state["parallel_envs"])
        self.population = Population.from_state_dict(state["population"])
        self.learners = {
            int(species_id): SpeciesLearner.from_state_dict(
                learner_state,
                self.device,
                hidden_size=self.hidden_size,
            )
            for species_id, learner_state in state["learners"].items()
        }
        self.gm_uc.load_state_dict(state["gm_uc"])
        self._restore_rng_state(state["rng"])

    def train(self) -> Path:
        """Run complete generations until the exact global physics budget."""
        checkpoint_frequency = int(
            self.config["checkpoint"]["every_generations"]
        )
        try:
            while self.environment_steps < self.target_environment_steps:
                environment = self._create_environment()
                epm = self.environments_per_species
                try:
                    observation, _ = environment.reset()
                    hidden = {
                        species.species_id: self.learners[
                            species.species_id
                        ].controller.policy.initial_hidden(
                            epm, species.graph, self.device
                        )
                        for species in self.population.species
                    }
                    episode_start = torch.ones(
                        self.parallel_envs,
                        dtype=torch.bool,
                        device=self.device,
                    )
                    episode_return = torch.zeros(
                        self.parallel_envs, device=self.device
                    )

                    update_metrics: list[dict[str, float]] = []
                    updates_per_generation = int(
                        self.training["updates_per_generation"]
                    )
                    for update_index in range(updates_per_generation):
                        (
                            rollouts,
                            observation,
                            hidden,
                            episode_start,
                            episode_return,
                        ) = self._collect_rollout(
                            environment,
                            observation,
                            hidden,
                            episode_start,
                            episode_return,
                        )
                        per_species_metrics = []
                        for species in self.population.species:
                            species_id = species.species_id
                            per_species_metrics.append(
                                self._ppo_update(
                                    species.graph,
                                    self.learners[species_id],
                                    rollouts[species_id],
                                )
                            )
                        update_metrics.extend(per_species_metrics)

                        # Log here, rather than only after population selection,
                        # so a new run exposes reward and PPO curves after its
                        # first update.  The x-axis is the shared, authoritative
                        # count of VSim environment steps.
                        wall_seconds = time.perf_counter() - self.started
                        update_log = summarize_training_update(
                            rollouts,
                            per_species_metrics,
                        )
                        completed_returns = [
                            value
                            for rollout in rollouts.values()
                            for value in rollout.completed_returns
                        ]
                        if completed_returns:
                            self.completed_reward_meter.update(
                                torch.tensor(
                                    completed_returns,
                                    dtype=torch.float32,
                                    device=self.device,
                                ).unsqueeze(1)
                            )
                        episode_reward = (
                            float(self.completed_reward_meter.get_mean())
                            if self.completed_reward_meter.current_size > 0
                            else None
                        )
                        iteration = (
                            self.population.generation * updates_per_generation
                            + update_index
                            + 1
                        )
                        update_log.update(
                            {
                                "benchmark/training/environment_steps": float(
                                    self.environment_steps
                                ),
                                "benchmark/training/wall_seconds": wall_seconds,
                                "benchmark/training/env_steps_per_second": (
                                    self.environment_steps
                                    / max(wall_seconds, 1.0e-8)
                                ),
                                "benchmark/training/parallel_envs": float(
                                    self.parallel_envs
                                ),
                                "benchmark/training/controller_environment_steps": float(
                                    self.controller_environment_steps
                                ),
                                "benchmark/training/selection_environment_steps": float(
                                    self.selection_environment_steps
                                ),
                                "benchmark/training/iteration": float(
                                    iteration
                                ),
                                "nge/progress/generation": float(
                                    self.population.generation
                                ),
                                "nge/progress/update_in_generation": float(
                                    update_index + 1
                                ),
                                "nge/progress/updates_per_generation": float(
                                    updates_per_generation
                                ),
                                "nge/population/unique_graphs": float(
                                    len(
                                        {
                                            species.graph.key
                                            for species in self.population.species
                                        }
                                    )
                                ),
                                "nge/population/mean_actuators": float(
                                    np.mean(
                                        [
                                            species.graph.num_actuators
                                            for species in self.population.species
                                        ]
                                    )
                                ),
                            }
                        )
                        self.logger.log_training_update(
                            update_log,
                            environment_steps=self.environment_steps,
                            iteration=iteration,
                            wall_seconds=wall_seconds,
                            episode_reward=episode_reward,
                        )
                        rollout_return_mean = float(
                            np.mean(
                                [
                                    rollout.rollout_return_estimate
                                    for rollout in rollouts.values()
                                ]
                            )
                        )
                        print(
                            "[nge] "
                            f"generation={self.population.generation} "
                            f"steps={self.environment_steps:,}/"
                            f"{self.target_environment_steps:,} "
                            f"rollout_return_estimate="
                            f"{rollout_return_mean:.3f}",
                            flush=True,
                        )
                finally:
                    environment.close()

                # NGE selects species on complete raw episode returns. This
                # separate, temporally deep pass replaces the censored
                # short-rollout estimate and consumes the same global budget.
                selection = self._evaluate_population_for_selection()
                completed_generation = (
                    self.environment_steps
                    // nge_generation_environment_steps(self.config)
                )
                fitness = selection.fitness
                self.population.assign_fitness(fitness)
                self.completed_reward_meter.update(
                    torch.tensor(
                        selection.completed_returns,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(1)
                )
                episode_reward = float(self.completed_reward_meter.get_mean())
                wall_seconds = time.perf_counter() - self.started
                iteration = (
                    self.population.generation + 1
                ) * updates_per_generation
                fitness_values = np.asarray(
                    list(fitness.values()),
                    dtype=np.float64,
                )
                episode_counts = np.asarray(
                    list(selection.episodes_per_species.values()),
                    dtype=np.float64,
                )
                metrics = {
                    "benchmark/training/environment_steps": float(
                        self.environment_steps
                    ),
                    "benchmark/training/wall_seconds": wall_seconds,
                    "benchmark/training/env_steps_per_second": (
                        self.environment_steps / max(wall_seconds, 1.0e-8)
                    ),
                    "benchmark/training/parallel_envs": float(
                        self.parallel_envs
                    ),
                    "benchmark/training/controller_environment_steps": float(
                        self.controller_environment_steps
                    ),
                    "benchmark/training/selection_environment_steps": float(
                        self.selection_environment_steps
                    ),
                    "benchmark/training/iteration": float(iteration),
                    "benchmark/resource/trainable_parameters": float(
                        sum(
                            parameter.numel()
                            for learner in self.learners.values()
                            for network in (
                                learner.controller.policy,
                                learner.controller.value,
                            )
                            for parameter in network.parameters()
                        )
                        + sum(
                            parameter.numel()
                            for parameter in self.gm_uc.model.parameters()
                        )
                    ),
                    "benchmark/resource/peak_device_bytes": float(
                        torch.cuda.max_memory_allocated(self.device)
                    ),
                    "benchmark/resource/peak_host_kib": float(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    ),
                    "nge/population/fitness_mean": float(
                        fitness_values.mean()
                    ),
                    "nge/population/fitness_best": float(
                        fitness_values.max()
                    ),
                    "nge/selection/return_mean": float(
                        fitness_values.mean()
                    ),
                    "nge/selection/return_std": float(fitness_values.std()),
                    "nge/selection/return_min": float(fitness_values.min()),
                    "nge/selection/return_max": float(fitness_values.max()),
                    "nge/selection/completed_episodes": float(
                        len(selection.completed_returns)
                    ),
                    "nge/selection/episodes_per_species_min": float(
                        episode_counts.min()
                    ),
                    "nge/selection/episodes_per_species_max": float(
                        episode_counts.max()
                    ),
                    "nge/selection/episode_length_mean": float(
                        np.mean(selection.completed_lengths)
                    ),
                    "nge/selection/environment_steps": float(
                        selection.environment_steps
                    ),
                    "nge/selection/parallel_envs": float(
                        self.population.size
                        * int(
                            self.selection_config[
                                "environments_per_species"
                            ]
                        )
                    ),
                    "rewards/selection_return_mean": float(
                        np.mean(selection.completed_returns)
                    ),
                    "rewards/selection_episode_length_mean": float(
                        np.mean(selection.completed_lengths)
                    ),
                    "rewards/completed_episodes": float(
                        len(selection.completed_returns)
                    ),
                    "nge/ppo/policy_loss": float(
                        np.mean([item["policy"] for item in update_metrics])
                    ),
                    "nge/ppo/value_loss": float(
                        np.mean([item["value"] for item in update_metrics])
                    ),
                    "nge/ppo/kl": float(
                        np.mean([item["kl"] for item in update_metrics])
                    ),
                }

                if self.environment_steps == self.target_environment_steps:
                    self.logger.log_training_update(
                        metrics,
                        environment_steps=self.environment_steps,
                        iteration=iteration,
                        wall_seconds=wall_seconds,
                        episode_reward=episode_reward,
                    )
                    final = self.save_checkpoint(final=True)
                    print(f"[nge] final checkpoint -> {final}", flush=True)
                    self._run_training_evaluation(
                        completed_generation=completed_generation,
                        final=True,
                    )
                    return final

                evolution = self.population.evolve(
                    self.gm_uc,
                    self.rng,
                    elimination_rate=float(
                        self.population_config["elimination_rate"]
                    ),
                    candidate_pool_size=int(
                        self.population_config["candidate_pool_size"]
                    ),
                    mutation_probabilities={
                        key: float(value)
                        for key, value in self.population_config[
                            "mutation_probabilities"
                        ].items()
                    },
                    node_perturb_probability=float(
                        self.population_config["node_perturb_probability"]
                    ),
                )
                self._inherit_after_selection(evolution.child_parent_ids)
                metrics.update(
                    {
                        "nge/population/generation": float(
                            self.population.generation
                        ),
                        "nge/population/eliminated": float(
                            len(evolution.eliminated_ids)
                        ),
                        "nge/gm_uc/loss": evolution.gm_uc_loss,
                        "nge/gm_uc/history_size": float(
                            len(self.gm_uc.history)
                        ),
                    }
                )
                self.logger.log_training_update(
                    metrics,
                    environment_steps=self.environment_steps,
                    iteration=iteration,
                    wall_seconds=wall_seconds,
                    episode_reward=episode_reward,
                )
                print(
                    "[nge] "
                    f"generation={self.population.generation} "
                    f"steps={self.environment_steps:,}/"
                    f"{self.target_environment_steps:,} "
                    f"selection_return={fitness_values.mean():.3f} "
                    f"complete_episodes={len(selection.completed_returns)}",
                    flush=True,
                )
                if self.population.generation % checkpoint_frequency == 0:
                    checkpoint = self.save_checkpoint()
                    print(f"[nge] checkpoint -> {checkpoint}", flush=True)
                self._run_training_evaluation(
                    completed_generation=completed_generation,
                    final=False,
                )
        finally:
            self.logger.close()
        raise RuntimeError("NGE training stopped without reaching its exact budget")
