"""Faithful BodyGen collection, PPO training, logging, and checkpointing.

BodyGen is episodic in a way the other baselines are not.  Each of twenty
logical streams first makes five topology decisions and one attribute
decision, then controls that body until the episode finishes.  VSim executes
those streams together.  Faster-falling lanes therefore keep consuming
physics while the longest lane finishes; those transitions are charged as
synchronisation waste but never enter PPO.

This module deliberately keeps that accounting beside the collector.  The
following identities hold at every checkpoint::

    environment_steps = trajectory_environment_steps + synchronisation_waste
    trajectory_environment_steps = ppo_used_steps + discarded_final_steps

Design decisions are stored MDP transitions, but are simulator-free and never
increment any physics-step counter.
"""
from __future__ import annotations

import copy
import itertools
import random
import re
import resource
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from .credit import (
    ATTRIBUTE,
    CONTROL,
    TOPOLOGY,
    ReturnNormalizer,
    enhanced_temporal_credit_assignment,
)
from .design import DesignBatchTrace, DesignTransition
from .mosat import BodyGenNetworks


FORMAT_VERSION = 1
OBSERVATION_SIZE = 893
ACTION_SIZE = 32
DESIGN_TRANSITIONS_PER_EPISODE = 6
REWARD_WINDOW = 100
FIXED_FORWARD_CHUNK = 10_000


def bodygen_worker_target(config: dict[str, Any]) -> int:
    """Return the camera-ready ``floor(min_batch_size / num_workers)``."""
    collection = config["collection"]
    return int(collection["minimum_batch_transitions"]) // int(
        collection["logical_streams"]
    )


def choose_wave_width(active_streams: int, remaining_steps: int) -> int:
    """Use all unfinished streams, narrowing only at the final step tail."""
    active_streams = int(active_streams)
    remaining_steps = int(remaining_steps)
    if active_streams <= 0 or remaining_steps <= 0:
        raise ValueError("active streams and remaining steps must be positive")
    return min(active_streams, remaining_steps)


def stage_grouped_permutation(
    permutation: np.ndarray,
    stages: np.ndarray,
) -> np.ndarray:
    """Apply upstream ``batch_design`` grouping after the random shuffle.

    The random order remains stable within topology, attribute, and control.
    Contiguous floor-minibatches therefore give each independent trunk the
    same update cadence as the camera-ready implementation.
    """
    permutation = np.asarray(permutation, dtype=np.int64)
    stages = np.asarray(stages, dtype=np.int64)
    if permutation.ndim != 1 or stages.ndim != 1:
        raise ValueError("permutation and stages must be one-dimensional")
    if permutation.size != stages.size:
        raise ValueError("permutation and stages must have equal length")
    if set(permutation.tolist()) != set(range(permutation.size)):
        raise ValueError("permutation must contain every transition once")
    if np.any((stages < TOPOLOGY) | (stages > CONTROL)):
        raise ValueError("stages must be topology, attribute, or control")
    shuffled_stages = stages[permutation]
    return np.concatenate(
        [
            permutation[shuffled_stages == stage]
            for stage in (TOPOLOGY, ATTRIBUTE, CONTROL)
        ]
    )


def load_bodygen_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("BodyGen config must be a YAML mapping")
    validate_bodygen_config(config)
    return config


def resume_configs_match(
    saved: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    """Permit execution/monitoring changes without changing the algorithm."""
    saved_algorithm = copy.deepcopy(saved)
    current_algorithm = copy.deepcopy(current)
    for key in ("native_evaluation", "training_evaluation", "logging"):
        saved_algorithm.pop(key, None)
        current_algorithm.pop(key, None)
    # Moving a checkpoint between CUDA devices changes where it executes, not
    # the BodyGen algorithm or the samples it is allowed to consume.
    saved_algorithm.get("runtime", {}).pop("device", None)
    current_algorithm.get("runtime", {}).pop("device", None)
    return saved_algorithm == current_algorithm


def _positive_integer(mapping: dict[str, Any], key: str, prefix: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{prefix}.{key} must be a positive integer")
    return int(value)


def _validate_evaluation_block(
    config: dict[str, Any],
    name: str,
    *,
    shared: bool,
) -> None:
    block = config[name]
    if not isinstance(block, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    if not isinstance(block.get("enabled"), bool):
        raise ValueError(f"{name}.enabled must be true or false")
    _positive_integer(block, "every_updates", name)
    if not isinstance(block.get("evaluate_final"), bool):
        raise ValueError(f"{name}.evaluate_final must be true or false")
    if shared:
        _positive_integer(block, "pairs", name)
        _positive_integer(block, "episodes_per_pair", name)
        seeds = block.get("seeds")
        if not isinstance(seeds, dict):
            raise ValueError(f"{name}.seeds must be a YAML mapping")
        for key in ("morphology", "rollout"):
            value = seeds.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}.seeds.{key} must be an integer")
    else:
        _positive_integer(block, "episodes", name)
        for key in ("morphology_seed", "rollout_seed"):
            value = block.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}.{key} must be an integer")


def validate_bodygen_config(config: dict[str, Any]) -> None:
    """Reject settings that break BodyGen fidelity or exact accounting."""
    if config.get("method") != "bodygen":
        raise ValueError("BodyGen training config must set method: bodygen")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    runtime = config["runtime"]
    if runtime.get("dtype") != "float64":
        raise ValueError("faithful BodyGen training requires runtime.dtype: float64")
    if not isinstance(runtime.get("development", False), bool):
        raise ValueError("runtime.development must be true or false")
    development = bool(runtime.get("development", False))

    budget = config["budget"]
    environment_steps = _positive_integer(budget, "environment_steps", "budget")
    parallel_cap = _positive_integer(budget, "parallel_envs", "budget")
    if parallel_cap > 4096:
        raise ValueError("BodyGen cannot exceed the shared 4,096-environment cap")

    environment = config["environment"]
    max_episode_length = _positive_integer(
        environment,
        "max_episode_length",
        "environment",
    )
    if not development and max_episode_length != 1000:
        raise ValueError(
            "faithful BodyGen requires environment.max_episode_length: 1000"
        )
    if list(environment.get("base_legs", ())) != [1, 4, 6]:
        raise ValueError("faithful BodyGen must begin from base_legs: [1, 4, 6]")
    expected_environment = {
        "ctrl_cost_weight": 0.5,
        "healthy_reward": 2.0,
        "reset_noise_scale": 1.0,
        "timestep": 0.01667,
        "frame_skip": 1,
        "spacing": 3.0,
        "max_contact_pairs_per_env": 64,
    }
    for key, expected in expected_environment.items():
        value = environment.get(key)
        if isinstance(expected, int):
            matches = (
                not isinstance(value, bool)
                and isinstance(value, int)
                and value == expected
            )
        else:
            try:
                matches = bool(np.isclose(float(value), expected))
            except (TypeError, ValueError):
                matches = False
        if not matches:
            raise ValueError(
                f"shared VSim task requires environment.{key}: {expected}"
            )
    try:
        healthy_y_range = np.asarray(
            environment["healthy_y_range"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "environment.healthy_y_range must be [0.3, 1.1]"
        ) from error
    if healthy_y_range.shape != (2,) or not np.allclose(
        healthy_y_range,
        [0.3, 1.1],
    ):
        raise ValueError(
            "shared VSim task requires environment.healthy_y_range: [0.3, 1.1]"
        )

    design = config["design"]
    if _positive_integer(design, "topology_steps", "design") != 5:
        raise ValueError("faithful BodyGen requires five topology steps")
    if _positive_integer(design, "attribute_steps", "design") != 1:
        raise ValueError("faithful BodyGen requires one attribute step")
    if _positive_integer(design, "max_effectors_per_limb", "design") != 3:
        raise ValueError("the shared grammar permits three effectors per limb")
    if _positive_integer(design, "root_slots", "design") != 8:
        raise ValueError("the shared grammar has eight root limb slots")
    if design.get("root_add_order") != "first_empty":
        raise ValueError("root Add must fill the first empty root slot")
    if _positive_integer(design, "minimum_effectors", "design") != 1:
        raise ValueError("every BodyGen morphology must keep one effector")

    collection = config["collection"]
    streams = _positive_integer(collection, "logical_streams", "collection")
    minimum_batch = _positive_integer(
        collection,
        "minimum_batch_transitions",
        "collection",
    )
    per_stream = _positive_integer(
        collection,
        "minimum_transitions_per_stream",
        "collection",
    )
    if streams != 20:
        raise ValueError("faithful BodyGen requires twenty logical streams")
    if not development and minimum_batch != 50_000:
        raise ValueError("faithful BodyGen requires a 50,000-transition batch")
    if per_stream != minimum_batch // streams:
        raise ValueError(
            "minimum_transitions_per_stream must equal floor("
            "minimum_batch_transitions / logical_streams)"
        )
    if collection.get("complete_episodes") is not True:
        raise ValueError("BodyGen collection must retain only complete episodes")
    if streams > parallel_cap:
        raise ValueError(
            "collection.logical_streams exceeds budget.parallel_envs cap"
        )

    network = config["network"]
    for key in (
        "hidden_size",
        "blocks",
        "attention_heads",
        "feed_forward_ratio",
        "topology_embeddings",
    ):
        _positive_integer(network, key, "network")
    if int(network["hidden_size"]) not in {32, 64, 128, 256}:
        raise ValueError(
            "network.hidden_size must be a paper-supported "
            "32, 64, 128, or 256"
        )
    if network["attention_heads"] != 1 or network["blocks"] != 3:
        raise ValueError("faithful MoSAT requires one head and three blocks")
    if network["feed_forward_ratio"] != 4:
        raise ValueError("faithful MoSAT requires feed_forward_ratio: 4")
    if int(network["topology_embeddings"]) != 256:
        raise ValueError("faithful TopoPE requires 256 embeddings")
    if network.get("layer_norm") not in {"none", "pre", "post"}:
        raise ValueError("network.layer_norm must be none, pre, or post")
    if network.get("activation") != "silu":
        raise ValueError("faithful MoSAT uses SiLU")
    if list(network.get("critic_hidden", ())) != [512, 256]:
        raise ValueError("faithful BodyGen critic_hidden must be [512, 256]")
    if not np.isclose(float(network["initial_control_log_std"]), -0.5):
        raise ValueError(
            "faithful Crawler BodyGen requires initial_control_log_std: -0.5"
        )

    training = config["training"]
    for key in (
        "gamma",
        "gae_lambda",
        "ppo_clip",
        "actor_learning_rate",
        "critic_learning_rate",
        "gradient_clip",
    ):
        value = float(training[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if not 0 < float(training["gamma"]) <= 1:
        raise ValueError("training.gamma must lie in (0, 1]")
    if not 0 < float(training["gae_lambda"]) <= 1:
        raise ValueError("training.gae_lambda must lie in (0, 1]")
    expected_scalars = {
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "ppo_clip": 0.2,
        "gradient_clip": 40.0,
    }
    for key, expected in expected_scalars.items():
        if not np.isclose(float(training[key]), expected):
            raise ValueError(
                f"faithful BodyGen requires training.{key}: {expected}"
            )
    if float(training["actor_learning_rate"]) not in {
        5.0e-5,
        1.0e-4,
        3.0e-4,
    }:
        raise ValueError(
            "training.actor_learning_rate is outside the paper sweep"
        )
    if float(training["critic_learning_rate"]) not in {1.0e-4, 3.0e-4}:
        raise ValueError(
            "training.critic_learning_rate is outside the paper sweep"
        )
    optimization_epochs = _positive_integer(
        training,
        "optimization_epochs",
        "training",
    )
    if not development and optimization_epochs != 10:
        raise ValueError("faithful BodyGen requires ten PPO optimization epochs")
    minibatch_size = _positive_integer(
        training,
        "minibatch_size",
        "training",
    )
    if not development and minibatch_size != 2048:
        raise ValueError("faithful BodyGen requires minibatch_size: 2048")
    for key in (
        "normalize_observations",
        "normalize_returns",
        "normalize_advantages",
    ):
        if not isinstance(training.get(key), bool):
            raise ValueError(f"training.{key} must be true or false")
        if not training[key]:
            raise ValueError(f"faithful BodyGen requires training.{key}: true")

    checkpoint = config["checkpoint"]
    _positive_integer(checkpoint, "every_updates", "checkpoint")
    _validate_evaluation_block(config, "native_evaluation", shared=False)
    _validate_evaluation_block(config, "training_evaluation", shared=True)

    logging = config["logging"]
    if logging.get("tensorboard") is not True:
        raise ValueError("TensorBoard must remain enabled")
    wandb = logging["wandb"]
    if wandb.get("mode") not in {"online", "offline"}:
        raise ValueError("logging.wandb.mode must be online or offline")
    if not isinstance(wandb.get("enabled"), bool):
        raise ValueError("logging.wandb.enabled must be true or false")
    if wandb["enabled"] and not wandb.get("project"):
        raise ValueError("W&B project is required when W&B is enabled")

    # Kept explicit because exact-budget support is meaningful only for a
    # positive counter; unlike NGE, BodyGen may end inside an episodic batch.
    if environment_steps < 1:
        raise ValueError("budget.environment_steps must be positive")


@dataclass
class Episode:
    """One complete six-design-step plus controller trajectory."""

    design: Any
    design_trace: Any
    observations: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    # Interior successor values come from the next stored observation. Only
    # the episode-boundary successor is needed explicitly for bootstrapping.
    final_next_observation: torch.Tensor | None = None
    rewards: list[torch.Tensor] = field(default_factory=list)
    terminated: list[torch.Tensor] = field(default_factory=list)
    truncated: list[torch.Tensor] = field(default_factory=list)

    @property
    def physics_steps(self) -> int:
        return len(self.rewards)

    @property
    def stored_transitions(self) -> int:
        return DESIGN_TRANSITIONS_PER_EPISODE + self.physics_steps

    @property
    def episode_return(self) -> float:
        if not self.rewards:
            return 0.0
        return float(torch.stack(self.rewards).sum())


def episode_bootstrap_observation(episode: Episode) -> torch.Tensor:
    """Return the correct value-bootstrap state for VSim's delayed done API.

    VSim reports done one call after reaching the terminal state. The collector
    records that notification call's pre-step observation as the preceding
    real transition's successor. Its returned observation is already a reset
    state and must never be used for bootstrapping.
    """

    if (
        not episode.observations
        or not episode.truncated
        or episode.final_next_observation is None
    ):
        raise ValueError("a completed episode needs a final successor")
    return episode.final_next_observation


def record_delayed_vsim_transition(
    episode: Episode,
    *,
    observation: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> bool:
    """Record one lane while correcting VSim's one-call-late done signal.

    A normal call stores its state, sampled action, and physical reward with a
    provisional non-terminal flag. When VSim reports done on the next call, it
    has already simulated one extra step and reset the lane. That call is not
    an MDP transition: its *pre-step* observation is the terminal successor of
    the previously stored transition, and its reset reward/action are ignored.
    The caller still charges the physical call as synchronization waste.

    Returns ``True`` only for the delayed done-notification call.
    """

    terminated_cpu = terminated.detach().to(device="cpu", dtype=torch.bool)
    truncated_cpu = truncated.detach().to(device="cpu", dtype=torch.bool)
    done = bool(terminated_cpu | truncated_cpu)
    if done:
        if not episode.rewards:
            raise RuntimeError(
                "VSim reported done before BodyGen stored a real transition"
            )
        episode.terminated[-1] = terminated_cpu
        episode.truncated[-1] = truncated_cpu
        episode.final_next_observation = observation.detach().cpu()
        return True

    episode.observations.append(observation.detach().cpu())
    episode.actions.append(action.detach().cpu())
    episode.rewards.append(reward.detach().cpu())
    episode.terminated.append(torch.zeros_like(terminated_cpu))
    episode.truncated.append(torch.zeros_like(truncated_cpu))
    return False


@dataclass(frozen=True)
class CollectionResult:
    episodes: tuple[Episode, ...]
    complete: bool
    physics_steps: int
    trajectory_physics_steps: int
    synchronization_waste_steps: int
    design_transitions: int
    peak_parallel_envs: int


@dataclass
class PreparedBatch:
    trace: DesignBatchTrace
    design_transitions: tuple[DesignTransition, ...]
    design_stages: torch.Tensor
    design_old_log_prob: torch.Tensor
    design_advantages: torch.Tensor
    design_returns: torch.Tensor
    control_observations: torch.Tensor
    control_designs: list[Any]
    control_actions: torch.Tensor
    control_old_log_prob: torch.Tensor
    control_advantages: torch.Tensor
    control_returns: torch.Tensor
    completed_returns: tuple[float, ...]

    @property
    def design_size(self) -> int:
        return int(self.design_advantages.numel())

    @property
    def control_size(self) -> int:
        return int(self.control_advantages.numel())

    @property
    def size(self) -> int:
        return self.design_size + self.control_size


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
                    "W&B is enabled in bodygen.yaml but wandb is unavailable"
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
            for metric in (
                "benchmark/training/environment_steps",
                "benchmark/training/update",
                "benchmark/training/wall_seconds",
            ):
                self.wandb_run.define_metric(metric)
            self.wandb_run.define_metric(
                "*",
                step_metric="benchmark/training/environment_steps",
            )
            self.wandb_run.define_metric(
                "rewards/iter",
                step_metric="benchmark/training/update",
            )
            self.wandb_run.define_metric(
                "rewards/time",
                step_metric="benchmark/training/wall_seconds",
            )

    def log_update(
        self,
        metrics: dict[str, float],
        *,
        environment_steps: int,
        update: int,
        wall_seconds: float,
        completed_reward: float | None,
    ) -> None:
        for name, value in metrics.items():
            if np.isfinite(value):
                self.writer.add_scalar(name, value, environment_steps)
        if completed_reward is not None:
            self.writer.add_scalar(
                "rewards/step",
                completed_reward,
                environment_steps,
            )
            self.writer.add_scalar("rewards/iter", completed_reward, update)
            self.writer.add_scalar("rewards/time", completed_reward, wall_seconds)
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
                    "benchmark/training/update": float(update),
                    "benchmark/training/wall_seconds": float(wall_seconds),
                }
            )
            if completed_reward is not None:
                payload.update(
                    {
                        "rewards/step": completed_reward,
                        "rewards/iter": completed_reward,
                        "rewards/time": completed_reward,
                    }
                )
            self.wandb_run.log(payload)

    def log_evaluation(
        self,
        tag: str,
        value: float,
        *,
        environment_steps: int,
        update: int,
        episodes: int,
    ) -> None:
        metrics = {
            tag: float(value),
            f"benchmark/{tag.replace('/', '_')}/episodes": float(episodes),
            "benchmark/training/update": float(update),
        }
        for name, metric_value in metrics.items():
            self.writer.add_scalar(name, metric_value, environment_steps)
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


def _flatten_parameters(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    return list(
        itertools.chain.from_iterable(module.parameters() for module in modules)
    )


def _standardize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return torch.zeros_like(values)
    # The camera-ready implementation adds no epsilon here.
    return (values - values.mean()) / values.std(unbiased=True)


class BodyGenTrainer:
    """Own all state needed to train, checkpoint, and resume BodyGen."""

    def __init__(
        self,
        config: dict[str, Any],
        run_dir: Path,
        *,
        resume: Path | None = None,
    ) -> None:
        validate_bodygen_config(config)
        self.config = copy.deepcopy(config)
        self.run_dir = run_dir.expanduser().resolve()
        self.device = torch.device(config["runtime"]["device"])
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "BodyGen VSim training requires an available CUDA device"
            )
        self.dtype = torch.float64
        self.seed = int(config["seed"])
        self.rng = np.random.default_rng(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.torch_generator = torch.Generator(device=self.device)
        self.torch_generator.manual_seed(self.seed)

        network = config["network"]
        self.networks = BodyGenNetworks(
            observation_size=OBSERVATION_SIZE,
            action_size=ACTION_SIZE,
            hidden_size=int(network["hidden_size"]),
            num_blocks=int(network["blocks"]),
            dtype=self.dtype,
            layer_norm=str(network["layer_norm"]),
            topology_embeddings=int(network["topology_embeddings"]),
            feed_forward_ratio=int(network["feed_forward_ratio"]),
            critic_hidden=tuple(int(x) for x in network["critic_hidden"]),
            initial_control_log_std=float(
                network["initial_control_log_std"]
            ),
        ).to(self.device)
        self.actor_parameters = _flatten_parameters(
            (
                self.networks.topology_actor,
                self.networks.attribute_actor,
                self.networks.control_actor,
            )
        )
        self.critic_parameters = _flatten_parameters(
            (
                self.networks.topology_critic,
                self.networks.attribute_critic,
                self.networks.control_critic,
            )
        )
        training = config["training"]
        self.actor_optimizer = torch.optim.Adam(
            self.actor_parameters,
            lr=float(training["actor_learning_rate"]),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_parameters,
            lr=float(training["critic_learning_rate"]),
        )
        self.design_return_normalizer = ReturnNormalizer(
            demean=False,
            dtype=self.dtype,
        ).to(self.device)
        self.control_return_normalizer = ReturnNormalizer(
            demean=False,
            dtype=self.dtype,
        ).to(self.device)

        self.target_environment_steps = int(
            config["budget"]["environment_steps"]
        )
        self.environment_steps = 0
        self.trajectory_environment_steps = 0
        self.ppo_used_environment_steps = 0
        self.discarded_trajectory_environment_steps = 0
        self.synchronization_waste_steps = 0
        self.used_design_transitions = 0
        self.discarded_design_transitions = 0
        self.peak_parallel_envs = 0
        self.completed_updates = 0
        self.completed_episodes = 0
        self.reward_window: deque[float] = deque(maxlen=REWARD_WINDOW)
        self.run_identity = re.sub(
            r"[^A-Za-z0-9_-]+",
            "-",
            f"bodygen-{self.run_dir.name}-s{self.seed}",
        )
        self.accumulated_wall_seconds = 0.0
        self.best_native_return = -float("inf")

        if resume is not None:
            self._load_checkpoint(resume)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        config_path = self.run_dir / "config.yaml"
        if resume is None:
            if config_path.exists():
                raise FileExistsError(
                    f"BodyGen run already exists: {self.run_dir}"
                )
            config_path.write_text(yaml.safe_dump(self.config, sort_keys=False))
        elif not config_path.is_file():
            raise FileNotFoundError(
                f"resume run has no config.yaml: {self.run_dir}"
            )

        self.logger = TrainingLogger(
            self.run_dir,
            self.config,
            self.run_identity,
        )
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)

    @property
    def wall_seconds(self) -> float:
        return self.accumulated_wall_seconds + (
            time.perf_counter() - self.started
        )

    def _environment_arguments(self) -> dict[str, Any]:
        saved = self.config["environment"]
        result: dict[str, Any] = {
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
                result[key] = saved[key]
        return result

    def _create_environment(
        self,
        width: int,
        initial_designs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Any:
        from envs.ant_envs.ant_codesign import AntCodesignEnv

        environment_seed = int(self.rng.integers(0, 2**31 - 1))
        return AntCodesignEnv(
            int(width),
            self.device,
            seed=environment_seed,
            initial_designs=initial_designs,
            **self._environment_arguments(),
        )

    @staticmethod
    def _design_arrays(designs: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = [design.to_arrays() for design in designs]
        if rows and isinstance(rows[0], dict):
            counts = np.stack([row["counts"] for row in rows])
            effectors = np.stack([row["eff_sub"] for row in rows])
            caps = np.stack([row["cap_sub"] for row in rows])
        else:
            counts, effectors, caps = (
                np.stack(values)
                for values in zip(*rows, strict=True)
            )
        return counts, effectors, caps

    def _collect_wave(
        self,
        stream_ids: list[int],
        *,
        remaining_steps: int,
    ) -> tuple[list[tuple[int, Episode]], int, int, int, int]:
        """Collect at most one first complete episode from each VSim lane."""
        width = choose_wave_width(len(stream_ids), remaining_steps)
        stream_ids = stream_ids[:width]
        designs, trace = self.networks.sample_designs(
            width,
            self.torch_generator,
            deterministic=False,
        )
        if len(designs) != width or len(trace.episodes) != width:
            raise RuntimeError("BodyGen design sampler returned the wrong batch size")

        counts, effectors, caps = self._design_arrays(designs)
        environment = self._create_environment(
            width,
            initial_designs=(counts, effectors, caps),
        )
        completed: list[tuple[int, Episode]] = []
        lane_episodes = [
            Episode(design=design, design_trace=trace.episodes[index])
            for index, design in enumerate(designs)
        ]
        live = torch.ones(width, dtype=torch.bool, device=self.device)
        wave_physics = 0
        try:
            observation, _ = environment.reset()

            while (
                bool(live.any())
                and wave_physics + width <= remaining_steps
            ):
                with torch.no_grad():
                    output = self.networks.control(
                        observation.to(self.device, dtype=self.dtype),
                        designs,
                        deterministic=False,
                        generator=self.torch_generator,
                    )
                raw_action = output.action
                environment_action = raw_action.clamp(-1.0, 1.0)
                environment_action = torch.where(
                    live[:, None],
                    environment_action,
                    torch.zeros_like(environment_action),
                )
                next_observation, reward, terminated, truncated, _ = (
                    environment.step(environment_action)
                )
                reward = reward.squeeze(-1) if reward.ndim > 1 else reward
                done = terminated | truncated
                wave_physics += width

                for lane in torch.nonzero(
                    live,
                    as_tuple=False,
                ).flatten().tolist():
                    episode = lane_episodes[lane]
                    notification = record_delayed_vsim_transition(
                        episode,
                        observation=observation[lane],
                        action=raw_action[lane],
                        reward=reward[lane],
                        terminated=terminated[lane],
                        truncated=truncated[lane],
                    )
                    if notification != bool(done[lane]):
                        raise RuntimeError(
                            "BodyGen/VSim done accounting is inconsistent"
                        )

                newly_done = live & done
                for lane in torch.nonzero(
                    newly_done,
                    as_tuple=False,
                ).flatten().tolist():
                    completed.append((stream_ids[lane], lane_episodes[lane]))
                live &= ~done
                observation = next_observation
        finally:
            environment.close()

        trajectory_physics = sum(
            episode.physics_steps for episode in lane_episodes
        )
        synchronization_waste = wave_physics - trajectory_physics
        if synchronization_waste < 0:
            raise RuntimeError("negative BodyGen synchronization waste")
        return (
            completed,
            wave_physics,
            trajectory_physics,
            synchronization_waste,
            width * DESIGN_TRANSITIONS_PER_EPISODE,
        )

    def collect_batch(self) -> CollectionResult:
        """Collect complete episodes to each worker's camera-ready threshold."""
        stream_count = int(self.config["collection"]["logical_streams"])
        worker_target = bodygen_worker_target(self.config)
        streams: dict[int, list[Episode]] = {
            stream: [] for stream in range(stream_count)
        }
        stored = {stream: 0 for stream in range(stream_count)}
        physics_steps = 0
        trajectory_steps = 0
        synchronization_waste = 0
        design_transitions = 0
        peak_width = 0

        while self.environment_steps + physics_steps < self.target_environment_steps:
            unfinished = [
                stream
                for stream in range(stream_count)
                if stored[stream] < worker_target
            ]
            if not unfinished:
                break
            remaining = (
                self.target_environment_steps
                - self.environment_steps
                - physics_steps
            )
            completed, wave_steps, useful, waste, design_steps = (
                self._collect_wave(unfinished, remaining_steps=remaining)
            )
            physics_steps += wave_steps
            trajectory_steps += useful
            synchronization_waste += waste
            design_transitions += design_steps
            peak_width = max(
                peak_width,
                choose_wave_width(len(unfinished), remaining),
            )
            for stream, episode in completed:
                streams[stream].append(episode)
                stored[stream] += episode.stored_transitions

        complete = all(value >= worker_target for value in stored.values())
        episodes = tuple(
            episode
            for stream in range(stream_count)
            for episode in streams[stream]
        )
        if physics_steps != trajectory_steps + synchronization_waste:
            raise RuntimeError("BodyGen collection step audit does not sum")
        return CollectionResult(
            episodes=episodes,
            complete=complete,
            physics_steps=physics_steps,
            trajectory_physics_steps=trajectory_steps,
            synchronization_waste_steps=synchronization_waste,
            design_transitions=design_transitions,
            peak_parallel_envs=peak_width,
        )

    def _raw_value(
        self,
        value: torch.Tensor,
        normalizer: ReturnNormalizer,
    ) -> torch.Tensor:
        value = value.reshape(-1)
        if self.config["training"]["normalize_returns"]:
            value = normalizer.unscale(value)
        return value.reshape(-1)

    def _value_target(
        self,
        value: torch.Tensor,
        normalizer: ReturnNormalizer,
    ) -> torch.Tensor:
        value = value.reshape(-1)
        if self.config["training"]["normalize_returns"]:
            value = normalizer.normalize(value, update=False)
        return value.reshape(-1)

    def _prepare_batch(self, episodes: tuple[Episode, ...]) -> PreparedBatch:
        """Recompute fixed probabilities and apply Enhanced-TCA once per batch.

        This follows upstream ordering: observations first update the shared
        normaliser, fixed log probabilities and critic values are then
        evaluated, and only then do the ten PPO passes begin.
        """
        if not episodes:
            raise RuntimeError("a complete BodyGen batch contains no episodes")
        trace = DesignBatchTrace(
            episodes=tuple(episode.design_trace for episode in episodes)
        )
        raw_observations = torch.cat(
            [
                torch.stack(episode.observations)
                for episode in episodes
            ]
        ).to(self.device, dtype=self.dtype)
        # Enhanced-TCA only needs the successor value at each episode boundary;
        # within an episode the next value is the following stored state's
        # value. Keeping all 893-channel successor observations would duplicate
        # the largest GPU tensor in the batch.
        final_next_observations = torch.stack(
            [episode_bootstrap_observation(episode) for episode in episodes]
        ).to(self.device, dtype=self.dtype)
        control_actions = torch.cat(
            [torch.stack(episode.actions) for episode in episodes]
        ).to(self.device, dtype=self.dtype)
        control_designs = [
            episode.design
            for episode in episodes
            for _ in range(episode.physics_steps)
        ]

        # Upstream has one observation normalizer shared by all actor/critic
        # trunks. Feed every design state and every controller state into that
        # single node-feature normalizer before fixed probabilities are
        # evaluated.
        design_states = [
            transition.design
            for episode in episodes
            for transition in episode.design_trace.transitions
        ]
        for start in range(0, len(design_states), FIXED_FORWARD_CHUNK):
            stop = min(start + FIXED_FORWARD_CHUNK, len(design_states))
            self.networks.update_observation_normalizer(
                design_states[start:stop],
                observations=None,
            )
        for start in range(0, len(control_designs), FIXED_FORWARD_CHUNK):
            stop = min(start + FIXED_FORWARD_CHUNK, len(control_designs))
            self.networks.update_observation_normalizer(
                control_designs[start:stop],
                observations=raw_observations[start:stop],
            )
        observations = raw_observations

        self.networks.eval()
        with torch.no_grad():
            fixed_design = self.networks.evaluate_design(trace)
            fixed_control_parts = []
            for start in range(
                0,
                len(control_designs),
                FIXED_FORWARD_CHUNK,
            ):
                stop = min(
                    start + FIXED_FORWARD_CHUNK,
                    len(control_designs),
                )
                fixed_control_parts.append(
                    self.networks.evaluate_control(
                        observations[start:stop],
                        control_designs[start:stop],
                        control_actions[start:stop],
                    )
                )
            final_control = self.networks.control(
                final_next_observations,
                [episode.design for episode in episodes],
                deterministic=True,
            )

        design_old_log_prob = fixed_design["log_prob"].reshape(-1).detach()
        design_values = self._raw_value(
            fixed_design["values"],
            self.design_return_normalizer,
        ).reshape(len(episodes), DESIGN_TRANSITIONS_PER_EPISODE)
        control_old_log_prob = torch.cat(
            [part["log_prob"].reshape(-1) for part in fixed_control_parts]
        ).detach()
        control_values = self._raw_value(
            torch.cat(
                [part["value"].reshape(-1) for part in fixed_control_parts]
            ),
            self.control_return_normalizer,
        )
        final_next_values = self._raw_value(
            final_control.value,
            self.control_return_normalizer,
        )

        design_advantages: list[torch.Tensor] = []
        design_returns: list[torch.Tensor] = []
        control_advantages: list[torch.Tensor] = []
        control_returns: list[torch.Tensor] = []
        control_offset = 0

        for episode_index, episode in enumerate(episodes):
            length = episode.physics_steps
            episode_control_values = control_values[
                control_offset : control_offset + length
            ]
            final_next_value = final_next_values[episode_index]

            values = torch.cat(
                (
                    design_values[episode_index],
                    episode_control_values,
                )
            )
            next_values = torch.empty_like(values)
            next_values[:-1] = values[1:]
            next_values[-1] = final_next_value
            rewards = torch.cat(
                (
                    torch.zeros(
                        DESIGN_TRANSITIONS_PER_EPISODE,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    torch.stack(episode.rewards).to(
                        self.device,
                        dtype=self.dtype,
                    ),
                )
            )
            terminated = torch.cat(
                (
                    torch.zeros(
                        DESIGN_TRANSITIONS_PER_EPISODE,
                        device=self.device,
                        dtype=torch.bool,
                    ),
                    torch.stack(episode.terminated).to(
                        self.device,
                        dtype=torch.bool,
                    ),
                )
            )
            truncated = torch.cat(
                (
                    torch.zeros(
                        DESIGN_TRANSITIONS_PER_EPISODE,
                        device=self.device,
                        dtype=torch.bool,
                    ),
                    torch.stack(episode.truncated).to(
                        self.device,
                        dtype=torch.bool,
                    ),
                )
            )
            stages = torch.tensor(
                [
                    int(transition.stage)
                    for transition in episode.design_trace.transitions
                ]
                + [CONTROL] * length,
                device=self.device,
                dtype=torch.long,
            )
            if stages.numel() != values.numel():
                raise RuntimeError(
                    "BodyGen design trace does not contain exactly six stages"
                )
            credit = enhanced_temporal_credit_assignment(
                rewards,
                values,
                next_values,
                terminated,
                truncated,
                stages,
                gamma=float(self.config["training"]["gamma"]),
                gae_lambda=float(self.config["training"]["gae_lambda"]),
                normalize_advantages=False,
            )
            advantages = credit.advantages.reshape(-1)
            returns = credit.returns.reshape(-1)
            design_advantages.append(
                advantages[:DESIGN_TRANSITIONS_PER_EPISODE]
            )
            design_returns.append(
                returns[:DESIGN_TRANSITIONS_PER_EPISODE]
            )
            control_advantages.append(
                advantages[DESIGN_TRANSITIONS_PER_EPISODE:]
            )
            control_returns.append(
                returns[DESIGN_TRANSITIONS_PER_EPISODE:]
            )
            control_offset += length

        flat_design_advantages = torch.cat(design_advantages)
        flat_design_returns = torch.cat(design_returns)
        flat_control_advantages = torch.cat(control_advantages)
        flat_control_returns = torch.cat(control_returns)
        if self.config["training"]["normalize_advantages"]:
            flat_design_advantages = _standardize(flat_design_advantages)
            flat_control_advantages = _standardize(flat_control_advantages)

        if self.config["training"]["normalize_returns"]:
            self.design_return_normalizer.update(flat_design_returns)
            self.control_return_normalizer.update(flat_control_returns)
        design_targets = self._value_target(
            flat_design_returns,
            self.design_return_normalizer,
        )
        control_targets = self._value_target(
            flat_control_returns,
            self.control_return_normalizer,
        )
        return PreparedBatch(
            trace=trace,
            design_transitions=tuple(
                transition
                for episode in episodes
                for transition in episode.design_trace.transitions
            ),
            design_stages=torch.tensor(
                [
                    int(transition.stage)
                    for episode in episodes
                    for transition in episode.design_trace.transitions
                ],
                device=self.device,
                dtype=torch.long,
            ),
            design_old_log_prob=design_old_log_prob,
            design_advantages=flat_design_advantages.detach(),
            design_returns=design_targets.detach(),
            control_observations=observations,
            control_designs=control_designs,
            control_actions=control_actions,
            control_old_log_prob=control_old_log_prob,
            control_advantages=flat_control_advantages.detach(),
            control_returns=control_targets.detach(),
            completed_returns=tuple(
                episode.episode_return for episode in episodes
            ),
        )

    def _selected_design_outputs(
        self,
        batch: PreparedBatch,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = indices.detach().cpu().tolist()
        evaluated = self.networks.evaluate_design_transitions(
            [batch.design_transitions[index] for index in selected]
        )
        return (
            evaluated["log_prob"].reshape(-1),
            evaluated["entropy"].reshape(-1),
            evaluated["values"].reshape(-1),
        )

    def _selected_control_outputs(
        self,
        batch: PreparedBatch,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = indices.detach().cpu().tolist()
        evaluated = self.networks.evaluate_control(
            batch.control_observations[indices],
            [batch.control_designs[index] for index in selected],
            batch.control_actions[indices],
        )
        return (
            evaluated["log_prob"].reshape(-1),
            evaluated["entropy"].reshape(-1),
            evaluated["value"].reshape(-1),
        )

    def _ppo_update(self, batch: PreparedBatch) -> dict[str, float]:
        """Run the camera-ready ten-pass, shuffled, floor-minibatch PPO."""
        self.networks.train()
        training = self.config["training"]
        clip = float(training["ppo_clip"])
        gradient_clip = float(training["gradient_clip"])
        minibatch_size = int(training["minibatch_size"])
        epochs = int(training["optimization_epochs"])
        design_size = batch.design_size
        total_size = batch.size
        if total_size <= minibatch_size:
            minibatches = 1
        else:
            # Upstream deliberately drops the shuffled remainder.
            minibatches = total_size // minibatch_size

        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approximate_kl": [],
        }
        stages = torch.cat(
            (
                batch.design_stages,
                torch.full(
                    (batch.control_size,),
                    CONTROL,
                    device=self.device,
                    dtype=torch.long,
                ),
            )
        ).detach().cpu().numpy()
        for _ in range(epochs):
            grouped = stage_grouped_permutation(
                self.rng.permutation(total_size),
                stages,
            )
            permutation = torch.as_tensor(
                grouped,
                device=self.device,
                dtype=torch.long,
            )
            for minibatch in range(minibatches):
                if total_size <= minibatch_size:
                    chosen = permutation
                else:
                    start = minibatch * minibatch_size
                    chosen = permutation[start : start + minibatch_size]
                design_indices = chosen[chosen < design_size]
                control_indices = chosen[chosen >= design_size] - design_size

                log_prob_parts: list[torch.Tensor] = []
                entropy_parts: list[torch.Tensor] = []
                value_parts: list[torch.Tensor] = []
                old_parts: list[torch.Tensor] = []
                advantage_parts: list[torch.Tensor] = []
                return_parts: list[torch.Tensor] = []
                if design_indices.numel():
                    log_prob, entropy, value = (
                        self._selected_design_outputs(
                            batch,
                            design_indices,
                        )
                    )
                    log_prob_parts.append(log_prob)
                    entropy_parts.append(entropy)
                    value_parts.append(value)
                    old_parts.append(
                        batch.design_old_log_prob[design_indices]
                    )
                    advantage_parts.append(
                        batch.design_advantages[design_indices]
                    )
                    return_parts.append(batch.design_returns[design_indices])
                if control_indices.numel():
                    log_prob, entropy, value = (
                        self._selected_control_outputs(
                            batch,
                            control_indices,
                        )
                    )
                    log_prob_parts.append(log_prob)
                    entropy_parts.append(entropy)
                    value_parts.append(value)
                    old_parts.append(
                        batch.control_old_log_prob[control_indices]
                    )
                    advantage_parts.append(
                        batch.control_advantages[control_indices]
                    )
                    return_parts.append(
                        batch.control_returns[control_indices]
                    )

                log_prob = torch.cat(log_prob_parts)
                entropy = torch.cat(entropy_parts)
                predicted_value = torch.cat(value_parts)
                old_log_prob = torch.cat(old_parts)
                advantage = torch.cat(advantage_parts)
                target_return = torch.cat(return_parts)

                value_loss = (predicted_value - target_return).square().mean()
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_parameters,
                    gradient_clip,
                )
                self.critic_optimizer.step()

                ratio = torch.exp(log_prob - old_log_prob)
                unclipped = ratio * advantage
                clipped = ratio.clamp(
                    1.0 - clip,
                    1.0 + clip,
                ) * advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_parameters,
                    gradient_clip,
                )
                self.actor_optimizer.step()

                metrics["policy_loss"].append(float(policy_loss.detach()))
                metrics["value_loss"].append(float(value_loss.detach()))
                metrics["entropy"].append(float(entropy.mean().detach()))
                metrics["approximate_kl"].append(
                    float((old_log_prob - log_prob).mean().detach())
                )

        return {
            f"bodygen/ppo/{name}": float(np.mean(values))
            for name, values in metrics.items()
        } | {
            "bodygen/ppo/actor_learning_rate": float(
                self.actor_optimizer.param_groups[0]["lr"]
            ),
            "bodygen/ppo/critic_learning_rate": float(
                self.critic_optimizer.param_groups[0]["lr"]
            ),
            "bodygen/ppo/stored_transitions": float(batch.size),
            "bodygen/ppo/design_transitions": float(batch.design_size),
            "bodygen/ppo/control_transitions": float(batch.control_size),
        }

    def _rng_state(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy_legacy": np.random.get_state(),
            "numpy_generator": copy.deepcopy(self.rng.bit_generator.state),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
            "torch_generator": self.torch_generator.get_state(),
        }

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy_legacy"])
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["numpy_generator"]
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        self.torch_generator.set_state(state["torch_generator"])

    def _check_step_audit(self) -> None:
        if (
            self.trajectory_environment_steps
            + self.synchronization_waste_steps
            != self.environment_steps
        ):
            raise RuntimeError(
                "BodyGen trajectory and synchronization counters do not sum"
            )
        if (
            self.ppo_used_environment_steps
            + self.discarded_trajectory_environment_steps
            != self.trajectory_environment_steps
        ):
            raise RuntimeError(
                "BodyGen used and discarded trajectory counters do not sum"
            )
        if self.peak_parallel_envs > int(
            self.config["budget"]["parallel_envs"]
        ):
            raise RuntimeError("BodyGen exceeded the parallel environment cap")

    def checkpoint_state(self) -> dict[str, Any]:
        self._check_step_audit()
        return {
            "format_version": FORMAT_VERSION,
            "method": "bodygen",
            "config": self.config,
            "training_seed": self.seed,
            "run_identity": self.run_identity,
            "networks": self.networks.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "design_return_normalizer": (
                self.design_return_normalizer.state_dict()
            ),
            "control_return_normalizer": (
                self.control_return_normalizer.state_dict()
            ),
            "environment_steps": self.environment_steps,
            "trajectory_environment_steps": (
                self.trajectory_environment_steps
            ),
            "ppo_used_environment_steps": self.ppo_used_environment_steps,
            "discarded_trajectory_environment_steps": (
                self.discarded_trajectory_environment_steps
            ),
            "synchronization_waste_steps": (
                self.synchronization_waste_steps
            ),
            "used_design_transitions": self.used_design_transitions,
            "discarded_design_transitions": (
                self.discarded_design_transitions
            ),
            "parallel_envs": self.peak_parallel_envs,
            "parallel_env_cap": int(self.config["budget"]["parallel_envs"]),
            "peak_parallel_envs": self.peak_parallel_envs,
            "completed_updates": self.completed_updates,
            "completed_episodes": self.completed_episodes,
            "reward_window": list(self.reward_window),
            "best_native_return": self.best_native_return,
            "wall_seconds": self.wall_seconds,
            "rng": self._rng_state(),
            "resume_boundary": "between_updates",
        }

    def save_checkpoint(
        self,
        *,
        final: bool = False,
        best: bool = False,
    ) -> Path:
        if final and best:
            raise ValueError("a checkpoint cannot be both final and best")
        if final:
            name = "final.pth"
        elif best:
            name = "best.pth"
        else:
            name = f"update_{self.completed_updates:04d}.pth"
        path = self.run_dir / "checkpoints" / name
        temporary = path.with_suffix(".tmp")
        torch.save(self.checkpoint_state(), temporary)
        temporary.replace(path)
        return path

    def _load_checkpoint(self, path: Path) -> None:
        # Keep checkpoint tensors on the host while restoring. Network and
        # optimizer loaders copy their state to the parameter device, whereas
        # Python/Torch RNG state must remain CPU byte tensors.
        state = torch.load(
            path.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if state.get("method") != "bodygen":
            raise ValueError("resume checkpoint is not BodyGen")
        if int(state.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError("unsupported BodyGen checkpoint format")
        if not resume_configs_match(state["config"], self.config):
            raise ValueError(
                "resume config changes BodyGen's algorithm or budget"
            )
        if state.get("resume_boundary") != "between_updates":
            raise ValueError(
                "BodyGen can resume only from a between-update checkpoint"
            )

        self.networks.load_state_dict(state["networks"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.design_return_normalizer.load_state_dict(
            state["design_return_normalizer"]
        )
        self.control_return_normalizer.load_state_dict(
            state["control_return_normalizer"]
        )
        self.run_identity = str(state["run_identity"])
        self.environment_steps = int(state["environment_steps"])
        self.trajectory_environment_steps = int(
            state["trajectory_environment_steps"]
        )
        self.ppo_used_environment_steps = int(
            state["ppo_used_environment_steps"]
        )
        self.discarded_trajectory_environment_steps = int(
            state["discarded_trajectory_environment_steps"]
        )
        self.synchronization_waste_steps = int(
            state["synchronization_waste_steps"]
        )
        self.used_design_transitions = int(
            state["used_design_transitions"]
        )
        self.discarded_design_transitions = int(
            state["discarded_design_transitions"]
        )
        self.peak_parallel_envs = int(state["peak_parallel_envs"])
        self.completed_updates = int(state["completed_updates"])
        self.completed_episodes = int(state["completed_episodes"])
        self.reward_window = deque(
            (float(value) for value in state["reward_window"]),
            maxlen=REWARD_WINDOW,
        )
        self.best_native_return = float(state["best_native_return"])
        self.accumulated_wall_seconds = float(state["wall_seconds"])
        self._restore_rng_state(state["rng"])
        self._check_step_audit()
        if self.environment_steps > self.target_environment_steps:
            raise ValueError(
                "resume checkpoint already exceeds the configured budget"
            )

    def _method_for_evaluation(
        self,
        checkpoint_path: Path,
        *,
        final: bool,
    ) -> Any:
        from .method import BodyGenMethod

        return BodyGenMethod(
            state=self.checkpoint_state(),
            device=self.device,
            run_dir=self.run_dir,
            run_config_path=self.run_dir / "config.yaml",
            run_config=self.config,
            checkpoint_path=checkpoint_path,
            checkpoint_label=(
                "final"
                if final
                else f"update_{self.completed_updates}"
            ),
        )

    def _evaluate(
        self,
        checkpoint_path: Path,
        *,
        native: bool,
        final: bool,
    ) -> float:
        from benchmarks.evaluate import evaluate_return

        method = self._method_for_evaluation(
            checkpoint_path,
            final=final,
        )
        if native:
            schedule = self.config["native_evaluation"]
            deterministic_sampler = getattr(
                method,
                "sample_pairs_deterministic",
                None,
            )
            if not callable(deterministic_sampler):
                raise RuntimeError(
                    "BodyGenMethod lacks sample_pairs_deterministic required "
                    "by native_evaluation"
                )
            method.sample_pairs = deterministic_sampler
            pairs = int(schedule["episodes"])
            episodes_per_pair = 1
            morphology_seed = int(schedule["morphology_seed"])
            rollout_seed = int(schedule["rollout_seed"])
        else:
            schedule = self.config["training_evaluation"]
            pairs = int(schedule["pairs"])
            episodes_per_pair = int(schedule["episodes_per_pair"])
            morphology_seed = int(schedule["seeds"]["morphology"])
            rollout_seed = int(schedule["seeds"]["rollout"])

        expected_return, _, episodes = evaluate_return(
            method,
            pairs=pairs,
            episodes_per_pair=episodes_per_pair,
            morphology_seed=morphology_seed,
            rollout_seed=rollout_seed,
        )
        evaluated_episodes = int(np.prod(episodes.returns.shape))
        self.logger.log_evaluation(
            "bodygen/native_return" if native else "rewards/step_eval",
            expected_return,
            environment_steps=self.environment_steps,
            update=self.completed_updates,
            episodes=evaluated_episodes,
        )
        return expected_return

    def _run_optional_evaluations(
        self,
        checkpoint_path: Path,
        *,
        final: bool,
    ) -> None:
        rng_state = self._rng_state()
        try:
            native = self.config["native_evaluation"]
            native_due = (
                self.completed_updates % int(native["every_updates"]) == 0
            )
            if native["enabled"] and (
                native_due or (final and native["evaluate_final"])
            ):
                native_return = self._evaluate(
                    checkpoint_path,
                    native=True,
                    final=final,
                )
                if native_return > self.best_native_return:
                    self.best_native_return = native_return
                    # ``best.pth`` is diagnostic only, but it should still be
                    # a deterministic resume boundary. Save the state from
                    # immediately before monitoring consumed any RNG.
                    self._restore_rng_state(rng_state)
                    self.save_checkpoint(best=True)

            shared = self.config["training_evaluation"]
            shared_due = (
                self.completed_updates % int(shared["every_updates"]) == 0
            )
            if shared["enabled"] and (
                shared_due or (final and shared["evaluate_final"])
            ):
                self._evaluate(
                    checkpoint_path,
                    native=False,
                    final=final,
                )
        finally:
            # Monitoring must not alter sampling, optimisation, or reset noise.
            self._restore_rng_state(rng_state)

    def _account_collection(self, collection: CollectionResult) -> None:
        self.environment_steps += collection.physics_steps
        self.trajectory_environment_steps += (
            collection.trajectory_physics_steps
        )
        self.synchronization_waste_steps += (
            collection.synchronization_waste_steps
        )
        self.peak_parallel_envs = max(
            self.peak_parallel_envs,
            collection.peak_parallel_envs,
        )
        self.completed_episodes += len(collection.episodes)
        if collection.complete:
            self.ppo_used_environment_steps += (
                collection.trajectory_physics_steps
            )
            self.used_design_transitions += collection.design_transitions
        else:
            self.discarded_trajectory_environment_steps += (
                collection.trajectory_physics_steps
            )
            self.discarded_design_transitions += (
                collection.design_transitions
            )
        self._check_step_audit()

    def _audit_metrics(self) -> dict[str, float]:
        """Canonical cumulative progress/resource metrics shared with NGE."""
        wall_seconds = self.wall_seconds
        return {
            "benchmark/training/environment_steps": float(
                self.environment_steps
            ),
            "benchmark/training/trajectory_environment_steps": float(
                self.trajectory_environment_steps
            ),
            "benchmark/training/ppo_used_environment_steps": float(
                self.ppo_used_environment_steps
            ),
            "benchmark/training/discarded_trajectory_environment_steps": float(
                self.discarded_trajectory_environment_steps
            ),
            "benchmark/training/synchronization_waste_steps": float(
                self.synchronization_waste_steps
            ),
            "benchmark/training/parallel_envs": float(
                self.peak_parallel_envs
            ),
            "benchmark/training/update": float(self.completed_updates),
            "benchmark/training/wall_seconds": wall_seconds,
            "benchmark/training/env_steps_per_second": float(
                self.environment_steps / max(wall_seconds, 1.0e-8)
            ),
            "benchmark/resource/trainable_parameters": float(
                sum(
                    parameter.numel()
                    for parameter in self.networks.parameters()
                    if parameter.requires_grad
                )
            ),
            "benchmark/resource/peak_device_bytes": float(
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda" and torch.cuda.is_available()
                else 0
            ),
            "benchmark/resource/peak_host_kib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "bodygen/steps/environment": float(self.environment_steps),
            "bodygen/steps/trajectory": float(
                self.trajectory_environment_steps
            ),
            "bodygen/steps/ppo_used": float(
                self.ppo_used_environment_steps
            ),
            "bodygen/steps/discarded_trajectory": float(
                self.discarded_trajectory_environment_steps
            ),
            "bodygen/steps/synchronization_waste": float(
                self.synchronization_waste_steps
            ),
            "bodygen/steps/used_design_transitions": float(
                self.used_design_transitions
            ),
            "bodygen/steps/discarded_design_transitions": float(
                self.discarded_design_transitions
            ),
        }

    def train(self) -> Path:
        """Train to the exact physics budget and return ``final.pth``."""
        final_path: Path | None = None
        try:
            while self.environment_steps < self.target_environment_steps:
                collection = self.collect_batch()
                self._account_collection(collection)
                if not collection.complete:
                    if self.environment_steps != self.target_environment_steps:
                        raise RuntimeError(
                            "incomplete BodyGen batch before exact budget end"
                        )
                    partial_returns = tuple(
                        episode.episode_return
                        for episode in collection.episodes
                    )
                    self.reward_window.extend(partial_returns)
                    rolling_reward = (
                        float(np.mean(self.reward_window))
                        if self.reward_window
                        else None
                    )
                    partial_metrics = {
                        **self._audit_metrics(),
                        "bodygen/collection/final_partial_discard": 1.0,
                        "bodygen/collection/complete_episodes": float(
                            len(partial_returns)
                        ),
                    }
                    if partial_returns:
                        partial_metrics.update(
                            {
                                "bodygen/collection/return_mean": float(
                                    np.mean(partial_returns)
                                ),
                                "bodygen/collection/return_std": float(
                                    np.std(partial_returns)
                                ),
                            }
                        )
                    self.logger.log_update(
                        partial_metrics,
                        environment_steps=self.environment_steps,
                        update=self.completed_updates,
                        wall_seconds=self.wall_seconds,
                        completed_reward=rolling_reward,
                    )
                    print(
                        "[bodygen] exact budget ended inside a collection; "
                        f"discarded {collection.trajectory_physics_steps:,} "
                        "trajectory physics steps",
                        flush=True,
                    )
                    break

                prepared = self._prepare_batch(collection.episodes)
                ppo_metrics = self._ppo_update(prepared)
                self.completed_updates += 1
                self.reward_window.extend(prepared.completed_returns)
                rolling_reward = (
                    float(np.mean(self.reward_window))
                    if self.reward_window
                    else None
                )
                metrics = {
                    **ppo_metrics,
                    **self._audit_metrics(),
                    "bodygen/collection/complete_episodes": float(
                        len(collection.episodes)
                    ),
                    "bodygen/collection/return_mean": float(
                        np.mean(prepared.completed_returns)
                    ),
                    "bodygen/collection/return_std": float(
                        np.std(prepared.completed_returns)
                    ),
                    "bodygen/collection/synchronization_waste_fraction": float(
                        collection.synchronization_waste_steps
                        / max(collection.physics_steps, 1)
                    ),
                    "bodygen/collection/design_transitions": float(
                        collection.design_transitions
                    ),
                }
                self.logger.log_update(
                    metrics,
                    environment_steps=self.environment_steps,
                    update=self.completed_updates,
                    wall_seconds=self.wall_seconds,
                    completed_reward=rolling_reward,
                )
                print(
                    "[bodygen] "
                    f"update={self.completed_updates} "
                    f"steps={self.environment_steps:,}/"
                    f"{self.target_environment_steps:,} "
                    f"episodes={len(collection.episodes)} "
                    f"reward={rolling_reward:.3f}",
                    flush=True,
                )

                checkpoint_path = (
                    self.run_dir
                    / "checkpoints"
                    / f"update_{self.completed_updates:04d}.pth"
                )
                if (
                    self.completed_updates
                    % int(self.config["checkpoint"]["every_updates"])
                    == 0
                ):
                    checkpoint_path = self.save_checkpoint()
                evaluations_enabled = (
                    self.config["native_evaluation"]["enabled"]
                    or self.config["training_evaluation"]["enabled"]
                )
                if evaluations_enabled:
                    # Evaluation reads the in-memory frozen state. It must not
                    # force an otherwise-unscheduled full resume checkpoint.
                    self._run_optional_evaluations(
                        checkpoint_path,
                        final=False,
                    )
                del prepared, collection
                torch.cuda.empty_cache()

            if self.environment_steps != self.target_environment_steps:
                raise RuntimeError(
                    "BodyGen stopped without consuming the exact physics budget"
                )
            final_path = self.save_checkpoint(final=True)
            self.logger.log_update(
                {
                    **self._audit_metrics(),
                    "bodygen/steps/final_checkpoint": 1.0,
                },
                environment_steps=self.environment_steps,
                update=self.completed_updates,
                wall_seconds=self.wall_seconds,
                completed_reward=None,
            )
            self._run_optional_evaluations(final_path, final=True)
            print(
                "[bodygen] complete "
                f"steps={self.environment_steps:,} "
                f"updates={self.completed_updates} "
                f"final={final_path}",
                flush=True,
            )
            return final_path
        finally:
            # Include this invocation's elapsed time in any later explicit
            # checkpoint and release log file handles even after VSim errors.
            self.accumulated_wall_seconds = self.wall_seconds
            self.started = time.perf_counter()
            self.logger.close()
