"""The complete benchmark flow: load, check, sample, roll out, score and save."""
from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .data import EpisodeResults, EvaluationPairs
from .metrics import calculate_metrics


@dataclass
class JobResult:
    """Everything retained from evaluating one trained run and checkpoint."""

    method_name: str
    run_name: str
    run_dir: Path
    run_config_path: Path
    checkpoint_label: str
    checkpoint_path: Path
    training_seed: int
    training_steps: int
    parallel_envs: int
    reporting_seed: int
    budget_compliant: bool
    seeds: dict[str, int]
    metrics: dict[str, float]
    summary: dict[str, Any]
    pairs: EvaluationPairs
    episodes: EpisodeResults
    diversity_pairs: EvaluationPairs
    provenance_paths: tuple[Path, ...]


def parse_run_job(
    specification: str,
    *,
    methods: set[str],
    default_method: str,
    default_checkpoints: str,
) -> tuple[str, str, str]:
    """Parse ``METHOD@CHECKPOINTS=RUN`` while keeping plain RUNs simple.

    The optional per-run checkpoint list lets one invocation compare methods
    whose checkpoint counters use different units, such as CoDesign epochs and
    NGE generations.
    """
    prefix, separator, run = specification.partition("=")
    method, marker, checkpoints = prefix.partition("@")
    if separator and method in methods:
        if not run:
            raise ValueError(f"run path is empty in {specification!r}")
        if marker and not checkpoints:
            raise ValueError(f"checkpoint list is empty in {specification!r}")
        return (
            method,
            run,
            checkpoints if marker else default_checkpoints,
        )
    return default_method, specification, default_checkpoints


def load_config(path: str | Path, *, preset: str | None = None) -> dict[str, Any]:
    """Load the single benchmark YAML and select its paper or smoke settings."""
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a YAML mapping")

    evaluation = config.get("evaluation", {})
    chosen = preset or evaluation.get("preset", "paper")
    presets = evaluation.get("presets", {})
    if chosen not in presets:
        raise ValueError(f"unknown evaluation preset: {chosen}")
    shared = {
        key: value
        for key, value in evaluation.items()
        if key not in {"preset", "presets"}
    }
    config["evaluation"] = {"preset": chosen, **shared, **presets[chosen]}
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Check only comparison rules whose violation could invalidate a result."""
    methods = {"codesign", "fixed_body", "uniform_action", "nge", "bodygen"}
    if config.get("method") not in methods:
        raise ValueError(
            "method must be codesign, fixed_body, uniform_action, nge or bodygen"
        )

    requirements = config.get("paper_run_requirements", {})
    if int(requirements.get("environment_steps", 0)) <= 0:
        raise ValueError(
            "paper_run_requirements.environment_steps must be positive"
        )
    if int(requirements.get("parallel_envs", 0)) <= 0:
        raise ValueError("paper_run_requirements.parallel_envs must be positive")
    training_seeds = requirements.get("training_seeds", [])
    if len(training_seeds) != 5 or len(set(training_seeds)) != 5:
        raise ValueError(
            "paper_run_requirements.training_seeds must contain five unique seeds"
        )

    evaluation = config.get("evaluation", {})
    for key in ("pairs", "episodes_per_pair", "top_k", "design_samples"):
        if int(evaluation.get(key, 0)) <= 0:
            raise ValueError(f"evaluation.{key} must be positive")
    if int(evaluation["top_k"]) > int(evaluation["pairs"]):
        raise ValueError("evaluation.top_k cannot exceed evaluation.pairs")
    seeds = evaluation.get("seeds", {})
    for name in ("morphology", "rollout", "diversity"):
        if isinstance(seeds.get(name), bool) or not isinstance(seeds.get(name), int):
            raise ValueError(f"evaluation.seeds.{name} must be an integer")

    logging = config.get("logging", {})
    if logging.get("tensorboard") is not True:
        raise ValueError("TensorBoard must remain enabled")
    wandb = logging.get("wandb", {})
    if wandb.get("enabled") and not wandb.get("project"):
        raise ValueError("logging.wandb.project is required when W&B is enabled")
    if wandb.get("mode") not in {"online", "offline"}:
        raise ValueError("logging.wandb.mode must be online or offline")


def evaluation_seeds(
    evaluation: dict[str, Any],
    training: int,
    reporting: int,
) -> dict[str, int]:
    """Return the exact seed integers written in the benchmark configuration."""
    configured = evaluation["seeds"]
    return {
        "training": training,
        "reporting": reporting,
        "morphology": int(configured["morphology"]),
        "rollout": int(configured["rollout"]),
        "diversity": int(configured["diversity"]),
    }


@torch.no_grad()
def run_episodes(
    method: Any,
    environment: Any,
    pairs: EvaluationPairs,
    episodes_per_pair: int,
) -> EpisodeResults:
    """Evaluate every fixed pair for exactly the requested completed episodes."""
    if int(environment.total_num_envs) != pairs.size:
        raise ValueError("VSim environment count must match the sampled pair count")

    method.install_pairs(environment, pairs)
    observation, _ = environment.reset()
    begin_rollout = getattr(method, "begin_rollout", None)
    if callable(begin_rollout):
        begin_rollout(pairs)
    device = observation.device
    shape = (pairs.size, episodes_per_pair)
    returns = torch.full(shape, torch.nan, device=device)
    falls = torch.full(shape, torch.nan, device=device)
    lengths = torch.full(shape, torch.nan, device=device)
    # Most shared controllers use float32, whereas faithful BodyGen keeps its
    # critic in the upstream float64 dtype. Allocate this lazily so recording a
    # critic value never narrows it or rejects an otherwise valid method.
    start_values: torch.Tensor | None = None

    current_return = torch.zeros(pairs.size, device=device)
    current_length = torch.zeros(pairs.size, device=device)
    completed = torch.zeros(pairs.size, dtype=torch.long, device=device)
    episode_start = torch.ones(pairs.size, dtype=torch.bool, device=device)
    rows = torch.arange(pairs.size, device=device)

    step_limit = (episodes_per_pair + 2) * int(environment.max_episode_length)
    for _ in range(step_limit):
        active = completed < episodes_per_pair
        if not bool(active.any()):
            break

        action, value = method.deterministic_action(observation)
        capture = episode_start & active
        if value is not None:
            if start_values is None:
                start_values = torch.full(
                    shape,
                    torch.nan,
                    device=device,
                    dtype=value.dtype,
                )
            if bool(capture.any()):
                start_values[rows[capture], completed[capture]] = (
                    value[capture].to(start_values)
                )

        observation, reward, terminated, truncated, _ = environment.step(action)
        reward = reward.squeeze(-1) if reward.ndim > 1 else reward
        done = terminated | truncated
        reset_controllers = getattr(method, "reset_controllers", None)
        if callable(reset_controllers):
            reset_controllers(done)

        # AntMultiMorphEnv exposes done one call after the terminal transition.
        # On this notification call the lane has already been reset and
        # ``reward`` belongs to that reset, not to the completed episode. The
        # real terminal reward/length were accumulated on the preceding call.
        finished = done & active
        if bool(finished.any()):
            row = rows[finished]
            episode = completed[finished]
            returns[row, episode] = current_return[finished]
            falls[row, episode] = terminated[finished].float()
            lengths[row, episode] = current_length[finished]
            completed[finished] += 1

        retained = active & ~done
        current_return += torch.where(retained, reward, 0)
        current_length += retained.float()
        current_return = torch.where(done, 0, current_return)
        current_length = torch.where(done, 0, current_length)
        episode_start = done

    if bool((completed != episodes_per_pair).any()):
        raise RuntimeError("not every pair completed the requested episodes")

    values = None
    if start_values is not None:
        if bool(torch.isnan(start_values).any()):
            raise RuntimeError("critic values were missing at some episode starts")
        values = start_values.cpu().numpy()
    return EpisodeResults(
        returns.cpu().numpy(),
        falls.cpu().numpy(),
        lengths.cpu().numpy(),
        values,
    )


def evaluate_return(
    method: Any,
    *,
    pairs: int,
    episodes_per_pair: int,
    morphology_seed: int,
    rollout_seed: int,
) -> tuple[float, EvaluationPairs, EpisodeResults]:
    """Run the common fixed-pair evaluation and return its expected return.

    This is the small reusable core shared by the final benchmark and optional
    evaluations during training. Evaluation simulator steps are deliberately
    not added to a method's training budget.
    """
    sampled_pairs = method.sample_pairs(int(pairs), int(morphology_seed))
    environment = method.create_environment(int(pairs), int(rollout_seed))
    try:
        episodes = run_episodes(
            method,
            environment,
            sampled_pairs,
            int(episodes_per_pair),
        )
    finally:
        close_environment = getattr(environment, "close", None)
        if callable(close_environment):
            close_environment()
    pair_returns = episodes.returns.mean(axis=1)
    expected_return = float(np.sum(pair_returns * sampled_pairs.weights))
    return expected_return, sampled_pairs, episodes


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(project_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _write_pairs(path: Path, result: JobResult) -> None:
    """Save the bodies and every episode for one comparison row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "counts": result.pairs.counts,
        "eff_sub": result.pairs.eff_sub,
        "cap_sub": result.pairs.cap_sub,
        "controller_ids": result.pairs.controller_ids,
        "weights": result.pairs.weights,
        "episode_returns": result.episodes.returns,
        "episode_falls": result.episodes.falls,
        "episode_lengths": result.episodes.lengths,
        "design_counts": result.diversity_pairs.counts,
        "design_eff_sub": result.diversity_pairs.eff_sub,
        "design_cap_sub": result.diversity_pairs.cap_sub,
    }
    if result.episodes.start_values is not None:
        arrays["episode_start_values"] = result.episodes.start_values
    np.savez_compressed(path, **arrays)


def _log_metrics(
    config: dict[str, Any],
    log_directory: Path,
    run_name: str,
    metrics: dict[str, float],
    step: int,
) -> None:
    """Write TensorBoard locally and mirror to W&B only when explicitly enabled."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("TensorBoard is required for benchmark evaluation") from error

    writer = SummaryWriter(str(log_directory))
    try:
        for name, value in metrics.items():
            writer.add_scalar(name, value, step)
        writer.flush()
    finally:
        writer.close()

    wandb_config = config["logging"]["wandb"]
    if not wandb_config["enabled"]:
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B is enabled but the wandb package is unavailable") from error
    run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config.get("entity"),
        group=wandb_config.get("group"),
        name=run_name,
        mode=wandb_config["mode"],
        config=config,
        reinit=True,
    )
    try:
        run.log(metrics, step=step)
    finally:
        run.finish()


def evaluate_method(
    config: dict[str, Any],
    method: Any,
) -> JobResult:
    """Evaluate one loaded run/checkpoint using the shared comparison rules."""
    requirements = config["paper_run_requirements"]
    required_steps = int(requirements["environment_steps"])
    required_envs = int(requirements["parallel_envs"])
    # ``parallel_envs`` is a shared resource ceiling, not a requirement that
    # every algorithm collect at the same width. Methods such as BodyGen need
    # temporal depth and may therefore use fewer simultaneous VSim lanes.
    environment_width_matches = 0 < method.parallel_envs <= required_envs
    budget_matches = (
        method.training_steps == required_steps
        and environment_width_matches
        and bool(getattr(method, "paper_eligible", True))
    )
    if config["checks"]["enforce_training_budget"] and not budget_matches:
        raise ValueError(
            "training budget mismatch: "
            f"checkpoint has {method.training_steps:,} steps/"
            f"{method.parallel_envs:,} envs; "
            f"benchmark requires exactly {required_steps:,} steps and "
            f"0 < peak envs <= {required_envs:,}, from a paper-eligible run"
        )

    reporting_seed = method.training_seed
    if config["checks"]["enforce_training_seed"]:
        if reporting_seed not in requirements["training_seeds"]:
            raise ValueError("training seed is not in the paper run requirements")

    evaluation = config["evaluation"]
    seeds = evaluation_seeds(
        evaluation,
        method.training_seed,
        reporting_seed,
    )
    started = time.perf_counter()
    expected_return, pairs, episodes = evaluate_return(
        method,
        pairs=int(evaluation["pairs"]),
        episodes_per_pair=int(evaluation["episodes_per_pair"]),
        morphology_seed=seeds["morphology"],
        rollout_seed=seeds["rollout"],
    )
    diversity_pairs = method.sample_designs(
        int(evaluation["design_samples"]),
        seeds["diversity"],
    )

    wall_seconds = time.perf_counter() - started
    metrics = calculate_metrics(
        pairs,
        episodes,
        diversity_pairs,
        top_k=int(evaluation["top_k"]),
    )
    if not np.isclose(metrics["benchmark/return/expected"], expected_return):
        raise RuntimeError("shared evaluation return calculation disagrees")
    metrics["rewards/step_eval"] = expected_return
    metrics.update(
        {
            "benchmark/eval/wall_seconds": wall_seconds,
            "benchmark/resource/trainable_parameters": float(
                method.trainable_parameters
            ),
        }
    )

    summary = {
        "protocol_version": config["protocol_version"],
        "method": method.name,
        "run": method.run_dir.name,
        "epoch": method.checkpoint_label,
        "training_seed": method.training_seed,
        "reporting_seed": reporting_seed,
        "training_environment_steps": method.training_steps,
        "budget_compliant": budget_matches,
        **metrics,
    }
    return JobResult(
        method_name=method.name,
        run_name=method.run_dir.name,
        run_dir=method.run_dir,
        run_config_path=method.run_config_path,
        checkpoint_label=method.checkpoint_label,
        checkpoint_path=method.checkpoint_path,
        training_seed=method.training_seed,
        training_steps=method.training_steps,
        parallel_envs=method.parallel_envs,
        reporting_seed=reporting_seed,
        budget_compliant=budget_matches,
        seeds=seeds,
        metrics=metrics,
        summary=summary,
        pairs=pairs,
        episodes=episodes,
        diversity_pairs=diversity_pairs,
        provenance_paths=tuple(
            Path(path) for path in getattr(method, "provenance_paths", ())
        ),
    )


def _safe_id(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    if not identifier:
        raise ValueError("result identifier contains no usable characters")
    return identifier


def evaluate_runs(
    config: dict[str, Any],
    methods: Any,
    *,
    project_root: Path,
    destination: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Evaluate multiple loaded jobs and write one side-by-side comparison bundle."""
    validate_config(config)
    device = torch.device(config["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"evaluation requested {device}, but CUDA is unavailable")

    if destination is None:
        name = config["output"].get("evaluation_id")
        if name is None:
            name = f"{time.strftime('%Y%m%d_%H%M%S')}_comparison"
        root = Path(config["output"]["root"]).expanduser()
        if not root.is_absolute():
            root = project_root / root
        destination = root / _safe_id(str(name))
    elif not destination.is_absolute():
        destination = project_root / destination
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"comparison output already exists: {destination}")

    summaries: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for method in methods:
        result = evaluate_method(config, method)
        job_id = _safe_id(
            f"{result.method_name}_{result.run_name}_{result.checkpoint_label}"
        )
        if job_id in used_ids:
            raise ValueError(f"duplicate benchmark job: {job_id}")
        used_ids.add(job_id)

        if not summaries:
            destination.mkdir(parents=True)
        raw_path = Path("pairs") / f"{job_id}.npz"
        _write_pairs(destination / raw_path, result)
        _log_metrics(
            config,
            destination / "tensorboard" / job_id,
            job_id,
            result.metrics,
            result.training_steps,
        )
        result.summary["raw_results"] = str(raw_path)
        summaries.append(result.summary)
        jobs.append(
            {
                "method": result.method_name,
                "run": result.run_name,
                "run_directory": str(result.run_dir),
                "run_config": str(result.run_config_path),
                "checkpoint": {
                    "label": result.checkpoint_label,
                    "path": str(result.checkpoint_path),
                    "sha256": _checkpoint_hash(result.checkpoint_path),
                },
                "training_seed": result.training_seed,
                "reporting_seed": result.reporting_seed,
                "training_environment_steps": result.training_steps,
                "parallel_environments": result.parallel_envs,
                "budget_compliant": result.budget_compliant,
                "seeds": result.seeds,
                "raw_results": str(raw_path),
                "provenance": {
                    str(path): _checkpoint_hash(path)
                    for path in getattr(result, "provenance_paths", ())
                },
            }
        )
        del result, method
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not summaries:
        raise ValueError("no runs were selected for benchmark evaluation")

    fields: list[str] = []
    for summary in summaries:
        fields.extend(key for key in summary if key not in fields)
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    manifest = {
        "protocol_version": config["protocol_version"],
        "git": _git_state(project_root),
        "resolved_config": config,
        "jobs": jobs,
    }
    (destination / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    return destination, summaries
