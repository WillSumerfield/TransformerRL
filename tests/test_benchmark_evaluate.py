from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from benchmarks.data import EvaluationPairs
from benchmarks.evaluate import (
    evaluation_seeds,
    evaluate_return,
    evaluate_runs,
    load_config,
    parse_run_job,
    run_episodes,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmarks/benchmark.yaml"


def _pairs(count: int = 2) -> EvaluationPairs:
    counts = np.zeros((count, 8), dtype=np.int64)
    counts[:, 0] = 1
    effectors = np.full((count, 8, 4), -1, dtype=np.int64)
    effectors[:, 0, 0] = np.arange(count) % 3
    return EvaluationPairs(
        counts,
        effectors,
        np.zeros((count, 8), dtype=np.int64),
        np.full(count, "fake"),
        np.full(count, 1 / count),
    )


class _Environment:
    max_episode_length = 3

    def __init__(self, count: int):
        self.total_num_envs = count
        self.local_steps = torch.zeros(count, dtype=torch.long)

    def reset(self):
        self.local_steps.zero_()
        return torch.zeros((self.total_num_envs, 4)), {}

    def step(self, _action):
        self.local_steps += 1
        reward = torch.arange(1, self.total_num_envs + 1, dtype=torch.float32)
        done = self.local_steps == 2
        terminated = done & (torch.arange(self.total_num_envs) % 2 == 0)
        truncated = done & ~terminated
        self.local_steps = torch.where(done, 0, self.local_steps)
        return (
            torch.zeros((self.total_num_envs, 4)),
            reward,
            terminated,
            truncated,
            {},
        )


class _ExclusiveEnvironment(_Environment):
    """Model VLearn's rule that only one GymSingleton may exist."""

    active = False

    def __init__(self, count: int):
        if self.__class__.active:
            raise RuntimeError("a second singleton environment was created")
        self.__class__.active = True
        super().__init__(count)

    def close(self):
        self.__class__.active = False


class _Method:
    name = "codesign"
    checkpoint_label = "final_ep_3000"
    training_seed = 42
    training_steps = 196_608_000
    parallel_envs = 4096
    trainable_parameters = 123

    def __init__(self, root: Path):
        self.run_dir = root
        self.run_config_path = root / "config.yaml"
        self.run_config_path.write_text("params: {}\n")
        self.checkpoint_path = root / "checkpoint.pth"
        self.checkpoint_path.write_bytes(b"checkpoint")
        self.installed = False

    def sample_pairs(self, count, _seed):
        return _pairs(count)

    def sample_designs(self, count, _seed):
        return _pairs(count)

    def create_environment(self, count, _seed):
        return _Environment(count)

    def install_pairs(self, _environment, _pairs):
        self.installed = True

    def deterministic_action(self, observation):
        return torch.zeros((len(observation), 1)), torch.full((len(observation),), 3.0)


class _ExclusiveMethod(_Method):
    def create_environment(self, count, _seed):
        return _ExclusiveEnvironment(count)


class BenchmarkEvaluationTests(unittest.TestCase):
    def test_run_job_can_override_checkpoints_for_one_method(self) -> None:
        methods = {"codesign", "fixed_body", "uniform_action", "nge"}

        self.assertEqual(
            parse_run_job(
                "nge@5,10=/tmp/nge-run",
                methods=methods,
                default_method="codesign",
                default_checkpoints="final",
            ),
            ("nge", "/tmp/nge-run", "5,10"),
        )
        self.assertEqual(
            parse_run_job(
                "/tmp/codesign-run",
                methods=methods,
                default_method="codesign",
                default_checkpoints="final",
            ),
            ("codesign", "/tmp/codesign-run", "final"),
        )

    def test_config_uses_the_seed_values_exactly_as_written(self) -> None:
        config = load_config(CONFIG, preset="smoke")
        self.assertEqual(config["method"], "codesign")
        self.assertEqual(config["evaluation"]["pairs"], 4)
        self.assertNotIn("presets", config["evaluation"])

        config["evaluation"]["seeds"] = {
            "morphology": 7,
            "rollout": 8,
            "diversity": 9,
        }
        seeds = evaluation_seeds(config["evaluation"], 42, 42)
        self.assertEqual(
            seeds,
            {
                "training": 42,
                "reporting": 42,
                "morphology": 7,
                "rollout": 8,
                "diversity": 9,
            },
        )

    def test_rollout_retains_every_completed_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            method = _Method(Path(temporary))
            results = run_episodes(method, _Environment(2), _pairs(), 2)
        self.assertTrue(method.installed)
        np.testing.assert_allclose(results.returns, [[2, 2], [4, 4]])
        np.testing.assert_allclose(results.falls, [[1, 1], [0, 0]])
        np.testing.assert_allclose(results.lengths, [[2, 2], [2, 2]])
        np.testing.assert_allclose(results.start_values, 3)

    def test_training_and_final_evaluation_share_expected_return(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            method = _Method(Path(temporary))
            expected_return, pairs, episodes = evaluate_return(
                method,
                pairs=2,
                episodes_per_pair=2,
                morphology_seed=1,
                rollout_seed=1,
            )

        self.assertEqual(expected_return, 3.0)
        self.assertEqual(pairs.size, 2)
        self.assertEqual(episodes.returns.shape, (2, 2))

    def test_multiple_runs_produce_one_comparison_table_and_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first = _Method(first_root)
            second = _Method(second_root)
            second.training_seed = 43
            config = load_config(CONFIG, preset="smoke")
            config["runtime"]["device"] = "cpu"
            config["output"] = {
                "root": str(root / "evals"),
                "evaluation_id": "contract",
            }
            destination, summaries = evaluate_runs(
                config,
                [first, second],
                project_root=ROOT,
            )

            self.assertTrue((destination / "manifest.yaml").is_file())
            self.assertTrue((destination / "summary.csv").is_file())
            self.assertTrue((destination / "tensorboard").is_dir())
            self.assertEqual(len(summaries), 2)
            self.assertEqual(
                summaries[0]["rewards/step_eval"],
                summaries[0]["benchmark/return/expected"],
            )
            raw_file = destination / summaries[0]["raw_results"]
            with np.load(raw_file, allow_pickle=False) as arrays:
                self.assertEqual(arrays["episode_returns"].shape, (4, 2))
            with (destination / "summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["run"] for row in rows], ["first", "second"])
            self.assertTrue(all(row["budget_compliant"] == "True" for row in rows))

    def test_training_budget_mismatch_is_rejected_before_vsim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            method = _Method(Path(temporary))
            method.training_steps = 50 * 4096 * 16
            config = load_config(CONFIG, preset="smoke")
            config["runtime"]["device"] = "cpu"
            with self.assertRaisesRegex(ValueError, "budget mismatch"):
                evaluate_runs(
                    config,
                    [method],
                    project_root=ROOT,
                    destination=Path(temporary) / "eval",
                )

    def test_vsim_environment_is_closed_between_checkpoint_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first = _ExclusiveMethod(first_root)
            second = _ExclusiveMethod(second_root)
            second.training_seed = 43
            config = load_config(CONFIG, preset="smoke")
            config["runtime"]["device"] = "cpu"
            try:
                _, summaries = evaluate_runs(
                    config,
                    [first, second],
                    project_root=ROOT,
                    destination=root / "comparison",
                )
            finally:
                _ExclusiveEnvironment.active = False
            self.assertEqual(len(summaries), 2)


if __name__ == "__main__":
    unittest.main()
