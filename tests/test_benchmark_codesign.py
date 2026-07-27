from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from benchmarks.codesign import (
    CodesignMethod,
    checkpoints_for_run,
    resolve_checkpoint,
    training_budget,
)
from benchmarks.evaluate import load_config
from benchmarks.fixed_body import FixedBodyMethod
from benchmarks.uniform_action import UniformActionMethod
from scripts import eval as legacy_eval
from transformer_rl.train_utils import _load_config
from transformer_rl.uniform_action_agent import UniformActionAgent

ROOT = Path(__file__).resolve().parents[1]


class _Inner:
    tdims = {"raw_tail_off": 2, "raw_tail_dim": 2}

    def __init__(self):
        self.sample_modes = []

    def sample(self, count: int, mode: str = "stochastic"):
        self.sample_modes.append(mode)
        counts = torch.zeros((count, 8))
        counts[:, 0] = 1
        effectors = torch.full((count, 8, 4), -1, dtype=torch.long)
        effectors[:, 0, 0] = torch.randint(0, 3, (count,))
        return {
            "counts": counts,
            "eff_sub": effectors,
            "cap_sub": torch.zeros((count, 8), dtype=torch.long),
        }


class _Network:
    def __init__(self):
        self.net = _Inner()

    def __call__(self, observation):
        value = observation["obs"].sum(dim=-1, keepdim=True)
        return observation["obs"][:, :2], None, value, None


class CodesignBenchmarkTests(unittest.TestCase):
    def test_direnv_exposes_all_vlearn_native_libraries(self) -> None:
        environment = os.environ.copy()
        environment.pop("LD_LIBRARY_PATH", None)
        result = subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                "unset LD_LIBRARY_PATH; "
                "PATH_add() { export PATH=\"$PWD/$1:$PATH\"; }; "
                "source .envrc; "
                "python -c 'import vlearn; print(\"vlearn ok\")'",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("vlearn ok", result.stdout)

    def test_activation_script_exposes_all_vlearn_native_libraries(self) -> None:
        environment = os.environ.copy()
        environment.pop("LD_LIBRARY_PATH", None)
        result = subprocess.run(
            [
                "zsh",
                "-c",
                "source scripts/activate_uv.sh; "
                "python -c 'import vlearn; print(\"vlearn ok\")'",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("vlearn ok", result.stdout)

    def test_launcher_bootstraps_the_current_vlearn_native_libraries(self) -> None:
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = ""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scripts.benchmark_eval; import vlearn; print('vlearn ok')",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("vlearn ok", result.stdout)

    def _method_without_checkpoint_loading(self) -> CodesignMethod:
        method = object.__new__(CodesignMethod)
        method.network = _Network()
        method.observation_normalizer = lambda value: value + 10
        method.device = torch.device("cpu")
        method.checkpoint_path = Path("checkpoint.pth")
        return method

    def test_control_step_matches_the_existing_evaluator(self) -> None:
        method = self._method_without_checkpoint_loading()
        config = load_config(ROOT / "configs/benchmarks/benchmark.yaml")
        tolerance = config["parity"]
        observation = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        expected_action, expected_value = legacy_eval._forward(
            method.network,
            method.observation_normalizer,
            observation,
        )
        action, value = method.deterministic_action(observation)
        torch.testing.assert_close(
            action,
            expected_action,
            rtol=tolerance["relative_tolerance"],
            atol=tolerance["absolute_tolerance"],
        )
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=tolerance["relative_tolerance"],
            atol=tolerance["absolute_tolerance"],
        )

    def test_stochastic_sampling_matches_legacy_sampling_at_the_same_seed(self) -> None:
        method = self._method_without_checkpoint_loading()
        expected = legacy_eval._sample(method.network, 6, "stochastic")
        actual = method._sample(6, legacy_eval.EVAL_SEED)
        np.testing.assert_array_equal(actual.eff_sub, expected["eff_sub"].numpy())

    def test_early_checkpoint_is_charged_only_for_steps_actually_consumed(self) -> None:
        run_config = {
            "params": {
                "config": {
                    "max_epochs": 3000,
                    "num_actors": 4096,
                    "horizon_length": 16,
                }
            }
        }
        steps, environments = training_budget(run_config, "ep_50")
        self.assertEqual(steps, 50 * 4096 * 16)
        self.assertEqual(environments, 4096)

    def test_bare_best_checkpoint_is_not_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint = run_dir / "codesign.pth"
            checkpoint.write_bytes(b"best")
            run_config = {"params": {"config": {"max_epochs": 3000}}}
            with self.assertRaisesRegex(ValueError, "bare best"):
                resolve_checkpoint(run_dir, run_config, checkpoint)

    def test_epoch_selection_matches_the_existing_evaluator_style(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint_dir = run_dir / "nn"
            checkpoint_dir.mkdir()
            (run_dir / "config.yaml").write_text(
                "params:\n"
                "  config:\n"
                "    max_epochs: 20\n"
            )
            for epoch in (10, 20):
                (checkpoint_dir / f"last_model_ep_{epoch}_rew_0.pth").write_bytes(b"x")
            config = {"run_config": None}
            selected = checkpoints_for_run(config, run_dir, "10,20")
            self.assertEqual([label for label, _ in selected], ["ep_10", "ep_20"])
            final = checkpoints_for_run(config, run_dir, "final")
            self.assertEqual(final[0][0], "final_ep_20")

    def test_fixed_body_always_returns_the_canonical_pair(self) -> None:
        method = object.__new__(FixedBodyMethod)
        method.network = _Network()
        method.network.net.max_limb_length = 4
        method.run_config = {"env": {"base_legs": [1, 4, 6]}}
        method.checkpoint_path = Path("fixed.pth")

        first = method.sample_pairs(3, seed=1)
        second = method.sample_pairs(3, seed=999)

        np.testing.assert_array_equal(first.counts, second.counts)
        np.testing.assert_array_equal(first.counts[0], [2, 0, 0, 2, 0, 2, 0, 0])
        np.testing.assert_array_equal(first.eff_sub[0, 0, :2], [0, 1])
        self.assertTrue(np.all(first.cap_sub == 0))

    def test_uniform_action_evaluation_uses_uniform_mode_and_exact_seed(self) -> None:
        method = object.__new__(UniformActionMethod)
        method.network = _Network()
        method.device = torch.device("cpu")
        method.checkpoint_path = Path("uniform.pth")

        first = method.sample_pairs(8, seed=17)
        second = method.sample_pairs(8, seed=17)

        self.assertEqual(method.network.net.sample_modes, ["uniform", "uniform"])
        np.testing.assert_array_equal(first.counts, second.counts)
        np.testing.assert_array_equal(first.eff_sub, second.eff_sub)
        np.testing.assert_array_equal(first.cap_sub, second.cap_sub)

    def test_uniform_action_inherits_the_codesign_controller_config(self) -> None:
        codesign = _load_config(ROOT / "configs/ppo_ant_codesign_single.yaml")
        uniform = _load_config(ROOT / "configs/ppo_ant_uniform_action.yaml")

        self.assertEqual(
            uniform["params"]["algo"]["name"],
            "uniform_action_continuous",
        )
        self.assertEqual(uniform["env"], codesign["env"])
        self.assertEqual(
            uniform["params"]["network"],
            codesign["params"]["network"],
        )
        self.assertEqual(
            uniform["params"]["config"],
            codesign["params"]["config"],
        )

    def test_uniform_training_resamples_without_a_generator_update(self) -> None:
        class FakeEnvironment:
            max_episode_length = 1
            total_num_envs = 2
            _sample_morphs = True

            def __init__(self):
                self.installed = None
                self.resample_count = 0

            def set_next(self, counts, effectors, caps):
                self.installed = (counts, effectors, caps)

            def resample(self):
                self.resample_count += 1

        class FakeGenerator:
            def __init__(self):
                self.modes = []

            def sample(self, count, mode):
                self.modes.append(mode)
                counts = torch.ones((count, 8), dtype=torch.long)
                effectors = torch.zeros((count, 8, 4), dtype=torch.long)
                caps = torch.zeros((count, 8), dtype=torch.long)
                return {
                    "counts": counts,
                    "presence": counts > 0,
                    "eff_sub": effectors,
                    "cap_sub": caps,
                }

        environment = FakeEnvironment()
        generator = FakeGenerator()
        agent = object.__new__(UniformActionAgent)
        agent.config = {"resample_interval": 1}
        agent.horizon_length = 1
        agent._steps_since_resample = 0
        agent._env = lambda: environment
        agent._window_Ri = lambda: torch.tensor([1.0, 2.0])
        agent._r_scale = 1.0
        agent._net = lambda: generator
        agent._gen_window = 0
        agent.epoch_num = 3
        agent.env_reset = lambda: torch.zeros((2, 1))
        agent.current_rewards = torch.ones(2)
        agent.current_lengths = torch.ones(2)
        agent._ep_ret = torch.ones(2)
        agent._win_ret_sum = torch.ones(2)
        agent._win_ret_cnt = torch.ones(2)
        agent._morph_meta = object()

        # No _resample_update attribute is installed: succeeding proves this
        # control does not invoke the learned generator update.
        with contextlib.redirect_stdout(io.StringIO()):
            agent._maybe_resample()

        self.assertEqual(generator.modes, ["uniform"])
        self.assertEqual(environment.resample_count, 1)
        self.assertEqual(agent._steps_since_resample, 0)
        self.assertTrue(torch.equal(agent._cur_counts, torch.ones((2, 8))))


if __name__ == "__main__":
    unittest.main()
