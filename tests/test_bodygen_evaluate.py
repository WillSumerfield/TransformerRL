from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from benchmarks.bodygen.method import (
    BodyGenMethod,
    checkpoints_for_bodygen_run,
    load_bodygen,
)
from benchmarks.bodygen.credit import ReturnNormalizer
from benchmarks.bodygen.mosat import ACTION_SIZE, OBSERVATION_SIZE, BodyGenNetworks
from benchmarks.evaluate import load_config


ROOT = Path(__file__).resolve().parents[1]
BODYGEN = ROOT / "benchmarks/bodygen"


class BodyGenEvaluatorTests(unittest.TestCase):
    @staticmethod
    def _network_config() -> dict:
        return {
            "hidden_size": 8,
            "blocks": 1,
            "layer_norm": "pre",
            "topology_embeddings": 256,
            "feed_forward_ratio": 2,
            "critic_hidden": [8],
            "initial_control_log_std": -0.5,
        }

    def _checkpoint_state(self) -> dict:
        network = self._network_config()
        config = {
            "network": network,
            "environment": {"base_legs": [1, 4, 6]},
            "training": {"normalize_returns": True},
        }
        model = BodyGenNetworks(
            observation_size=OBSERVATION_SIZE,
            action_size=ACTION_SIZE,
            hidden_size=network["hidden_size"],
            num_blocks=network["blocks"],
            layer_norm=network["layer_norm"],
            topology_embeddings=network["topology_embeddings"],
            feed_forward_ratio=network["feed_forward_ratio"],
            critic_hidden=network["critic_hidden"],
            dtype=torch.float64,
        )
        control_normalizer = ReturnNormalizer(dtype=torch.float64)
        return {
            "method": "bodygen",
            "format_version": 1,
            "config": config,
            "training_seed": 42,
            "environment_steps": 196_608_000,
            "peak_parallel_envs": 20,
            "networks": model.state_dict(),
            "control_return_normalizer": control_normalizer.state_dict(),
        }

    def _method(self, root: Path) -> BodyGenMethod:
        config = {
            "network": self._network_config(),
            "environment": {"base_legs": [1, 4, 6]},
            "training": {"normalize_returns": True},
        }
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        checkpoint = root / "checkpoint.pth"
        torch.save(self._checkpoint_state(), checkpoint)
        return BodyGenMethod(
            state=self._checkpoint_state(),
            device=torch.device("cpu"),
            run_dir=root,
            run_config_path=config_path,
            run_config=config,
            checkpoint_path=checkpoint,
            checkpoint_label="final",
        )

    def test_final_and_numeric_checkpoints_use_native_update_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            checkpoints = run / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "update_0100.pth").write_bytes(b"update")
            (checkpoints / "update_0200.pth").write_bytes(b"update")
            (checkpoints / "final.pth").write_bytes(b"final")

            config = {"run_root": run.parent}
            numeric = checkpoints_for_bodygen_run(config, run, "200,100")
            final = checkpoints_for_bodygen_run(config, run, "final")

        self.assertEqual(
            [label for label, _ in numeric],
            ["update_100", "update_200"],
        )
        self.assertEqual(final[0][0], "final")

    def test_benchmark_config_exposes_bodygen_as_a_plain_method(self) -> None:
        config = load_config(
            ROOT / "configs/benchmarks/benchmark.yaml",
            preset="smoke",
        )
        self.assertEqual(
            config["bodygen"]["run_root"],
            "runs/benchmarks/bodygen/bodygen_mosat",
        )
        self.assertEqual(config["bodygen"]["checkpoint"], "final")

    def test_provenance_files_are_complete_and_hash_the_local_licence(self) -> None:
        provenance = yaml.safe_load((BODYGEN / "upstream.yaml").read_text())
        self.assertEqual(
            provenance["implementation"]["commit"],
            "4e0bdc0c1e528174e19bbe62633f5316f6283db6",
        )
        self.assertEqual(provenance["license"]["spdx"], "Apache-2.0")
        licence = BODYGEN / provenance["license"]["local_copy"]
        self.assertEqual(
            hashlib.sha256(licence.read_bytes()).hexdigest(),
            provenance["license"]["sha256"],
        )
        self.assertTrue((BODYGEN / "ADAPTATIONS.md").is_file())

    def test_sampling_replays_the_literal_seed_and_keeps_equal_weight_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            method = self._method(Path(temporary))
            first = method.sample_pairs(6, 17)
            second = method.sample_pairs(6, 17)

        np.testing.assert_array_equal(first.counts, second.counts)
        np.testing.assert_array_equal(first.eff_sub, second.eff_sub)
        np.testing.assert_array_equal(first.cap_sub, second.cap_sub)
        np.testing.assert_array_equal(
            first.controller_ids,
            np.full(6, "shared"),
        )
        np.testing.assert_allclose(first.weights, 1.0 / 6)

    def test_shared_controller_is_paired_with_every_installed_design(self) -> None:
        class Environment:
            def set_next(self, counts, effectors, caps):
                self.installed = (counts, effectors, caps)

            def resample(self):
                self.resampled = True

        with tempfile.TemporaryDirectory() as temporary:
            method = self._method(Path(temporary))
            pairs = method.sample_pairs(2, 9)
            environment = Environment()
            method.install_pairs(environment, pairs)
            method.begin_rollout(pairs)
            method.control_return_normalizer.count.fill_(1)
            method.control_return_normalizer.variance.fill_(4.0)
            observation = torch.zeros((2, OBSERVATION_SIZE))
            raw_value = method.networks.control(
                observation,
                method._installed_designs,
                deterministic=True,
            ).value
            action, value = method.deterministic_action(
                observation
            )

        self.assertTrue(environment.resampled)
        self.assertEqual(action.shape, (2, ACTION_SIZE))
        self.assertEqual(value.shape, (2,))
        self.assertTrue(torch.all(action.abs() <= 1.0))
        torch.testing.assert_close(
            value,
            raw_value * (2.0 + 1.0e-8),
        )

    def test_foreign_controller_ids_are_rejected_before_vsim_rebuild(self) -> None:
        class Environment:
            def set_next(self, *_args):
                raise AssertionError("invalid pairs must not reach the environment")

        with tempfile.TemporaryDirectory() as temporary:
            method = self._method(Path(temporary))
            pairs = method.sample_pairs(2, 9)
            pairs.controller_ids[0] = "species_1"
            with self.assertRaisesRegex(ValueError, "shared controller"):
                method.install_pairs(Environment(), pairs)

    def test_loader_restores_budget_and_native_network_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            checkpoint_dir = run / "checkpoints"
            checkpoint_dir.mkdir()
            run_config = {
                "network": self._network_config(),
                "environment": {"base_legs": [1, 4, 6]},
                "training": {"normalize_returns": True},
            }
            (run / "config.yaml").write_text(yaml.safe_dump(run_config))
            torch.save(self._checkpoint_state(), checkpoint_dir / "final.pth")

            method = load_bodygen(
                {"run_dir": run, "checkpoint": "final"},
                torch.device("cpu"),
            )

        self.assertEqual(method.training_steps, 196_608_000)
        self.assertEqual(method.parallel_envs, 20)
        self.assertEqual(method.checkpoint_label, "final")
        self.assertTrue(all(path.is_file() for path in method.provenance_paths))

    def test_checkpoint_cannot_be_loaded_with_a_different_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            config = self._checkpoint_state()["config"]
            changed = {
                **config,
                "network": {
                    **config["network"],
                    "layer_norm": "post",
                },
            }
            with self.assertRaisesRegex(ValueError, "does not match"):
                BodyGenMethod(
                    state=self._checkpoint_state(),
                    device=torch.device("cpu"),
                    run_dir=run,
                    run_config_path=run / "config.yaml",
                    run_config=changed,
                    checkpoint_path=run / "checkpoint.pth",
                    checkpoint_label="final",
                )

    def test_checkpoint_can_move_between_cuda_devices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            state = self._checkpoint_state()
            state["config"]["runtime"] = {"device": "cuda:0"}
            run_config = {
                **state["config"],
                "runtime": {"device": "cuda:1"},
            }
            method = BodyGenMethod(
                state=state,
                device=torch.device("cpu"),
                run_dir=run,
                run_config_path=run / "config.yaml",
                run_config=run_config,
                checkpoint_path=run / "checkpoint.pth",
                checkpoint_label="final",
            )

        self.assertEqual(method.checkpoint_label, "final")


if __name__ == "__main__":
    unittest.main()
