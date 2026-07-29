"""Focused CPU tests for BodyGen's design, MoSAT and Enhanced-TCA core."""
from __future__ import annotations

import math
import copy
import itertools
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import torch

from benchmarks.bodygen.credit import (
    ReturnNormalizer,
    enhanced_temporal_credit_assignment,
)
from benchmarks.bodygen.design import (
    ADD,
    ATTRIBUTE,
    CONTROL,
    DELETE,
    NO_CHANGE,
    TOPOLOGY,
    BodyGenDesign,
    DesignBatchTrace,
    DesignNode,
    apply_attribute_actions,
    apply_topology_actions,
    design_node_features,
    topology_id,
)
from benchmarks.bodygen.mosat import NODE_FEATURE_SIZE, BodyGenNetworks
from benchmarks.bodygen.training import (
    BodyGenTrainer,
    CollectionResult,
    Episode,
    bodygen_worker_target,
    choose_wave_width,
    episode_bootstrap_observation,
    load_bodygen_config,
    record_delayed_vsim_transition,
    resume_configs_match,
    stage_grouped_permutation,
    validate_bodygen_config,
)
from transformer_rl.vocab import CANON_CAP, N_CAP, N_EFF


class BodyGenDesignTests(unittest.TestCase):
    def test_canonical_body_and_first_empty_root_add(self) -> None:
        body = BodyGenDesign.canonical()
        counts, effectors, caps = body.to_arrays()
        np.testing.assert_array_equal(
            counts,
            np.asarray([2, 0, 0, 2, 0, 2, 0, 0]),
        )
        self.assertEqual(effectors.shape, (8, 4))
        self.assertTrue(np.all(caps == CANON_CAP))

        actions = [NO_CHANGE] * body.num_nodes
        actions[0] = ADD
        changed = apply_topology_actions(body, actions)
        changed_counts, _, _ = changed.to_arrays()
        np.testing.assert_array_equal(
            changed_counts,
            np.asarray([2, 1, 0, 2, 0, 2, 0, 0]),
        )
        # The new limb was not present in the action snapshot, so it cannot
        # immediately grow a second node in the same wave.
        self.assertEqual(changed.effectors[1], (0,))

    def test_new_nodes_act_on_the_next_topology_wave(self) -> None:
        body = BodyGenDesign.canonical()
        first_actions = [NO_CHANGE] * body.num_nodes
        first_actions[0] = ADD
        body = apply_topology_actions(body, first_actions)

        second_actions = [NO_CHANGE] * body.num_nodes
        new_node = body.nodes.index(DesignNode(limb=1, depth=0))
        second_actions[new_node] = ADD
        body = apply_topology_actions(body, second_actions)
        self.assertEqual(len(body.effectors[1]), 2)

    def test_five_root_waves_expose_all_eight_slots(self) -> None:
        body = BodyGenDesign.canonical()
        for _ in range(5):
            actions = [NO_CHANGE] * body.num_nodes
            actions[0] = ADD
            body = apply_topology_actions(body, actions)
        counts, _, _ = body.to_arrays()
        self.assertTrue(np.all(counts > 0))
        self.assertEqual(int(counts.sum()), 11)

        # A saturated terminal Add remains the native invalid-action no-op.
        limb_zero_tip = body.nodes.index(DesignNode(0, 1))
        grow = [NO_CHANGE] * body.num_nodes
        grow[limb_zero_tip] = ADD
        body = apply_topology_actions(body, grow)
        self.assertEqual(len(body.effectors[0]), 3)
        saturated_tip = body.nodes.index(DesignNode(0, 2))
        invalid = [NO_CHANGE] * body.num_nodes
        invalid[saturated_tip] = ADD
        self.assertEqual(apply_topology_actions(body, invalid), body)

    def test_snapshot_deletes_never_remove_the_final_effector(self) -> None:
        body = BodyGenDesign(
            effectors=((0,), (1,), (), (), (), (), (), ()),
            caps=(CANON_CAP,) * 8,
        )
        actions = [NO_CHANGE, DELETE, DELETE]
        result = apply_topology_actions(body, actions)
        self.assertEqual(result.num_effectors, 1)
        # Root-first/limb-major application retains the later limb.
        self.assertEqual(result.effectors[0], ())
        self.assertEqual(result.effectors[1], (1,))

        only_one = BodyGenDesign(
            effectors=((0,), (), (), (), (), (), (), ()),
            caps=(CANON_CAP,) * 8,
        )
        unchanged = apply_topology_actions(
            only_one, [NO_CHANGE, DELETE]
        )
        self.assertEqual(unchanged, only_one)

    def test_topope_is_stable_unique_base_nine_and_bounded(self) -> None:
        expected = {
            DesignNode(None, -1): 0,
            DesignNode(0, 0): 1,
            DesignNode(0, 1): 10,
            DesignNode(0, 2): 91,
            DesignNode(7, 0): 8,
            DesignNode(7, 1): 17,
            DesignNode(7, 2): 98,
        }
        for node, identifier in expected.items():
            self.assertEqual(topology_id(node), identifier)

        all_ids = [
            topology_id(DesignNode(limb, depth))
            for limb in range(8)
            for depth in range(3)
        ]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertLess(max(all_ids), 256)

        before = BodyGenDesign.canonical()
        actions = [NO_CHANGE] * before.num_nodes
        actions[0] = ADD
        after = apply_topology_actions(before, actions)
        before_id = before.topology_ids()[
            before.nodes.index(DesignNode(3, 1))
        ]
        after_id = after.topology_ids()[
            after.nodes.index(DesignNode(3, 1))
        ]
        self.assertEqual(before_id, after_id)

    def test_attribute_step_selects_every_effector_and_terminal_cap(self) -> None:
        body = BodyGenDesign.canonical()
        effectors = tuple(
            index % N_EFF for index in range(body.num_effectors)
        )
        caps = tuple(
            (index + 1) % N_CAP
            for index in range(len(body.terminal_nodes))
        )
        result = apply_attribute_actions(body, effectors, caps)
        flat_effectors = tuple(
            kind for chain in result.effectors for kind in chain
        )
        self.assertEqual(flat_effectors, effectors)
        self.assertEqual(
            tuple(result.caps[node.limb] for node in result.terminal_nodes),
            caps,
        )

    def test_design_features_need_no_simulator_or_validity_flags(self) -> None:
        body = BodyGenDesign.canonical()
        features = design_node_features(body)
        self.assertEqual(features.shape, (body.num_nodes, 21))
        self.assertEqual(features.dtype, torch.float64)
        self.assertEqual(features[0, 0].item(), 1.0)
        self.assertEqual(features[0, 2].item(), 1.0)


class MoSATTests(unittest.TestCase):
    def make_small_networks(self) -> BodyGenNetworks:
        torch.manual_seed(3)
        return BodyGenNetworks(
            hidden_size=8,
            num_blocks=1,
            critic_hidden=(16, 8),
        )

    def test_six_trunks_are_independent_double_precision_mosat(self) -> None:
        networks = self.make_small_networks()
        trunks = (
            networks.topology_actor.trunk,
            networks.attribute_actor.trunk,
            networks.control_actor.trunk,
            networks.topology_critic.trunk,
            networks.attribute_critic.trunk,
            networks.control_critic.trunk,
        )
        self.assertEqual(len({id(trunk) for trunk in trunks}), 6)
        parameters = [
            next(trunk.parameters()) for trunk in trunks
        ]
        self.assertTrue(
            all(parameter.dtype == torch.float64 for parameter in parameters)
        )
        self.assertEqual(networks.control_actor.log_std.numel(), 1)

    def test_parameters_are_initialized_natively_in_float64(self) -> None:
        torch.manual_seed(23)
        expected = torch.nn.Linear(
            NODE_FEATURE_SIZE,
            8,
            dtype=torch.float64,
        ).weight.detach().clone()
        torch.manual_seed(23)
        networks = BodyGenNetworks(
            hidden_size=8,
            num_blocks=1,
            critic_hidden=(16, 8),
        )
        torch.testing.assert_close(
            networks.topology_actor.trunk.input.weight,
            expected,
            rtol=0,
            atol=0,
        )

    def test_padding_is_finite_and_excluded_from_mosat_output(self) -> None:
        networks = self.make_small_networks()
        large = BodyGenDesign.canonical()
        small = BodyGenDesign(
            effectors=((0,), (), (), (), (), (), (), ()),
            caps=(CANON_CAP,) * 8,
        )
        batch = networks._batch((large, small))
        hidden = networks.topology_actor.trunk(
            batch.features,
            batch.mask,
            batch.topology_ids,
        )
        self.assertEqual(hidden.shape, (2, large.num_nodes, 8))
        self.assertTrue(torch.isfinite(hidden).all())
        self.assertTrue(
            torch.equal(
                hidden[1, small.num_nodes :],
                torch.zeros_like(hidden[1, small.num_nodes :]),
            )
        )

    def test_seeded_design_sampling_and_body_level_probabilities(self) -> None:
        networks = self.make_small_networks()
        # Uniform topology/attribute heads make the exact body-level sum easy
        # to audit independently of the attention activations.
        for head in (
            networks.topology_actor.logits,
            networks.attribute_actor.effector_logits,
            networks.attribute_actor.cap_logits,
        ):
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)

        first_generator = torch.Generator().manual_seed(17)
        first_designs, first_trace = networks.sample_designs(
            3, first_generator
        )
        second_generator = torch.Generator().manual_seed(17)
        second_designs, second_trace = networks.sample_designs(
            3, second_generator
        )
        self.assertEqual(first_designs, second_designs)
        self.assertEqual(first_trace, second_trace)

        statistics = networks.evaluate_design(first_trace)
        self.assertEqual(statistics["log_prob"].shape, (3, 6))
        self.assertEqual(statistics["entropy"].shape, (3, 6))
        self.assertEqual(statistics["values"].shape, (3, 6))
        expected_first = -BodyGenDesign.canonical().num_nodes * math.log(3)
        self.assertAlmostEqual(
            statistics["log_prob"][0, 0].item(),
            expected_first,
            places=10,
        )
        self.assertTrue(torch.isfinite(statistics["values"]).all())

        selected_transitions = [
            first_trace.episodes[0].transitions[0],
            first_trace.episodes[0].transitions[5],
            first_trace.episodes[1].transitions[2],
        ]
        selected = networks.evaluate_design_transitions(
            selected_transitions
        )
        expected_indices = ((0, 0), (0, 5), (1, 2))
        for name in ("log_prob", "entropy", "values"):
            expected = torch.stack(
                [statistics[name][row, step] for row, step in expected_indices]
            )
            torch.testing.assert_close(selected[name], expected)

        combined = DesignBatchTrace.concatenate(
            (
                DesignBatchTrace((first_trace.select(0),)),
                DesignBatchTrace((first_trace.select(1),)),
            )
        )
        self.assertEqual(len(combined), 2)

    def test_control_distribution_sums_only_active_effectors(self) -> None:
        networks = self.make_small_networks()
        torch.nn.init.zeros_(networks.control_actor.action_mean.weight)
        torch.nn.init.zeros_(networks.control_actor.action_mean.bias)
        with torch.no_grad():
            networks.control_actor.log_std.zero_()
        body = BodyGenDesign.canonical()
        observations = torch.zeros(1, 893, dtype=torch.float64)
        output = networks.control(
            observations, (body,), deterministic=True
        )
        self.assertEqual(output.mean.shape, (1, 32))
        self.assertEqual(output.action_mask.sum().item(), body.num_effectors)
        self.assertTrue(torch.equal(output.action, output.mean))

        statistics = networks.evaluate_control(
            observations,
            (body,),
            torch.zeros_like(output.mean),
        )
        per_action_entropy = 0.5 * (1.0 + math.log(2.0 * math.pi))
        self.assertAlmostEqual(
            statistics["entropy"].item(),
            body.num_effectors * per_action_entropy,
            places=10,
        )
        self.assertEqual(statistics["value"].shape, (1,))

        # Upstream's learned scalar is intentionally unbounded. A defensive
        # clamp would change the policy and gradients in long runs.
        with torch.no_grad():
            networks.control_actor.log_std.fill_(3.0)
        unclipped = networks.control(
            observations,
            (body,),
            deterministic=True,
        )
        self.assertTrue(
            torch.equal(
                unclipped.log_std[unclipped.action_mask],
                torch.full(
                    (body.num_effectors,),
                    3.0,
                    dtype=torch.float64,
                ),
            )
        )


class EnhancedTCATests(unittest.TestCase):
    def test_design_is_undiscounted_while_control_uses_gae(self) -> None:
        result = enhanced_temporal_credit_assignment(
            rewards=torch.tensor([0.0, 0.0, 1.0, 2.0]),
            values=torch.zeros(4),
            next_values=torch.zeros(4),
            terminated=torch.tensor([False, False, False, True]),
            truncated=torch.zeros(4, dtype=torch.bool),
            stages=torch.tensor([TOPOLOGY, ATTRIBUTE, CONTROL, CONTROL]),
            gamma=0.5,
            gae_lambda=1.0,
            normalize_advantages=False,
        )
        torch.testing.assert_close(
            result.design_returns,
            torch.tensor([3.0, 3.0, 3.0, 2.0]),
        )
        torch.testing.assert_close(
            result.advantages,
            torch.tensor([3.0, 3.0, 2.0, 2.0]),
        )
        torch.testing.assert_close(
            result.returns,
            torch.tensor([3.0, 3.0, 2.0, 2.0]),
        )

    def test_truncation_bootstraps_but_termination_does_not(self) -> None:
        common = {
            "rewards": torch.tensor([1.0]),
            "values": torch.tensor([0.0]),
            "next_values": torch.tensor([10.0]),
            "stages": torch.tensor([CONTROL]),
            "gamma": 0.5,
            "gae_lambda": 1.0,
            "normalize_advantages": False,
        }
        truncated = enhanced_temporal_credit_assignment(
            terminated=torch.tensor([False]),
            truncated=torch.tensor([True]),
            **common,
        )
        terminated = enhanced_temporal_credit_assignment(
            terminated=torch.tensor([True]),
            truncated=torch.tensor([False]),
            **common,
        )
        self.assertEqual(truncated.advantages.item(), 6.0)
        self.assertEqual(terminated.advantages.item(), 1.0)
        # The undiscounted design-return accumulator never bootstraps.
        self.assertEqual(truncated.design_returns.item(), 1.0)

    def test_design_and_control_return_scales_are_independent(self) -> None:
        design_normalizer = ReturnNormalizer()
        control_normalizer = ReturnNormalizer()
        enhanced_temporal_credit_assignment(
            rewards=torch.tensor([0.0, 4.0], dtype=torch.float64),
            values=torch.zeros(2, dtype=torch.float64),
            next_values=torch.zeros(2, dtype=torch.float64),
            terminated=torch.tensor([False, True]),
            truncated=torch.tensor([False, False]),
            stages=torch.tensor([TOPOLOGY, CONTROL]),
            normalize_advantages=False,
            design_normalizer=design_normalizer,
            control_normalizer=control_normalizer,
        )
        self.assertEqual(design_normalizer.count.item(), 1)
        self.assertEqual(control_normalizer.count.item(), 1)
        self.assertIsNot(design_normalizer, control_normalizer)


class BodyGenTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = load_bodygen_config(
            cls.root / "configs/benchmarks/bodygen.yaml"
        )

    def test_camera_ready_config_and_worker_partition(self) -> None:
        self.assertEqual(
            self.config["budget"]["environment_steps"],
            196_608_000,
        )
        self.assertEqual(
            self.config["collection"]["logical_streams"],
            20,
        )
        self.assertEqual(bodygen_worker_target(self.config), 2_500)
        self.assertEqual(
            self.config["training"]["minibatch_size"],
            2_048,
        )
        self.assertEqual(
            self.config["network"]["layer_norm"],
            "pre",
        )
        self.assertEqual(self.config["checkpoint"]["every_updates"], 100)

    def test_exact_budget_wave_width_never_overshoots(self) -> None:
        self.assertEqual(choose_wave_width(20, 100), 20)
        self.assertEqual(choose_wave_width(20, 47), 20)
        self.assertEqual(choose_wave_width(20, 7), 7)
        self.assertEqual(choose_wave_width(7, 42), 7)
        for active in range(1, 21):
            for remaining in range(1, 100):
                width = choose_wave_width(active, remaining)
                self.assertEqual(width, min(active, remaining))

    def test_collection_wave_logs_complete_returns_on_step_and_time_axes(
        self,
    ) -> None:
        class CapturingLogger:
            def __init__(self) -> None:
                self.calls: list[dict[str, float]] = []

            def log_collection_rewards(self, **values: float) -> None:
                self.calls.append(values)

        trainer = BodyGenTrainer.__new__(BodyGenTrainer)
        trainer.config = {
            "collection": {
                "logical_streams": 2,
                "minimum_batch_transitions": 16,
            }
        }
        trainer.environment_steps = 100
        trainer.target_environment_steps = 1_000
        trainer.completed_updates = 3
        trainer.reward_window = deque(maxlen=100)
        trainer.accumulated_wall_seconds = 0.0
        trainer.started = time.perf_counter()
        trainer.logger = CapturingLogger()

        first = Episode(design=None, design_trace=None)
        first.rewards = [torch.tensor(1.0), torch.tensor(2.0)]
        second = Episode(design=None, design_trace=None)
        second.rewards = [torch.tensor(4.0), torch.tensor(5.0)]
        trainer._collect_wave = lambda stream_ids, remaining_steps: (
            [(0, first), (1, second)],
            10,
            4,
            6,
            12,
        )

        collection = trainer.collect_batch()

        self.assertTrue(collection.complete)
        self.assertEqual(list(trainer.reward_window), [3.0, 9.0])
        self.assertEqual(len(trainer.logger.calls), 1)
        logged = trainer.logger.calls[0]
        self.assertEqual(logged["environment_steps"], 110)
        self.assertEqual(logged["update"], 3)
        self.assertEqual(logged["completed_episodes"], 2)
        self.assertEqual(logged["rolling_reward"], 6.0)
        self.assertEqual(logged["wave_reward"], 6.0)

    def test_batch_design_groups_each_stage_after_shuffle(self) -> None:
        stages = np.asarray(
            [TOPOLOGY, CONTROL, ATTRIBUTE, TOPOLOGY, CONTROL]
        )
        shuffled = np.asarray([4, 2, 1, 3, 0])
        grouped = stage_grouped_permutation(shuffled, stages)
        np.testing.assert_array_equal(grouped, [3, 0, 2, 4, 1])

    def test_delayed_vsim_done_excludes_the_reset_transition(self) -> None:
        initial = torch.tensor([1.0])
        terminal = torch.tensor([2.0])
        episode = Episode(
            design=None,
            design_trace=None,
        )
        self.assertFalse(
            record_delayed_vsim_transition(
                episode,
                observation=initial,
                action=torch.tensor([0.25]),
                reward=torch.tensor(7.0),
                terminated=torch.tensor(False),
                truncated=torch.tensor(False),
            )
        )
        self.assertTrue(
            record_delayed_vsim_transition(
                episode,
                observation=terminal,
                action=torch.tensor([99.0]),
                reward=torch.tensor(99.0),
                terminated=torch.tensor(False),
                truncated=torch.tensor(True),
            )
        )

        self.assertEqual(episode.physics_steps, 1)
        self.assertEqual(float(episode.rewards[0]), 7.0)
        torch.testing.assert_close(episode.actions[0], torch.tensor([0.25]))
        self.assertTrue(bool(episode.truncated[0]))
        self.assertFalse(bool(episode.terminated[0]))
        torch.testing.assert_close(
            episode_bootstrap_observation(episode),
            terminal,
        )

    def test_validator_locks_non_swept_fidelity_settings(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["training"]["gamma"] = 0.99
        with self.assertRaisesRegex(ValueError, "training.gamma"):
            validate_bodygen_config(invalid)

        invalid = copy.deepcopy(self.config)
        invalid["training"]["actor_learning_rate"] = 2.0e-4
        with self.assertRaisesRegex(ValueError, "paper sweep"):
            validate_bodygen_config(invalid)

        invalid = copy.deepcopy(self.config)
        invalid["network"]["topology_embeddings"] = 512
        with self.assertRaisesRegex(ValueError, "256"):
            validate_bodygen_config(invalid)

        invalid = copy.deepcopy(self.config)
        invalid["environment"]["ctrl_cost_weight"] = 0.25
        with self.assertRaisesRegex(ValueError, "ctrl_cost_weight"):
            validate_bodygen_config(invalid)

        invalid = copy.deepcopy(self.config)
        invalid["environment"]["max_episode_length"] = 500
        with self.assertRaisesRegex(ValueError, "max_episode_length"):
            validate_bodygen_config(invalid)

    def test_development_mode_relaxes_only_batch_sizing(self) -> None:
        smoke = copy.deepcopy(self.config)
        smoke["runtime"]["development"] = True
        smoke["collection"]["minimum_batch_transitions"] = 120
        smoke["collection"]["minimum_transitions_per_stream"] = 6
        smoke["environment"]["max_episode_length"] = 4
        smoke["training"]["optimization_epochs"] = 1
        smoke["training"]["minibatch_size"] = 64
        validate_bodygen_config(smoke)

        smoke["training"]["ppo_clip"] = 0.3
        with self.assertRaisesRegex(ValueError, "training.ppo_clip"):
            validate_bodygen_config(smoke)

    def test_resume_allows_monitoring_but_not_algorithm_changes(self) -> None:
        monitored = copy.deepcopy(self.config)
        monitored["training_evaluation"]["enabled"] = True
        monitored["logging"]["wandb"]["enabled"] = True
        monitored["logging"]["wandb"]["project"] = "test"
        self.assertTrue(resume_configs_match(self.config, monitored))

        moved = copy.deepcopy(self.config)
        moved["runtime"]["device"] = "cuda:1"
        self.assertTrue(resume_configs_match(self.config, moved))

        changed = copy.deepcopy(self.config)
        changed["network"]["hidden_size"] = 128
        self.assertFalse(resume_configs_match(self.config, changed))

    def test_collection_accounting_separates_used_and_discarded_steps(
        self,
    ) -> None:
        trainer = BodyGenTrainer.__new__(BodyGenTrainer)
        trainer.config = {"budget": {"parallel_envs": 4096}}
        trainer.environment_steps = 0
        trainer.trajectory_environment_steps = 0
        trainer.ppo_used_environment_steps = 0
        trainer.discarded_trajectory_environment_steps = 0
        trainer.synchronization_waste_steps = 0
        trainer.used_design_transitions = 0
        trainer.discarded_design_transitions = 0
        trainer.peak_parallel_envs = 0
        trainer.completed_episodes = 0

        trainer._account_collection(
            CollectionResult(
                episodes=(),
                complete=True,
                physics_steps=100,
                trajectory_physics_steps=80,
                synchronization_waste_steps=20,
                design_transitions=120,
                peak_parallel_envs=20,
            )
        )
        trainer._account_collection(
            CollectionResult(
                episodes=(),
                complete=False,
                physics_steps=10,
                trajectory_physics_steps=7,
                synchronization_waste_steps=3,
                design_transitions=12,
                peak_parallel_envs=2,
            )
        )
        self.assertEqual(trainer.environment_steps, 110)
        self.assertEqual(trainer.trajectory_environment_steps, 87)
        self.assertEqual(trainer.ppo_used_environment_steps, 80)
        self.assertEqual(
            trainer.discarded_trajectory_environment_steps,
            7,
        )
        self.assertEqual(trainer.synchronization_waste_steps, 23)
        self.assertEqual(trainer.used_design_transitions, 120)
        self.assertEqual(trainer.discarded_design_transitions, 12)
        self.assertEqual(trainer.peak_parallel_envs, 20)

    def _checkpoint_trainer(self, run_dir: Path) -> BodyGenTrainer:
        trainer = BodyGenTrainer.__new__(BodyGenTrainer)
        trainer.config = copy.deepcopy(self.config)
        trainer.config["network"]["hidden_size"] = 32
        trainer.device = torch.device("cpu")
        trainer.dtype = torch.float64
        network = trainer.config["network"]
        trainer.networks = BodyGenNetworks(
            hidden_size=network["hidden_size"],
            num_blocks=network["blocks"],
            layer_norm=network["layer_norm"],
            topology_embeddings=network["topology_embeddings"],
            feed_forward_ratio=network["feed_forward_ratio"],
            critic_hidden=tuple(network["critic_hidden"]),
            initial_control_log_std=network["initial_control_log_std"],
        )
        trainer.actor_parameters = list(
            itertools.chain(
                trainer.networks.topology_actor.parameters(),
                trainer.networks.attribute_actor.parameters(),
                trainer.networks.control_actor.parameters(),
            )
        )
        trainer.critic_parameters = list(
            itertools.chain(
                trainer.networks.topology_critic.parameters(),
                trainer.networks.attribute_critic.parameters(),
                trainer.networks.control_critic.parameters(),
            )
        )
        trainer.actor_optimizer = torch.optim.Adam(
            trainer.actor_parameters,
            lr=5.0e-5,
        )
        trainer.critic_optimizer = torch.optim.Adam(
            trainer.critic_parameters,
            lr=3.0e-4,
        )
        trainer.design_return_normalizer = ReturnNormalizer()
        trainer.control_return_normalizer = ReturnNormalizer()
        trainer.seed = 42
        trainer.rng = np.random.default_rng(42)
        trainer.torch_generator = torch.Generator().manual_seed(42)
        trainer.target_environment_steps = 196_608_000
        trainer.environment_steps = 100
        trainer.trajectory_environment_steps = 80
        trainer.ppo_used_environment_steps = 80
        trainer.discarded_trajectory_environment_steps = 0
        trainer.synchronization_waste_steps = 20
        trainer.used_design_transitions = 120
        trainer.discarded_design_transitions = 0
        trainer.peak_parallel_envs = 20
        trainer.completed_updates = 1
        trainer.completed_episodes = 20
        trainer.reward_window = deque([3.0, 5.0], maxlen=100)
        trainer.run_identity = "bodygen-test-s42"
        trainer.accumulated_wall_seconds = 2.0
        trainer.started = time.perf_counter()
        trainer.best_native_return = -float("inf")
        trainer.run_dir = run_dir
        (run_dir / "checkpoints").mkdir(parents=True)
        return trainer

    def test_checkpoint_restores_complete_seeded_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            original = self._checkpoint_trainer(run_dir)
            torch.manual_seed(123)
            checkpoint = original.save_checkpoint()
            saved = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(saved["parallel_envs"], 20)
            self.assertEqual(saved["parallel_env_cap"], 4096)
            audit = original._audit_metrics()
            for name in (
                "benchmark/training/env_steps_per_second",
                "benchmark/training/parallel_envs",
                "benchmark/resource/trainable_parameters",
                "benchmark/resource/peak_device_bytes",
                "benchmark/resource/peak_host_kib",
                "bodygen/steps/environment",
                "bodygen/steps/synchronization_waste",
            ):
                self.assertIn(name, audit)
            expected_numpy = int(original.rng.integers(0, 1_000_000))
            expected_generator = torch.rand(
                4,
                generator=original.torch_generator,
            )
            expected_torch = torch.rand(4)

            restored = self._checkpoint_trainer(run_dir / "restored")
            restored._load_checkpoint(checkpoint)
            self.assertEqual(
                int(restored.rng.integers(0, 1_000_000)),
                expected_numpy,
            )
            torch.testing.assert_close(
                torch.rand(4, generator=restored.torch_generator),
                expected_generator,
            )
            torch.testing.assert_close(torch.rand(4), expected_torch)
            self.assertEqual(restored.environment_steps, 100)
            self.assertEqual(restored.completed_updates, 1)
            self.assertEqual(list(restored.reward_window), [3.0, 5.0])


if __name__ == "__main__":
    unittest.main()
