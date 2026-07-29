from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from benchmarks.nge.gm_uc import GraphMutationWithUncertainty
from benchmarks.nge.graph import MUTATION_NAMES, NGEGraph, mutate
from benchmarks.nge.method import NGEMethod, checkpoints_for_nge_run
from benchmarks.nge.nervenet import (
    ControllerState,
    N_ACTIONS,
    OBSERVATION_SIZE,
    NerveNetPlusPlus,
    PHYSICAL_OBSERVATION_SIZE,
    RunningMeanStd,
)
from benchmarks.nge.population import Population
from benchmarks.nge.training import (
    Rollout,
    NGETrainer,
    SpeciesLearner,
    TrainingLogger,
    load_nge_config,
    nge_generation_environment_steps,
    resume_configs_match,
    summarize_selection_episodes,
    summarize_training_update,
    validate_nge_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmarks/nge.yaml"
TUNE_CONFIG = ROOT / "configs/benchmarks/tune_nge.yaml"


class _FakeCodesignEnvironment:
    def __init__(self) -> None:
        self.installed = None
        self.rebuilds = 0

    def set_next(self, counts, effectors, caps) -> None:
        self.installed = (counts, effectors, caps)

    def resample(self) -> None:
        self.rebuilds += 1


class _FakeSelectionPolicy:
    def __init__(self) -> None:
        self.hidden_inputs: list[float] = []

    def eval(self) -> None:
        pass

    def initial_hidden(self, environments, graph, device):
        return torch.zeros((environments, 1, 1), device=device)

    def forward_step(self, observation, graph, hidden):
        self.hidden_inputs.append(float(hidden[0, 0, 0]))
        batch = observation.shape[0]
        mean = torch.zeros((batch, N_ACTIONS), device=observation.device)
        log_std = torch.zeros_like(mean)
        return mean, log_std, hidden + 1


class _FakeSelectionEnvironment:
    def __init__(self) -> None:
        self.index = 0
        self.closed = False
        self.rewards = [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
        self.dones = [
            [False, False],
            [True, False],
            [False, True],
            [True, False],
        ]

    def reset(self):
        return torch.zeros((2, OBSERVATION_SIZE)), {}

    def step(self, action):
        reward = torch.tensor(self.rewards[self.index])
        terminated = torch.tensor(self.dones[self.index])
        self.index += 1
        observation = torch.zeros((2, OBSERVATION_SIZE))
        truncated = torch.zeros(2, dtype=torch.bool)
        return observation, reward, terminated, truncated, {}

    def close(self) -> None:
        self.closed = True


class NGEGraphTests(unittest.TestCase):
    def test_canonical_graph_round_trips_through_benchmark_arrays(self) -> None:
        graph = NGEGraph.canonical((1, 4, 6))
        counts, effectors, caps = graph.to_arrays()
        restored = NGEGraph.from_arrays(counts, effectors, caps)

        self.assertEqual(restored, graph)
        self.assertEqual(graph.num_actuators, 6)
        np.testing.assert_array_equal(counts, [2, 0, 0, 2, 0, 2, 0, 0])
        self.assertEqual(graph.node_attributes().shape, (7, 16))

    def test_all_four_mutations_are_explicit_and_grammar_valid(self) -> None:
        base = NGEGraph.canonical()
        for index, operation in enumerate(MUTATION_NAMES):
            probabilities = {
                name: float(name == operation) for name in MUTATION_NAMES
            }
            result = mutate(
                base,
                np.random.default_rng(index + 1),
                probabilities,
            )
            counts, effectors, caps = result.graph.to_arrays()
            self.assertEqual(result.operation, operation)
            self.assertNotEqual(result.graph, base)
            self.assertTrue(np.all((counts >= 0) & (counts <= 3)))
            self.assertGreater(counts.sum(), 0)
            NGEGraph.from_arrays(counts, effectors, caps)


class NerveNetTests(unittest.TestCase):
    def test_controller_is_graph_independent_and_outputs_only_active_slots(self) -> None:
        graph = NGEGraph.canonical()
        policy = NerveNetPlusPlus(64)
        observation = torch.zeros((3, OBSERVATION_SIZE))
        hidden = policy.initial_hidden(3, graph, torch.device("cpu"))

        mean, log_std, next_hidden = policy.forward_step(
            observation, graph, hidden
        )

        self.assertEqual(mean.shape, (3, N_ACTIONS))
        self.assertEqual(log_std.shape, (3, N_ACTIONS))
        self.assertEqual(next_hidden.shape, (3, graph.num_nodes, 64))
        inactive = sorted(set(range(N_ACTIONS)) - set(graph.action_slots()))
        torch.testing.assert_close(mean[:, inactive], torch.zeros(3, len(inactive)))
        self.assertFalse(torch.equal(hidden, next_hidden))

    def test_child_inherits_every_policy_value_and_normalizer_tensor(self) -> None:
        torch.manual_seed(5)
        parent = ControllerState.create(torch.device("cpu"), hidden_size=64)
        parent.normalizer.mean.fill_(2.0)
        child = parent.inherited_copy()

        for parent_parameter, child_parameter in zip(
            parent.policy.parameters(), child.policy.parameters()
        ):
            torch.testing.assert_close(parent_parameter, child_parameter)
            self.assertNotEqual(parent_parameter.data_ptr(), child_parameter.data_ptr())
        for parent_parameter, child_parameter in zip(
            parent.value.parameters(), child.value.parameters()
        ):
            torch.testing.assert_close(parent_parameter, child_parameter)
        torch.testing.assert_close(parent.normalizer.mean, child.normalizer.mean)

    def test_generation_rebuild_resets_optimizer_but_preserves_controller_and_lr(
        self,
    ) -> None:
        config = load_nge_config(CONFIG)
        parent = SpeciesLearner.create(
            torch.device("cpu"),
            config["training"],
            hidden_size=64,
        )
        adapted_lr = parent.initial_policy_lr / 8.0
        parent.current_policy_lr = adapted_lr
        for group in parent.policy_optimizer.param_groups:
            group["lr"] = adapted_lr

        child = parent.inherited_copy()
        parent.reset_optimizers()

        for left, right in zip(
            parent.controller.policy.parameters(),
            child.controller.policy.parameters(),
        ):
            torch.testing.assert_close(left, right)
        self.assertEqual(parent.current_policy_lr, adapted_lr)
        self.assertEqual(child.current_policy_lr, adapted_lr)
        self.assertEqual(parent.policy_optimizer.param_groups[0]["lr"], adapted_lr)
        self.assertEqual(child.policy_optimizer.param_groups[0]["lr"], adapted_lr)
        self.assertEqual(len(parent.policy_optimizer.state), 0)
        self.assertEqual(len(child.policy_optimizer.state), 0)


class GMUCPopulationTests(unittest.TestCase):
    def test_one_dropout_sample_is_reusable_across_candidate_graphs(self) -> None:
        graph = NGEGraph.canonical()
        gm_uc = GraphMutationWithUncertainty(
            torch.device("cpu"), gradient_steps=1
        )
        rng = np.random.default_rng(11)
        masks = gm_uc.model.sample_masks(rng, torch.device("cpu"))
        gm_uc.model.eval()
        first = gm_uc.model.forward_graph(graph, masks)
        second = gm_uc.model.forward_graph(graph, masks)
        torch.testing.assert_close(first, second)

    def test_population_eliminates_worst_and_children_use_surviving_parents(self) -> None:
        population = Population.initial(4, NGEGraph.canonical())
        population.assign_fitness({1: -10.0, 2: 1.0, 3: 2.0, 4: 3.0})
        gm_uc = GraphMutationWithUncertainty(
            torch.device("cpu"),
            batch_size=4,
            gradient_steps=1,
        )
        result = population.evolve(
            gm_uc,
            np.random.default_rng(3),
            elimination_rate=0.25,
            candidate_pool_size=8,
            mutation_probabilities={name: 0.25 for name in MUTATION_NAMES},
            node_perturb_probability=0.1,
        )

        self.assertEqual(result.eliminated_ids, (1,))
        self.assertEqual(population.size, 4)
        self.assertEqual(population.generation, 1)
        self.assertTrue(set(result.child_parent_ids.values()) <= {2, 3, 4})
        self.assertEqual(len(gm_uc.history), 4)


class NGEConfigAndMethodTests(unittest.TestCase):
    def test_training_reward_log_uses_only_completed_episode_returns(self) -> None:
        def rollout(
            fitness: float,
            raw_step_reward: float,
            completed_returns: tuple[float, ...],
        ) -> Rollout:
            scalar = torch.zeros((1, 1))
            return Rollout(
                observations=torch.zeros((1, 1, OBSERVATION_SIZE)),
                actions=torch.zeros((1, 1, N_ACTIONS)),
                old_log_prob=scalar,
                values=scalar,
                advantages=scalar,
                returns=scalar,
                episode_starts=torch.ones((1, 1), dtype=torch.bool),
                hidden=torch.zeros((1, 1, 1, 64)),
                rollout_return_estimate=fitness,
                raw_step_reward_mean=raw_step_reward,
                completed_returns=completed_returns,
            )

        rollouts = {
            1: rollout(100.0, 1.0, (10.0,)),
            2: rollout(200.0, 3.0, (20.0,)),
        }
        ppo_metrics = [
            {
                "policy": 1.0,
                "value": 2.0,
                "entropy": 3.0,
                "kl": 4.0,
                "learning_rate": 5.0,
            },
            {
                "policy": 3.0,
                "value": 4.0,
                "entropy": 5.0,
                "kl": 6.0,
                "learning_rate": 7.0,
            },
        ]

        metrics = summarize_training_update(rollouts, ppo_metrics)

        self.assertEqual(metrics["rewards/rollout_return_estimate"], 150.0)
        self.assertEqual(metrics["rewards/raw_step_mean"], 2.0)
        self.assertEqual(metrics["rewards/completed_episodes"], 2.0)
        self.assertNotIn("rewards/step", metrics)

        no_completion = {
            1: rollout(100.0, 1.0, ()),
            2: rollout(200.0, 3.0, ()),
        }
        metrics = summarize_training_update(no_completion, ppo_metrics)
        self.assertNotIn("rewards/step", metrics)

    def test_reward_views_use_the_same_value_on_rl_games_axes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            config = load_nge_config(CONFIG)
            logger = TrainingLogger(run_dir, config, "test-run")
            logger.log_training_update(
                {"rewards/rollout_return_estimate": 150.0},
                environment_steps=131_072,
                iteration=1,
                wall_seconds=7.0,
                episode_reward=15.0,
            )
            logger.close()

            events = EventAccumulator(str(run_dir / "tensorboard"))
            events.Reload()
            expected_steps = {
                "rewards/step": 131_072,
                "rewards/iter": 1,
                "rewards/time": 7,
            }
            for tag, expected_step in expected_steps.items():
                scalar = events.Scalars(tag)[0]
                self.assertEqual(scalar.value, 15.0)
                self.assertEqual(scalar.step, expected_step)

    def test_training_evaluation_has_its_own_step_aligned_reward_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            config = load_nge_config(CONFIG)
            logger = TrainingLogger(run_dir, config, "test-run")
            logger.log_training_evaluation(
                expected_return=123.5,
                environment_steps=2_621_440,
                completed_generation=1,
                evaluated_episodes=32,
            )
            logger.close()

            events = EventAccumulator(str(run_dir / "tensorboard"))
            events.Reload()
            reward = events.Scalars("rewards/step_eval")[0]
            self.assertEqual(reward.value, 123.5)
            self.assertEqual(reward.step, 2_621_440)

    def test_selection_fitness_uses_only_complete_episode_returns(self) -> None:
        result = summarize_selection_episodes(
            {
                1: [10.0, 20.0],
                2: [7.0],
            },
            {
                1: [5, 8],
                2: [3],
            },
            environment_steps=64,
        )

        self.assertEqual(result.fitness, {1: 15.0, 2: 7.0})
        self.assertEqual(result.completed_returns, (10.0, 20.0, 7.0))
        self.assertEqual(result.completed_lengths, (5, 8, 3))
        self.assertEqual(result.episodes_per_species, {1: 2, 2: 1})
        self.assertEqual(result.environment_steps, 64)

    def test_selection_rejects_a_species_without_a_complete_episode(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "species 2"):
            summarize_selection_episodes(
                {1: [10.0], 2: []},
                {1: [5], 2: []},
                environment_steps=64,
            )

    def test_selection_collector_uses_only_first_complete_episode_per_lane(
        self,
    ) -> None:
        trainer = NGETrainer.__new__(NGETrainer)
        trainer.device = torch.device("cpu")
        trainer.population = Population.initial(2, NGEGraph.canonical())
        trainer.selection_config = {
            "environments_per_species": 1,
            "rollout_steps": 4,
            "action_mode": "deterministic",
        }
        trainer.environment_steps = 100
        trainer.controller_environment_steps = 100
        trainer.selection_environment_steps = 0
        policies = {}
        trainer.learners = {}
        normalizer_counts = {}
        for species in trainer.population.species:
            policy = _FakeSelectionPolicy()
            normalizer = RunningMeanStd(
                PHYSICAL_OBSERVATION_SIZE,
                torch.device("cpu"),
            )
            policies[species.species_id] = policy
            normalizer_counts[species.species_id] = normalizer.count
            trainer.learners[species.species_id] = SimpleNamespace(
                controller=SimpleNamespace(
                    policy=policy,
                    normalizer=normalizer,
                )
            )
        environment = _FakeSelectionEnvironment()
        trainer._create_environment = lambda num_envs: environment

        result = trainer._evaluate_population_for_selection()

        # Species 1 finishes again with return 7 later in the fixed window.
        # That auto-reset episode is deliberately not another ranking sample.
        self.assertEqual(result.fitness, {1: 3.0, 2: 60.0})
        self.assertEqual(result.completed_returns, (3.0, 60.0))
        self.assertEqual(result.completed_lengths, (2, 3))
        self.assertEqual(result.episodes_per_species, {1: 1, 2: 1})
        self.assertEqual(trainer.environment_steps, 108)
        self.assertEqual(trainer.controller_environment_steps, 100)
        self.assertEqual(trainer.selection_environment_steps, 8)
        self.assertTrue(environment.closed)
        self.assertEqual(policies[1].hidden_inputs, [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(policies[2].hidden_inputs, [0.0, 1.0, 2.0, 0.0])
        for species in trainer.population.species:
            self.assertEqual(
                trainer.learners[
                    species.species_id
                ].controller.normalizer.count,
                normalizer_counts[species.species_id],
            )

    def _method(self, root: Path) -> NGEMethod:
        config = load_nge_config(CONFIG)
        population = Population.initial(2, NGEGraph.canonical())
        population.assign_fitness({1: 1.0, 2: 2.0})
        learners = {}
        for species in population.species:
            torch.manual_seed(species.species_id)
            controller = ControllerState.create(
                torch.device("cpu"), hidden_size=64
            )
            learners[species.species_id] = {
                "controller": controller.state_dict()
            }
        gm_uc = GraphMutationWithUncertainty(torch.device("cpu"))
        state = {
            "format_version": 1,
            "method": "nge",
            "training_seed": 42,
            "environment_steps": 196_608_000,
            "parallel_envs": 4096,
            "population": population.state_dict(),
            "learners": learners,
            "gm_uc": gm_uc.state_dict(),
        }
        config_path = root / "config.yaml"
        config_path.write_text("")
        checkpoint = root / "final.pth"
        checkpoint.write_bytes(b"placeholder")
        return NGEMethod(
            state=state,
            device=torch.device("cpu"),
            run_dir=root,
            run_config_path=config_path,
            run_config=config,
            checkpoint_path=checkpoint,
            checkpoint_label="final",
        )

    def test_paper_config_has_exact_counted_selection_budget(self) -> None:
        config = load_nge_config(CONFIG)
        validate_nge_config(config)
        generation_steps = nge_generation_environment_steps(config)
        controller_steps = (
            config["budget"]["parallel_envs"]
            * config["training"]["rollout_steps"]
            * config["training"]["updates_per_generation"]
        )
        selection_steps = (
            config["population"]["size"]
            * config["selection_evaluation"]["environments_per_species"]
            * config["selection_evaluation"]["rollout_steps"]
        )
        self.assertEqual(controller_steps, 2_490_368)
        self.assertEqual(selection_steps, 131_072)
        self.assertEqual(generation_steps, 2_621_440)
        self.assertEqual(config["budget"]["environment_steps"] // generation_steps, 75)
        self.assertEqual(
            config["budget"]["environment_steps"] % generation_steps,
            0,
        )
        self.assertEqual(
            set(config["population"]["mutation_probabilities"]),
            set(MUTATION_NAMES),
        )
        self.assertFalse(config["training_evaluation"]["enabled"])
        self.assertEqual(config["training_evaluation"]["every_generations"], 1)

    def test_training_evaluation_schedule_is_validated(self) -> None:
        config = copy.deepcopy(load_nge_config(CONFIG))
        config["training_evaluation"]["every_generations"] = 0

        with self.assertRaisesRegex(
            ValueError,
            "training_evaluation.every_generations",
        ):
            validate_nge_config(config)

    def test_old_config_without_optional_training_evaluation_remains_valid(
        self,
    ) -> None:
        config = copy.deepcopy(load_nge_config(CONFIG))
        del config["training_evaluation"]

        validate_nge_config(config)

    def test_resume_allows_only_training_evaluation_changes(self) -> None:
        saved = copy.deepcopy(load_nge_config(CONFIG))
        current = copy.deepcopy(saved)
        current["training_evaluation"]["enabled"] = True
        current["training_evaluation"]["every_generations"] = 5
        self.assertTrue(resume_configs_match(saved, current))

        current["training"]["gamma"] = 0.5
        self.assertFalse(resume_configs_match(saved, current))

    def test_config_requires_temporal_depth_for_a_complete_selection_episode(
        self,
    ) -> None:
        config = load_nge_config(CONFIG)
        config = copy.deepcopy(config)
        config["selection_evaluation"]["rollout_steps"] = config[
            "environment"
        ]["max_episode_length"]

        with self.assertRaisesRegex(ValueError, "greater than"):
            validate_nge_config(config)

    def test_legacy_run_config_fails_with_a_restart_explanation(self) -> None:
        config = load_nge_config(CONFIG)
        config = copy.deepcopy(config)
        del config["selection_evaluation"]

        with self.assertRaisesRegex(ValueError, "cannot be resumed"):
            validate_nge_config(config)

    def test_config_requires_the_upstream_per_species_sample_count(self) -> None:
        config = load_nge_config(CONFIG)
        config = copy.deepcopy(config)
        config["training"]["rollout_steps"] = 16

        with self.assertRaisesRegex(ValueError, "PPO batch"):
            validate_nge_config(config)

    def test_config_requires_budget_for_complete_generation_cost(self) -> None:
        config = load_nge_config(CONFIG)
        config = copy.deepcopy(config)
        config["budget"]["environment_steps"] -= 1

        with self.assertRaisesRegex(ValueError, "complete generation cost"):
            validate_nge_config(config)

    def test_tuning_contract_targets_valid_yaml_paths_and_proxy_budget(
        self,
    ) -> None:
        tuning = yaml.safe_load(TUNE_CONFIG.read_text())
        config = load_nge_config(CONFIG)
        self.assertEqual(tuning["study"]["candidates"], 30)
        self.assertEqual(tuning["study"]["tuning_seeds"], [142, 143, 144])
        self.assertFalse(tuning["study"]["early_pruning"])

        for parameter in tuning["params"]:
            value = config
            for key in parameter["path"].split("."):
                self.assertIn(key, value, parameter["path"])
                value = value[key]

        proxy = copy.deepcopy(config)
        proxy["budget"]["environment_steps"] = tuning["study"][
            "proxy_environment_steps"
        ]
        validate_nge_config(proxy)
        self.assertEqual(
            proxy["budget"]["environment_steps"]
            // nge_generation_environment_steps(proxy),
            30,
        )

    def test_native_pair_sampling_is_seeded_and_routes_own_controllers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            method = self._method(Path(temporary))
            first = method.sample_pairs(8, 17)
            second = method.sample_pairs(8, 17)
            np.testing.assert_array_equal(
                first.controller_ids, second.controller_ids
            )
            self.assertTrue(
                set(first.controller_ids) <= {"species_1", "species_2"}
            )

            environment = _FakeCodesignEnvironment()
            method.install_pairs(environment, first)
            method.begin_rollout(first)
            action, value = method.deterministic_action(
                torch.zeros((8, OBSERVATION_SIZE))
            )
            self.assertEqual(environment.rebuilds, 1)
            self.assertEqual(action.shape, (8, N_ACTIONS))
            self.assertEqual(value.shape, (8,))
            method.reset_controllers(torch.ones(8, dtype=torch.bool))
            for hidden in method._hidden.values():
                torch.testing.assert_close(hidden, torch.zeros_like(hidden))

    def test_numeric_checkpoint_selection_means_nge_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            checkpoint_dir = run / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "generation_0005.pth").write_bytes(b"x")
            (checkpoint_dir / "final.pth").write_bytes(b"x")
            config = {"run_root": str(run.parent)}
            numeric = checkpoints_for_nge_run(config, run, "5")
            final = checkpoints_for_nge_run(config, run, "final")
            self.assertEqual(numeric[0][0], "generation_5")
            self.assertEqual(final[0][0], "final")


if __name__ == "__main__":
    unittest.main()
