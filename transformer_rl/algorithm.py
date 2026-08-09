"""`CodesignAlgorithm`: the shared-trunk PPG codesign agent, as a package `Algorithm`.

This is the conformance layer, not a second trainer. `CodesignAgent` still owns its network, its
optimizer and its rollout loop exactly as it does under `scripts/train_codesign_single.py`; what
this adds is the shape `codesigner.optimize` drives -- a `run()` that returns periodically, the two
artifacts, and checkpoints carrying enough provenance to be read somewhere else (D18, D19).

**One `run()` is one resample window.** That is where the generator updates, where each body's
return is measured, where the population turns over, and where a rebuild crash costs the most -- so
it is both the natural reporting boundary and the right checkpoint cadence (D24). The window is
reached by stepping `LoggingA2CAgent._train_iter`, the same loop the script path drains in one go.

The Task arrives unsized, since the env count, the body count and the bodies themselves are all the
algorithm's to choose (D7); `_start` calls `setup` on the first `run()`.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from codesigner.interfaces import Algorithm, ControlPolicy, MorphologyGenerator, Morphology

from . import runtime
from .artifacts import TransformerControlPolicy, TransformerMorphologyGenerator
from .morphology import seed_body
from .train_utils import (_adjust_minibatch, _compose_identity, _deep_merge, _load_config,
                          _resolve_task)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The identity fields base.yaml deliberately leaves out, because the entry point owns them -- which
# network builder and model are looked up, and which algorithm the run is. The task-shaped fields
# (env_name, run name, train_dir) are NOT here: they are composed from the config's `env.task` by
# `_compose_identity`, the same call `run_training` makes, so the two entry points cannot disagree
# about where an ant codesign run writes.
_NETWORK_NAME = "multimorph_limb_transformer"
_MODEL_NAME = "transformer_masked_a2c_logstd"
_FAMILY = "codesign"
_EXPERIMENT = "codesign_single_transformer"


class CodesignAlgorithm(Algorithm):
    """Single-network codesign: control and morphology generation on one shared trunk."""

    def __init__(self, config: str | Path, run_name: Optional[str] = None,
                 overrides: Optional[dict] = None, seed: Optional[int] = None,
                 name: str = "shared_trunk_codesign"):
        super().__init__(name, "PPG codesign: control and generator heads over one transformer "
                               "trunk, updated jointly at every rebuild boundary.")
        self.config_path = Path(config)
        self.seed = seed
        self._overrides = overrides or {}
        self._run_name = run_name
        self._explicit_save_freq = "save_frequency" in (
            self._overrides.get("params", {}).get("config", {}))
        self._cfg = self._load()
        self._agent = None
        self._iter = None
        self._model = None
        self._policy = None
        self._generator = None

    # ---- configuration ---------------------------------------------------------------

    def _load(self) -> dict:
        """The resolved run config: `extends` chain, then defaults/base.yaml underneath, then this
        algorithm's own overrides on top. Same resolution order the training script uses, so a
        config behaves identically either way it is run."""
        with open(_PROJECT_ROOT / "configs" / "defaults" / "base.yaml") as f:
            base = yaml.safe_load(f)
        cfg = _deep_merge(base, _load_config(self.config_path))
        cfg = _deep_merge(cfg, self._overrides)
        if self.seed is not None:
            cfg["params"]["seed"] = self.seed

        self._task_key, self._task_class = _resolve_task(cfg)
        identity = _compose_identity(self._task_key, _FAMILY, _EXPERIMENT)

        params = cfg["params"]
        params["config"]["env_name"] = identity["env_name"]
        params["config"]["name"] = identity["name"]
        params.setdefault("network", {})["name"] = _NETWORK_NAME
        params.setdefault("model", {})["name"] = _MODEL_NAME
        ppo = params["config"]
        ppo.setdefault("use_diagnostics", True)
        ppo.setdefault("train_dir", identity["train_dir"])
        ppo["full_experiment_name"] = (self._run_name
                                       or datetime.now().strftime("%d-%H-%M-%S"))
        return cfg

    def _set_checkpoint_cadence(self, ppo: dict) -> None:
        """Checkpoint cadence tracks rebuild cadence (D24).

        `resample_interval` is in episodes and an epoch is `horizon_length` env-steps, so this is
        how many epochs one window spans -- the same arithmetic `CodesignAgent` uses to size its LR
        warmup. The rebuild crash has no auto-resume, so the checkpoint interval is the maximum a
        crash can cost; matching it to the window means at most one window is ever lost.

        Read off the live Task rather than the config, because `max_episode_length` has a Task-owned
        default a config need not restate. This supersedes the config's own `save_frequency`, which
        is spaced for readable checkpoints rather than for crash exposure; pass one in `overrides`
        to keep that instead.
        """
        interval = ppo.get("resample_interval", 0)
        if not interval or self._explicit_save_freq:
            return
        epochs_per_window = max(1, -(-interval * self.task.max_episode_length
                                     // ppo["horizon_length"]))
        ppo["save_frequency"] = epochs_per_window

    @property
    def library_kwargs(self) -> dict:
        return dict(self._cfg.get("env", {}).get("module_library_kwargs", {}))

    def make_library(self):
        """The run's ONE module library, named by the config (D14). Built here so the caller can
        hand the same instance to `optimize`, which hands it back through
        `assign_task_and_modlib` -- one object, not two that agree today."""
        from codesigner.components.modular_libraries import REGISTRY

        env_cfg = self._cfg.get("env", {})
        return REGISTRY[env_cfg.get("module_library", "simple")](**self.library_kwargs)

    @property
    def task_key(self) -> str:
        """The config's registry key for this run's Task -- the same string a checkpoint records."""
        return self._task_key

    def make_task(self, **overrides):
        """Construct (but do not size) the Task the config names, from its `env:` block.

        Two-phase construction (D7): everything here is a simulation parameter. The keys this
        module owns -- which task, which library, which seed body, how many bodies -- are stripped,
        because they describe the *run*, not the simulator, and the Task would reject them.
        """
        import torch
        import vlearn as v

        env_cfg = dict(self._cfg.get("env", {}))
        for key in ("task", "module_library", "module_library_kwargs", "base_morphology",
                    "num_morphs"):
            env_cfg.pop(key, None)
        kwargs = {
            "device": torch.device("cuda:0"),
            "rendering": False,
            "raise_exception": False,
            "enable_scene_query": False,
            # No offset between the world frame and the scene root; every task places its own bodies.
            "rootOffset": (v.Vec3(0, 0, 0), v.Quat(0, 0, 0, 1)),
            **env_cfg,
            **overrides,
        }
        return self._task_class(**kwargs)

    # ---- startup ---------------------------------------------------------------------

    def _num_morphs(self, num_actors: int) -> int:
        return self._cfg.get("env", {}).get("num_morphs") or num_actors

    def _start(self) -> None:
        """Size the Task, build the rl_games stack, and stop just short of training."""
        import torch
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.ivecenv import IVecEnv
        from rl_games.algos_torch import model_builder as mb_module
        from rl_games.torch_runner import Runner
        from vlearn.torch_utils.wrappers import NewToOldAPICompatilibity

        from .codesign_agent import CodesignAgent
        from .models import (MultiMorphLimbTransformerBuilder, MultiMorphValueBuilder,
                             TransformerMaskedNorm, TransformerMaskedValue)

        cfg = self._cfg
        ppo = cfg["params"]["config"]
        env_cfg = cfg.get("env", {})

        # The library the driver handed us is the run's, not a fresh one -- the whole point of D14.
        library = self.modlib
        base_morphology = seed_body(library, **env_cfg.get("base_morphology", {}))
        runtime.set_run(library=library, base_morphology=base_morphology)

        # `num_morphs` bodies share `num_actors` envs and the division rounds down, so snap before
        # rl_games sizes its rollout buffers off a count the Task will never reach (D15-D17).
        num_morphs = self._num_morphs(ppo["num_actors"])
        snapped = (ppo["num_actors"] // num_morphs) * num_morphs
        if snapped != ppo["num_actors"]:
            print(f"[env] num_actors {ppo['num_actors']} -> {snapped}: {num_morphs} bodies x "
                  f"{ppo['num_actors'] // num_morphs} envs each", flush=True)
            ppo["num_actors"] = snapped
        if snapped < num_morphs:
            raise ValueError(f"num_actors={snapped} over num_morphs={num_morphs} leaves no envs "
                             "per body")
        _adjust_minibatch(ppo, ppo["num_actors"], ppo["horizon_length"])
        self._set_checkpoint_cadence(ppo)

        task, seed = self.task, cfg["params"].get("seed")

        def create_envs(n, **kw):
            built = task.setup(library, n, num_morphs, [base_morphology] * num_morphs, seed=seed)
            assert built == n, f"Task built {built} envs, rl_games is sized for {n}"
            runtime.set_run(obs_layout=task.obs_layout())
            return NewToOldAPICompatilibity(task)

        class VlearnEnv(IVecEnv):
            def __init__(self, config_dict, config_name, num_actors, **kwargs):
                self.envs = config_dict[config_name]["env_creator"](num_actors, **kwargs)
                self.num_actors = task.total_num_envs

            def step(self, actions):
                return self.envs.step(actions)

            def reset(self):
                return self.envs.reset()

            def get_env_info(self):
                import gymnasium.spaces
                from vlearn.spaces import Box

                def convert(space):
                    assert isinstance(space, Box), f"unexpected space {type(space)}"
                    return gymnasium.spaces.Box(low=space.low, high=space.high, shape=space.shape)

                return {"observation_space": convert(task.observation_space),
                        "action_space": convert(task.action_space),
                        "value_size": getattr(task, "value_size", 1)}

        env_name = ppo["env_name"]
        env_configurations.register(env_name, {"vecenv_type": "VLEARN", "env_creator": create_envs})
        vecenv.register("VLEARN", lambda config_name, num_actors, **kw:
                        VlearnEnv(env_configurations.configurations, config_name, num_actors, **kw))

        mb_module.register_network(_NETWORK_NAME, MultiMorphLimbTransformerBuilder)
        mb_module.register_network('multimorph_limb_value', MultiMorphValueBuilder)
        mb_module.register_model('transformer_masked_a2c_logstd', TransformerMaskedNorm)
        mb_module.register_model('transformer_masked_value', TransformerMaskedValue)

        captured = {}

        def build_agent(**kwargs):
            agent = CodesignAgent(**kwargs)
            # Deferred: run_train() constructs, restores and compiles the agent, then calls train().
            # We want everything but the last step, because this Algorithm drives the loop itself --
            # one window per run(). Replicating run_train's restore/compile handling instead would
            # be a copy that silently rots the first time rl_games changes it.
            agent._defer_train = True
            captured["agent"] = agent
            return agent

        runner = Runner()
        runner.algo_factory.register_builder('codesign_continuous', build_agent)
        runner.load(cfg)
        runner.run_train({"train": True})

        self._agent = captured["agent"]
        self._agent._on_iteration = self._on_iteration    # the driver attached before we existed
        self._iter = self._agent._train_iter()
        self._model = self._agent.model
        self._policy = TransformerControlPolicy(self._model)
        self._generator = TransformerMorphologyGenerator(self._agent._net(), library)
        torch.cuda.empty_cache()

    def _build_inference_model(self) -> None:
        """The network alone -- no optimizer, no rollout buffers, no rl_games agent.

        What `evaluate` needs. Loading a checkpoint to *score* it should not stand up a trainer:
        the Task it is handed is already sized for evaluation, and `_start` would call `setup` on it
        a second time with the training env count. So this builds the same model the agent holds,
        by the same builder and model class the config names, and stops there.
        """
        import numpy as np
        import torch

        from .models import MultiMorphLimbTransformerBuilder, TransformerMaskedNorm

        assert self.modlib is not None, \
            "assign_task_and_modlib before building the model; the network's widths are the " \
            "library's"
        env_cfg = self._cfg.get("env", {})
        layout = self.obs_layout()      # the checkpoint's if one is loaded, else the live Task's
        runtime.set_run(library=self.modlib,
                        base_morphology=seed_body(self.modlib,
                                                  **env_cfg.get("base_morphology", {})),
                        obs_layout=layout)

        ppo = self._cfg["params"]["config"]
        builder = MultiMorphLimbTransformerBuilder()
        builder.load(self._cfg["params"]["network"])
        model = TransformerMaskedNorm(builder).build({
            # Action width = padded module slots + the task's root axes (one action per module
            # slot, plus one per actuated world-mount axis). Read off the layout, not the Task's
            # action space, which does not exist until the Task is sized.
            "actions_num": layout["n_modules"] + layout["n_root_axes"],
            "input_shape": (layout["obs_total"],),
            "num_seqs": 1,
            "value_size": 1,
            "normalize_value": ppo.get("normalize_value", False),
            "normalize_input": ppo.get("normalize_input", True),
        })
        self._model = model.to(torch.device("cuda:0"))
        self._policy = TransformerControlPolicy(self._model)
        self._generator = TransformerMorphologyGenerator(self._model.a2c_network.net, self.modlib)
        torch.cuda.empty_cache()

    # ---- the driver contract ---------------------------------------------------------

    def run(self):
        if self._agent is None:
            self._start()
        try:
            reward, _epoch = next(self._iter)
        except StopIteration:                    # the loop yields before returning, so only a
            self._agent._train_finished = True   # second call past the end lands here
            reward = self._agent.mean_rewards
        reward = float(reward)
        # rl_games spells "no episode has finished yet" as -1e9. Reported as -inf instead: it means
        # the same thing to the driver's best-so-far comparison, and reads as absent rather than as
        # a real, catastrophic reward.
        return (float("-inf") if reward <= -1e9 else reward), self._policy, self._generator

    def is_finished(self) -> bool:
        return self._agent is not None and self._agent._train_finished

    def attach_progress(self, callback) -> None:
        super().attach_progress(callback)
        if self._agent is not None:                      # re-attached after the agent was built
            self._agent._on_iteration = callback

    def control_policy(self) -> ControlPolicy:
        if self._policy is None:
            self._build_inference_model()
        return self._policy

    def morphology_generator(self) -> MorphologyGenerator:
        if self._generator is None:
            self._build_inference_model()
        return self._generator

    def current_morphologies(self) -> Optional[List[Morphology]]:
        """The window's built bodies -- what the returns just reported were actually measured on,
        not a fresh draw from the generator."""
        if self._agent is None or self._agent._cur_counts is None:
            return None
        from .morphology import designs_from_arrays
        a = self._agent
        return designs_from_arrays(self.modlib, a._cur_counts, a._cur_eff, a._cur_cap)

    # ---- checkpoints -----------------------------------------------------------------

    def checkpoint_payload(self) -> dict:
        """The artifacts' weights, not the run's resume state (D19).

        `model` carries both heads and the observation normalizer, since they share one trunk and a
        policy loaded without its normalizer reads a different observation than it trained on. The
        generator's window counter rides along because it is what a reader needs to know whether the
        generator is still on its warmup teacher.
        """
        a = self._agent
        return {"model": a.model.state_dict(), "gen_window": a._gen_window, "epoch": a.epoch_num}

    def load_checkpoint_payload(self, payload: dict) -> None:
        if self._model is None:
            self._build_inference_model()
        # `_orig_mod.` is torch.compile's prefix. Stripped rather than matched, so a checkpoint
        # written from a compiled run loads into an uncompiled model and vice versa -- the two
        # differ only in the wrapper, and refusing the pair would be refusing over nothing.
        state = {k.removeprefix("_orig_mod."): v for k, v in payload["model"].items()}
        self._model.load_state_dict(state)
        if self._agent is not None:
            self._agent._gen_window = int(payload.get("gen_window", 0))
