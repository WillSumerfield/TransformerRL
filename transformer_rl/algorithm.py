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
from codesigner.metrics import DESIGN_QUALITY_PREDICTOR, SPREAD_CONTROL, provides

from . import runtime
from .artifacts import TransformerControlPolicy, TransformerMorphologyGenerator
from .morphology import arrays_from_designs, designs_from_arrays, seed_body
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
                 name: str = "shared_trunk_codesign", experiment: str = _EXPERIMENT):
        super().__init__(name, "PPG codesign: control and generator heads over one transformer "
                               "trunk, updated jointly at every rebuild boundary.")
        self.config_path = Path(config)
        self.seed = seed
        # Which experiment directory this run belongs to, i.e. `runs/<task>_codesign/<experiment>`.
        # A parameter and not the module constant because it is a property of the RUN, not of the
        # algorithm class: experiment 5's baselines are this algorithm constrained from outside and
        # must not land in the codesign study's directory. The launcher derives its own
        # done-detection and resume from that path, so the two disagreeing is silent -- every run
        # reads "fresh" forever while writing somewhere else entirely.
        self._experiment = experiment
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
        # Weights restored before the agent existed (the `refine_control` ordering), replayed onto
        # the agent's model once `_start` has built one over the top of them.
        self._pending_state = None
        self._pending_gen_window = 0
        # How far the run being continued had got. A resume rebuilds the agent from scratch and
        # rl_games counts epochs on the agent, so without this a crashed run would restart its
        # epoch counter at zero and spend a SECOND full `max_epochs` -- doubling the frame budget
        # the cross-method comparison asserts, and doing it silently.
        self._pending_epoch = 0
        self._evaluated = None          # the bodies the last run()'s returns were earned on

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
        identity = _compose_identity(self._task_key, _FAMILY, self._experiment)

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

    # ---- what a subclass swaps out ---------------------------------------------------
    # A baseline that shares this run's network, config, Task and rl_games stack but not its body
    # source differs in exactly three things: which agent closes the window, what the FIRST window
    # is built on, and which artifact answers "draw me a body". Named here so such an arm is a
    # subclass with three short overrides rather than a second copy of `_start`.

    def _agent_class(self):
        from .codesign_agent import CodesignAgent
        return CodesignAgent

    def _initial_bodies(self, base_morphology: Morphology, num_morphs: int) -> List[Morphology]:
        """Window 0's population, when no fixed set was imposed. The seed body, everywhere: the
        generator has learnt nothing yet, so window 0 is the teacher's."""
        return [base_morphology] * num_morphs

    def _make_generator(self, net, library) -> MorphologyGenerator:
        return TransformerMorphologyGenerator(net, library)

    def _start(self) -> None:
        """Size the Task, build the rl_games stack, and stop just short of training."""
        import torch
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.ivecenv import IVecEnv
        from rl_games.algos_torch import model_builder as mb_module
        from rl_games.torch_runner import Runner
        from vlearn.torch_utils.wrappers import NewToOldAPICompatilibity

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
        # A fixed set is a WHITELIST, not a layout: it says what may be built and never how many
        # groups each body gets, so it is tiled across the population already sized above. One fixed
        # body therefore means every group runs it, rather than one group running and the rest idle.
        bodies = (self._initial_bodies(base_morphology, num_morphs)
                  if self.fixed_morphologies is None else
                  [self.fixed_morphologies[i % len(self.fixed_morphologies)]
                   for i in range(num_morphs)])

        def create_envs(n, **kw):
            built = task.setup(library, n, num_morphs, bodies, seed=seed)
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
        agent_class = self._agent_class()

        def build_agent(**kwargs):
            agent = agent_class(**kwargs)
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
        if self._pending_state is not None:
            # `refine_control` restores the payload before any agent exists, and `run_train` has
            # just built a fresh network over the top of it -- so the restore is replayed here or
            # the refinement silently starts from random weights.
            _load_state(self._agent.model, self._pending_state)
            self._agent._gen_window = self._pending_gen_window
            self._pending_state = None
        self._agent.epoch_num = self._pending_epoch
        self._agent._built_morphs = list(bodies)      # what `create_envs` just stood the sim up on
        self._agent._carry_ep_returns = True          # the driver reads Episode return every tick
        if self.fixed_morphologies is not None:
            self._pin_bodies(bodies)
        self._iter = self._agent._train_iter()
        self._model = self._agent.model
        self._policy = TransformerControlPolicy(self._model)
        self._generator = self._make_generator(self._agent._net(), library)
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
        self._generator = self._make_generator(self._model.a2c_network.net, self.modlib)
        torch.cuda.empty_cache()

    # ---- the driver contract ---------------------------------------------------------

    def run(self):
        if self._agent is None:
            self._start()
        window = self._agent._gen_window
        try:
            reward, _epoch = next(self._iter)
        except StopIteration:                    # the loop yields before returning, so only a
            self._agent._train_finished = True   # second call past the end lands here
            reward = self._agent.mean_rewards
        # What the returns the driver is about to take were earned on. A run that ended AT a window
        # boundary closed one -- the sim already holds the next window's bodies, so the set that ran
        # is the previous one; a run that ended because training did (max_epochs) never swapped, so
        # the set that ran is the one still standing. Read here rather than in
        # `current_morphologies` because only `run` can tell the two apart.
        a = self._agent
        self._evaluated = a._ran_morphs if a._gen_window != window else a._built_morphs
        reward = float(reward)
        # rl_games spells "no episode has finished yet" as -1e9. Reported as -inf instead: it means
        # the same thing to the driver's best-so-far comparison, and reads as absent rather than as
        # a real, catastrophic reward.
        return (float("-inf") if reward <= -1e9 else reward), self._policy, self._generator

    def is_finished(self) -> bool:
        """Whether this run is over -- answerable BEFORE the agent exists, which is when a resume
        asks. `_resume_from` checks this to refuse continuing a finished run, and an algorithm that
        always said "no" there would answer by running the whole budget again."""
        if self._agent is None:
            return self._pending_epoch >= self._cfg["params"]["config"]["max_epochs"]
        return self._agent._train_finished

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

    def _pin_bodies(self, bodies: List[Morphology]) -> None:
        """Write `bodies` (one per morph group) into the agent's window state and stop it resampling.

        The agent tracks its built bodies per ENV, not per group, and the Task lays groups out as
        contiguous blocks of `envs_per_morph` -- the same layout it publishes as `design_layout` --
        so the per-group list expands by `repeat_interleave` and not by tiling, which would pair
        every env with the wrong body wherever a group holds more than one.
        """
        a = self._agent
        counts, eff, cap = arrays_from_designs(self.modlib, bodies, a._max_len,
                                               a._cur_counts.device)
        per_group = a._cur_counts.shape[0] // len(bodies)
        a._cur_counts = counts.repeat_interleave(per_group, dim=0)
        a._cur_eff = eff.repeat_interleave(per_group, dim=0)
        a._cur_cap = cap.repeat_interleave(per_group, dim=0)
        a._fixed = True
        # The authoritative record of what stands, kept beside the arrays because the arrays round
        # an uncapped limb to a bare cap (see `arrays_from_designs`) and `current_morphologies` is a
        # statement about what the returns were measured ON.
        a._built_morphs = list(bodies)

    def apply_fixed_morphologies(self, bodies: List[Morphology]) -> None:
        """Enter the fixed-morphology phase: build only `bodies`, and start counting again.

        Before `_start` this is only a record -- `_start` reads `fixed_morphologies` and stands the
        Task up on it directly, which is both cheaper and the only order that works, since the first
        build is what sizes rl_games' rollout buffers. After `_start` it rebuilds onto the set.

        What actually holds the body still is `CodesignAgent._fixed`, set by `_pin_bodies`: the
        window boundary keeps ticking (it is what bounds one `run()`) but the resample draws
        nothing, updates nothing and rebuilds nothing. Control PPO is untouched and runs every
        epoch, which is the whole point of the phase.

        The generator is NOT frozen and is not claimed to be -- the two heads share a trunk, so
        control updates move it. That is the drift the package's contract explicitly permits: this
        run returns no generator, and the checkpoint it started from is not written to.
        """
        if self._agent is None:
            # Before `_start`, entering the phase is only a record -- but it is still a new phase,
            # so a restored epoch count belongs to the run this one starts FROM and must not carry.
            # This is what keeps `refine_control` at a clean 0..max_epochs on a checkpoint whose
            # training run had already spent its budget.
            self._pending_epoch = 0
            return
        a = self._agent
        env = a._env()
        tiled = [bodies[i % len(bodies)] for i in range(env.n_morphs)]
        env.resample(tiled)
        a.obs = a.env_reset()
        self._pin_bodies(tiled)
        # A new phase, not a continuation: `is_finished` is checked before the first `run()`, so an
        # algorithm that just exhausted its codesign budget would otherwise refuse to refine at all.
        a._train_finished = False
        a.epoch_num = 0

    def current_morphologies(self) -> Optional[List[Morphology]]:
        """The window's built bodies -- what the returns just reported were actually measured on,
        not a fresh draw from the generator.

        One entry per DESIGN, matching the Task's `design_layout`, because that is what the driver
        pairs its per-design returns against. Handed back as the list given to `setup`/`resample`
        rather than decoded out of the agent's arrays: the arrays cannot express an uncapped limb,
        and by the time a window closes they describe the NEXT window's design anyway.
        """
        return self._evaluated

    # ---- capabilities ----------------------------------------------------------------

    @property
    def _reward_scale(self) -> float:
        """The reward shaper's scale, read off the config rather than off the agent so the two
        capabilities work on an inference-only build too (`evaluate`, and every protocol)."""
        shaper = self._cfg["params"]["config"].get("reward_shaper") or {}
        return float(shaper.get("scale_value", 1.0))

    @provides(SPREAD_CONTROL)
    def spread_at(self, spread: float, n: int) -> List[Morphology]:
        """`n` bodies at a normalized spread, drawn on the generator's own frontier MDP.

        The knob is an inverse temperature on the GenAct logits, `beta = (1 - spread) / spread`:
        spread 1 is `beta = 0`, zero logits, a uniform draw over the grammar-valid token set at
        every step, and it is monotone downwards from there because flattening a categorical only
        widens it. Half-way is `beta = 1` -- the trained distribution -- which is a useful thing to
        know when reading a ladder but is not what the axis is reported on.

        Spread 0 is the one value NOT taken from the sampler. `beta = inf` is the argmax at every
        step, but the MDP visits still-growable limbs in a RANDOM order, so n greedy draws are
        near-identical rather than identical and level 0 would carry real design variance. The
        contract asks for the deterministic draw of one, repeated, and `PerturbationLadder` reads
        its noise floor off level 0 on exactly that basis -- every design being the same design is
        what makes the spread of scores there pure episode noise. So it is drawn once and repeated.

        One honest caveat about the top end. Uniform over the valid TOKENS is not identical to
        `ModuleLibrary.random_morphology`, uniform over BODIES: walking the MDP with flat logits
        induces its own chain-length distribution. It does not affect the ladder, whose top rung is
        the library's draw regardless of the algorithm and whose rungs are reported by measured
        Morphology distance, but a caller reading `spread_at(1.0, n)` as the library's uniform draw
        would be reading it slightly wrong.
        """
        gen = self.morphology_generator()
        if spread <= 0.0:
            return gen.generate(1, deterministic=True) * n
        trace = gen.net.sample(n, beta=(1.0 - spread) / spread)
        return designs_from_arrays(gen.library, trace["counts"].long(),
                                   trace["eff_sub"], trace["cap_sub"])

    @provides(DESIGN_QUALITY_PREDICTOR)
    def predict_design_quality(self, designs: List[Morphology]) -> List[float]:
        """GenCrit at each finished body -- what the generator believes it is worth before it runs.

        `v(full)` is the value head read at the COMPLETE design rather than at a prefix, which is
        the only point on the frontier MDP where the prediction is about a body a Task could build.

        In Episode-return units, as the capability demands. GenCrit regresses
        `_window_Ri() * _r_scale`, so its output is in the control critic's shaped units and the
        scale divides back out. That inversion is exact rather than approximate here: every config
        in this project shapes reward by `scale_value` alone, with no shift and no clipping, so the
        whole of the shaping is one multiplication.
        """
        import torch

        net = self.morphology_generator().net
        device = next(net.parameters()).device
        counts, eff, cap = arrays_from_designs(self.modlib, designs, net.max_limb_length, device)
        with torch.no_grad():
            H = net._encode_design(counts, cap, eff)          # (M, n_tokens, d); CLS is token 0
            v = net.gencrit_head(H[:, 0]).squeeze(-1)
        return (v / self._reward_scale).tolist()

    # ---- provenance ------------------------------------------------------------------

    @property
    def config(self) -> dict:
        """What this run was configured with, resolved rather than as written.

        The resolved dict, because that is what actually ran: a config that `extends` another is
        two files and a merge order, and a reader months later has neither. The three fields beside
        it are what the resolution cannot recover -- which file was named, what the caller layered
        on top, and the seed, which is the commonest reason two otherwise identical records differ.
        """
        return {"config_path": str(self.config_path), "seed": self.seed,
                "run_name": self._run_name, "overrides": self._overrides, "resolved": self._cfg}

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
        _load_state(self._model, payload["model"])
        self._pending_gen_window = int(payload.get("gen_window", 0))
        self._pending_epoch = int(payload.get("epoch", 0))
        if self._agent is not None:
            self._agent._gen_window = self._pending_gen_window
            self._agent.epoch_num = self._pending_epoch
        else:
            # No agent yet -- the `refine_control` / `optimize_control` ordering, where the payload
            # is restored before anything is trained. Held so `_start` can replay it onto the model
            # `run_train` is about to build.
            self._pending_state = payload["model"]


def _load_state(model, state: dict) -> None:
    """Load weights into `model` regardless of which side torch.compile wrapped.

    `_orig_mod.` is torch.compile's prefix. Normalized both ways rather than matched, so a
    checkpoint written from a compiled run loads into an uncompiled model and vice versa -- the two
    differ only in the wrapper, and refusing the pair would be refusing over nothing.
    """
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    if any(k.startswith("_orig_mod.") for k in model.state_dict()):
        state = {f"_orig_mod.{k}": v for k, v in state.items()}
    model.load_state_dict(state)
