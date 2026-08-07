"""MultiGroupEnvironmentGpu: EnvironmentGpu extended to support multiple EnvironmentGroups."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "vlearn-main" / "train"))

import numpy as np
import random
import torch
from math import ceil
from abc import ABC, abstractmethod
from hashlib import sha256
from typing import Union
from vlearn.spaces import Box

import vlearn as v


class RenderFinished(Exception):
    """Sentinel raised by render() to signal an intended viewer shutdown.

    Distinguishes a clean stop (window closed, --num-episodes cap, video budget)
    from a real crash. Caught at the play loop's runner.run() and in _run_random
    (see docs/adr/0003); any other Exception propagates as a genuine error.
    """


class MultiGroupEnvironmentGpu(ABC):

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        max_episode_length: int,
        gravity: v.Vec3,
        timestep: float,
        frame_skip: int,
        spacing: float,
        max_contact_pairs_per_env: int,
        rendering: bool = False,
        enable_scene_query: bool = False,
        treat_warning_as_error: bool = False,
        initial_is_paused: bool = False,
        initial_render_substep: bool = True,
        print_hash: bool = False,
        up_axis: v.Vec3 = v.Vec3(0.0, 1.0, 0.0),
        with_window: bool = True,
        max_contact_patches_per_env: int = -1,
        max_contact_points_per_patch: int = 4,
        update_scene_dependent_components_in_step: bool = True,
        seed: int = None,
        verbose: bool = True,
        raise_exception: bool = True,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        assert device.type == "cuda"

        self.num_envs = num_envs
        self.device = device
        self.rendering = rendering
        self.enable_scene_query = enable_scene_query
        self.max_episode_length = max_episode_length
        self.frame_skip = frame_skip
        self.spacing = spacing
        self.gravity = gravity
        self.print_hash = print_hash
        self.raise_exception = raise_exception

        self._observation_space = None
        self._privileged_observation_space = None
        self._action_space = None

        if seed is None:
            seed = torch.seed() % (2**32)

        assert torch.cuda.is_available()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        if max_contact_patches_per_env == -1:
            max_contact_patches_per_env = max_contact_pairs_per_env

        # Saved so the gym can be torn down and recreated in place (morphology resampling needs a
        # full rebuild — geometry is baked at finalize). total_num_envs is invariant, so the
        # contact-buffer sizes are fixed at construction.
        self._gym_params = dict(
            treat_warning_as_error=treat_warning_as_error,
            up_axis=up_axis,
            with_window=with_window,
            max_contact_pairs=max_contact_pairs_per_env * self.total_num_envs,
            max_contact_patches=max_contact_patches_per_env * self.total_num_envs,
            max_contact_points=max_contact_points_per_patch
            * max_contact_patches_per_env
            * self.total_num_envs,
            update_scene_dependent_components_in_step=update_scene_dependent_components_in_step,
            cuda_device=device.index,
            seed=seed,
            verbose=verbose,
        )
        self._timestep = timestep
        self._initial_is_paused = initial_is_paused
        self._initial_render_substep = initial_render_substep

        self.up_axis = up_axis
        self.up_axis_rotation = v.shortest_rotation(up_axis, v.Vec3(0, 1, 0))

        self._create_gym()

        # Multi-group support
        self.env_groups = []

    def _create_gym(self):
        """Create the gym + render and apply timestep/gravity. Called at init and on rebuild."""
        self.gym = v.create_gym(
            self.rendering, self.enable_scene_query,
            enable_graph_captures=True, **self._gym_params,
        )
        self.gym.set_timestep(self._timestep)
        self.gym_render = self.gym.get_render() if self.rendering else None
        self._render_finished = False
        if self.rendering:
            self.gym_render.set_paused(self._initial_is_paused)
            self.render_substep = self._initial_render_substep
        self.gym.set_gravity(self.gravity)

    # ---- Space accessors -------------------------------------------------------

    @property
    def observation_space(self):
        if self._observation_space is None:
            raise AttributeError("observation_space has not been set yet")
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space):
        if not isinstance(space, Box):
            raise TypeError("observation_space must be Box")
        self._observation_space = space

    @property
    def action_space(self):
        if self._action_space is None:
            raise AttributeError("action_space has not been set yet")
        return self._action_space

    @action_space.setter
    def action_space(self, space):
        if not isinstance(space, Box):
            raise TypeError("action_space must be Box")
        self._action_space = space

    # ---- Multi-group construction ----------------------------------------------

    def create_env_group(
        self,
        env_def_handle: v.EnvironmentDefHandle,
        num_envs_in_group: int,
        group_offset: v.Vec3 = None,
        env_rot: v.Quat = v.Quat(0, 0, 0, 1),
    ) -> v.EnvironmentGroup:
        """Create one EnvironmentGroup and place envs on a grid with group_offset."""
        if group_offset is None:
            group_offset = v.Vec3(0, 0, 0)

        assert self.gym.get_environment_def(env_def_handle).is_finalized()

        env_group = self.gym.create_environment_group(env_def_handle, [num_envs_in_group])
        self.env_groups.append(env_group)

        env_set = list(env_group.get_environment_sets())[0]
        cols = max(1, ceil(num_envs_in_group ** 0.5))
        for i in range(num_envs_in_group):
            col = i % cols
            row = i // cols
            pos = v.Vec3(
                group_offset.x + col * self.spacing,
                0.0,
                group_offset.z + row * self.spacing,
            )
            env_handle = env_set.get_environment_handle(i)
            env = env_set.get_environment(env_handle)
            env.set_transform(v.Transform(env_rot, pos))

        return env_group

    # ---- Buffer allocation -----------------------------------------------------

    def allocate_buffers(self):
        self._obs_buf = torch.zeros(
            (self.total_num_envs,) + self.observation_space.shape,
            dtype=torch.float32, device=self.device,
        )
        self._rew_buf = torch.zeros(self.total_num_envs, dtype=torch.float32, device=self.device)
        self._term_buf = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        self._next_term_buf = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        self._trunc_buf = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        self._next_trunc_buf = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        self._progress_buf = torch.zeros(self.total_num_envs, dtype=torch.long, device=self.device)
        self._act_buf = torch.zeros(
            (self.total_num_envs,) + self.action_space.shape,
            device=self.device, dtype=torch.float32,
        )
        self._reset_buf = torch.ones(self.total_num_envs, dtype=torch.bool, device=self.device)
        self._info = {"extras": {}}

    # ---- Buffer read-only properties ------------------------------------------

    @property
    def obs_buf(self):
        return self._obs_buf

    @property
    def rew_buf(self):
        return self._rew_buf

    @property
    def term_buf(self):
        return self._term_buf

    @property
    def next_term_buf(self):
        return self._next_term_buf

    @property
    def trunc_buf(self):
        return self._trunc_buf

    @property
    def next_trunc_buf(self):
        return self._next_trunc_buf

    @property
    def progress_buf(self):
        return self._progress_buf

    @property
    def act_buf(self):
        return self._act_buf

    @property
    def reset_buf(self):
        return self._reset_buf

    @property
    def info(self):
        return self._info

    # ---- Step loop ------------------------------------------------------------

    def step(self, actions):
        if self.print_hash:
            print("actions: {}".format(sha256(actions.cpu().numpy().tobytes()).hexdigest()))

        if self.rendering and not self.render_substep:
            self.render()

        self.pre_physics_step(actions)

        for _ in range(self.frame_skip):
            if self.rendering and self.render_substep:
                self.render()
            self.pre_gym_step()
            self.gym.step()
            self.post_gym_step()

        if self.rendering:
            self.gym_render.set_step(False)

        self.post_physics_step()

        return self.obs_buf, self.rew_buf, self.term_buf, self.trunc_buf, self.info

    @abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        pass

    def pre_gym_step(self):
        pass

    def post_gym_step(self):
        pass

    @abstractmethod
    def post_physics_step(self):
        pass

    def reset(self, compute_kinematics=True):
        self.reset_buf[:] = True
        self.reset_idx()
        if compute_kinematics:
            self.gym.compute_kinematics()
        return self.obs_buf.clone(), {}

    @abstractmethod
    def reset_idx(self):
        pass

    def inference_mode_post_init_callback(self):
        pass

    # ---- Utilities ------------------------------------------------------------

    @property
    def total_num_envs(self) -> int:
        if isinstance(self.num_envs, int):
            return self.num_envs
        elif isinstance(self.num_envs, list):
            return sum(self.num_envs)
        raise ValueError(f"self.num_envs must be int or list[int], got {type(self.num_envs)}")

    @property
    def render_finished(self) -> bool:
        return self._render_finished

    @render_finished.setter
    def render_finished(self, new_val: bool) -> None:
        assert isinstance(new_val, bool)
        if new_val:
            self._render_finished = True

    @property
    def render_substep(self) -> bool:
        return self._render_substep

    @render_substep.setter
    def render_substep(self, new_val: bool) -> None:
        assert self.rendering
        self._render_substep = new_val
        if self._render_substep:
            self.gym_render.set_render_timestep(self.timestep)
        else:
            self.gym_render.set_render_timestep(self.dt)

    def render(self):
        self.render_finished = self.gym_render.render(self.render_callback)
        if self.render_finished and self.raise_exception:
            raise RenderFinished

    def render_callback(self):
        pass

    @property
    def timestep(self) -> float:
        return self.gym.get_timestep()

    @property
    def dt(self) -> float:
        return self.timestep * self.frame_skip
