import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "vlearn-main" / "train"))

import torch
import vlearn as v

from ..codesign_environment import CodesignEnvironmentGpu
from ..modular_libraries.simple import Morphology, SimpleModuleLibrary


class GraspCodesignEnv(CodesignEnvironmentGpu):

    # World-mounted hand: vertical (y) + lateral (x) actuated approach axes over the grasp target.
    _MODULE_LIBRARY = SimpleModuleLibrary(root_axes=['y', 'x'])
    _BASE_MORPHOLOGY = Morphology.from_design(
        _MODULE_LIBRARY,
        effector_types={
            1: ["swing", "knee"],
            4: ["swing", "knee"],
            6: ["swing", "knee"],
        },
        cap_types={
            1: "bare",
            4: "bare",
            6: "bare",
        }
    )

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        max_episode_length: int = 1000,
        gravity: v.Vec3 = v.Vec3(0, -9.81, 0),
        timestep: float = 0.01667,
        frame_skip: int = 1,
        spacing: float = 3.0,
        max_contact_pairs_per_env: int = 64,
        reset_noise_scale: float = 1.0,
        ctrl_cost_weight: float = 0.5,
        healthy_reward: float = 2.0,
        healthy_y_range: tuple = (0.3, 1.1),
        module_library=None,
        base_morphology: Morphology = None,
        **kwargs,
    ):
        super().__init__(
            num_envs=num_envs,
            device=device,
            max_episode_length=max_episode_length,
            gravity=gravity,
            timestep=timestep,
            frame_skip=frame_skip,
            spacing=spacing,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            reset_noise_scale=reset_noise_scale,
            module_library=module_library or self._MODULE_LIBRARY,
            base_morphology=base_morphology or self._BASE_MORPHOLOGY,
            **kwargs
        )

        # Custom reward variables
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward_val = healthy_reward
        self.healthy_y_range = healthy_y_range

    def compute_reward_termination_truncation(self):
        self.reward_term_trunc_helper_jit(
            self._rew_buf,
            self._act_buf,
            self._next_term_buf,
            self._next_trunc_buf,
            self._get_root_pose,
            self.old_root_pos_buf,
            self._progress_buf,
            self.healthy_y_range,
            self.healthy_reward_val,
            self.dt,
            self.ctrl_cost_weight,
            self.max_episode_length,
        )

    @staticmethod
    def reward_term_trunc_helper(
        rew_buf: torch.Tensor,
        act_buf: torch.Tensor, 
        term_buf: torch.Tensor,
        trunc_buf: torch.Tensor,
        root_pose: torch.Tensor, 
        old_root_pos: torch.Tensor, 
        progress: torch.Tensor, 
        healthy_y_range: tuple[float, float],
        healthy_reward: float, 
        dt: float, 
        ctrl_cost_weight: float, 
        max_episode_length: int
    ):
        pass

    def set_root_pose(self, group: dict, start: int, end: int):
        """Write the root pose to the set buffers (self._set_root_pose) for all envs. Called at reset and after resample()."""
        
        morph = group["morph"]
        longest = max((morph.num_modules(n) for n in group["active"]), default=0)
        offset = 2.0
        if longest > 2:
            offset += morph.module_lengths[group["active"][0]][0] * (longest - 2)
        self._set_root_pose[start:end, 5] += offset
        self._height_offset[start:end] = offset
        self._global_dof_mask[start:end] = group["dof_mask"]
