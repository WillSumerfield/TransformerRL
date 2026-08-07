import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "vlearn-main" / "train"))

import torch
import vlearn as v
from envs.ant_environment_common import (
    compute_reward_termination_truncation_helper,
)

from ..codesign_environment import CodesignEnvironmentGpu
from ..modular_libraries.simple import Morphology, SimpleModuleLibrary


class AntCodesignEnv(CodesignEnvironmentGpu):

    _MODULE_LIBRARY = SimpleModuleLibrary(root_axes=None)   # free-floating base
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
        # Reward needs only root pose plus the already-global act/old-root/progress buffers, so it
        # runs whole-tensor (the helper is elementwise per env).
        # The healthy Y band is ASYMMETRIC in body height: the CEILING scales with the spawn lift (a
        # long-legged ant stands taller), but the FLOOR is ABSOLUTE — a fallen ant's torso is near the
        # ground regardless of leg length (only the limbs are longer, not the torso). So we pass a
        # Y-normalized pose to the helper (giving the correct relative ceiling + untouched forward
        # reward from X), then revive any body the shifted floor wrongly killed while its torso is
        # still above the absolute floor. (Shifting BOTH bounds killed standing tall ants at normY<0.3.)
        root_pose = self._get_root_pose.clone()
        root_pose[:, 5] -= self._height_offset

        self.reward_term_trunc_helper_jit(
            self._rew_buf,
            self._next_term_buf,
            self._next_trunc_buf,
            self._act_buf,
            root_pose,
            self.old_root_pos_buf,
            self._progress_buf,
            self.healthy_y_range,
            self.healthy_reward_val,
            self.dt,
            self.ctrl_cost_weight,
            self.max_episode_length,
        )
        
        lo = self.healthy_y_range[0]

        # helper killed it on the (shifted) floor, but the real torso is above the absolute floor:
        wrongly_low = self._next_term_buf & (root_pose[:, 5] < lo) & (self._get_root_pose[:, 5] >= lo)
        self._next_term_buf[:] = self._next_term_buf & ~wrongly_low
        self._rew_buf[:] = self._rew_buf + self.healthy_reward_val * wrongly_low.to(self._rew_buf.dtype)   # restore healthy bonus

    @staticmethod
    def reward_term_trunc_helper(
        rew_buf: torch.Tensor,
        term_buf: torch.Tensor,
        trunc_buf: torch.Tensor,
        act: torch.Tensor, 
        root_pose: torch.Tensor, 
        old_root_pos: torch.Tensor, 
        progress: torch.Tensor, 
        healthy_y_range: tuple[float, float],
        healthy_reward: float, 
        dt: float, 
        ctrl_cost_weight: float, 
        max_episode_length: int
    ):
        rew_buf[:], term_buf[:], trunc_buf[:] = compute_reward_termination_truncation_helper(
            act, root_pose, old_root_pos, progress, healthy_y_range,
            healthy_reward, dt, ctrl_cost_weight, max_episode_length
        )

    def set_root_pose(self, group: dict, start: int, end: int):
        """Write the root pose to the set buffers (self._set_root_pose) for all envs. Called at reset and after resample()."""
        
        # Spawn higher when the longest limb exceeds the default 2-module chain, so long legs
        # don't clip through the ground at reset. Constant +one module length per extra module (read
        # off the morphology itself, not a module_library constant -- module length is currently
        # uniform across effector types); no lift when the longest limb is <= 2 modules. Index 5 =
        # root Y (up-axis) in [qx,qy,qz,qw, px,py,pz]. store_initial_conditions returns a fixed
        # height, so we add the offset to the reset pose (which governs — envs reset before stepping).
        morph = group["morph"]
        longest = max((morph.num_modules(n) for n in group["active"]), default=0)
        offset = 0.0
        if longest > 2:
            offset = morph.module_lengths[group["active"][0]][0] * (longest - 2)
        self._set_root_pose[start:end, 5] += offset
        self._height_offset[start:end] = offset
        self._global_dof_mask[start:end] = group["dof_mask"]
