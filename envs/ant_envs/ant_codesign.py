"""AntCodesignEnv: simultaneous morphology + controller training via one group per morphology."""
import sys
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "vlearn-main" / "train"))

import numpy as np
import torch
import vlearn as v
from vlearn.spaces import Box
from vlearn.torch_utils.command_handlers import ArticulationKinematicStateHandler
from envs.ant_environment_common import (
    store_initial_conditions_helper,
    compute_reward_termination_truncation_helper,
)
from envs.common import create_plane, reset_noise_helper

from ..codesign_environment import CodesignEnvironmentGpu
from .build_vsim import build_ant_vsim, write_vsim_tempfile


_N_DOFS_FULL = 16   # 8 hips + 8 ankles (DOF order: h1,a1,...,h8,a8)
_N_LEGS      = 8
_OBS_BASE    = 107  # 1+4+3+3+16+16+16+8*6
_MASK_DIM    = 16
_OBS_TOTAL   = _OBS_BASE + _MASK_DIM  # 123


def _stable_morphologies(
    min_legs: int = 3,
    max_legs: int = 8,
    max_gap_deg: float = 135.0,
) -> list[frozenset]:
    """Return stable morphologies with min_legs..max_legs legs and no circular gap > max_gap_deg."""
    result = []
    for mask in range(1, 256):
        active = frozenset(i + 1 for i in range(8) if (mask >> i) & 1)
        if not (min_legs <= len(active) <= max_legs):
            continue
        angles = sorted((n - 1) * 45.0 for n in active)
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append(360.0 - angles[-1] + angles[0])
        if max(gaps) <= max_gap_deg:
            result.append(active)
    return result


class AntCodesignEnv(CodesignEnvironmentGpu):
    """One EnvironmentGroup per stable morphology. Obs is 123D; actions are always 16D."""

    @property
    def unwrapped(self):
        return self

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        rendering: bool = False,
        enable_scene_query: bool = False,
        max_episode_length: int = 1000,
        ctrl_cost_weight: float = 0.5,
        healthy_reward: float = 1.0,
        healthy_y_range: tuple = (0.3, 1.1),
        reset_noise_scale: float = 1.0,
        gravity: v.Vec3 = v.Vec3(0, -9.81, 0),
        timestep: float = 0.01667,
        frame_skip: int = 1,
        spacing: float = 2.0,
        max_contact_pairs_per_env: int = 64,
        with_window: bool = True,
        seed: int = None,
        raise_exception: bool = True,
        morphologies: list[frozenset] | None = None,
        **kwargs,
    ):
        self._morphologies = morphologies if morphologies is not None else _stable_morphologies()
        self.n_morphs = len(self._morphologies)
        self.envs_per_morph = max(1, num_envs // self.n_morphs)
        total_envs = self.n_morphs * self.envs_per_morph

        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward_val = healthy_reward
        self.healthy_y_range = healthy_y_range
        self.reset_noise_scale = reset_noise_scale

        super().__init__(
            num_envs=total_envs,
            device=device,
            rendering=rendering,
            enable_scene_query=enable_scene_query,
            max_episode_length=max_episode_length,
            timestep=timestep,
            frame_skip=frame_skip,
            spacing=spacing,
            gravity=gravity,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            with_window=with_window,
            seed=seed,
            raise_exception=raise_exception,
            verbose=False,
        )

        self.observation_space = Box(
            low=np.full(_OBS_TOTAL, np.finfo("f").min, dtype=np.float32),
            high=np.full(_OBS_TOTAL, np.finfo("f").max, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=np.full(_N_DOFS_FULL, -1.0, dtype=np.float32),
            high=np.full(_N_DOFS_FULL, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._tempfiles = []
        self.groups = []
        self.create_envs()
        self.allocate_buffers()

        create_plane(self.gym)
        self.gym.set_num_solver_iterations(8)
        self.gym.finalize()

        if self.rendering:
            epm = self.envs_per_morph
            cols = max(1, ceil(epm ** 0.5))
            rows = max(1, ceil(epm / cols))
            sx = (cols + 2) * self.spacing
            sz = (rows + 2) * self.spacing
            n_gc = max(1, ceil(len(self._morphologies) ** 0.5))
            s = n_gc * max(sx, sz) * 0.5
            self.gym_render.reset_camera(v.Vec3(s, 2 * s, s), v.Vec3(1e-5, -1, 0))

    # ---- Environment construction ---------------------------------------------

    def create_envs(self):
        up_axis     = v.Vec3(0, 1, 0)
        rot0        = v.shortest_rotation(v.Vec3(0, 0, 1), up_axis)
        init_tf     = v.Transform(rot0, 0.75 * up_axis)

        epm          = self.envs_per_morph
        cols         = max(1, ceil(epm ** 0.5))
        rows_per_grp = max(1, ceil(epm / cols))
        stride_x     = (cols + 2) * self.spacing
        stride_z     = (rows_per_grp + 2) * self.spacing
        n_grp_cols   = max(1, ceil(len(self._morphologies) ** 0.5))

        for gi, morph in enumerate(self._morphologies):
            active = sorted(morph)
            n_dofs = len(active) * 2

            xml     = build_ant_vsim(morph)
            tmpfile = write_vsim_tempfile(xml)
            self._tempfiles.append(tmpfile)

            name           = f"ant_morph_{gi}"
            env_def_handle = self.gym.create_environment_def(name)
            env_def        = self.gym.get_environment_def(env_def_handle)
            env_def.import_definitions(str(tmpfile), False)

            arti_def_handle = env_def.get_articulation_def_handle_by_name("torso")
            arti_handle     = env_def.create_articulation(arti_def_handle, init_tf, "crawler")
            art_def         = env_def.get_articulation_def(arti_def_handle)
            art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)
            env_def.finalize()

            group_offset = v.Vec3((gi % n_grp_cols) * stride_x, 0, (gi // n_grp_cols) * stride_z)
            env_group    = self.create_env_group(env_def_handle, epm, group_offset)

            # Scatter maps into full 16-DOF space (DOF order: h1,a1,...,h8,a8)
            dof_indices = torch.tensor(
                [j for n in active for j in (2 * (n - 1), 2 * (n - 1) + 1)],
                dtype=torch.long, device=self.device,
            )  # (n_dofs,)

            # Scatter map for sensor forces into 48D slot (8 legs × 6)
            sensor_indices = torch.tensor(
                [j for n in active for j in range(6 * (n - 1), 6 * (n - 1) + 6)],
                dtype=torch.long, device=self.device,
            )  # (n_active_legs * 6,)

            # Limb mask: 1 for active DOFs, 0 otherwise
            limb_mask = torch.zeros(_N_DOFS_FULL, dtype=torch.float32, device=self.device)
            limb_mask[dof_indices] = 1.0

            self.groups.append({
                "morph":           morph,
                "active":          active,
                "n_dofs":          n_dofs,
                "env_group":       env_group,
                "art_def":         art_def,
                "arti_def_handle": arti_def_handle,
                "arti_handle":     arti_handle,
                "env_def_handle":  env_def_handle,
                "dof_indices":     dof_indices,
                "sensor_indices":  sensor_indices,
                "limb_mask":       limb_mask,
            })

    # ---- Buffer allocation ----------------------------------------------------

    def allocate_buffers(self):
        super().allocate_buffers()
        N   = self.total_num_envs
        EPM = self.envs_per_morph

        self.old_root_pos_buf = torch.zeros((N, 7), dtype=torch.float32, device=self.device)

        all_motor_cmds  = []
        all_sensor_cmds = []

        for gi, g in enumerate(self.groups):
            start = gi * EPM
            end   = start + EPM
            n_dofs = g["n_dofs"]

            # Initial conditions
            dof_pos_init, root_trans_init, _ = store_initial_conditions_helper(
                g["art_def"], self.device
            )
            g["dof_pos_init"] = torch.tile(dof_pos_init, (EPM, 1))  # (EPM, n_dofs)
            g["dof_vel_init"] = torch.zeros_like(g["dof_pos_init"])

            root_tf_init = torch.tensor(
                [root_trans_init.q.x, root_trans_init.q.y,
                 root_trans_init.q.z, root_trans_init.q.w,
                 root_trans_init.p.x, root_trans_init.p.y,
                 root_trans_init.p.z],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0).expand(EPM, -1).clone()  # (EPM, 7)
            root_vl_init = torch.zeros(EPM, 6, dtype=torch.float32, device=self.device)
            g["root_tf_init"] = root_tf_init
            g["root_vl_init"] = root_vl_init

            # Motor command buffer (EPM × n_dofs)
            motor_buf = torch.zeros((EPM, n_dofs), dtype=torch.float32, device=self.device)
            g["motor_buf"] = motor_buf
            motor_cmd = g["env_group"].create_motor_control_command(
                v.wrap_gpu_buffer(motor_buf), g["arti_handle"]
            )
            g["motor_cmd"] = motor_cmd
            all_motor_cmds.append(motor_cmd)

            # Kinematic state handler
            kh = ArticulationKinematicStateHandler(
                self.device,
                g["env_group"],
                g["arti_handle"],
                link_index_range=(0, 1),
                masks_buffer=self._reset_buf[start:end],
            )
            kh.set_link_pose_buf[:] = root_tf_init
            kh.set_link_vel_buf[:]  = root_vl_init
            g["kine_handler"] = kh

            # Force sensor commands
            env_def     = self.gym.get_environment_def(g["env_def_handle"])
            articulation = env_def.get_articulation(g["arti_handle"])
            n_sensors   = g["art_def"].get_num_force_sensor_defs()
            force_buffers = []
            for si in range(n_sensors):
                sensor_handle = articulation.get_force_sensor_handle(si)
                buf = torch.zeros((EPM, 6), dtype=torch.float32, device=self.device)
                force_buffers.append(buf)
                cmd = g["env_group"].create_force_sensor_command(
                    v.wrap_gpu_buffer(buf), sensor_handle
                )
                all_sensor_cmds.append(cmd)
            g["force_buffers"] = force_buffers
            g["n_sensors"]     = n_sensors

            # Pre-set constant limb_mask in obs_buf (slot 107:123, never overwritten in loop)
            self._obs_buf[start:end, _OBS_BASE:_OBS_TOTAL] = g["limb_mask"].unsqueeze(0)

        # Batch all motor / sensor commands across all groups into single GPU arrays
        self.all_motor_cmd_array  = self.gym.create_gpu_array(all_motor_cmds)
        self.all_sensor_cmd_array = self.gym.create_gpu_array(all_sensor_cmds) \
            if all_sensor_cmds else None

    # ---- Step methods ---------------------------------------------------------

    def pre_physics_step(self, actions: torch.Tensor):
        EPM = self.envs_per_morph
        for gi, g in enumerate(self.groups):
            start, end = gi * EPM, (gi + 1) * EPM
            # Save root pose before physics (from last get_articulation_kinematic_states)
            self.old_root_pos_buf[start:end] = g["kine_handler"].get_link_pose_buf
            # Mask inactive joints in action; write active DOFs to motor buffer
            self._act_buf[start:end] = actions[start:end] * g["limb_mask"]
            g["motor_buf"][:] = self._act_buf[start:end, g["dof_indices"]]
        self.gym.set_motor_forces(self.all_motor_cmd_array)

    def post_physics_step(self):
        self._progress_buf += 1
        self._reset_buf[:]      = self._next_term_buf | self._next_trunc_buf
        self._term_buf[:]       = self._next_term_buf
        self._trunc_buf[:]      = self._next_trunc_buf
        self.reset_idx()
        self.compute_observations(self._act_buf)
        self.compute_reward_termination_truncation(self._act_buf)

    # ---- Obs / reward ---------------------------------------------------------

    def compute_observations(self, actions: torch.Tensor):
        all_handlers = [g["kine_handler"] for g in self.groups]
        ArticulationKinematicStateHandler.get_articulation_kinematic_states(all_handlers)

        if self.all_sensor_cmd_array is not None:
            self.gym.get_sensor_forces(self.all_sensor_cmd_array)

        EPM = self.envs_per_morph
        for gi, g in enumerate(self.groups):
            start, end = gi * EPM, (gi + 1) * EPM
            obs = self._obs_buf[start:end]  # view into global obs_buf

            root_pos = g["kine_handler"].get_link_pose_buf  # (EPM, 7)
            root_vel = g["kine_handler"].get_link_vel_buf   # (EPM, 6)
            dof_pos  = g["kine_handler"].get_dof_pos_buf    # (EPM, n_dofs)
            dof_vel  = g["kine_handler"].get_dof_vel_buf    # (EPM, n_dofs)

            # Torso state
            obs[:, 0:1]  = root_pos[:, 5:6]   # y position
            obs[:, 1:5]  = root_pos[:, 0:4]   # quaternion
            obs[:, 5:8]  = root_vel[:, 3:6]   # linear velocity
            obs[:, 8:11] = root_vel[:, 0:3]   # angular velocity

            # Scatter DOF data into 16-slot fields (inactive stay 0)
            obs[:, 11:27].zero_()
            obs[:, 27:43].zero_()
            obs[:, 43:59].zero_()
            obs[:, 59:107].zero_()

            if g["n_dofs"] > 0:
                obs[:, 11 + g["dof_indices"]] = dof_pos
                obs[:, 27 + g["dof_indices"]] = dof_vel

            # Last actions: act_buf already masked to zero for inactive joints
            obs[:, 43:59] = self._act_buf[start:end]

            # Force sensors scattered into 8-leg slot
            if g["n_sensors"] > 0 and g["force_buffers"]:
                raw = torch.cat(g["force_buffers"], dim=-1)  # (EPM, n_sensors*6)
                obs[:, 59 + g["sensor_indices"]] = raw

            # [107:123] = limb_mask — set once at allocate time, preserved here

    def compute_reward_termination_truncation(self, actions: torch.Tensor):
        EPM = self.envs_per_morph
        for gi, g in enumerate(self.groups):
            start, end = gi * EPM, (gi + 1) * EPM
            rew, term, trunc = compute_reward_termination_truncation_helper(
                self._act_buf[start:end],
                g["kine_handler"].get_link_pose_buf,
                self.old_root_pos_buf[start:end],
                self._progress_buf[start:end],
                self.healthy_y_range,
                self.healthy_reward_val,
                self.dt,
                self.ctrl_cost_weight,
                self.max_episode_length,
            )
            self._rew_buf[start:end]       = rew
            self._next_term_buf[start:end] = term
            self._next_trunc_buf[start:end] = trunc

    # ---- Reset ----------------------------------------------------------------

    def reset_idx(self):
        if not self._reset_buf.any():
            return

        EPM = self.envs_per_morph
        # If any env is still live, read physics state so we can preserve it below.
        # Skip on a full reset (e.g. initial reset before any step) — no valid state exists yet.
        has_live_envs = not self._reset_buf.all()
        if has_live_envs:
            ArticulationKinematicStateHandler.get_articulation_kinematic_states(
                [g["kine_handler"] for g in self.groups]
            )

        for gi, g in enumerate(self.groups):
            start, end = gi * EPM, (gi + 1) * EPM
            reset_mask = self._reset_buf[start:end]
            if not reset_mask.any():
                continue

            kh = g["kine_handler"]

            new_dof_pos = g["dof_pos_init"] + reset_noise_helper(
                g["dof_pos_init"], self.reset_noise_scale, 0.4, 0.2
            )
            new_dof_vel = reset_noise_helper(
                g["dof_vel_init"], self.reset_noise_scale, 0.2, 0.1
            )
            if has_live_envs:
                m = reset_mask.unsqueeze(-1)  # (EPM, 1)
                kh.set_dof_pos_buf[:]  = torch.where(m, new_dof_pos,       kh.get_dof_pos_buf)
                kh.set_dof_vel_buf[:]  = torch.where(m, new_dof_vel,       kh.get_dof_vel_buf)
                kh.set_link_pose_buf[:] = torch.where(m, g["root_tf_init"], kh.get_link_pose_buf)
                kh.set_link_vel_buf[:] = torch.where(m, g["root_vl_init"],  kh.get_link_vel_buf)
            else:
                kh.set_dof_pos_buf[:]  = new_dof_pos
                kh.set_dof_vel_buf[:]  = new_dof_vel
                kh.set_link_pose_buf[:] = g["root_tf_init"]
                kh.set_link_vel_buf[:]  = g["root_vl_init"]

            ArticulationKinematicStateHandler.set_articulation_kinematic_states([kh])

            # Update old_root_pos and clear act/progress for resetting envs
            self.old_root_pos_buf[start:end] = torch.where(
                reset_mask.view(-1, 1), g["root_tf_init"], self.old_root_pos_buf[start:end]
            )
            self._act_buf[start:end] = torch.where(
                reset_mask.view(-1, 1),
                torch.zeros_like(self._act_buf[start:end]),
                self._act_buf[start:end],
            )
            self._progress_buf[start:end] = torch.where(
                reset_mask,
                torch.zeros_like(self._progress_buf[start:end]),
                self._progress_buf[start:end],
            )

    def reset(self):
        self._reset_buf[:] = True
        self.reset_idx()
        self.gym.compute_kinematics()
        self.compute_observations(self._act_buf)
        return self.obs_buf.clone(), {}

    # ---- Cleanup --------------------------------------------------------------

    def __del__(self):
        for f in self._tempfiles:
            try:
                f.unlink()
            except OSError:
                pass
