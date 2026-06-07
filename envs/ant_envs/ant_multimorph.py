"""AntMultiMorphEnv: train one controller across many morphologies, one group per morphology."""
import gc
import random
import sys
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "vlearn-main" / "train"))

import numpy as np
import torch
import vlearn as v
from vlearn.spaces import Box
from envs.ant_environment_common import (
    store_initial_conditions_helper,
    compute_reward_termination_truncation_helper,
)
from envs.common import create_plane

from ..multigroup_environment import MultiGroupEnvironmentGpu
from .build_vsim import Morphology, write_vsim, HIP_RANGE, ANKLE_RANGE


_N_DOFS_FULL = 16   # 8 hips + 8 ankles (DOF order: h1,a1,...,h8,a8)
_N_LEGS      = 8
_OBS_BASE    = 107  # 1+4+3+3+16+16+16+8*6 (physical obs)
_LEN_DIM     = 16   # 8 hip + 8 ankle segment lengths (raw; RMS-normalized by the policy)
_MASK_DIM    = 16
# obs layout: [0:107] physical, [107:123] lengths, [123:139] dof mask
_OBS_TOTAL   = _OBS_BASE + _LEN_DIM + _MASK_DIM  # 139

# Empty grid cells of padding between adjacent morphology sets (in units of `spacing`).
_SET_GAP_CELLS = 4


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


def morph_split(
    morphs: list[frozenset],
    train_pct: float,
    seed: int,
    test_set: bool = False,
) -> list[frozenset]:
    """Return train or test morphologies via stratified split by leg count.

    Stratifies by leg count so each stratum contributes proportionally to both halves.
    Requires seed; caller must validate before calling.
    """
    import random as _random
    rng = _random.Random(seed)
    by_legs: dict[int, list[frozenset]] = {}
    for m in morphs:
        by_legs.setdefault(len(m), []).append(m)

    train, test = [], []
    for strat in sorted(by_legs):
        group = by_legs[strat][:]
        rng.shuffle(group)
        n_train = max(1, int(len(group) * train_pct))
        n_train = min(n_train, len(group) - 1)  # ensure at least 1 in test
        train.extend(group[:n_train])
        test.extend(group[n_train:])

    return test if test_set else train


def sample_morphologies(num: int, seed: int = None, rng: "random.Random" = None) -> list[Morphology]:
    """Sample `num` full morphologies: leg count uniform in 3..8, topology uniform within that count,
    each active leg's hip/ankle length uniform in its range. Pass a persistent `rng` for a
    reproducible resample stream, or a `seed` for a one-off draw."""
    rng = rng if rng is not None else random.Random(seed)
    by_legs: dict[int, list[frozenset]] = {}
    for m in _stable_morphologies():
        by_legs.setdefault(len(m), []).append(m)
    leg_counts = sorted(by_legs)
    out = []
    for _ in range(num):
        legs = rng.choice(by_legs[rng.choice(leg_counts)])
        hip = {n: rng.uniform(*HIP_RANGE) for n in legs}
        ank = {n: rng.uniform(*ANKLE_RANGE) for n in legs}
        out.append(Morphology(legs, hip, ank))
    return out


def _as_morphology(m) -> Morphology:
    return m if isinstance(m, Morphology) else Morphology.from_legs(m)


class AntMultiMorphEnv(MultiGroupEnvironmentGpu):
    """One EnvironmentGroup per morphology. Obs is 139D; actions are always 16D."""

    @property
    def unwrapped(self):
        return self

    def follow_sets(self) -> list[list[int]]:
        EPM = self.envs_per_morph
        return [list(range(gi * EPM, (gi + 1) * EPM)) for gi in range(len(self.groups))]

    def follow_world_pos(self, idx: int) -> v.Vec3:
        EPM = self.envs_per_morph
        gi, i = idx // EPM, idx % EPM
        g = self.groups[gi]
        if "env_transforms" not in g:
            env_set = list(g["env_group"].get_environment_sets())[0]
            g["env_transforms"] = [
                env_set.get_environment(env_set.get_environment_handle(j)).get_transform()
                for j in range(EPM)
            ]
        p = self._get_root_pose[idx, 4:7]  # idx is the global env index
        local = v.Vec3(float(p[0]), float(p[1]), float(p[2]))
        return g["env_transforms"][i].transform(local)

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        rendering: bool = False,
        enable_scene_query: bool = False,
        max_episode_length: int = 1000,
        ctrl_cost_weight: float = 0.5,
        healthy_reward: float = 2.0,
        healthy_y_range: tuple = (0.3, 1.1),
        reset_noise_scale: float = 1.0,
        gravity: v.Vec3 = v.Vec3(0, -9.81, 0),
        timestep: float = 0.01667,
        frame_skip: int = 1,
        spacing: float = 3.0,
        max_contact_pairs_per_env: int = 64,
        with_window: bool = True,
        seed: int = None,
        raise_exception: bool = True,
        morphologies: list | None = None,
        sample_morphs: bool = False,
        train_pct: float = 1.0,
        test_set: bool = False,
        **kwargs,
    ):
        # Full ant: sample `num_envs` variable-length bodies (one per env). Otherwise use the given
        # topology set (or the stable set), optionally train/test-split, at default lengths.
        self._sample_morphs = sample_morphs
        self._sample_seed = seed
        if sample_morphs:
            if seed is None:
                raise ValueError("seed required when sample_morphs=True")
            # Persistent rng so the initial draw + every resample form one reproducible stream.
            self._morph_rng = random.Random(seed)
            self._morphologies = self._draw_morphs(num_envs)
        else:
            morphs = morphologies if morphologies is not None else _stable_morphologies()
            if train_pct < 1.0:
                if seed is None:
                    raise ValueError("seed required when train_pct < 1.0")
                morphs = morph_split(morphs, train_pct, seed, test_set)
            self._morphologies = [_as_morphology(m) for m in morphs]
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

    def create_envs(self):
        up_axis     = v.Vec3(0, 1, 0)
        rot0        = v.shortest_rotation(v.Vec3(0, 0, 1), up_axis)
        init_tf     = v.Transform(rot0, 0.75 * up_axis)

        epm          = self.envs_per_morph
        cols         = max(1, ceil(epm ** 0.5))
        rows_per_grp = max(1, ceil(epm / cols))
        stride_x     = (cols + _SET_GAP_CELLS) * self.spacing
        stride_z     = (rows_per_grp + _SET_GAP_CELLS) * self.spacing
        n_grp_cols   = max(1, ceil(len(self._morphologies) ** 0.5))

        for gi, morph in enumerate(self._morphologies):
            active = sorted(morph.legs)
            n_dofs = len(active) * 2

            tmpfile = write_vsim(morph, gi)

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

            # DOF mask: 1 for active DOFs, 0 otherwise
            dof_mask = torch.zeros(_N_DOFS_FULL, dtype=torch.float32, device=self.device)
            dof_mask[dof_indices] = 1.0

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
                "dof_mask":       dof_mask,
            })

    def allocate_buffers(self):
        super().allocate_buffers()
        N   = self.total_num_envs
        EPM = self.envs_per_morph

        self.old_root_pos_buf = torch.zeros((N, 7), dtype=torch.float32, device=self.device)

        # Root pose/vel are uniform-width across morphologies, so they live in single global
        # tensors; each group's command is backed by a contiguous row-slice (a vsim command can
        # wrap a tensor view). Downstream obs/reward then read them whole-tensor.
        self._get_root_pose = torch.zeros((N, 7), dtype=torch.float32, device=self.device)
        self._get_root_vel  = torch.zeros((N, 6), dtype=torch.float32, device=self.device)
        # Root reset is constant (init pose, no noise), so these set buffers are filled once here
        # and never touched in the step loop. _set_root_vel stays zero.
        self._set_root_pose = torch.zeros((N, 7), dtype=torch.float32, device=self.device)
        self._set_root_vel  = torch.zeros((N, 6), dtype=torch.float32, device=self.device)
        self._global_dof_mask = torch.zeros((N, _N_DOFS_FULL), dtype=torch.float32, device=self.device)
        # Per-env segment lengths, constant per body: [hip_leg1..8, ankle_leg1..8], 0 for inactive legs.
        self._global_lengths = torch.zeros((N, _LEN_DIM), dtype=torch.float32, device=self.device)

        # DOF/sensor data is ragged (per-morphology width + a per-morphology slot permutation), so
        # it can't share a rectangular tensor. Each quantity gets ONE flat buffer holding every
        # group's data end-to-end; each group's command wraps a reshaped contiguous slice of it.
        # Precomputed indices map flat <-> padded obs slots so the per-step scatter/gather is one
        # whole-tensor op:
        #   _dof_gather_idx[e,c] : flat index feeding canonical DOF slot c of env e (inactive slots
        #                          gather index 0 and are zeroed by _global_dof_mask; serves pos+vel)
        #   _motor_src_idx[p]    : index into act.view(-1) feeding flat-motor slot p
        #   _sensor_gather_idx / _sensor_mask : the same for the 48 force-sensor slots
        n_sensors_per = [g["art_def"].get_num_force_sensor_defs() for g in self.groups]
        dof_off, sen_off, d, s = [], [], 0, 0
        for gi, g in enumerate(self.groups):
            dof_off.append(d); d += EPM * g["n_dofs"]
            sen_off.append(s); s += EPM * n_sensors_per[gi] * 6
        FLAT_DOF, FLAT_SEN = d, s

        self._flat_get_dof_pos = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_get_dof_vel = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_set_dof_pos = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_set_dof_vel = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_motor       = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_dof_init    = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_sensor      = torch.zeros(FLAT_SEN, dtype=torch.float32, device=self.device)
        self._has_sensors      = FLAT_SEN > 0

        self._dof_gather_idx    = torch.zeros((N, _N_DOFS_FULL), dtype=torch.long, device=self.device)
        self._motor_src_idx     = torch.zeros(FLAT_DOF, dtype=torch.long, device=self.device)
        self._sensor_gather_idx = torch.zeros((N, _N_LEGS * 6), dtype=torch.long, device=self.device)
        self._sensor_mask       = torch.zeros((N, _N_LEGS * 6), dtype=torch.float32, device=self.device)

        all_motor_cmds, all_sensor_cmds, all_get_cmds, all_set_cmds = [], [], [], []
        ar_epm = torch.arange(EPM, device=self.device)
        zero_l = torch.zeros((), dtype=torch.long, device=self.device)

        def chunk(flat, off, w):
            return flat[off:off + EPM * w].view(EPM, w)

        for gi, g in enumerate(self.groups):
            start = gi * EPM
            end   = start + EPM
            n_dofs = g["n_dofs"]
            doff = dof_off[gi]

            # Initial conditions
            dof_pos_init, root_trans_init, _ = store_initial_conditions_helper(
                g["art_def"], self.device
            )
            g["dof_pos_init"] = torch.tile(dof_pos_init, (EPM, 1))  # (EPM, n_dofs)
            g["dof_vel_init"] = torch.zeros_like(g["dof_pos_init"])

            self._set_root_pose[start:end] = torch.tensor(
                [root_trans_init.q.x, root_trans_init.q.y,
                 root_trans_init.q.z, root_trans_init.q.w,
                 root_trans_init.p.x, root_trans_init.p.y,
                 root_trans_init.p.z],
                dtype=torch.float32, device=self.device,
            )
            self._global_dof_mask[start:end] = g["dof_mask"]

            morph = g["morph"]
            lvec = torch.zeros(_LEN_DIM, dtype=torch.float32, device=self.device)
            for n in g["active"]:
                lvec[n - 1]           = morph.hip_lengths[n]
                lvec[_N_LEGS + n - 1] = morph.ankle_lengths[n]
            self._global_lengths[start:end] = lvec

            self._flat_dof_init[doff:doff + EPM * n_dofs] = g["dof_pos_init"].reshape(-1)

            # Commands aliased to reshaped flat slices (root buffers stay global row-slices).
            all_motor_cmds.append(g["env_group"].create_motor_control_command(
                v.wrap_gpu_buffer(chunk(self._flat_motor, doff, n_dofs)), g["arti_handle"]
            ))
            all_get_cmds.append(g["env_group"].create_articulation_kinematic_state_command(
                v.wrap_gpu_buffer(chunk(self._flat_get_dof_pos, doff, n_dofs)),
                v.wrap_gpu_buffer(chunk(self._flat_get_dof_vel, doff, n_dofs)),
                v.wrap_gpu_buffer(self._get_root_pose[start:end]),
                v.wrap_gpu_buffer(self._get_root_vel[start:end]),
                g["arti_handle"], link_index_range=(0, 1),
                transform_type=v.TransformType.MODEL, frame_type=v.FrameType.ENVIRONMENT,
            ))
            all_set_cmds.append(g["env_group"].create_articulation_kinematic_state_command(
                v.wrap_gpu_buffer(chunk(self._flat_set_dof_pos, doff, n_dofs)),
                v.wrap_gpu_buffer(chunk(self._flat_set_dof_vel, doff, n_dofs)),
                v.wrap_gpu_buffer(self._set_root_pose[start:end]),
                v.wrap_gpu_buffer(self._set_root_vel[start:end]),
                g["arti_handle"], link_index_range=(0, 1),
                transform_type=v.TransformType.MODEL, frame_type=v.FrameType.ENVIRONMENT,
                masks_buffer=v.wrap_gpu_buffer(self._reset_buf[start:end]),
            ))

            # DOF gather index (canonical slot -> flat) + motor source index (flat -> act).
            dof_idx = g["dof_indices"]                       # (n_dofs,) canonical slot per packed k
            col_k = torch.full((_N_DOFS_FULL,), -1, dtype=torch.long, device=self.device)
            col_k[dof_idx] = torch.arange(n_dofs, device=self.device)  # canonical -> packed k (-1)
            self._dof_gather_idx[start:end] = torch.where(
                (col_k >= 0).unsqueeze(0),
                doff + ar_epm.unsqueeze(1) * n_dofs + col_k.clamp(min=0).unsqueeze(0),
                zero_l,
            )
            motor_block = (start + ar_epm).unsqueeze(1) * _N_DOFS_FULL + dof_idx.unsqueeze(0)
            self._motor_src_idx[doff:doff + EPM * n_dofs] = motor_block.reshape(-1)

            # Force-sensor commands (one per sensor) + sensor gather index.
            env_def      = self.gym.get_environment_def(g["env_def_handle"])
            articulation = env_def.get_articulation(g["arti_handle"])
            soff = sen_off[gi]
            for si in range(n_sensors_per[gi]):
                sensor_handle = articulation.get_force_sensor_handle(si)
                all_sensor_cmds.append(g["env_group"].create_force_sensor_command(
                    v.wrap_gpu_buffer(chunk(self._flat_sensor, soff + si * EPM * 6, 6)), sensor_handle
                ))
            g["n_sensors"] = n_sensors_per[gi]
            sen_idx = g["sensor_indices"]                    # (n_sensors*6,) canonical 48-slot per (si,comp)
            base48 = torch.full((_N_LEGS * 6,), -1, dtype=torch.long, device=self.device)
            p = torch.arange(sen_idx.numel(), device=self.device)
            base48[sen_idx] = (p // 6) * (EPM * 6) + (p % 6)   # canonical 48-slot -> within-group flat base
            self._sensor_mask[start:end] = (base48 >= 0).float().unsqueeze(0)
            self._sensor_gather_idx[start:end] = torch.where(
                (base48 >= 0).unsqueeze(0),
                soff + base48.clamp(min=0).unsqueeze(0) + ar_epm.unsqueeze(1) * 6,
                zero_l,
            )

        # Constant length + dof_mask blocks in obs; whole-tensor, set once.
        self._obs_buf[:, _OBS_BASE:_OBS_BASE + _LEN_DIM]   = self._global_lengths
        self._obs_buf[:, _OBS_BASE + _LEN_DIM:_OBS_TOTAL]  = self._global_dof_mask

        # Batch commands across all groups into single GPU arrays.
        self.all_motor_cmd_array  = self.gym.create_gpu_array(all_motor_cmds)
        self._get_cmd_array       = self.gym.create_gpu_array(all_get_cmds)
        self._set_cmd_array       = self.gym.create_gpu_array(all_set_cmds)
        self.all_sensor_cmd_array = self.gym.create_gpu_array(all_sensor_cmds) \
            if all_sensor_cmds else None

    def pre_physics_step(self, actions: torch.Tensor):
        # Snapshot root pose before physics (used by reward), mask inactive joints, and gather
        # active DOFs into the flat motor buffer via one precomputed index.
        self.old_root_pos_buf[:] = self._get_root_pose
        self._act_buf[:] = actions * self._global_dof_mask
        self._flat_motor[:] = self._act_buf.reshape(-1)[self._motor_src_idx]
        self.gym.set_motor_forces(self.all_motor_cmd_array)

    def post_physics_step(self):
        self._progress_buf += 1
        self._reset_buf[:]      = self._next_term_buf | self._next_trunc_buf
        self._term_buf[:]       = self._next_term_buf
        self._trunc_buf[:]      = self._next_trunc_buf
        self.reset_idx()
        self.compute_observations(self._act_buf)
        self.compute_reward_termination_truncation(self._act_buf)

    def compute_observations(self, actions: torch.Tensor):
        self.gym.get_articulation_kinematic_states(self._get_cmd_array)
        if self.all_sensor_cmd_array is not None:
            self.gym.get_sensor_forces(self.all_sensor_cmd_array)

        obs = self._obs_buf
        # DOF/sensor gathers pull each padded obs slot from its flat source; inactive slots gather
        # index 0 and are zeroed by the mask. _dof_gather_idx serves both pos and vel.
        obs[:, 0:1]  = self._get_root_pose[:, 5:6]   # y position
        obs[:, 1:5]  = self._get_root_pose[:, 0:4]   # quaternion
        obs[:, 5:8]  = self._get_root_vel[:, 3:6]    # linear velocity
        obs[:, 8:11] = self._get_root_vel[:, 0:3]    # angular velocity
        obs[:, 11:27] = self._flat_get_dof_pos[self._dof_gather_idx] * self._global_dof_mask
        obs[:, 27:43] = self._flat_get_dof_vel[self._dof_gather_idx] * self._global_dof_mask
        obs[:, 43:59] = self._act_buf   # last actions (already masked)
        if self._has_sensors:
            obs[:, 59:107] = self._flat_sensor[self._sensor_gather_idx] * self._sensor_mask
        else:
            obs[:, 59:107].zero_()

        # [107:123] = lengths, [123:139] = dof_mask — set once at allocate time, preserved here

    def compute_reward_termination_truncation(self, actions: torch.Tensor):
        # Reward needs only root pose plus the already-global act/old-root/progress buffers, so it
        # runs whole-tensor (the helper is elementwise per env).
        rew, term, trunc = compute_reward_termination_truncation_helper(
            self._act_buf,
            self._get_root_pose,
            self.old_root_pos_buf,
            self._progress_buf,
            self.healthy_y_range,
            self.healthy_reward_val,
            self.dt,
            self.ctrl_cost_weight,
            self.max_episode_length,
        )
        self._rew_buf[:]        = rew
        self._next_term_buf[:]  = term
        self._next_trunc_buf[:] = trunc

    def reset_idx(self):
        # Set buffers are filled for every env; each set command's reset_buf mask makes vsim apply
        # the write only to resetting envs, so no per-env gather or branch is needed. Root reset is
        # constant (the set buffers filled at allocate); only DOF gets fresh noise.
        s = self.reset_noise_scale
        self._flat_set_dof_pos[:] = self._flat_dof_init + s * (
            torch.rand_like(self._flat_set_dof_pos) * 0.4 - 0.2
        )
        self._flat_set_dof_vel[:] = s * (torch.rand_like(self._flat_set_dof_vel) * 0.2 - 0.1)

        m = self._reset_buf.view(-1, 1)
        self.old_root_pos_buf[:] = torch.where(m, self._set_root_pose, self.old_root_pos_buf)
        self._act_buf[:]         = torch.where(m, 0.0, self._act_buf)
        self._progress_buf[:]    = torch.where(self._reset_buf, 0, self._progress_buf)

        self.gym.set_articulation_kinematic_states(self._set_cmd_array)

    def reset(self):
        self._reset_buf[:] = True
        self.reset_idx()
        self.gym.compute_kinematics()
        self.compute_observations(self._act_buf)
        return self.obs_buf.clone(), {}

    def resample(self):
        """Draw a fresh sampled body set and rebuild the sim in place (full gym rebuild).

        vsim bakes link geometry at finalize, so new segment lengths require tearing the gym down
        and recreating it; see docs/morphology_resampling_cost.md. The caller must reset afterwards
        (the env is left rebuilt-but-unreset). Only valid for the sampled (full ant) configuration.
        """
        if not getattr(self, "_sample_morphs", False):
            raise RuntimeError("resample() requires sample_morphs=True")
        self._morphologies = self._draw_morphs(self.total_num_envs)
        self._rebuild()

    def _draw_morphs(self, num: int) -> list:
        """Draw `num` sampled bodies (full-ant config). Overridable so experiment subclasses can
        change the sampling distribution (e.g. AntBinaryLegEnv) while reusing the build/resample
        machinery."""
        return sample_morphologies(num, rng=self._morph_rng)

    def _rebuild(self):
        # Drop every gym-backed reference so delete_gym frees cleanly, then recreate the scene
        # exactly as __init__ does after gym creation.
        self._get_cmd_array = self._set_cmd_array = None
        self.all_motor_cmd_array = self.all_sensor_cmd_array = None
        self.groups = []
        self.env_groups = []
        self.gym = self.gym_render = None
        gc.collect()

        v.delete_gym()
        self._create_gym()
        self.groups = []
        self.create_envs()
        self.allocate_buffers()
        create_plane(self.gym)
        self.gym.set_num_solver_iterations(8)
        self.gym.finalize()

