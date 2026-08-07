"""CodesignEnvironmentGpu: train one controller across many morphologies, one group per morphology."""
import gc
import sys
from math import ceil
from pathlib import Path
from abc import abstractmethod

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "vlearn-main" / "train"))

import numpy as np
import torch
import vlearn as v
from vlearn.spaces import Box
from envs.common import create_plane
from envs.ant_environment_common import store_initial_conditions_helper

from codesigner.components.interfaces import ModuleType

from .multigroup_environment import MultiGroupEnvironmentGpu
from transformer_rl.vocab import N_SUB
from .build_vsim import write_vsim
from .modular_libraries.simple import Morphology, designs_from_arrays


_N_LIMBS      = 8
_MAX_LEN      = 4                                # up to 4 modules per limb (Phase 1); tensor-grammar
                                                  # constant, independent of which module_library is
                                                  # plugged in (mirrors net.max_limb_length elsewhere)
_N_DOFS_FULL  = _N_LIMBS * _MAX_LEN              # 32 = 8 limbs x 4 modules, 1 DOF each.
_N_SENSOR     = _N_LIMBS * 6                     # 48: one terminal contact sensor per limb, 6 comps
_LEN_DIM      = _N_DOFS_FULL                     # 32 module lengths (kept for diversity harness)
_MASK_DIM     = _N_DOFS_FULL                     # 32 DOF mask: 1 per EFFECTOR slot
# Obs offsets (_o_root, _o_sin, ... _obs_total) are INSTANCE attributes computed in __init__, not
# module constants here: the root block's width depends on module_library.root_axes, which is fixed
# per env/config but differs between e.g. AntCodesignEnv (root_axes=None) and GraspCodesignEnv
# (root_axes=[...]). See __init__ for the derivation.

_SET_GAP_CELLS = 4                               # gap between env groups in the grid (in multiples of spacing)


def _slot(n: int, d: int) -> int:
    """Canonical padded slot for limb n (1..8), module depth d (1..4). Depth-major."""
    return (d - 1) * _N_LIMBS + (n - 1)


def _parse_joint(name: str) -> tuple:
    """'joint_{n}_{d}' -> (n, d)."""
    _, ns, ds = name.split("_")
    return int(ns), int(ds)


def _is_root_joint(name: str) -> bool:
    """'root_{axis}' joints (build_vsim's root-mount prismatic chain) vs. 'joint_{n}_{d}' limbs."""
    return name.startswith("root_")


def _root_axis_of(name: str) -> str:
    return name.removeprefix("root_")


@torch.jit.script
def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(...,4) quaternion (x,y,z,w, normalized) -> (...,3,3) rotation matrix (columns = body axes
    in world frame). Used for the 6D rotation (first two columns) + relative-geometry obs (2a)."""
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
    ], dim=-1)
    if q.ndim == 2:
        return R.reshape(q.shape[0], 3, 3)
    if q.ndim == 3:
        return R.reshape(q.shape[0], q.shape[1], 3, 3)
    else:
        raise ValueError(f"Unexpected q.ndim={q.ndim}")


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def designs_from_arrays(module_library, counts_2d, eff_sub_3d, cap_sub_2d, n_limbs) -> list:
    """Parse the generator's designed body grid (int subtype ids, generator/tensor-side vocabulary)
    into per-env (effector_types, cap_types) specs keyed by ring slot, with TYPE NAMES:
      counts_2d  (N, n_limbs)            effectors per limb, 0 = limb absent
      eff_sub_3d (N, n_limbs, max_len)   effector subtype id per depth (only [:count] is read)
      cap_sub_2d (N, n_limbs)            cap subtype id per limb (-1 where the limb was never capped)
    All torch-free (numpy / nested lists). A limb with count 0 is absent regardless of its cap."""
    eff_names = module_library.names(ModuleType.EFFECTOR)
    cap_names = module_library.names(ModuleType.CAP)
    out = []
    for e in range(len(counts_2d)):
        eff, caps = {}, {}
        for j in range(n_limbs):
            k = int(counts_2d[e][j])
            if k <= 0:
                continue
            assert k <= module_library._MAX_EFFECTORS, \
                f"effector count {k} > MAX_EFFECTORS={module_library._MAX_EFFECTORS}"
            eff[j + 1] = [eff_names[int(eff_sub_3d[e][j][d])] for d in range(k)]
            c = int(cap_sub_2d[e][j])
            caps[j + 1] = module_library._CANON_CAP if c < 0 else cap_names[c]
        assert eff, "0-module body; generator must guarantee >=1 effector"
        out.append((eff, caps))
    return out


class CodesignEnvironmentGpu(MultiGroupEnvironmentGpu):
    """One EnvironmentGroup per morphology. The 32-slot depth-major limb layout (only active
    EFFECTOR slots carry a DOF) is padded to a fixed width; the root block on top of it is not --
    its width is exact, computed once per instance from module_library.root_axes (see __init__'s
    self._o_* offsets)."""

    # --------------------------------- Abstract Interface ---------------------------------

    @abstractmethod
    def compute_reward_termination_truncation(self):
        """Compute reward + termination + truncation for all envs. Normally just calls the jit helper, but can be overridden for custom reward shaping.
        Outputs results to self._rew_buf, self._next_term_buf, self._next_trunc_buf."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def reward_term_trunc_helper():
        """Compute reward + termination + truncation for all envs. Must be JIT compilable (no Python control flow, no object attributes)."""
        raise NotImplementedError

    @abstractmethod
    def set_root_pose(self, group: dict, start: int, end: int):
        """Write the root pose to the set buffers (self._set_root_pose) for all envs. Called at reset and after resample()."""
        raise NotImplementedError

    
    # ------------------------------------ Construction ------------------------------------

    @property
    def unwrapped(self):
        return self

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
        n_morphs: int = 1,
        morphologies: list[] = None,
        value_size: int = 1,
        reset_noise_scale: float = 1.0,
        dof_reset_pos_bias:  float = -0.2,
        dof_reset_pos_scale: float = 0.4,
        dof_reset_vel_bias:  float = -0.1,
        dof_reset_vel_scale: float = 0.2,
        module_library=None,
        base_morphology: Morphology = None,
        **kwargs,
    ):
        self.module_library = module_library
        self.base_morphology = base_morphology
        # morphologies: explicit per-group body list (dev/eval tooling -- e.g. a module gallery or a
        # fixed eval set), 1 env-group per entry. Default None -> n_morphs copies of the sampled
        # (or base) body, as usual; sampling/resample() still redraws from the DEFAULT set only.
        self._init_morphologies = morphologies
        self.n_morphs = len(morphologies) if morphologies is not None else n_morphs
        self.envs_per_morph = max(1, num_envs // self.n_morphs)
        total_envs = self.n_morphs * self.envs_per_morph

        self.value_size = value_size            # reported to rl_games env_info (always 1)
        self.reset_noise_scale = reset_noise_scale

        # Obs/action layout: instance attrs, not module constants -- the root block's width and the
        # action width both depend on module_library.root_axes (fixed per env, never resampled).
        root_axes = self.module_library.root_axes
        self._n_root_axes = len(root_axes) if root_axes else 0
        self._n_dofs_ext = _N_DOFS_FULL + self._n_root_axes   # +1 canonical slot per root axis
        self._o_root   = 13 + 3 * self._n_root_axes           # y(1)+rot6d(6)+linvel(3)+angvel(3)+3/axis
        self._o_sin    = self._o_root
        self._o_cos    = self._o_sin + _N_DOFS_FULL
        self._o_vel    = self._o_cos + _N_DOFS_FULL
        self._o_act    = self._o_vel + _N_DOFS_FULL
        self._o_relpos = self._o_act + _N_DOFS_FULL
        self._o_relrot = self._o_relpos + 3 * _N_DOFS_FULL
        self._o_relvel = self._o_relrot + 6 * _N_DOFS_FULL
        self._o_sensor = self._o_relvel + 6 * _N_DOFS_FULL
        self._obs_base = self._o_sensor + _N_SENSOR
        self._o_mask   = self._obs_base + _LEN_DIM
        self._o_cap    = self._o_mask + _MASK_DIM
        self._o_sub    = self._o_cap + _N_DOFS_FULL
        self._obs_total = self._o_sub + N_SUB * _N_DOFS_FULL

        self.dof_reset_pos_bias  = dof_reset_pos_bias
        self.dof_reset_pos_scale = dof_reset_pos_scale
        self.dof_reset_vel_bias  = dof_reset_vel_bias
        self.dof_reset_vel_scale = dof_reset_vel_scale

        super().__init__(
            num_envs=total_envs,
            device=device,
            max_episode_length=max_episode_length,
            gravity=gravity,
            timestep=timestep,
            frame_skip=frame_skip,
            spacing=spacing,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            **kwargs,
        )

        self.observation_space = Box(
            low=np.full(self._obs_total, np.finfo("f").min, dtype=np.float32),
            high=np.full(self._obs_total, np.finfo("f").max, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=np.full(self._n_dofs_ext, -1.0, dtype=np.float32),
            high=np.full(self._n_dofs_ext, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self.groups = []
        self._morphologies = self._init_morphologies or [self.base_morphology] * self.n_morphs
        self.create_envs()
        self.allocate_buffers()

        create_plane(self.gym)
        self.gym.set_num_solver_iterations(8)
        self.gym.finalize()

        # Compile the jit helpers. Attribute checker stops for vectorized envs from recompiling each instance.
        if not hasattr(self.reward_term_trunc_helper, "_jit_compiled"):
            self.reward_term_trunc_helper._jit_compiled = torch.jit.script(self.reward_term_trunc_helper)
        self.reward_term_trunc_helper_jit = staticmethod(self.reward_term_trunc_helper._jit_compiled)

        if not hasattr(self.obs_helper, "_jit_compiled"):
            self.obs_helper._jit_compiled = torch.jit.script(self.obs_helper)
        self.obs_helper_jit = staticmethod(self.obs_helper._jit_compiled)

        if not hasattr(self.reset_helper, "_jit_compiled"):
            self.reset_helper._jit_compiled = torch.jit.script(self.reset_helper)
        self.reset_helper_jit = staticmethod(self.reset_helper._jit_compiled)

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
            active = sorted(morph.limbs)

            tmpfile = write_vsim(self.module_library.build_robot(morph), gi)

            name           = f"morph_{gi}"
            env_def_handle = self.gym.create_environment_def(name)
            env_def        = self.gym.get_environment_def(env_def_handle)
            # fixed=True pins the articulation's base link to its placement transform instead of
            # leaving it free-floating -- needed whenever the root is world-mounted (root_axes is
            # not None), whether or not any prismatic root-mount joints exist (root_axes=[]).
            env_def.import_definitions(str(tmpfile), fixed=self.module_library.root_axes is not None)

            arti_def_handle = env_def.get_articulation_def_handle_by_name("root")
            arti_handle     = env_def.create_articulation(arti_def_handle, init_tf, "robot")
            art_def         = env_def.get_articulation_def(arti_def_handle)
            art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)
            env_def.finalize()

            group_offset = v.Vec3((gi % n_grp_cols) * stride_x, 0, (gi // n_grp_cols) * stride_z)
            env_group    = self.create_env_group(env_def_handle, epm, group_offset)

            # DOF scatter into the padded canonical space (32 limb slots + one per root axis). We do
            # NOT assume vsim's packed DOF order: query each DOF's joint name and map packed k ->
            # canonical slot -- 'joint_{n}_{d}' -> _slot(n,d); 'root_{axis}' (build_vsim's root-mount
            # chain) -> _N_DOFS_FULL + that axis's position in module_library.root_axes.
            n_dofs = art_def.get_num_joint_dof_defs()
            dof_names = [art_def.get_joint_dof_def_name(k) for k in range(n_dofs)]
            root_order = {ax: i for i, ax in enumerate(self.module_library.root_axes or [])}
            dof_indices = torch.tensor(
                [_N_DOFS_FULL + root_order[_root_axis_of(nm)] if _is_root_joint(nm)
                 else _slot(*_parse_joint(nm)) for nm in dof_names],
                dtype=torch.long, device=self.device,
            )  # (n_dofs,) canonical slot per packed DOF k

            # Sensor scatter into 48D slot (8 limbs x 6). One terminal contact sensor per limb,
            # emitted in ascending-active order (build_vsim), so sensor si -> active[si].
            sensor_indices = torch.tensor(
                [j for n in active for j in range(6 * (n - 1), 6 * (n - 1) + 6)],
                dtype=torch.long, device=self.device,
            )  # (n_active_limbs * 6,)

            # DOF mask: 1 for active DOFs, 0 otherwise. Root-axis slots always land here too (every
            # group's articulation has the same root-mount joints), so they read back as always-1
            # with no special-casing.
            dof_mask = torch.zeros(self._n_dofs_ext, dtype=torch.float32, device=self.device)
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
        # Per-env spawn-height lift (offset added to the reset Y for long-limbed bodies). Used to make
        # the healthy_y_range CEILING relative to standing height so taller ants aren't killed at spawn
        # (the floor stays absolute — see compute_reward_termination_truncation).
        self._height_offset = torch.zeros(N, dtype=torch.float32, device=self.device)
        # Extended canonical DOF width: 32 limb slots + one per configured root axis (always active,
        # no padding -- root_axes is fixed per env, unlike limb presence).
        self._global_dof_mask = torch.zeros((N, self._n_dofs_ext), dtype=torch.float32, device=self.device)
        # Per-env segment lengths, constant per body: [hip_leg1..8, ankle_leg1..8], 0 for inactive limbs.
        self._global_lengths = torch.zeros((N, _LEN_DIM), dtype=torch.float32, device=self.device)
        # Phase-5 per-module type ids, also constant per body (see self._o_cap / self._o_sub).
        self._global_is_cap = torch.zeros((N, _N_DOFS_FULL), dtype=torch.float32, device=self.device)
        self._global_sub_oh = torch.zeros((N, _N_DOFS_FULL * N_SUB), dtype=torch.float32,
                                          device=self.device)

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
        n_links_per   = [g["art_def"].get_num_link_defs() for g in self.groups]  # 1 root + 1/module
        dof_off, sen_off, link_off, d, s, L = [], [], [], 0, 0, 0
        for gi, g in enumerate(self.groups):
            dof_off.append(d); d += EPM * g["n_dofs"]
            sen_off.append(s); s += EPM * n_sensors_per[gi] * 6
            link_off.append(L); L += EPM * n_links_per[gi]   # in ROWS (1 row = one link's pose/vel)
        FLAT_DOF, FLAT_SEN, FLAT_LINK = d, s, L

        self._flat_get_dof_pos = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_get_dof_vel = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_set_dof_pos = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_set_dof_vel = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_motor       = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_dof_init    = torch.zeros(FLAT_DOF, dtype=torch.float32, device=self.device)
        self._flat_sensor      = torch.zeros(FLAT_SEN, dtype=torch.float32, device=self.device)
        self._has_sensors      = FLAT_SEN > 0
        # Per-link pose (7 = quat xyzw + pos xyz) / velocity (6 = ang xyz + lin xyz) in ENV frame, one
        # ROW per (env, link), groups concatenated. Feeds the relative-to-parent geometry obs (2a).
        self._flat_get_link_pose = torch.zeros(FLAT_LINK * 7, dtype=torch.float32, device=self.device)
        self._flat_get_link_vel  = torch.zeros(FLAT_LINK * 6, dtype=torch.float32, device=self.device)

        self._dof_gather_idx    = torch.zeros((N, self._n_dofs_ext), dtype=torch.long, device=self.device)
        self._motor_src_idx     = torch.zeros(FLAT_DOF, dtype=torch.long, device=self.device)
        self._sensor_gather_idx = torch.zeros((N, _N_LIMBS * 6), dtype=torch.long, device=self.device)
        self._sensor_mask       = torch.zeros((N, _N_LIMBS * 6), dtype=torch.float32, device=self.device)
        # Canonical module slot -> flat link ROW for the module's own link (_link_gather_idx) and its
        # PARENT link (_parent_gather_idx; parent of depth-1 = root). Inactive slots gather row 0 and
        # are zeroed by _global_dof_mask. gather via _flat_get_link_pose.view(-1,7)[idx].
        self._link_gather_idx   = torch.zeros((N, _N_DOFS_FULL), dtype=torch.long, device=self.device)
        self._parent_gather_idx = torch.zeros((N, _N_DOFS_FULL), dtype=torch.long, device=self.device)

        all_motor_cmds, all_sensor_cmds, all_get_cmds, all_set_cmds, all_get_link_cmds = [], [], [], [], []
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

            self.set_root_pose(g, start, end)

            morph = g["morph"]
            lvec = torch.zeros(_LEN_DIM, dtype=torch.float32, device=self.device)
            capvec = torch.zeros(_N_DOFS_FULL, dtype=torch.float32, device=self.device)
            subvec = torch.zeros((_N_DOFS_FULL, N_SUB), dtype=torch.float32, device=self.device)
            ml = morph.module_library
            eff_names, cap_names = ml.names(ModuleType.EFFECTOR), ml.names(ModuleType.CAP)
            for n in g["active"]:
                for d, ln in enumerate(morph.module_lengths[n], start=1):  # depth-major slot(n,d)
                    lvec[_slot(n, d)] = ln
                for d, t in enumerate(morph.effector_types[n], start=1):
                    subvec[_slot(n, d), eff_names.index(t)] = 1.0
                cd = morph.num_modules(n) + 1              # the cap rides the depth==count slot
                capvec[_slot(n, cd)] = 1.0                 # (<= MAX_EFFECTORS+1 == _MAX_LEN)
                subvec[_slot(n, cd), cap_names.index(morph.cap_of(n))] = 1.0
            self._global_lengths[start:end] = lvec
            self._global_is_cap[start:end] = capvec
            self._global_sub_oh[start:end] = subvec.reshape(-1)

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
            # Separate ALL-link GET for the relative-geometry obs (root path above stays untouched).
            # dof buffers are required args but re-fetch the same values (harmless); we only read the
            # link pose/vel slices. Rows [lo : lo+EPM*n_links) of the flat link buffers.
            nl = n_links_per[gi]
            lo = link_off[gi]
            all_get_link_cmds.append(g["env_group"].create_articulation_kinematic_state_command(
                v.wrap_gpu_buffer(chunk(self._flat_get_dof_pos, doff, n_dofs)),
                v.wrap_gpu_buffer(chunk(self._flat_get_dof_vel, doff, n_dofs)),
                v.wrap_gpu_buffer(self._flat_get_link_pose[lo * 7:(lo + EPM * nl) * 7].view(EPM, nl * 7)),
                v.wrap_gpu_buffer(self._flat_get_link_vel[lo * 6:(lo + EPM * nl) * 6].view(EPM, nl * 6)),
                g["arti_handle"], link_index_range=(0, nl),
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
            col_k = torch.full((self._n_dofs_ext,), -1, dtype=torch.long, device=self.device)
            col_k[dof_idx] = torch.arange(n_dofs, device=self.device)  # canonical -> packed k (-1)
            self._dof_gather_idx[start:end] = torch.where(
                (col_k >= 0).unsqueeze(0),
                doff + ar_epm.unsqueeze(1) * n_dofs + col_k.clamp(min=0).unsqueeze(0),
                zero_l,
            )
            motor_block = (start + ar_epm).unsqueeze(1) * self._n_dofs_ext + dof_idx.unsqueeze(0)
            self._motor_src_idx[doff:doff + EPM * n_dofs] = motor_block.reshape(-1)

            # Link gather (canonical module slot -> flat link ROW) + parent-link gather (parent of a
            # depth-1 module = root). Link k named 'mod_{n}_{d}' -> slot(n,d); 'root' -> root.
            names   = [g["art_def"].get_link_def_name(k) for k in range(nl)]
            root_k = names.index("root")
            slot_to_k = torch.full((_N_DOFS_FULL,), -1, dtype=torch.long, device=self.device)
            for k, nm in enumerate(names):
                # 'mod_{n}_{d}' only — 'root' has no slot and 'cap_{n}_{d}' links carry no
                # relative-geometry obs (a cap is on a FIXED joint, so its pose is constant, and
                # every rel-* block is masked by the DOF mask which is 0 at the cap slot).
                if not nm.startswith("mod_"):
                    continue
                _, ns, ds = nm.split("_")
                slot_to_k[_slot(int(ns), int(ds))] = k
            parent_k = torch.full((_N_DOFS_FULL,), root_k, dtype=torch.long, device=self.device)
            for c in range(_N_DOFS_FULL):
                if slot_to_k[c] < 0:
                    continue
                depth = c // _N_LIMBS + 1                       # inverse of _slot (depth-major)
                if depth > 1:
                    parent_k[c] = slot_to_k[_slot(c % _N_LIMBS + 1, depth - 1)]
            active_c   = (slot_to_k >= 0).unsqueeze(0)
            child_row  = lo + ar_epm.unsqueeze(1) * nl + slot_to_k.clamp(min=0).unsqueeze(0)  # (EPM, n_dof)
            parent_row = lo + ar_epm.unsqueeze(1) * nl + parent_k.unsqueeze(0)
            self._link_gather_idx[start:end]   = torch.where(active_c, child_row,  zero_l)
            self._parent_gather_idx[start:end] = torch.where(active_c, parent_row, zero_l)

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
            base48 = torch.full((_N_LIMBS * 6,), -1, dtype=torch.long, device=self.device)
            p = torch.arange(sen_idx.numel(), device=self.device)
            base48[sen_idx] = (p // 6) * (EPM * 6) + (p % 6)   # canonical 48-slot -> within-group flat base
            self._sensor_mask[start:end] = (base48 >= 0).float().unsqueeze(0)
            self._sensor_gather_idx[start:end] = torch.where(
                (base48 >= 0).unsqueeze(0),
                soff + base48.clamp(min=0).unsqueeze(0) + ar_epm.unsqueeze(1) * 6,
                zero_l,
            )

        # Constant length + dof_mask + type blocks in obs; whole-tensor, set once. The reported DOF
        # mask feature is limb-only (_MASK_DIM == _N_DOFS_FULL): root-axis slots aren't modules, so
        # they're excluded here even though _global_dof_mask itself is extended for gather/motor use.
        self._obs_buf[:, self._obs_base:self._o_mask] = self._global_lengths
        self._obs_buf[:, self._o_mask:self._o_cap]    = self._global_dof_mask[:, :_N_DOFS_FULL]
        self._obs_buf[:, self._o_cap:self._o_sub]     = self._global_is_cap
        self._obs_buf[:, self._o_sub:self._obs_total] = self._global_sub_oh

        # Batch commands across all groups into single GPU arrays.
        self.all_motor_cmd_array  = self.gym.create_gpu_array(all_motor_cmds)
        self._get_cmd_array       = self.gym.create_gpu_array(all_get_cmds)
        self._get_link_cmd_array  = self.gym.create_gpu_array(all_get_link_cmds)
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
        self.compute_observations()
        self.compute_reward_termination_truncation()

    def compute_observations(self):
        self.gym.get_articulation_kinematic_states(self._get_cmd_array)       # root pose/vel
        self.gym.get_articulation_kinematic_states(self._get_link_cmd_array)  # all-link pose/vel
        if self.all_sensor_cmd_array is not None:
            self.gym.get_sensor_forces(self.all_sensor_cmd_array)

        self.obs_helper_jit(
            self._obs_buf, self._act_buf,
            self._get_root_pose, self._get_root_vel,
            self._flat_get_dof_pos, self._flat_get_dof_vel,
            self._flat_get_link_pose, self._flat_get_link_vel,
            self._flat_sensor, self._has_sensors,
            self._dof_gather_idx, self._link_gather_idx,
            self._parent_gather_idx, self._sensor_gather_idx,
            self._global_dof_mask, self._sensor_mask,
            _N_DOFS_FULL, self._n_root_axes,
            self._o_root, self._o_sin, self._o_cos, self._o_vel, self._o_act,
            self._o_relpos, self._o_relrot, self._o_relvel, self._o_sensor, self._obs_base,
        )

    def reset_idx(self):
        # Set buffers are filled for every env; each set command's reset_buf mask makes vsim apply
        # the write only to resetting envs, so no per-env gather or branch is needed. Root reset is
        # constant (the set buffers filled at allocate); only DOF gets fresh noise.
        
        self.reset_helper_jit(
            self._reset_buf, self._act_buf,
            self._flat_set_dof_pos, self._flat_set_dof_vel,
            self._flat_dof_init, self._progress_buf,
            self.reset_noise_scale,
            self.dof_reset_pos_bias, self.dof_reset_pos_scale, 
            self.dof_reset_vel_bias, self.dof_reset_vel_scale
        )

        self.gym.set_articulation_kinematic_states(self._set_cmd_array)

    def reset(self):
        self._reset_buf[:] = True
        self.reset_idx()
        self.gym.compute_kinematics()
        self.compute_observations()
        return self.obs_buf.clone(), {}

    def resample(self):
        """Draw a fresh sampled body set and rebuild the sim in place (full gym rebuild).

        vsim bakes link geometry at finalize, so new segment lengths require tearing the gym down
        and recreating it; see docs/guides/morphology_resampling_cost.md. The caller must reset afterwards
        (the env is left rebuilt-but-unreset). Only valid for the sampled (full ant) configuration.
        """
        self._morphologies = self._draw_morphs(self.total_num_envs)
        self._rebuild()

    def _rebuild(self):
        # DRAIN all in-flight async work before delete_gym frees vsim's pinned host buffers.
        # A lingering async op referencing a just-freed buffer causes a rare rebuild-time
        # use-after-free: vsim "FATAL: Error deallocating pinned host memory". torch.cuda.synchronize()
        # alone is insufficient (vsim drives its own pipeline), so also drain vsim via a device
        # error-check. Cost ~0.2ms vs ~14s per rebuild. (end_streaming was dropped: we never
        # start_streaming, so it only hit vsim's "never started" early-return + printed a WARN.)
        torch.cuda.synchronize()
        for _fn in ("_check_for_cuda_errors",):
            try:
                getattr(self.gym, _fn)()
            except Exception as _e:
                print(f"[rebuild-drain] {_fn}() skipped: {_e!r}", flush=True)
        # Drop every gym-backed reference so delete_gym frees cleanly, then recreate the scene
        # exactly as __init__ does after gym creation.
        self._get_cmd_array = self._set_cmd_array = self._get_link_cmd_array = None
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

    def set_next(self, counts, eff_sub, cap_sub):
        """Store the next window's per-env design. Effective at the next rebuild.
          counts  (N, 8)          effectors per limb (0 = limb absent), column j -> limb j+1
          eff_sub (N, 8, max_len) effector subtype per depth (only [:count] is read)
          cap_sub (N, 8)          cap subtype per limb (-1 -> the canonical bare cap)"""
        self._next_bodies = designs_from_arrays(self.module_library, _np(counts), _np(eff_sub),
                                                _np(cap_sub), _N_LIMBS)

    def _draw_morphs(self, num):
        assert len(self._next_bodies) == num, f"set_next gave {len(self._next_bodies)} bodies but env has {num} envs"
        return [Morphology.from_design(self.module_library, eff, caps) for eff, caps in self._next_bodies]

    # ------------------------------------ JIT Helpers ------------------------------------
    
    @staticmethod
    def obs_helper(obs_buf: torch.Tensor, act_buf: torch.Tensor,
                   get_root_pose: torch.Tensor, get_root_vel: torch.Tensor,
                   get_dof_pos: torch.Tensor, get_dof_vel: torch.Tensor,
                   get_link_pose: torch.Tensor, get_link_vel: torch.Tensor,
                   get_sensor: torch.Tensor, has_sensors: bool,
                   dof_gather_idx: torch.Tensor, link_gather_idx: torch.Tensor,
                   parent_gather_idx: torch.Tensor, sensor_gather_idx: torch.Tensor,
                   global_dof_mask: torch.Tensor, sensor_mask: torch.Tensor,
                   n_dofs_full: int, n_root_axes: int,
                   o_root: int, o_sin: int, o_cos: int, o_vel: int, o_act: int,
                   o_relpos: int, o_relrot: int, o_relvel: int, o_sensor: int, obs_base: int,
        ):
        """Compute the full observation buffer from the vsim GET buffers. Must be JIT compilable (no
        Python control flow beyond a plain `if` on a scalar int, no object attributes) -- offsets are
        passed in as plain ints (not baked as compile-time defaults) since they vary per env instance
        (root_axes-dependent), while the compiled graph itself is identical for every instance."""
        m = global_dof_mask                                   # (N, n_dofs_ext) active mask; root-
        m_limb = m[:, :n_dofs_full]                            # axis columns (if any) are always 1
        # ── Root token: y + 6D rotation (first two cols of R) + lin/ang velocity (world frame) ──
        Rr = _quat_to_rotmat(get_root_pose[:, 0:4])           # (N,3,3)
        obs_buf[:, 0:1]   = get_root_pose[:, 5:6]                 # y (up-axis)
        obs_buf[:, 1:7]   = torch.cat([Rr[..., 0], Rr[..., 1]], dim=-1)  # rot6d: col0, col1
        obs_buf[:, 7:10]  = get_root_vel[:, 3:6]                  # linear velocity
        obs_buf[:, 10:13] = get_root_vel[:, 0:3]                  # angular velocity

        # ── Per-module DOF handle: joint sin/cos + velocity + last action (inactive slots -> 0) ──
        dof_pos = get_dof_pos[dof_gather_idx] * m               # (N, n_dofs_ext)
        dof_vel = get_dof_vel[dof_gather_idx] * m
        obs_buf[:, o_sin:o_cos] = torch.sin(dof_pos[:, :n_dofs_full]) * m_limb
        obs_buf[:, o_cos:o_vel] = torch.cos(dof_pos[:, :n_dofs_full]) * m_limb  # cos(0)=1 gated to 0
        obs_buf[:, o_vel:o_act] = dof_vel[:, :n_dofs_full]
        obs_buf[:, o_act:o_relpos] = act_buf[:, :n_dofs_full]     # last actions (already masked)

        # ── Root-axis extension: raw position/velocity/last-action per configured axis, grouped
        # right after the y/rot6d/linvel/angvel fields (root block width == o_root, no padding). ──
        if n_root_axes > 0:
            p0 = o_root - 3 * n_root_axes
            obs_buf[:, p0:p0 + n_root_axes]                     = dof_pos[:, n_dofs_full:]
            obs_buf[:, p0 + n_root_axes:p0 + 2 * n_root_axes]   = dof_vel[:, n_dofs_full:]
            obs_buf[:, p0 + 2 * n_root_axes:p0 + 3 * n_root_axes] = act_buf[:, n_dofs_full:]

        # ── Relative-to-parent geometry (parent-local frame), from the all-link pose/vel buffers ──
        lp = get_link_pose.view(-1, 7)                  # (FLAT_LINK, 7) [quat xyzw, pos]
        lv = get_link_vel.view(-1, 6)                   # (FLAT_LINK, 6) [ang, lin]
        Pc, Pp = lp[link_gather_idx], lp[parent_gather_idx]   # (N, n_dof, 7)
        Vc, Vp = lv[link_gather_idx], lv[parent_gather_idx]   # (N, n_dof, 6)
        Rc, Rp = _quat_to_rotmat(Pc[..., 0:4]), _quat_to_rotmat(Pp[..., 0:4])
        RpT = Rp.transpose(-1, -2)                                 # world -> parent frame
        pc, pp = Pc[..., 4:7], Pp[..., 4:7]
        wc, vc = Vc[..., 0:3], Vc[..., 3:6]
        wp, vp = Vp[..., 0:3], Vp[..., 3:6]
        rel_R = RpT @ Rc                                           # child orientation in parent frame
        rel_rot = torch.cat([rel_R[..., 0], rel_R[..., 1]], dim=-1)                  # 6D (N,n_dof,6)
        rel_pos = (RpT @ (pc - pp).unsqueeze(-1)).squeeze(-1)                        # (N,n_dof,3)
        rel_ang = (RpT @ (wc - wp).unsqueeze(-1)).squeeze(-1)                        # joint ang vel
        rel_lin = (RpT @ (vc - vp - torch.cross(wp, pc - pp, dim=-1)).unsqueeze(-1)).squeeze(-1)
        mm = m_limb.unsqueeze(-1)
        obs_buf[:, o_relpos:o_relrot] = (rel_pos * mm).reshape(obs_buf.shape[0], -1)       # slot-major
        obs_buf[:, o_relrot:o_relvel] = (rel_rot * mm).reshape(obs_buf.shape[0], -1)
        obs_buf[:, o_relvel:o_sensor] = (torch.cat([rel_lin, rel_ang], dim=-1) * mm).reshape(obs_buf.shape[0], -1)

        if has_sensors:
            obs_buf[:, o_sensor:obs_base] = get_sensor[sensor_gather_idx] * sensor_mask
        else:
            obs_buf[:, o_sensor:obs_base].zero_()

        # [obs_base : obs_total] = lengths | dof_mask | is_cap | subtype one-hot — all constant per
        # body, set once at allocate and preserved here.

    @staticmethod
    def reset_helper(reset_buf: torch.Tensor, act_buf: torch.Tensor,
                     set_dof_pos: torch.Tensor, set_dof_vel: torch.Tensor,
                     dof_init: torch.Tensor, progress_buf: torch.Tensor, 
                     reset_noise_scale: float,
                     dof_reset_pos_bias: float, dof_reset_pos_scale: float, 
                     dof_reset_vel_bias: float, dof_reset_vel_scale: float
        ):
        """Reset the DOF and root state of all envs flagged in reset_buf. Must be JIT compilable (no Python control flow, no object attributes)."""
        s = reset_noise_scale
        set_dof_pos[:] = dof_init + s * (torch.rand_like(set_dof_pos) * dof_reset_pos_scale + dof_reset_pos_bias)
        set_dof_vel[:] = s * (torch.rand_like(set_dof_vel) * dof_reset_vel_scale + dof_reset_vel_bias)

        m = reset_buf.view(-1, 1)
        act_buf[:]      = torch.where(m, 0.0, act_buf)
        progress_buf[:] = torch.where(reset_buf, 0, progress_buf)
