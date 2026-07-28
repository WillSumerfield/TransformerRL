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
from transformer_rl.vocab import N_SUB, CAP_BARE
from .build_vsim import (Morphology, write_vsim, HIP_RANGE, ANKLE_RANGE, MAX_LIMB_LENGTH,
                         MAX_EFFECTORS, DEFAULT_ANKLE)


_N_LIMBS      = 8
_MAX_LEN      = MAX_LIMB_LENGTH                 # up to 4 modules per limb (Phase 1)
_N_DOFS_FULL  = _N_LIMBS * _MAX_LEN            # 32 = 8 limbs x 4 modules, 1 DOF each.
# Canonical padded DOF/length/mask slot for (limb n, module depth d, both 1-based): DEPTH-MAJOR,
# slot = (d-1)*8 + (n-1). Depth d in [1..4]; d=1 is the swing module, d>=2 the knee chain. Length-2
# bodies occupy depths 1,2 -> the [0:16] band. Aligns with the eff0(pos1)/eff1(pos2+) token blocks.
_N_SENSOR    = _N_LIMBS * 6                     # 48: one terminal contact sensor per limb, 6 comps
# Phase-2 (2a) obs layout (offsets), must match transformer_rl/tokenize.token_dims:
#   root(13) | sin cos vel act (n_dof each) | relpos(3nd) relrot(6nd) relvel(6nd) | sensors(48)
#   | lengths(32) | dofmask(32).  Rotations are 6D; per-module geometry is relative-to-parent.
_O_ROOT      = 13                               # y(1) + rot6d(6) + linvel(3) + angvel(3)
_O_SIN       = _O_ROOT                           # 13  joint sin(theta)
_O_COS       = _O_SIN + _N_DOFS_FULL             # 45  joint cos(theta)
_O_VEL       = _O_COS + _N_DOFS_FULL             # 77  joint velocity
_O_ACT       = _O_VEL + _N_DOFS_FULL             # 109 last action
_O_RELPOS    = _O_ACT + _N_DOFS_FULL             # 141 rel-pos (3/module)
_O_RELROT    = _O_RELPOS + 3 * _N_DOFS_FULL      # 237 rel-rot 6D (6/module)
_O_RELVEL    = _O_RELROT + 6 * _N_DOFS_FULL      # 429 rel-vel lin+ang (6/module)
_O_SENSOR    = _O_RELVEL + 6 * _N_DOFS_FULL      # 621 per-limb terminal contact (48)
_OBS_BASE    = _O_SENSOR + _N_SENSOR             # 669  (end of physical obs; start of lengths)
_LEN_DIM     = _N_DOFS_FULL                      # 32 module lengths (kept for diversity harness)
_MASK_DIM    = _N_DOFS_FULL                      # 32 DOF mask: 1 per EFFECTOR slot
# Phase-5 per-module TYPE ids, constant per body like lengths/mask. They sit in the RAW TAIL
# [_O_MASK : _OBS_TOTAL] that models._restore_mask_tail re-inserts UNNORMALIZED — every channel is
# exactly {0,1} and is read back with a `> 0` threshold. CATEGORY is derived, not stored:
# effector <=> mask>0, cap <=> is_cap>0, pad <=> neither.
_O_MASK      = _OBS_BASE + _LEN_DIM              # 701
_O_CAP       = _O_MASK + _MASK_DIM               # 733 is_cap flag, 1 per limb at its cap slot
_O_SUB       = _O_CAP + _N_DOFS_FULL             # 765 subtype one-hot, [slot][N_SUB]
_OBS_TOTAL   = _O_SUB + N_SUB * _N_DOFS_FULL     # 893


def _slot(n: int, d: int) -> int:
    """Canonical padded slot for limb n (1..8), module depth d (1..4). Depth-major."""
    return (d - 1) * _N_LIMBS + (n - 1)


def _parse_joint(name: str) -> tuple:
    """'joint_{n}_{d}' -> (n, d)."""
    _, ns, ds = name.split("_")
    return int(ns), int(ds)


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
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R

# Empty grid cells of padding between adjacent morphology sets (in units of `spacing`).
_SET_GAP_CELLS = 4


def _stable_morphologies(
    min_limbs: int = 3,
    max_limbs: int = 8,
    max_gap_deg: float = 135.0,
) -> list[frozenset]:
    """Return stable morphologies with min_limbs..max_limbs limbs and no circular gap > max_gap_deg."""
    result = []
    for mask in range(1, 256):
        active = frozenset(i + 1 for i in range(8) if (mask >> i) & 1)
        if not (min_limbs <= len(active) <= max_limbs):
            continue
        angles = sorted((n - 1) * 45.0 for n in active)
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append(360.0 - angles[-1] + angles[0])
        if max(gaps) <= max_gap_deg:
            result.append(active)
    return result


def sample_morphologies(num: int, seed: int = None, rng: "random.Random" = None) -> list[Morphology]:
    """Sample `num` full morphologies: limb count uniform in 3..8, topology uniform within that count,
    each active limb's hip/ankle length uniform in its range. Pass a persistent `rng` for a
    reproducible resample stream, or a `seed` for a one-off draw."""
    rng = rng if rng is not None else random.Random(seed)
    by_limbs: dict[int, list[frozenset]] = {}
    for m in _stable_morphologies():
        by_limbs.setdefault(len(m), []).append(m)
    limb_counts = sorted(by_limbs)
    out = []
    for _ in range(num):
        limbs = rng.choice(by_limbs[rng.choice(limb_counts)])
        # Full-ant sampler stays length-2 (one hip + one ankle module) in Phase 1; only the
        # codesign env grows variable-length limbs. module_lengths = {limb -> [hip, ankle]}.
        out.append(Morphology({n: [rng.uniform(*HIP_RANGE), rng.uniform(*ANKLE_RANGE)]
                               for n in limbs}))
    return out


def _as_morphology(m) -> Morphology:
    return m if isinstance(m, Morphology) else Morphology.from_legs(m)


class AntMultiMorphEnv(MultiGroupEnvironmentGpu):
    """One EnvironmentGroup per morphology. Obs/action widths are padded to the fixed 32-slot
    depth-major layout (see the _O_* offsets above); only active EFFECTOR slots carry a DOF."""

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
        value_size: int = 1,
        **kwargs,
    ):
        # Full ant: sample `num_envs` variable-length bodies (one per env). Otherwise use the given
        # topology set (or the stable set) at default lengths.
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
            self._morphologies = [_as_morphology(m) for m in morphs]
        self.n_morphs = len(self._morphologies)
        self.envs_per_morph = max(1, num_envs // self.n_morphs)
        total_envs = self.n_morphs * self.envs_per_morph

        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward_val = healthy_reward
        self.healthy_y_range = healthy_y_range
        self.reset_noise_scale = reset_noise_scale
        self.value_size = value_size            # reported to rl_games env_info (always 1)
        self._obs_total = _OBS_TOTAL

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
            low=np.full(self._obs_total, np.finfo("f").min, dtype=np.float32),
            high=np.full(self._obs_total, np.finfo("f").max, dtype=np.float32),
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

            # DOF scatter into the padded 32-DOF space. We do NOT assume vsim's packed DOF order:
            # query each DOF's joint name ('joint_{n}_{d}') and map packed k -> canonical slot(n,d).
            # (Probed order is per-limb ascending, depth ascending; querying makes it robust.)
            n_dofs = art_def.get_num_joint_dof_defs()
            dof_indices = torch.tensor(
                [_slot(*_parse_joint(art_def.get_joint_dof_def_name(k))) for k in range(n_dofs)],
                dtype=torch.long, device=self.device,
            )  # (n_dofs,) canonical slot per packed DOF k

            # Sensor scatter into 48D slot (8 limbs x 6). One terminal contact sensor per limb,
            # emitted in ascending-active order (build_vsim), so sensor si -> active[si].
            sensor_indices = torch.tensor(
                [j for n in active for j in range(6 * (n - 1), 6 * (n - 1) + 6)],
                dtype=torch.long, device=self.device,
            )  # (n_active_limbs * 6,)

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
        # Per-env spawn-height lift (offset added to the reset Y for long-limbed bodies). Used to make
        # the healthy_y_range CEILING relative to standing height so taller ants aren't killed at spawn
        # (the floor stays absolute — see compute_reward_termination_truncation).
        self._height_offset = torch.zeros(N, dtype=torch.float32, device=self.device)
        self._global_dof_mask = torch.zeros((N, _N_DOFS_FULL), dtype=torch.float32, device=self.device)
        # Per-env segment lengths, constant per body: [hip_leg1..8, ankle_leg1..8], 0 for inactive limbs.
        self._global_lengths = torch.zeros((N, _LEN_DIM), dtype=torch.float32, device=self.device)
        # Phase-5 per-module type ids, also constant per body (see _O_CAP / _O_SUB).
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
        n_links_per   = [g["art_def"].get_num_link_defs() for g in self.groups]  # 1 torso + 1/module
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

        self._dof_gather_idx    = torch.zeros((N, _N_DOFS_FULL), dtype=torch.long, device=self.device)
        self._motor_src_idx     = torch.zeros(FLAT_DOF, dtype=torch.long, device=self.device)
        self._sensor_gather_idx = torch.zeros((N, _N_LIMBS * 6), dtype=torch.long, device=self.device)
        self._sensor_mask       = torch.zeros((N, _N_LIMBS * 6), dtype=torch.float32, device=self.device)
        # Canonical module slot -> flat link ROW for the module's own link (_link_gather_idx) and its
        # PARENT link (_parent_gather_idx; parent of depth-1 = torso). Inactive slots gather row 0 and
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
            # Spawn higher when the longest limb exceeds the default 2-module chain, so long legs
            # don't clip through the ground at reset. Constant +DEFAULT_ANKLE (one default knee-module
            # length) per extra module; no lift when the longest limb is <= 2 modules. Index 5 = root
            # Y (up-axis) in [qx,qy,qz,qw, px,py,pz]. store_initial_conditions returns a fixed height,
            # so we add the offset to the reset pose (which governs — envs reset before stepping).
            longest = max((g["morph"].num_modules(n) for n in g["active"]), default=0)
            offset = DEFAULT_ANKLE * max(0, longest - 2)
            self._set_root_pose[start:end, 5] += offset
            self._height_offset[start:end] = offset
            self._global_dof_mask[start:end] = g["dof_mask"]

            morph = g["morph"]
            lvec = torch.zeros(_LEN_DIM, dtype=torch.float32, device=self.device)
            capvec = torch.zeros(_N_DOFS_FULL, dtype=torch.float32, device=self.device)
            subvec = torch.zeros((_N_DOFS_FULL, N_SUB), dtype=torch.float32, device=self.device)
            for n in g["active"]:
                for d, ln in enumerate(morph.module_lengths[n], start=1):  # depth-major slot(n,d)
                    lvec[_slot(n, d)] = ln
                for d, t in enumerate(morph.effector_types[n], start=1):
                    subvec[_slot(n, d), t] = 1.0
                cd = morph.num_modules(n) + 1              # the cap rides the depth==count slot
                capvec[_slot(n, cd)] = 1.0                 # (<= MAX_EFFECTORS+1 == _MAX_LEN)
                subvec[_slot(n, cd), morph.cap_of(n)] = 1.0
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
            col_k = torch.full((_N_DOFS_FULL,), -1, dtype=torch.long, device=self.device)
            col_k[dof_idx] = torch.arange(n_dofs, device=self.device)  # canonical -> packed k (-1)
            self._dof_gather_idx[start:end] = torch.where(
                (col_k >= 0).unsqueeze(0),
                doff + ar_epm.unsqueeze(1) * n_dofs + col_k.clamp(min=0).unsqueeze(0),
                zero_l,
            )
            motor_block = (start + ar_epm).unsqueeze(1) * _N_DOFS_FULL + dof_idx.unsqueeze(0)
            self._motor_src_idx[doff:doff + EPM * n_dofs] = motor_block.reshape(-1)

            # Link gather (canonical module slot -> flat link ROW) + parent-link gather (parent of a
            # depth-1 module = torso). Link k named 'mod_{n}_{d}' -> slot(n,d); 'torso' -> root.
            names   = [g["art_def"].get_link_def_name(k) for k in range(nl)]
            torso_k = names.index("torso")
            slot_to_k = torch.full((_N_DOFS_FULL,), -1, dtype=torch.long, device=self.device)
            for k, nm in enumerate(names):
                # 'mod_{n}_{d}' only — 'torso' has no slot and 'cap_{n}_{d}' links carry no
                # relative-geometry obs (a cap is on a FIXED joint, so its pose is constant, and
                # every rel-* block is masked by the DOF mask which is 0 at the cap slot).
                if not nm.startswith("mod_"):
                    continue
                _, ns, ds = nm.split("_")
                slot_to_k[_slot(int(ns), int(ds))] = k
            parent_k = torch.full((_N_DOFS_FULL,), torso_k, dtype=torch.long, device=self.device)
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

        # Constant length + dof_mask + type blocks in obs; whole-tensor, set once.
        self._obs_buf[:, _OBS_BASE:_O_MASK] = self._global_lengths
        self._obs_buf[:, _O_MASK:_O_CAP]    = self._global_dof_mask
        self._obs_buf[:, _O_CAP:_O_SUB]     = self._global_is_cap
        self._obs_buf[:, _O_SUB:_OBS_TOTAL] = self._global_sub_oh

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
        self.compute_observations(self._act_buf)
        self.compute_reward_termination_truncation(self._act_buf)

    def compute_observations(self, actions: torch.Tensor):
        self.gym.get_articulation_kinematic_states(self._get_cmd_array)       # root pose/vel
        self.gym.get_articulation_kinematic_states(self._get_link_cmd_array)  # all-link pose/vel
        if self.all_sensor_cmd_array is not None:
            self.gym.get_sensor_forces(self.all_sensor_cmd_array)

        obs = self._obs_buf
        m = self._global_dof_mask                                   # (N, n_dof) active mask
        # ── Root token: y + 6D rotation (first two cols of R) + lin/ang velocity (world frame) ──
        Rr = _quat_to_rotmat(self._get_root_pose[:, 0:4])           # (N,3,3)
        obs[:, 0:1]   = self._get_root_pose[:, 5:6]                 # y (up-axis)
        obs[:, 1:7]   = torch.cat([Rr[..., 0], Rr[..., 1]], dim=-1)  # rot6d: col0, col1
        obs[:, 7:10]  = self._get_root_vel[:, 3:6]                  # linear velocity
        obs[:, 10:13] = self._get_root_vel[:, 0:3]                  # angular velocity

        # ── Per-module DOF handle: joint sin/cos + velocity + last action (inactive slots -> 0) ──
        dof_pos = self._flat_get_dof_pos[self._dof_gather_idx] * m
        obs[:, _O_SIN:_O_COS] = torch.sin(dof_pos) * m
        obs[:, _O_COS:_O_VEL] = torch.cos(dof_pos) * m             # cos(0)=1 gated to 0 for inactive
        obs[:, _O_VEL:_O_ACT] = self._flat_get_dof_vel[self._dof_gather_idx] * m
        obs[:, _O_ACT:_O_RELPOS] = self._act_buf                   # last actions (already masked)

        # ── Relative-to-parent geometry (parent-local frame), from the all-link pose/vel buffers ──
        lp = self._flat_get_link_pose.view(-1, 7)                  # (FLAT_LINK, 7) [quat xyzw, pos]
        lv = self._flat_get_link_vel.view(-1, 6)                   # (FLAT_LINK, 6) [ang, lin]
        Pc, Pp = lp[self._link_gather_idx], lp[self._parent_gather_idx]   # (N, n_dof, 7)
        Vc, Vp = lv[self._link_gather_idx], lv[self._parent_gather_idx]   # (N, n_dof, 6)
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
        mm = m.unsqueeze(-1)
        obs[:, _O_RELPOS:_O_RELROT] = (rel_pos * mm).reshape(obs.shape[0], -1)       # slot-major
        obs[:, _O_RELROT:_O_RELVEL] = (rel_rot * mm).reshape(obs.shape[0], -1)
        obs[:, _O_RELVEL:_O_SENSOR] = (torch.cat([rel_lin, rel_ang], dim=-1) * mm).reshape(obs.shape[0], -1)

        if self._has_sensors:
            obs[:, _O_SENSOR:_OBS_BASE] = self._flat_sensor[self._sensor_gather_idx] * self._sensor_mask
        else:
            obs[:, _O_SENSOR:_OBS_BASE].zero_()

        # [_OBS_BASE : _OBS_TOTAL] = lengths | dof_mask | is_cap | subtype one-hot — all constant per
        # body, set once at allocate and preserved here.

    def compute_reward_termination_truncation(self, actions: torch.Tensor):
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
        rew, term, trunc = compute_reward_termination_truncation_helper(
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
        wrongly_low = term & (root_pose[:, 5] < lo) & (self._get_root_pose[:, 5] >= lo)
        term = term & ~wrongly_low
        rew  = rew + self.healthy_reward_val * wrongly_low.to(rew.dtype)   # restore healthy bonus
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
        and recreating it; see docs/guides/morphology_resampling_cost.md. The caller must reset afterwards
        (the env is left rebuilt-but-unreset). Only valid for the sampled (full ant) configuration.
        """
        if not getattr(self, "_sample_morphs", False):
            raise RuntimeError("resample() requires sample_morphs=True")
        self._morphologies = self._draw_morphs(self.total_num_envs)
        self._rebuild()

    def _draw_morphs(self, num: int) -> list:
        """Draw `num` sampled bodies (full-ant config). Overridable so subclasses (e.g.
        AntCodesignEnv) can change the sampling distribution while reusing the build/resample
        machinery."""
        return sample_morphologies(num, rng=self._morph_rng)

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

