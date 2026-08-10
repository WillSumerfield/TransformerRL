import math
import torch

ROOT_DIM = 11   # y(1) + quat(4) + linvel(3) + angvel(3)
EFF0_DIM   = 5    # pos(1) + vel(1) + last_action(1) + sin(1) + cos(1)
EFF1_DIM = 11   # pos(1) + vel(1) + cfrc(6) + last_action(1) + sin(1) + cos(1)
EFF0_DIM_8   = 6  # 8-limb adds eff0 segment length
EFF1_DIM_8 = 12 # 8-limb adds eff1 segment length

OBS_DIM_4  = 59
MASK_DIM_4 = 8
OBS_DIM_8  = 107  # physical obs; then lengths[107:123], then dof mask[123:139]
LEN_DIM_8  = 16   # 8 eff0 + 8 eff1 segment lengths (raw; normalized by the policy input normalizer)
MASK_DIM_8 = 16

_LIMB_ENC_4 = torch.tensor(
    [[math.sin((2*i + 1) * math.pi / 4), math.cos((2*i + 1) * math.pi / 4)]
     for i in range(4)],
    dtype=torch.float32,
)  # angles: pi/4, 3pi/4, 5pi/4, 7pi/4

_LIMB_ENC_8 = torch.tensor(
    [[math.sin(i * math.pi / 4), math.cos(i * math.pi / 4)] for i in range(8)],
    dtype=torch.float32,
)


def _tokenize(obs, n_limbs, obs_dim, mask_dim, limb_enc, len_dim=0):
    B = obs.shape[0]

    n_dof     = 2 * n_limbs
    root     = obs[:, 0:11]
    dof_pos   = obs[:, 11 : 11 + n_dof]
    dof_vel   = obs[:, 11 + n_dof : 11 + 2 * n_dof]
    acts      = obs[:, 11 + 2 * n_dof : 11 + 3 * n_dof]
    sensors   = obs[:, 11 + 3 * n_dof : obs_dim]

    # Per-limb segment lengths (8-limb only): obs[obs_dim : obs_dim+n_limbs] eff0, then eff1. The mask
    # follows the length block, so its offset shifts by len_dim.
    eff0_len   = obs[:, obs_dim : obs_dim + n_limbs] if len_dim else None
    eff1_len = obs[:, obs_dim + n_limbs : obs_dim + 2 * n_limbs] if len_dim else None
    mask_off  = obs_dim + len_dim
    raw_mask  = (obs[:, mask_off : mask_off + mask_dim]
                 if obs.shape[1] >= mask_off + mask_dim
                 else torch.ones(B, mask_dim, device=obs.device))

    limb_active = (raw_mask[:, 0::2] > 0).float()  # (B, n_limbs) from eff0 slots
    enc = limb_enc.to(obs.device).unsqueeze(0).expand(B, -1, -1)  # (B, n_limbs, 2)
    enc = enc * limb_active.unsqueeze(-1)  # zero sin/cos for inactive limbs

    eff0_tokens = torch.stack([
        torch.cat([dof_pos[:, 2*i:2*i+1], dof_vel[:, 2*i:2*i+1], acts[:, 2*i:2*i+1], enc[:, i, :]]
                  + ([eff0_len[:, i:i+1]] if len_dim else []), dim=-1)
        for i in range(n_limbs)
    ], dim=1)  # (B, n_limbs, EFF0_DIM[+1])

    eff1_tokens = torch.stack([
        torch.cat([dof_pos[:, 2*i+1:2*i+2], dof_vel[:, 2*i+1:2*i+2], sensors[:, 6*i:6*i+6],
                   acts[:, 2*i+1:2*i+2], enc[:, i, :]]
                  + ([eff1_len[:, i:i+1]] if len_dim else []), dim=-1)
        for i in range(n_limbs)
    ], dim=1)  # (B, n_limbs, EFF1_DIM[+1])

    active_mask = (torch.cat([raw_mask[:, 0::2], raw_mask[:, 1::2]], dim=-1) > 0).float()  # (B, mask_dim)
    return root, eff0_tokens, eff1_tokens, active_mask


def tokenize_4(obs):
    return _tokenize(obs, 4, OBS_DIM_4, MASK_DIM_4, _LIMB_ENC_4)


def tokenize_8(obs):
    return _tokenize(obs, 8, OBS_DIM_8, MASK_DIM_8, _LIMB_ENC_8, len_dim=LEN_DIM_8)


# ── Phase 2 (2a): relative-geometry module tokens, read off the Task's published layout ──
# A limb is a chain of up to max_depth modules; each module is ONE uniform token. Rotations are 6D
# (first two cols of the 3x3); per-module geometry is RELATIVE-TO-PARENT (parent-local frame),
# computed ENV-SIDE and written into obs BEFORE the RunningMeanStd normalizer (so it only ever sees
# already-6D features).
#   obs: [ global region ][ module region ]
#        global: root_height | root_rot6d | root_linvel | root_angvel | root_axis_{pos,vel,act}
#                | whatever the Task declared (a handle's position, a target pose, a phase)
#        module: sin | cos | vel | act | relpos | relrot | relvel | sensor | has_sensor
#                | length | mask | cap | sub  -- each `dim * n_modules`, depth-major
# The WHOLE global region is the root token's content: the policy needs the scene the objective
# describes, and where a field came from is not a distinction it acts on (ADR-0019).
# `length` stays in obs (constant per body; consumed by the diversity harness, NOT the token).
# `mask`, `cap`, `sub` and `has_sensor` are the layout's STRUCTURAL fields — read back with a `> 0`
# threshold, so models._restore_mask_tail re-inserts them UNNORMALIZED after RunningMeanStd (a
# normalized constant channel collapses to ~0 and would be misread as absent). CATEGORY is derived,
# not stored: effector <=> mask>0, cap <=> cap>0, pad <=> neither.
#
# EVERY offset below comes from `Task.obs_layout()` (D23) — this file owns only what the bytes
# become. Fields are looked up BY NAME in their group, so nothing here depends on where a region
# starts or on what the task put in it.
# Per-module token content, in token order: (layout field name, per-module width). Widths sum to
# MODULE_DIM. `sensor` is a plain module field now: the library places it on a module and the
# package says which, so there is no contact slot left to reconstruct on this side.
_MOD_BLOCKS = (("sin", 1), ("cos", 1), ("vel", 1), ("act", 1), ("sensor", 6),
               ("relpos", 3), ("relrot", 6), ("relvel", 6))
MODULE_DIM = 25   # token content: sin, cos, vel, act, cfrc(6), relpos(3), relrot6d(6), relvel(6)


def token_counts(layout: dict) -> dict:
    """Token-side counts the layout does not carry, because they are the model's, not the Task's."""
    n_dof = layout["n_modules"]
    return dict(n_module_tokens=n_dof,                       # one token per padded module slot
                n_tokens=1 + layout["n_slots"] + n_dof)      # CLS + start*n_slots + module*n_dof


def limb_enc(n_limbs: int) -> torch.Tensor:
    """sin/cos of each limb's placement angle i*45deg (matches build_vsim._DIR / _LIMB_ENC_8).
    Still used for the model's per-limb START-anchor (angle_proj); NOT a per-module token feature."""
    return torch.tensor([[math.sin(i * math.pi / 4), math.cos(i * math.pi / 4)] for i in range(n_limbs)],
                        dtype=torch.float32)


def _mod(layout: dict, name: str):
    """(start, per-module width) of a module field. Named lookup, so nothing here depends on where
    the module region begins or on what else the Task declared."""
    entry = layout["module"][name]
    return entry["off"], entry["dim"]


def contact_mask(obs, layout: dict):
    """(B, n_dof) float: which module slots carry a force sensor, depth-major.

    Read straight off the layout's `has_sensor` field rather than reconstructed. Placement used to be
    re-derived here from cap/subtype — "the cap when it is a real body, else the terminal effector" —
    which was the library's rule restated on this side of the boundary and free to drift from it.
    A sensorless slot and a sensor reading zero are the same six zeros, so this is the only thing
    separating them; the FD aux head supervises contact where a sensor actually is.
    """
    off, dim = _mod(layout, "has_sensor")
    return obs[:, off:off + dim * layout["n_modules"]].reshape(obs.shape[0], -1)


def global_span(layout: dict) -> tuple:
    """(start, stop) of the global region — the root token's whole content.

    The region is contiguous by construction, so this is one slice: the robot's root state and every
    field the Task declared about its objective, in one vector the policy reads without decomposing
    it (ADR-0019). Zero-width fields (`root_axis_*` on a free-floating robot) contribute nothing.
    """
    entries = layout["global"].values()
    return min(e["off"] for e in entries), max(e["off"] + e["dim"] for e in entries)


def tokenize_modules(obs, layout: dict, enc: torch.Tensor = None):
    """Tokenize a depth-major relative-geometry obs into
    (root, module_tokens, active_mask, cap_mask, sub_oh).
      root:          (B, global region width) — the whole global region, so a task's own fields ride
                     the root token with the robot's root state and nothing distinguishes them.
      module_tokens: (B, n_slots*max_depth, MODULE_DIM=25) canonical depth-major slot order:
                     [sin, cos, vel, act, cfrc(6), relpos(3), relrot6d(6), relvel(6)].
      active_mask:   (B, n_dof) 1 for real EFFECTORS (== DOFs; caps are actionless, so 0).
      cap_mask:      (B, n_dof) 1 at each present limb's cap slot (depth == #effectors).
      sub_oh:        (B, n_dof, n_sub) subtype one-hot; all-zero on pad slots.
    `layout` is the Task's `obs_layout()`. cfrc is now a plain module field, read straight off the
    slot the library placed the sensor on — sensorless slots read zero, as they did before, but the
    zeros come from the package instead of a mask this file re-derived.
    `enc` is accepted for call compat but unused (per-module limb-direction feature dropped in 2a).
    Inactive slots stay 0."""
    B = obs.shape[0]
    n, max_len, n_dof, n_sub = (layout["n_slots"], layout["max_depth"],
                                layout["n_modules"], layout["n_sub"])
    g_start, g_stop = global_span(layout)
    root = obs[:, g_start:g_stop]

    dm = lambda x, w: x.reshape(B, max_len, n, w)                 # depth-major [B, d, n, w]; slot-major flat
    parts = []
    for name, w in _MOD_BLOCKS:
        off, dim = _mod(layout, name)
        assert dim == w, f"{name} is {dim} wide per module, token expects {w}"
        parts.append(dm(obs[:, off:off + w * n_dof], w))
    sin_d, cos_d, vel_d, act_d, cfrc, relpos_d, relrot_d, relvel_d = parts

    mask_off, _ = _mod(layout, "mask")
    cap_off, _ = _mod(layout, "cap")
    sub_off, sub_w = _mod(layout, "sub")
    mask_d = obs[:, mask_off:mask_off + n_dof].reshape(B, max_len, n)     # (B, d, n)
    eff = mask_d > 0
    cap_d = (obs[:, cap_off:cap_off + n_dof] > 0).reshape(B, max_len, n)
    sub_d = dm(obs[:, sub_off:sub_off + sub_w * n_dof], sub_w)            # (B, d, n, n_sub)

    module = torch.cat([sin_d, cos_d, vel_d, act_d, cfrc, relpos_d, relrot_d, relvel_d], dim=-1)
    module_tokens = module.reshape(B, n_dof, MODULE_DIM)         # (B, n_dof, 25) depth-major slot order
    active_mask   = eff.reshape(B, n_dof).float()
    cap_mask      = cap_d.reshape(B, n_dof).float()
    sub_oh        = sub_d.reshape(B, n_dof, n_sub)
    return root, module_tokens, active_mask, cap_mask, sub_oh
