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
#   obs: root | [sin cos vel act] (n_modules each) | [relpos(3) relrot(6) relvel(6)] (per module)
#        | sensors(n_slots*6) | lengths | mask | is_cap | sub_oh(n_modules*n_sub) | task extra
# `lengths` stays in obs (constant per body; consumed by the diversity harness, NOT the token).
# Phase 5: per-module TYPE ids ride obs as the is_cap flag + subtype one-hot (constant per body,
# like mask/lengths). They live in the RAW TAIL [mask_off : extra_off] that models._restore_mask_tail
# re-inserts UNNORMALIZED after RunningMeanStd — every channel there is exactly {0,1}, so the
# `> 0` threshold reads them back bit-exactly (a normalized constant channel collapses to ~0 and
# would be misread as absent). CATEGORY is derived, not stored: effector <=> mask>0, cap <=> is_cap>0,
# pad <=> neither.
#
# EVERY offset below comes from `Task.obs_layout()` (D23). This file used to derive the same layout
# a second time from (n_limbs, max_len) and the two had to be kept in step by hand; the package owns
# where the bytes are, and this file owns only what they become. The block offsets are read
# individually rather than walked from a base, so the root block widening for a world-mounted task
# moves everything downstream correctly instead of silently shearing the module blocks.
# per-module dynamic obs blocks, in token-content order: (layout offset key, width). cfrc is NOT
# here — it rides the shared per-limb sensor block, routed to each limb's contact module below.
# Widths sum to 19; the token adds cfrc(6) on top, hence MODULE_DIM below.
_MOD_BLOCKS = (("sin_off", 1), ("cos_off", 1), ("vel_off", 1), ("act_off", 1),
               ("relpos_off", 3), ("relrot_off", 6), ("relvel_off", 6))
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


def _contact_slots(eff, cap_d, sub_d, count, depth0, cap_bare: int):
    """(B, d, n) bool: the slot carrying each limb's contact force — its cap when that cap is a
    morphology cap (foot/pad/ball, which is what actually touches the ground), else its terminal
    effector (the bare cap has no body, so contact stays put == phase-1 behaviour)."""
    morph_cap = cap_d & (sub_d[..., cap_bare] <= 0)               # a cap that is a real body
    is_terminal = eff & (depth0 == (count.unsqueeze(1) - 1))
    return torch.where(morph_cap.any(dim=1, keepdim=True), morph_cap, is_terminal)


def contact_mask(obs, layout: dict, cap_bare: int):
    """(B, n_dof) float contact-slot mask in depth-major slot order — the routing tokenize_modules
    uses for cfrc, exposed so the FD aux head can supervise contact on the SAME slot that observes
    it. Returns the phase-1 terminal-effector mask when obs predates the Phase-5 type tail.
    `layout` is the Task's `obs_layout()`; cap_bare = the active ModuleLibrary's bare-cap subtype id
    (caller-supplied, since "bare" is our choice of default cap, not something the library names)."""
    B = obs.shape[0]
    n, max_len, n_dof, n_sub = (layout["n_slots"], layout["max_depth"],
                                layout["n_modules"], layout["n_sub"])
    mask_off, cap_off, sub_off = layout["mask_off"], layout["cap_off"], layout["sub_off"]
    eff = (obs[:, mask_off:mask_off + n_dof] > 0).reshape(B, max_len, n)
    if obs.shape[1] >= layout["obs_total"]:
        cap_d = (obs[:, cap_off:cap_off + n_dof] > 0).reshape(B, max_len, n)
        sub_d = obs[:, sub_off:sub_off + n_sub * n_dof].reshape(B, max_len, n, n_sub)
    else:
        cap_d = torch.zeros_like(eff)
        sub_d = torch.zeros(B, max_len, n, n_sub, device=obs.device)
    depth0 = torch.arange(max_len, device=obs.device).view(1, max_len, 1)
    return _contact_slots(eff, cap_d, sub_d, eff.sum(1), depth0, cap_bare).reshape(B, n_dof).float()


def tokenize_modules(obs, layout: dict, cap_bare: int, enc: torch.Tensor = None):
    """Tokenize a depth-major relative-geometry obs into
    (root, module_tokens, active_mask, cap_mask, sub_oh).
      root:          (B, layout["root_dim"]) = y + rot6d(6) + linvel(3) + angvel(3), then pos/vel/act
                     per actuated root axis if the task mounts its root on any.
      module_tokens: (B, n_slots*max_depth, MODULE_DIM=25) canonical depth-major slot order:
                     [sin, cos, vel, act, cfrc(6), relpos(3), relrot6d(6), relvel(6)].
      active_mask:   (B, n_dof) 1 for real EFFECTORS (== DOFs; caps are actionless, so 0).
      cap_mask:      (B, n_dof) 1 at each present limb's cap slot (depth == #effectors).
      sub_oh:        (B, n_dof, n_sub) subtype one-hot; all-zero on pad slots.
    `layout` is the Task's `obs_layout()`. cfrc rides the limb's CONTACT slot: the cap when it is a
    morphology cap (foot/pad/ball), else the terminal effector (bare cap == phase-1 behaviour).
    `enc` is accepted for call compat but unused (per-module limb-direction feature dropped in 2a).
    Inactive slots stay 0."""
    B = obs.shape[0]
    n, max_len, n_dof, n_sub = (layout["n_slots"], layout["max_depth"],
                                layout["n_modules"], layout["n_sub"])
    sens_off = layout["sensor_off"]
    dev = obs.device
    root = obs[:, layout["root_off"]:layout["root_off"] + layout["root_dim"]]

    dm = lambda x, w: x.reshape(B, max_len, n, w)                 # depth-major [B, d, n, w]; slot-major flat
    parts = [dm(obs[:, layout[k]:layout[k] + w * n_dof], w) for k, w in _MOD_BLOCKS]
    sin_d, cos_d, vel_d, act_d, relpos_d, relrot_d, relvel_d = parts

    has_tail = obs.shape[1] >= layout["obs_total"]
    mask_off, cap_off, sub_off = layout["mask_off"], layout["cap_off"], layout["sub_off"]
    raw_mask = obs[:, mask_off:mask_off + n_dof] if has_tail else torch.ones(B, n_dof, device=dev)
    mask_d = raw_mask.reshape(B, max_len, n)                     # (B, d, n)
    eff = mask_d > 0

    count = eff.sum(dim=1)                                        # (B, n_slots) effector-chain length
    depth0 = torch.arange(max_len, device=dev).view(1, max_len, 1)
    if has_tail:
        cap_d = (obs[:, cap_off:cap_off + n_dof] > 0).reshape(B, max_len, n)
        sub_d = dm(obs[:, sub_off:sub_off + n_sub * n_dof], n_sub)        # (B, d, n, n_sub)
    else:                                                         # pre-phase-5 obs: canonical body
        cap_d = torch.zeros(B, max_len, n, dtype=torch.bool, device=dev)
        sub_d = torch.zeros(B, max_len, n, n_sub, device=dev)

    # contact slot: morphology cap if this limb has one, else the terminal effector.
    contact = _contact_slots(eff, cap_d, sub_d, count, depth0, cap_bare)    # (B, d, n)
    sens = obs[:, sens_off:sens_off + n * 6].view(B, n, 6)
    cfrc = torch.where(contact.unsqueeze(-1),
                       sens.unsqueeze(1).expand(B, max_len, n, 6),
                       torch.zeros(B, max_len, n, 6, device=dev))   # (B, d, n, 6)

    module = torch.cat([sin_d, cos_d, vel_d, act_d, cfrc, relpos_d, relrot_d, relvel_d], dim=-1)
    module_tokens = module.reshape(B, n_dof, MODULE_DIM)         # (B, n_dof, 25) depth-major slot order
    active_mask   = eff.reshape(B, n_dof).float()
    cap_mask      = cap_d.reshape(B, n_dof).float()
    sub_oh        = sub_d.reshape(B, n_dof, n_sub)
    return root, module_tokens, active_mask, cap_mask, sub_oh
