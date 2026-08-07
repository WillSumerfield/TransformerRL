import math
import torch

from .vocab import N_SUB

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


# ── Phase 2 (2a): relative-geometry module tokens, parameterized by (n_limbs, max_len) ──
# Single source of truth for the depth-major obs layout (must match ant_multimorph._O_* / _slot,
# derived from the SAME (n_limbs, max_len)). A limb is a chain of up to max_len modules; each module
# is ONE uniform token. Rotations are 6D (first two cols of the 3x3); per-module geometry is
# RELATIVE-TO-PARENT (parent-local frame), computed ENV-SIDE and written into obs BEFORE the
# RunningMeanStd normalizer (so it only ever sees already-6D features).
#   obs: root(13) | [sin cos vel act] (n_dof each) | [relpos(3) relrot(6) relvel(6)] (per module)
#        | sensors(n_limbs*6) | lengths(n_dof) | mask(n_dof) | is_cap(n_dof) | sub_oh(n_dof*4)
# `lengths` stays in obs (constant per body; consumed by the diversity harness, NOT the token) — so
# obs_base/len_dim keep phase-1 semantics and their downstream readers are unchanged.
# Phase 5: per-module TYPE ids ride obs as the is_cap flag + subtype one-hot (constant per body,
# like mask/lengths). They live in the RAW TAIL [mask_off : obs_total] that models._restore_mask_tail
# re-inserts UNNORMALIZED after RunningMeanStd — every channel there is exactly {0,1}, so the
# `> 0` threshold reads them back bit-exactly (a normalized constant channel collapses to ~0 and
# would be misread as absent). CATEGORY is derived, not stored: effector <=> mask>0, cap <=> is_cap>0,
# pad <=> neither.
ROOT_DIM_P2 = 13   # y(1) + rot6d(6) + linvel(3) + angvel(3)   (quat->6D; vs phase-1 ROOT_DIM=11)
# per-module dynamic obs blocks (SoA, depth-major slot order): (name, width). cfrc is NOT here — it
# rides the shared per-limb sensor block, routed to each limb's terminal module below.
_MOD_BLOCKS = (("sin", 1), ("cos", 1), ("vel", 1), ("act", 1),
               ("relpos", 3), ("relrot", 6), ("relvel", 6))
MODULE_DYN = 19   # sum of _MOD_BLOCKS widths — per-module obs dims (excludes terminal cfrc)
MODULE_DIM = 25   # token content: sin, cos, vel, act, cfrc(6), relpos(3), relrot6d(6), relvel(6)


def token_dims(n_limbs: int, max_len: int) -> dict:
    """Derived obs/token dims. slot(n,d)=(d-1)*n_limbs+(n-1) depth-major; sensors stay per-limb (6).
    obs_base = start of the lengths block (== phase-1 semantics, so mask_off = obs_base + len_dim).
    raw_tail_off/raw_tail_dim span mask+is_cap+sub_oh — the block models.py keeps UNNORMALIZED."""
    n_dof = n_limbs * max_len
    sens_off = ROOT_DIM_P2 + MODULE_DYN * n_dof                    # start of per-limb contact sensors
    obs_base = sens_off + n_limbs * 6                             # start of lengths block
    mask_off = obs_base + n_dof                                   # dof mask (phase-1 offset kept)
    cap_off = mask_off + n_dof                                    # is_cap flag  (5a)
    sub_off = cap_off + n_dof                                     # subtype one-hot, [slot][N_SUB]
    obs_total = sub_off + N_SUB * n_dof
    return dict(n_limbs=n_limbs, max_len=max_len, n_dof=n_dof, sens_off=sens_off, obs_base=obs_base,
                len_dim=n_dof, mask_dim=n_dof, mask_off=mask_off, cap_off=cap_off, sub_off=sub_off,
                raw_tail_off=mask_off, raw_tail_dim=obs_total - mask_off, obs_total=obs_total,
                n_module_tokens=n_dof, n_tokens=1 + n_limbs + n_dof)  # CLS + start*n + module*n_dof


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


def contact_mask(obs, n_limbs: int, max_len: int, cap_bare: int):
    """(B, n_dof) float contact-slot mask in depth-major slot order — the routing tokenize_modules
    uses for cfrc, exposed so the FD aux head can supervise contact on the SAME slot that observes
    it. Returns the phase-1 terminal-effector mask when obs predates the Phase-5 type tail.
    cap_bare = the active ModuleLibrary's CAP_BARE subtype id (caller-supplied, not hardcoded)."""
    B, d = obs.shape[0], token_dims(n_limbs, max_len)
    n_dof = d["n_dof"]
    eff = (obs[:, d["mask_off"]:d["mask_off"] + n_dof] > 0).reshape(B, max_len, n_limbs)
    if obs.shape[1] >= d["obs_total"]:
        cap_d = (obs[:, d["cap_off"]:d["cap_off"] + n_dof] > 0).reshape(B, max_len, n_limbs)
        sub_d = obs[:, d["sub_off"]:d["sub_off"] + N_SUB * n_dof].reshape(B, max_len, n_limbs, N_SUB)
    else:
        cap_d = torch.zeros_like(eff)
        sub_d = torch.zeros(B, max_len, n_limbs, N_SUB, device=obs.device)
    depth0 = torch.arange(max_len, device=obs.device).view(1, max_len, 1)
    return _contact_slots(eff, cap_d, sub_d, eff.sum(1), depth0, cap_bare).reshape(B, n_dof).float()


def tokenize_modules(obs, n_limbs: int, max_len: int, cap_bare: int, enc: torch.Tensor = None):
    """Tokenize a depth-major relative-geometry obs into
    (root, module_tokens, active_mask, cap_mask, sub_oh).
      root:          (B, ROOT_DIM_P2=13) = y + rot6d(6) + linvel(3) + angvel(3).
      module_tokens: (B, n_limbs*max_len, MODULE_DIM=25) canonical depth-major slot order:
                     [sin, cos, vel, act, cfrc(6), relpos(3), relrot6d(6), relvel(6)].
      active_mask:   (B, n_dof) 1 for real EFFECTORS (== DOFs; caps are actionless, so 0).
      cap_mask:      (B, n_dof) 1 at each present limb's cap slot (depth == #effectors).
      sub_oh:        (B, n_dof, N_SUB) subtype one-hot; all-zero on pad slots.
    cfrc rides the limb's CONTACT slot: the cap when it is a morphology cap (foot/pad/ball), else
    the terminal effector (bare cap == phase-1 behaviour). `enc` is accepted for call compat but
    unused (per-module limb-direction feature dropped in 2a). Inactive slots stay 0."""
    B = obs.shape[0]
    d = token_dims(n_limbs, max_len)
    n_dof, sens_off = d["n_dof"], d["sens_off"]
    dev = obs.device
    root = obs[:, 0:ROOT_DIM_P2]

    dm = lambda x, w: x.reshape(B, max_len, n_limbs, w)           # depth-major [B, d, n, w]; slot-major flat
    o, parts = ROOT_DIM_P2, []
    for _, w in _MOD_BLOCKS:
        parts.append(dm(obs[:, o:o + w * n_dof], w)); o += w * n_dof
    sin_d, cos_d, vel_d, act_d, relpos_d, relrot_d, relvel_d = parts

    has_tail = obs.shape[1] >= d["obs_total"]
    mask_off, cap_off, sub_off = d["mask_off"], d["cap_off"], d["sub_off"]
    raw_mask = obs[:, mask_off:mask_off + n_dof] if has_tail else torch.ones(B, n_dof, device=dev)
    mask_d = raw_mask.reshape(B, max_len, n_limbs)               # (B, d, n)
    eff = mask_d > 0

    count = eff.sum(dim=1)                                        # (B, n_limbs) effector-chain length
    depth0 = torch.arange(max_len, device=dev).view(1, max_len, 1)
    if has_tail:
        cap_d = (obs[:, cap_off:cap_off + n_dof] > 0).reshape(B, max_len, n_limbs)
        sub_d = dm(obs[:, sub_off:sub_off + N_SUB * n_dof], N_SUB)        # (B, d, n, N_SUB)
    else:                                                         # pre-phase-5 obs: canonical body
        cap_d = torch.zeros(B, max_len, n_limbs, dtype=torch.bool, device=dev)
        sub_d = torch.zeros(B, max_len, n_limbs, N_SUB, device=dev)

    # contact slot: morphology cap if this limb has one, else the terminal effector.
    contact = _contact_slots(eff, cap_d, sub_d, count, depth0, cap_bare)    # (B, d, n)
    sens = obs[:, sens_off:sens_off + n_limbs * 6].view(B, n_limbs, 6)
    cfrc = torch.where(contact.unsqueeze(-1),
                       sens.unsqueeze(1).expand(B, max_len, n_limbs, 6),
                       torch.zeros(B, max_len, n_limbs, 6, device=dev))   # (B, d, n, 6)

    module = torch.cat([sin_d, cos_d, vel_d, act_d, cfrc, relpos_d, relrot_d, relvel_d], dim=-1)
    module_tokens = module.reshape(B, n_dof, MODULE_DIM)         # (B, n_dof, 25) depth-major slot order
    active_mask   = eff.reshape(B, n_dof).float()
    cap_mask      = cap_d.reshape(B, n_dof).float()
    sub_oh        = sub_d.reshape(B, n_dof, N_SUB)
    return root, module_tokens, active_mask, cap_mask, sub_oh
