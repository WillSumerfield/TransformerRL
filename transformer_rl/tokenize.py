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


# ── Phase 2 (2a): relative-geometry module tokens, parameterized by (n_limbs, max_len) ──
# Single source of truth for the depth-major obs layout (must match ant_multimorph._O_* / _slot,
# derived from the SAME (n_limbs, max_len)). A limb is a chain of up to max_len modules; each module
# is ONE uniform token. Rotations are 6D (first two cols of the 3x3); per-module geometry is
# RELATIVE-TO-PARENT (parent-local frame), computed ENV-SIDE and written into obs BEFORE the
# RunningMeanStd normalizer (so it only ever sees already-6D features).
#   obs: root(13) | [sin cos vel act] (n_dof each) | [relpos(3) relrot(6) relvel(6)] (per module)
#        | sensors(n_limbs*6) | lengths(n_dof) | mask(n_dof)
# `lengths` stays in obs (constant per body; consumed by the diversity harness, NOT the token) — so
# obs_base/len_dim/obs_total keep phase-1 semantics and their downstream readers are unchanged.
ROOT_DIM_P2 = 13   # y(1) + rot6d(6) + linvel(3) + angvel(3)   (quat->6D; vs phase-1 ROOT_DIM=11)
# per-module dynamic obs blocks (SoA, depth-major slot order): (name, width). cfrc is NOT here — it
# rides the shared per-limb sensor block, routed to each limb's terminal module below.
_MOD_BLOCKS = (("sin", 1), ("cos", 1), ("vel", 1), ("act", 1),
               ("relpos", 3), ("relrot", 6), ("relvel", 6))
MODULE_DYN = 19   # sum of _MOD_BLOCKS widths — per-module obs dims (excludes terminal cfrc)
MODULE_DIM = 25   # token content: sin, cos, vel, act, cfrc(6), relpos(3), relrot6d(6), relvel(6)


def token_dims(n_limbs: int, max_len: int) -> dict:
    """Derived obs/token dims. slot(n,d)=(d-1)*n_limbs+(n-1) depth-major; sensors stay per-limb (6).
    obs_base = start of the lengths block (== phase-1 semantics, so mask_off = obs_base + len_dim)."""
    n_dof = n_limbs * max_len
    sens_off = ROOT_DIM_P2 + MODULE_DYN * n_dof                    # start of per-limb contact sensors
    obs_base = sens_off + n_limbs * 6                             # start of lengths block
    return dict(n_limbs=n_limbs, max_len=max_len, n_dof=n_dof, sens_off=sens_off, obs_base=obs_base,
                len_dim=n_dof, mask_dim=n_dof, obs_total=obs_base + 2 * n_dof,
                n_module_tokens=n_dof, n_tokens=1 + n_limbs + n_dof)  # CLS + start*n + module*n_dof


def limb_enc(n_limbs: int) -> torch.Tensor:
    """sin/cos of each limb's placement angle i*45deg (matches build_vsim._DIR / _LIMB_ENC_8).
    Still used for the model's per-limb START-anchor (angle_proj); NOT a per-module token feature."""
    return torch.tensor([[math.sin(i * math.pi / 4), math.cos(i * math.pi / 4)] for i in range(n_limbs)],
                        dtype=torch.float32)


def tokenize_modules(obs, n_limbs: int, max_len: int, enc: torch.Tensor = None):
    """Tokenize a depth-major relative-geometry obs into (root, module_tokens, active_mask).
      root:          (B, ROOT_DIM_P2=13) = y + rot6d(6) + linvel(3) + angvel(3).
      module_tokens: (B, n_limbs*max_len, MODULE_DIM=25) canonical depth-major slot order:
                     [sin, cos, vel, act, cfrc(6), relpos(3), relrot6d(6), relvel(6)].
      active_mask:   (B, n_limbs*max_len) 1 for real modules. cfrc rides ONLY on each limb's
                     terminal (last active) module. `enc` is accepted for call compat but unused
                     (per-module limb-direction feature dropped in 2a). Inactive slots stay 0."""
    B = obs.shape[0]
    d = token_dims(n_limbs, max_len)
    n_dof, sens_off, obs_base = d["n_dof"], d["sens_off"], d["obs_base"]
    dev = obs.device
    root = obs[:, 0:ROOT_DIM_P2]

    dm = lambda x, w: x.reshape(B, max_len, n_limbs, w)           # depth-major [B, d, n, w]; slot-major flat
    o, parts = ROOT_DIM_P2, []
    for _, w in _MOD_BLOCKS:
        parts.append(dm(obs[:, o:o + w * n_dof], w)); o += w * n_dof
    sin_d, cos_d, vel_d, act_d, relpos_d, relrot_d, relvel_d = parts

    mask_off = obs_base + d["len_dim"]                            # skip the lengths block
    raw_mask = (obs[:, mask_off:mask_off + n_dof] if obs.shape[1] >= mask_off + n_dof
                else torch.ones(B, n_dof, device=dev))
    mask_d = raw_mask.reshape(B, max_len, n_limbs)               # (B, d, n)

    count = mask_d.sum(dim=1)                                     # (B, n_limbs) chain length
    depth0 = torch.arange(max_len, device=dev).view(1, max_len, 1)
    is_terminal = (mask_d > 0) & (depth0 == (count.unsqueeze(1) - 1))   # (B, d, n) last active module
    sens = obs[:, sens_off:sens_off + n_limbs * 6].view(B, n_limbs, 6)
    cfrc = torch.where(is_terminal.unsqueeze(-1),
                       sens.unsqueeze(1).expand(B, max_len, n_limbs, 6),
                       torch.zeros(B, max_len, n_limbs, 6, device=dev))   # (B, d, n, 6)

    module = torch.cat([sin_d, cos_d, vel_d, act_d, cfrc, relpos_d, relrot_d, relvel_d], dim=-1)
    module_tokens = module.reshape(B, n_dof, MODULE_DIM)         # (B, n_dof, 25) depth-major slot order
    active_mask   = (mask_d > 0).reshape(B, n_dof).float()
    return root, module_tokens, active_mask
