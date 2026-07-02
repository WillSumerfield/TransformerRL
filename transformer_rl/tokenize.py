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


# ── Phase 1: variable-length limbs, uniform module tokens, parameterized by (n_limbs, max_len) ──
# Single source of truth for the 32-DOF depth-major obs layout (must match ant_multimorph._slot and
# its _OBS_* constants, which derive from the SAME (n_limbs, max_len)). A limb is a chain of up to
# max_len modules; each module is one uniform token.
MODULE_DIM = 12   # pos(1)+vel(1)+last_action(1)+sin(1)+cos(1)+module_len(1)+cfrc(6)


def token_dims(n_limbs: int, max_len: int) -> dict:
    """Derived obs/token dims. slot(n,d)=(d-1)*n_limbs+(n-1) depth-major; sensors stay per-limb (6)."""
    n_dof = n_limbs * max_len
    obs_base = ROOT_DIM + 3 * n_dof + n_limbs * 6          # root + dofpos+dofvel+lastact + sensors
    return dict(n_limbs=n_limbs, max_len=max_len, n_dof=n_dof, obs_base=obs_base,
                len_dim=n_dof, mask_dim=n_dof, obs_total=obs_base + 2 * n_dof,
                n_module_tokens=n_dof, n_tokens=1 + n_limbs + n_dof)  # CLS + start*n + module*n_dof


def limb_enc(n_limbs: int) -> torch.Tensor:
    """sin/cos of each limb's placement angle i*45deg (matches build_vsim._DIR / _LIMB_ENC_8)."""
    return torch.tensor([[math.sin(i * math.pi / 4), math.cos(i * math.pi / 4)] for i in range(n_limbs)],
                        dtype=torch.float32)


def tokenize_modules(obs, n_limbs: int, max_len: int, enc: torch.Tensor = None):
    """Tokenize a depth-major variable-length obs into (root, module_tokens, active_mask).
      module_tokens: (B, n_limbs*max_len, MODULE_DIM) in canonical depth-major slot order.
      active_mask:   (B, n_limbs*max_len) 1 for real modules. cfrc rides ONLY on each limb's
                     terminal (last active) module.
    """
    B = obs.shape[0]
    d = token_dims(n_limbs, max_len)
    n_dof, obs_base = d["n_dof"], d["obs_base"]
    dev = obs.device
    root    = obs[:, 0:ROOT_DIM]
    dof_pos = obs[:, ROOT_DIM              : ROOT_DIM + n_dof]
    dof_vel = obs[:, ROOT_DIM + n_dof      : ROOT_DIM + 2 * n_dof]
    acts    = obs[:, ROOT_DIM + 2 * n_dof  : ROOT_DIM + 3 * n_dof]
    sensors = obs[:, ROOT_DIM + 3 * n_dof  : obs_base]              # (B, n_limbs*6) per-limb contact
    lengths = obs[:, obs_base               : obs_base + n_dof]
    mask_off = obs_base + n_dof
    raw_mask = (obs[:, mask_off : mask_off + n_dof] if obs.shape[1] >= mask_off + n_dof
                else torch.ones(B, n_dof, device=dev))

    dm = lambda x: x.view(B, max_len, n_limbs)                     # depth-major [B, d, n]
    pos_d, vel_d, act_d, len_d, mask_d = dm(dof_pos), dm(dof_vel), dm(acts), dm(lengths), dm(raw_mask)

    enc = (limb_enc(n_limbs) if enc is None else enc).to(dev)      # (n_limbs, 2)
    # per-module gate: sin/cos ride only on ACTIVE modules (inactive slots stay fully zero)
    enc_dn = enc.view(1, 1, n_limbs, 2) * (mask_d > 0).unsqueeze(-1).float()   # (B, d, n, 2)

    count = mask_d.sum(dim=1)                                      # (B, n_limbs) chain length
    depth0 = torch.arange(max_len, device=dev).view(1, max_len, 1)
    is_terminal = (mask_d > 0) & (depth0 == (count.unsqueeze(1) - 1))   # (B, d, n) last active module
    sens = sensors.view(B, n_limbs, 6)
    cfrc = torch.where(is_terminal.unsqueeze(-1),
                       sens.unsqueeze(1).expand(B, max_len, n_limbs, 6),
                       torch.zeros(B, max_len, n_limbs, 6, device=dev))

    module = torch.cat([pos_d.unsqueeze(-1), vel_d.unsqueeze(-1), act_d.unsqueeze(-1),
                        enc_dn, len_d.unsqueeze(-1), cfrc], dim=-1)     # (B, d, n, 12)
    module_tokens = module.reshape(B, n_dof, MODULE_DIM)          # depth-major slot order
    active_mask   = (mask_d > 0).reshape(B, n_dof).float()
    return root, module_tokens, active_mask
