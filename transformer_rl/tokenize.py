import math
import torch

TORSO_DIM = 11   # y(1) + quat(4) + linvel(3) + angvel(3)
HIP_DIM   = 5    # pos(1) + vel(1) + last_action(1) + sin(1) + cos(1)
ANKLE_DIM = 11   # pos(1) + vel(1) + cfrc(6) + last_action(1) + sin(1) + cos(1)

OBS_DIM_4  = 59
MASK_DIM_4 = 8
OBS_DIM_8  = 107
MASK_DIM_8 = 16

_LEG_ENC_4 = torch.tensor(
    [[math.sin((2*i + 1) * math.pi / 4), math.cos((2*i + 1) * math.pi / 4)]
     for i in range(4)],
    dtype=torch.float32,
)  # angles: pi/4, 3pi/4, 5pi/4, 7pi/4

_LEG_ENC_8 = torch.tensor(
    [[math.sin(i * math.pi / 4), math.cos(i * math.pi / 4)] for i in range(8)],
    dtype=torch.float32,
)


def _tokenize(obs, n_legs, obs_dim, mask_dim, leg_enc):
    B = obs.shape[0]
    enc = leg_enc.to(obs.device).unsqueeze(0).expand(B, -1, -1)  # (B, n_legs, 2)

    n_dof     = 2 * n_legs
    torso     = obs[:, 0:11]
    dof_pos   = obs[:, 11 : 11 + n_dof]
    dof_vel   = obs[:, 11 + n_dof : 11 + 2 * n_dof]
    acts      = obs[:, 11 + 2 * n_dof : 11 + 3 * n_dof]
    sensors   = obs[:, 11 + 3 * n_dof : obs_dim]
    raw_mask  = (obs[:, obs_dim : obs_dim + mask_dim]
                 if obs.shape[1] >= obs_dim + mask_dim
                 else torch.ones(B, mask_dim, device=obs.device))

    hip_tokens = torch.stack([
        torch.cat([dof_pos[:, 2*i:2*i+1], dof_vel[:, 2*i:2*i+1],
                   acts[:, 2*i:2*i+1], enc[:, i, :]], dim=-1)
        for i in range(n_legs)
    ], dim=1)  # (B, n_legs, 5)

    ankle_tokens = torch.stack([
        torch.cat([dof_pos[:, 2*i+1:2*i+2], dof_vel[:, 2*i+1:2*i+2],
                   sensors[:, 6*i:6*i+6], acts[:, 2*i+1:2*i+2], enc[:, i, :]], dim=-1)
        for i in range(n_legs)
    ], dim=1)  # (B, n_legs, 11)

    active_mask = torch.cat([raw_mask[:, 0::2], raw_mask[:, 1::2]], dim=-1)  # (B, mask_dim)
    return torso, hip_tokens, ankle_tokens, active_mask


def tokenize_4(obs):
    return _tokenize(obs, 4, OBS_DIM_4, MASK_DIM_4, _LEG_ENC_4)


def tokenize_8(obs):
    return _tokenize(obs, 8, OBS_DIM_8, MASK_DIM_8, _LEG_ENC_8)
