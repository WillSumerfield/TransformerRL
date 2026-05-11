"""Ant-v5 / CustomAnt-v5 observation tokenization.

Obs layout (105 core dims, optionally +8 limb-mask dims = 113):
  [0]      torso z
  [1:5]    torso quat
  [5:13]   8 joint angles  (h1, a1, h2, a2, h3, a3, h4, a4)  ← qpos order
  [13:16]  torso linvel
  [16:19]  torso angvel
  [19:27]  8 joint velocities (same order)
  [27:33]  torso cfrc_ext
  [33:105] per-leg cfrc blocks of [*_leg, aux, foot] x 4 legs (each 6 dim)
  [105:113] limb mask in qpos order (0.0=inactive, 1.0=active) — optional

Body mapping (verified): hip_N joint actuates aux_N; ankle_N joint actuates foot_N;
*_leg_N is welded to torso. So *_leg cfrcs get absorbed into the torso token, and
hip/ankle tokens carry only the cfrc of the body they directly actuate.
"""
import torch

N_LEGS = 4
TORSO_DIM = 41   # 1 + 4 + 3 + 3 + 6 + 4*6
HIP_DIM = 10     # 1 + 1 + 6 + 2 (angle, vel, cfrc, sin/cos rotation)
ANKLE_DIM = 10   # same
OBS_DIM = 105
MASK_DIM = 8
ROT_DIM = 8      # sin+cos for 4 legs, interleaved: [s1,c1,s2,c2,s3,c3,s4,c4]

# Reorder mask from qpos order [h1,a1,h2,a2,h3,a3,h4,a4]
# to natural token order       [h1,h2,h3,h4,a1,a2,a3,a4]
_QPOS_TO_NAT = [0, 2, 4, 6, 1, 3, 5, 7]


def tokenize(obs: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    obs: [B, 105], [B, 113], or [B, 121]
    Returns:
      tokens: {torso:[B,1,41], hip:[B,4,10], ankle:[B,4,10]}
      active_mask: [B, 8] float in natural order [h1,h2,h3,h4,a1,a2,a3,a4]
                   1.0=active, 0.0=inactive (all-ones when obs is 105-dim)
    """
    B = obs.shape[0]

    if obs.shape[1] > OBS_DIM:
        mask_qpos = obs[:, OBS_DIM:OBS_DIM + MASK_DIM]            # [B, 8] qpos order
        mask_nat  = mask_qpos[:, _QPOS_TO_NAT]                    # [B, 8] natural order
        active_mask = (mask_nat > 0.5).float()
    else:
        active_mask = torch.ones(B, 8, device=obs.device)

    rot_start = OBS_DIM + MASK_DIM                                # 113
    if obs.shape[1] >= rot_start + ROT_DIM:                       # 121-dim obs
        rot = obs[:, rot_start:rot_start + ROT_DIM].view(B, N_LEGS, 2)  # [B, 4, 2]
    else:
        rot = torch.zeros(B, N_LEGS, 2, device=obs.device)
        rot[:, :, 1] = 1.0                                        # cos=1 (angle=0) fallback

    obs = obs[:, :OBS_DIM]

    # joint angles/vels: (hip, ankle) interleaved per leg
    j_ang = obs[:, 5:13].view(B, N_LEGS, 2)
    j_vel = obs[:, 19:27].view(B, N_LEGS, 2)

    # cfrc: 13 non-world bodies; first is torso, then 4 legs of 3 bodies each
    cfrc_torso = obs[:, 27:33]                            # [B, 6]
    cfrc_legs = obs[:, 33:105].view(B, N_LEGS, 3, 6)      # [B, 4, 3, 6]
    cfrc_upper = cfrc_legs[:, :, 0, :]                    # *_leg  [B, 4, 6]
    cfrc_aux   = cfrc_legs[:, :, 1, :]                    # hip   [B, 4, 6]
    cfrc_foot  = cfrc_legs[:, :, 2, :]                    # ankle [B, 4, 6]

    torso = torch.cat([
        obs[:, 0:1], obs[:, 1:5], obs[:, 13:16], obs[:, 16:19],
        cfrc_torso, cfrc_upper.reshape(B, -1),
    ], dim=-1).unsqueeze(1)                               # [B, 1, 41]

    hip   = torch.cat([j_ang[:, :, 0:1], j_vel[:, :, 0:1], cfrc_aux,  rot], dim=-1)  # [B, 4, 10]
    ankle = torch.cat([j_ang[:, :, 1:2], j_vel[:, :, 1:2], cfrc_foot, rot], dim=-1)  # [B, 4, 10]

    return {"torso": torso, "hip": hip, "ankle": ankle}, active_mask


def _verify_body_mapping() -> None:
    """Assert Ant's kinematic tree matches what tokenize() assumes."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import envs  # noqa: F401
    import gymnasium as gym
    env = gym.make("CustomAnt-v5")
    m = env.unwrapped.model
    # hip_N on aux_N (body 3,6,9,12); ankle_N on foot (body 4,7,10,13)
    expected = {"hip_1": 3, "ankle_1": 4, "hip_2": 6, "ankle_2": 7,
                "hip_3": 9, "ankle_3": 10, "hip_4": 12, "ankle_4": 13}
    for i in range(m.njnt):
        n = m.joint(i).name
        if n in expected:
            assert m.jnt_bodyid[i] == expected[n], f"{n}: got body {m.jnt_bodyid[i]}, expected {expected[n]}"
    env.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import envs  # noqa: F401
    import gymnasium as gym

    _verify_body_mapping()

    env = gym.make("CustomAnt-v5")
    o, _ = env.reset(seed=0)                              # o is 113-dim
    o = torch.from_numpy(o).float().unsqueeze(0).repeat(3, 1)  # [3, 113]
    tokens, mask = tokenize(o)
    for k, v in tokens.items():
        assert not torch.isnan(v).any()
        print(f"{k:6s} {tuple(v.shape)}")
    print(f"mask   {tuple(mask.shape)}")
    assert tokens["torso"].shape == (3, 1, TORSO_DIM),  tokens["torso"].shape
    assert tokens["hip"].shape   == (3, N_LEGS, HIP_DIM),   tokens["hip"].shape
    assert tokens["ankle"].shape == (3, N_LEGS, ANKLE_DIM), tokens["ankle"].shape
    assert mask.shape == (3, 8)
    assert mask.all(), "full-mask env should have all-ones mask"
    print("ok")
    env.close()
