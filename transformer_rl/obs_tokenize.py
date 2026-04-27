"""Ant-v5 observation tokenization.

Obs layout (105):
  [0]      torso z
  [1:5]    torso quat
  [5:13]   8 joint angles  (h1, a1, h2, a2, h3, a3, h4, a4)
  [13:16]  torso linvel
  [16:19]  torso angvel
  [19:27]  8 joint velocities (same order)
  [27:33]  torso cfrc_ext
  [33:105] per-leg cfrc blocks of [*_leg, aux, foot] x 4 legs (each 6 dim)

Body mapping (verified): hip_N joint actuates aux_N; ankle_N joint actuates foot_N;
*_leg_N is welded to torso. So *_leg cfrcs get absorbed into the torso token, and
hip/ankle tokens carry only the cfrc of the body they directly actuate.
"""
import torch

N_LEGS = 4
TORSO_DIM = 41   # 1 + 4 + 3 + 3 + 6 + 4*6
HIP_DIM = 8      # 1 + 1 + 6
ANKLE_DIM = 8    # 1 + 1 + 6


def tokenize(obs: torch.Tensor) -> dict[str, torch.Tensor]:
    """obs: [B, 105] float -> {torso:[B,1,41], hip:[B,4,8], ankle:[B,4,8]}."""
    B = obs.shape[0]

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

    hip   = torch.cat([j_ang[:, :, 0:1], j_vel[:, :, 0:1], cfrc_aux],  dim=-1)  # [B, 4, 8]
    ankle = torch.cat([j_ang[:, :, 1:2], j_vel[:, :, 1:2], cfrc_foot], dim=-1)  # [B, 4, 8]

    return {"torso": torso, "hip": hip, "ankle": ankle}


def _verify_body_mapping() -> None:
    """Assert Ant's kinematic tree matches what tokenize() assumes."""
    import gymnasium as gym
    env = gym.make("Ant-v5")
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
    import gymnasium as gym
    _verify_body_mapping()
    env = gym.make("Ant-v5")
    o, _ = env.reset(seed=0)
    o = torch.from_numpy(o).float().unsqueeze(0).repeat(3, 1)  # [3, 105]
    out = tokenize(o)
    for k, v in out.items():
        assert not torch.isnan(v).any()
        print(f"{k:6s} {tuple(v.shape)}")
    assert out["torso"].shape == (3, 1, TORSO_DIM)
    assert out["hip"].shape   == (3, N_LEGS, HIP_DIM)
    assert out["ankle"].shape == (3, N_LEGS, ANKLE_DIM)
    print("ok")
    env.close()
