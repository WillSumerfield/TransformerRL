import numpy as np
import gymnasium


class LimbMaskObsWrapper(gymnasium.Wrapper):
    """Appends limb mask (8) and leg rotation sin/cos (8) to 105-dim obs → 121-dim.

    Layout of appended dims:
      [105:113] limb mask in qpos order (0.0/1.0)
      [113:121] leg rotation: [sin_1, cos_1, sin_2, cos_2, sin_3, cos_3, sin_4, cos_4]
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        low  = np.concatenate([env.observation_space.low,  np.zeros(8), np.full(8, -1.0)])
        high = np.concatenate([env.observation_space.high, np.ones(8),  np.ones(8)])
        self.observation_space = gymnasium.spaces.Box(
            low=low, high=high, dtype=np.float64
        )

    def _rot_features(self, info: dict) -> np.ndarray:
        angles = info.get("leg_angles", np.zeros(4))
        return np.array([v for a in angles for v in (np.sin(a), np.cos(a))])

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        mask = info.get("limb_mask", np.ones(8, dtype=np.float64))
        return np.concatenate([obs, mask, self._rot_features(info)]), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        mask = info.get("limb_mask", np.ones(8, dtype=np.float64))
        return np.concatenate([obs, mask, self._rot_features(info)]), rew, term, trunc, info
