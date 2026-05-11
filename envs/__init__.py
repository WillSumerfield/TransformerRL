import numpy as np
from gymnasium.envs.registration import register
from .limb_mask_wrapper import LimbMaskObsWrapper

register(
    id="CustomAnt-v5",
    entry_point="envs.ant_v5:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)


def sample_valid_mask(rng=None) -> np.ndarray:
    """Uniform over 5 morphologies: all legs intact, or exactly one leg removed.

    Returns an 8-element bool array in qpos order:
      [hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4]
    """
    rng = rng or np.random.default_rng()
    mask = np.ones(8, dtype=bool)
    leg = rng.integers(5)       # 0=all intact, 1-4=remove that leg
    if leg > 0:
        i = leg - 1
        mask[2 * i] = False     # hip_i
        mask[2 * i + 1] = False # ankle_i
    return mask
