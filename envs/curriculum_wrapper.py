import numpy as np
import gymnasium

from envs import sample_valid_mask


class CurriculumWrapper(gymnasium.Wrapper):
    """p(random morphology) = 0 until onset_start_steps, then ramps linearly to 1 by onset_end_steps."""

    def __init__(self, env: gymnasium.Env, onset_start_steps: int, onset_end_steps: int, num_envs: int):
        super().__init__(env)
        self._onset_start = onset_start_steps
        self._onset_end   = onset_end_steps
        self._steps = 0

    @property
    def p_mask(self) -> float:
        if self._steps < self._onset_start:
            return 0.0
        if self._steps >= self._onset_end:
            return 1.0
        return (self._steps - self._onset_start) / (self._onset_end - self._onset_start)

    def reset(self, *, seed=None, options=None):
        if np.random.random() < self.p_mask:
            mask = sample_valid_mask()
        else:
            mask = np.ones(8, dtype=bool)
        options = {**(options or {}), "limb_mask": mask}
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        result = self.env.step(action)
        self._steps += 1
        return result
