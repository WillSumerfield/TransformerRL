import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import gymnasium as gym

import envs
from envs import sample_valid_mask, LimbMaskObsWrapper

# ---------------------------------------------------------------------------
# Base env (105-dim obs, mask in info)
# ---------------------------------------------------------------------------

env = gym.make("CustomAnt-v5")

# 1. full mask — baseline
obs, info = env.reset(options={"limb_mask": np.ones(8, dtype=bool)})
assert obs.shape == (105,), f"expected (105,), got {obs.shape}"
assert np.all(info["limb_mask"] == 1.0), "mask in info should be all-ones"
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (105,)
    assert isinstance(reward, float)
    assert np.all(info["limb_mask"] == 1.0)
print("1. full mask (base env): OK")

# 2. one-leg-removed mask — leg 4 (indices 6,7) off
mask2 = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
obs, info = env.reset(options={"limb_mask": mask2})
assert obs.shape == (105,)
assert obs[11] == 0.0 and obs[12] == 0.0, f"hip_4/ankle_4 qpos not zeroed: {obs[11:13]}"
assert obs[25] == 0.0 and obs[26] == 0.0, f"hip_4/ankle_4 qvel not zeroed: {obs[25:27]}"
assert np.array_equal(info["limb_mask"], mask2.astype(float)), "mask in info mismatch"
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs[11] == 0.0 and obs[12] == 0.0
    assert obs[25] == 0.0 and obs[26] == 0.0
print("2. one-leg-removed mask (base env): OK")

# 3. half mask — first 4 joints on, last 4 off
mask = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
obs, info = env.reset(options={"limb_mask": mask})
assert np.all(obs[9:13]  == 0.0), f"joints 4-7 qpos not zeroed: {obs[9:13]}"
assert np.all(obs[23:27] == 0.0), f"joints 4-7 qvel not zeroed: {obs[23:27]}"
assert not np.all(obs[5:9] == 0.0), "active joints should be nonzero"
assert np.array_equal(info["limb_mask"], mask.astype(float)), "mask in info mismatch"
print("3. half mask (base env): OK")

# 4. invalid mask — hip off, ankle on → ValueError
try:
    env.reset(options={"limb_mask": np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)})
    assert False, "expected ValueError"
except ValueError as e:
    print(f"4. invalid mask correctly raised ValueError: {e}")

# 5. re-reset back to full mask — model restores correctly
obs, info = env.reset(options={"limb_mask": np.ones(8, dtype=bool)})
assert not np.all(obs[5:13] == 0.0), "after restore, joint obs should be nonzero"
print("5. re-reset with full mask (base env): OK")

# 6. random_morphology env auto-samples valid masks
env_rng = gym.make("CustomAnt-v5", random_morphology=True)
for _ in range(10):
    obs, info = env_rng.reset()
    assert obs.shape == (105,)
    mask_bits = info["limb_mask"]
    for hip_i, ank_i in [(0,1),(2,3),(4,5),(6,7)]:
        assert not (mask_bits[hip_i] == 0.0 and mask_bits[ank_i] == 1.0), \
            f"invalid mask sampled: {mask_bits}"
    assert mask_bits[[0,2,4,6]].sum() > 0, "at least 1 hip must be active"
env_rng.close()
print("6. random_morphology (base env): OK")

# 7. sample_valid_mask() correctness
rng = np.random.default_rng(42)
for _ in range(200):
    m = sample_valid_mask(rng)
    assert m.dtype == bool and m.shape == (8,)
    for hip_i, ank_i in [(0,1),(2,3),(4,5),(6,7)]:
        assert not (not m[hip_i] and m[ank_i]), "ankle on without hip"
    assert m[[0,2,4,6]].any(), "at least 1 hip must be active"
print("7. sample_valid_mask: OK")

env.close()

# ---------------------------------------------------------------------------
# Wrapped env (LimbMaskObsWrapper → 113-dim obs, raw mask appended)
# ---------------------------------------------------------------------------

wenv = LimbMaskObsWrapper(gym.make("CustomAnt-v5"))

# 8. full mask wrapped — obs is 121-dim, mask all-ones, sin/cos at original angles
obs, info = wenv.reset(options={"limb_mask": np.ones(8, dtype=bool)})
assert obs.shape == (121,), f"expected (121,), got {obs.shape}"
assert np.all(obs[105:113] == 1.0), "wrapped mask bits should be all-ones"
assert "leg_angles" in info, "leg_angles missing from info"
assert obs.shape[0] == 121, "rotation features should be present"
print("8. full mask (wrapped env): OK")

# 9. partial mask wrapped — mask dims match, rotation dims non-trivial
mask = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
obs, info = wenv.reset(options={"limb_mask": mask})
assert obs.shape == (121,)
assert np.array_equal(obs[105:113], mask.astype(float)), \
    f"wrapped mask mismatch: {obs[105:113]} vs {mask.astype(float)}"
# sin^2 + cos^2 == 1 for each leg
for i in range(4):
    s, c = obs[113 + 2*i], obs[113 + 2*i + 1]
    assert abs(s**2 + c**2 - 1.0) < 1e-9, f"leg {i}: sin^2+cos^2 != 1"
print("9. partial mask (wrapped env): OK")

# 10. step preserves mask and rotation in wrapped obs
obs, _, _, _, info = wenv.step(wenv.action_space.sample())
assert obs.shape == (121,)
assert np.array_equal(obs[105:113], mask.astype(float)), "mask must persist across steps"
print("10. step mask persistence (wrapped env): OK")

wenv.close()
print("PASS")
