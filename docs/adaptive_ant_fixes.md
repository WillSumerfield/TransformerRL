# Adaptive Ant Fixes

This document records the bugs found and fixed while making the **adaptive ant**
(`AntAdaptiveEnv`, a codesign env restricted to morphology `{2,4,6,8}`) train
equivalently to the **classic ant** (`AntEnv`), plus the masking machinery that
makes the shared 8-leg transformer work. It is organized symptom-first: skim the
quick-reference table, then read the entry for whatever you're chasing.

---

## Quick reference

| Symptom | Root cause | Fix | File |
|---|---|---|---|
| Adaptive ≠ classic on a synced rollout | vsim emits DOFs in reverse leg order; scatter assumed ascending | Declare hip/ankle joints in reverse so DOFs come out ascending | `envs/ant_envs/build_vsim.py` |
| Entropy cliffs from `-57 → -137` after ~1 min | Input normalizer collapses the constant limb-mask to 0 (fp32 rounding); every leg reads "inactive" | Mask-passthrough model: leave `obs[107:123]` un-normalized | `transformer_rl/models.py`, configs |
| Spiky `a_loss`, jerky reward; LR pinned at `0.01` | `σ=e⁻¹⁰` on inactive dims poisons `policy_kl` (−0.5/dim → ≈−4 KL), so adaptive LR cranks to max | Inactive `log_std = 0` (σ=1) instead of `-10` | `transformer_rl/models.py` |
| Inactive `log_std_param` drifts under entropy bonus | Entropy gradient hit every dim, masked or not | Gradient-gate: `log_std = mask_dof * log_std_param` | `transformer_rl/models.py` |
| (ruled out) Suspected pos-emb index mismatch | — | Proven to be pure relabeling; no effect | `scripts/test_pos_emb_permutation.py` |

**Net effect:** adaptive ant with morph `{2,4,6,8}` is now bit-for-bit equivalent
to classic ant through a forward pass *and* 10 physics steps (actions, value,
reward, losses), and trains with a stable, KL-controlled learning rate.

---

## Background: how the adaptive ant is wired

Understanding three layout facts makes every entry below clearer.

- **Two ants, two obs sizes.** Classic `AntEnv` is a fixed 4-leg ant: **59-D obs,
  8-D actions**, network `LegTransformer(n_legs=4)`. Adaptive `AntAdaptiveEnv`
  inherits the codesign stack: **123-D obs, 16-D actions** (always padded to
  8 legs), network `DynamicLegTransformer(n_legs=8)`.

- **The 8-leg obs layout (123-D):**
  ```
  [0:11]    torso (y, quat, lin vel, ang vel)
  [11:27]   dof_pos   (16 slots, inactive = 0)
  [27:43]   dof_vel   (16 slots, inactive = 0)
  [43:59]   last_act  (16 slots, inactive = 0)
  [59:107]  force sensors (8 legs × 6)
  [107:123] limb_mask (16 bits: 1 = active DOF, 0 = inactive)   ← the crux
  ```
  The limb mask is written **once** at allocation and never touched during
  stepping. The tokenizer (`tokenize_8`) and the policy builder both read
  `obs[107:123]` purely through a `> 0` boolean test to decide which legs exist.

- **`{2,4,6,8}` ≡ classic.** build_vsim places leg `n` at angle `(n-1)·45°`, so
  legs `{2,4,6,8}` sit at `45/135/225/315°` — exactly the classic ant's four
  legs. That physical identity is what lets us assert exact parity.

---

## Detailed entries

### Symptom: adaptive ant ≠ classic ant on a synchronized rollout

**Observed.** We built a parity harness (`scripts/test_ant_parity.py`) that resets
classic ant, copies its kinematic state into adaptive ant (`{2,4,6,8}`), and feeds
both the same `DynamicLegTransformer` weights. Obs matched at reset, the forward
pass matched — but after **one physics step**, state diverged hugely:

```
torso  [0:11]    3.9e-01
dofpos [11:27]   2.9e-02
dofvel [27:43]   1.7e+00     ← not float noise; structurally wrong
```

**Investigation.** Obs/forward parity at reset ruled out tokenization and the
network. The divergence appeared only after `step()`, pointing at the env
applying forces to the *wrong joints*. We printed the articulation DOF names:

```
classic  : [hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4]  (ascending)
adaptive : [hip_8, ankle_8, hip_6, ankle_6, hip_4, ankle_4, hip_2, ankle_2]  (DESCENDING)
```

**Root cause.** vsim builds its DOF order by a depth-first traversal that visits a
torso's child joints in **reverse declaration order**. `build_vsim` declared hips
ascending (`hip_2, hip_4, …`), so vsim emitted them descending. But
`AntCodesignEnv.compute_observations` scatters `dof_pos[i]` into the obs slot for
the *i-th active leg in ascending order* — so leg N's reading landed in the obs
slot (and motor) of the diametrically opposite leg. Parity at *reset* still
passed because both the bad sync and the bad scatter cancelled when all values are
near zero; the error only manifests once physics moves the joints.

**The fix** (`build_vsim.py:211-219`): declare hip and ankle joints in
**reverse** active order, so vsim's reverse-DFS produces DOFs in ascending order,
matching the scatter assumption.

```python
for n in reversed(active):
    parts.append(_hip_joint(n, _LEG_DATA[n - 1]))
for n in reversed(active):
    parts.append(_ankle_joint(n, _LEG_DATA[n - 1]))
```

After the fix the adaptive DOF order is `[hip_2, ankle_2, hip_4, …]` and all parity
checks drop to `0.000e+00`.

**Why it's useful.** Without this, *every* codesign morphology was training on
mirror-scrambled observations — the policy had to learn to compensate for a
permutation, and any checkpoint trained pre-fix is on corrupted data.

**Tradeoffs.** None functionally; it only changes XML emission order. It does
invalidate pre-fix checkpoints (they learned the scrambled mapping).

---

### Symptom: entropy cliffs from `-57` to `-137` after ~1 minute of training

**Observed.** Training entropy sat at the expected `≈ -57.3`, then dropped in one
straight step to `≈ -137.3` and stayed there.

The numbers decode exactly. Entropy of a diagonal Gaussian is
`Σ (1.4189 + log_stdᵢ)`:
- Healthy adaptive: 8 active dims (`log_std=0`) + 8 inactive (`log_std=-10`) =
  `8·1.4189 + 8·(1.4189-10) = -57.3`.
- `-137.3 = 16·(1.4189-10)` → **all 16 dims** at `log_std=-10`, i.e. every leg
  read as inactive.

**Investigation.** The mask is constant per env, and `log_std=-10` is applied to
any dim where `(obs[107:123] > 0)` is false. So the cliff meant the *active* mask
entries had become ≤ 0. The only thing between the env (which writes raw `{0,1}`)
and the network is rl_games' `RunningMeanStd`. Simulating it with the real batch
size reproduced the cliff precisely:

```
epoch 127:  1-mean=3.0e-8   norm_active=1.88e-5   >0? True
epoch 128:  1-mean=2.9e-8   norm_active=0.0       >0? False   ← collapse
```

**Root cause.** `RunningMeanStd.forward` computes `(x - running_mean.float()) / …`.
The mask's running mean converges toward `1.0`; once it gets within `2⁻²⁵≈3e-8`,
the **float32 cast rounds it to exactly `1.0`**, so `(1.0 - 1.0) = 0` and every
active mask dim normalizes to `0`. The `> 0` checks then flip all legs to inactive
→ all `log_std = -10` → entropy `-137`, and the policy outputs ≈0 with no
exploration. It's a hard threshold (hence "one straight drop"), not a slow decay.
A 50-epoch test missed it because `1-mean` was still `~5e-6`.

**The fix** (`transformer_rl/models.py:87`): a model subclass
`TransformerMaskedNorm(ModelA2CContinuousLogStd)` that keeps the stock normalizer
but overrides `norm_obs` to restore the raw mask after normalization:

```python
def norm_obs(self, observation):
    normed = super().norm_obs(observation)        # standard RMS (jit, 123-D)
    if self.normalize_input:
        normed = normed.clone()
        normed[..., -DYN_MASK_DIM:] = observation[..., -DYN_MASK_DIM:]   # restore raw {0,1}
    return normed
```

It's registered as `transformer_masked_a2c_logstd` (`train_utils.py:365`) and
selected via `model.name` in `ppo_ant_adaptive.yaml` and `ppo_ant_dynamic_p1.yaml`.
The mask size comes from the existing `MASK_DIM_8` constant (shared with the
tokenizer — single source of truth), and a build-time assert enforces
`obs == OBS_DIM_8 + MASK_DIM_8 (123)`.

**Why this design.** `norm_obs` is the single normalization chokepoint (training
*and* the player both call it), while checkpoint save/load and the `.eval()` that
freezes stats touch `running_mean_std` directly — so keeping the stock RMS object
and overriding only `norm_obs` means save/load/eval all keep working untouched. No
custom RMS class, no attribute-swapping.

**Verification.** `scripts/test_mask_norm.py` drives a standard and the masked
model to the collapse point on identical obs and asserts: standard collapses the
active mask to 0 (bug reproduced), masked preserves raw `{0,1}` exactly, masked
still normalizes the physical dims, and `(mask>0)` recovers `{2,4,6,8}`.

**Tradeoffs.** The mask dims still get (unused) running stats computed for them —
negligible cost, and someone inspecting `running_mean_std` would see mask stats
that look normalized but aren't applied. Purely cosmetic.

---

### Symptom: spiky `a_loss` and jerky reward (entropy now correct)

**Observed.** With the mask fix in, entropy held but the actor loss was far
spikier than classic and reward was jerky.

**Investigation.** Spiky `a_loss` with an otherwise-healthy setup points at the
**adaptive learning-rate controller** getting a bad KL. We checked rl_games'
`policy_kl` (`torch_ext.py:27`) and the `AdaptiveScheduler` (`schedulers.py:26`).

**Root cause.** `policy_kl` has `ε=1e-5` in its denominator:
`c2 = (σ₀² + Δμ²) / (2(σ₁² + ε))`. For an inactive dim, `σ=e⁻¹⁰`, so
`σ²≈2e-9 ≪ ε`, and `c2 → 0` instead of the `0.5` it should be for identical
distributions — leaving `c1(≈0) + 0 + c3(−0.5) = −0.4999` per inactive dim.
With 8 inactive dims that's a **≈−4.0** offset on the summed KL. The scheduler:
```python
if kl_dist < 0.5 * kl_threshold:   # 0.0075
    lr = min(lr * 1.5, 1e-2)        # max_lr
```
sees `kl_dist ≈ active_KL − 4 ≈ −3.99` every epoch, so it multiplies LR by 1.5
each epoch until it pins at `max_lr = 1e-2` (~14× the configured `7e-4`, reached
in ~7 epochs). A too-large pinned LR makes the policy overshoot → spiky `a_loss`,
jerky reward. (Classic has no inactive dims, so its KL is honest.)

**The fix** (`transformer_rl/models.py:37,75`): set inactive `log_std = 0` (σ=1)
instead of `-10`. Combined with the gradient-gating, both builders use:

```python
log_std = mask_dof * self.log_std_param   # active: log_std_param; inactive: 0 (σ=1)
```

The env already zeros inactive actions (`_act_buf = actions * limb_mask`), so the
inactive σ is **irrelevant to dynamics** — but a moderate σ keeps `policy_kl`
well-conditioned. After the fix, an unchanged 16-D policy reports `KL ≈ 8e-5`
(was `−3.99`), so the controller tracks the real active-dim KL.

**Why it's useful.** This is what actually restores stable training; the LR now
stays in its intended band instead of saturating.

**Tradeoffs.** Reported entropy moves from `≈ -57` to `≈ +22.7` (inactive dims now
contribute `+1.42` each instead of `-8.58`). Purely cosmetic — the entropy
*gradient* is still active-only (see next entry). If matching classic's `≈11.35`
matters, mask entropy/neglogp to active dims in the model forward.

---

### Symptom: inactive `log_std_param` drifts during training

**Observed / reasoning.** With the old formula `log_std = log_std_param − 10·(1−mask)`,
the entropy bonus (`−entropy_coef · entropy`) has `∂entropy/∂log_std_paramᵢ = 1`
for *every* dim, masked or not. So entropy maximization slowly pushed the inactive
`log_std_param` entries upward even though they never affect behavior.

**The fix.** The same `log_std = mask_dof * self.log_std_param` gates the gradient:
`∂log_std/∂log_std_paramᵢ = mask_dofᵢ`, so inactive entries receive zero gradient.
Verified in `test_ant_parity.py`: after `(-entropy).backward()`, active
`log_std_param.grad = 1.0`, inactive `= 0.0`.

**Tradeoffs.** None. Note this fix predated the σ=1 change and the two now share
one line; either way the gating property holds.

---

### Ruled out: position-embedding index mismatch

Before the real causes were nailed down, we suspected the 8-leg transformer's
position embeddings (active legs use `pos_emb[2,4,6,8]`) vs the 4-leg's
(`pos_emb[1,2,3,4]`) caused divergent training. `scripts/test_pos_emb_permutation.py`
permutes the rows and remaps `pos_ids`, producing **bit-identical** outputs — so
pos-emb indexing is pure relabeling and cannot cause a systematic gap. Recorded so
we don't re-investigate it.

---

## How the masking generalizes to multi-morphology (3- and 4-leg mixes)

All three masking changes are **per-env**, so a batch mixing a 3-leg morph
(6 active DOFs) and a 4-leg morph (8 active DOFs) works without modification:

- **Mask-passthrough** always restores raw `{0,1}`, so `(mask>0)` recovers each
  env's true active set. Still needed even with varied masks: any DOF active in
  *every* morph is constant again and would collapse under plain RMS.
- **Gradient-gating** becomes meaningful — `log_std_paramᵢ` gets gradient
  proportional to the fraction of envs whose morph uses DOF `i`; morphs without it
  don't pull on it.
- **σ=1 inactive** is *more* important here: with the old `-10`, each morph
  injected a different KL offset (4-leg → `-4.0`, 3-leg → `-5.0`), so the
  batch-mean KL was a morph-distribution-dependent mess. With σ=1 every inactive
  dim contributes `≈0` regardless of count.

**Separate (non-masking) concern:** one shared policy/value/LR spans dynamically
different bodies with different reward scales; `normalize_value` /
`normalize_advantage` pool across all of them, and the adaptive LR adapts to the
*mean* KL. If per-morph imbalance shows up, look at per-morph advantage
normalization or reward scaling — not the mask plumbing.

---

## Tests added

| Test | What it guards | How to run |
|---|---|---|
| `scripts/test_ant_parity.py` | classic ↔ adaptive`{2,4,6,8}` bit-parity over 10 steps (obs, mu, value, reward, term/trunc, all loss components) + mask preservation + log_std gradient-gating | `uv run python scripts/test_ant_parity.py` |
| `scripts/test_mask_norm.py` | mask survives input normalization (standard model collapses, masked model preserves raw `{0,1}` and still normalizes physical dims) | `uv run python scripts/test_mask_norm.py` |
| `scripts/test_pos_emb_permutation.py` | pos-emb index choice is pure relabeling (outputs identical) | `uv run python scripts/test_pos_emb_permutation.py` |

`test_ant_parity.py` runs each env in its own subprocess because vlearn's gym is a
process singleton; the classic phase dumps a trajectory, the adaptive phase syncs
and replays it.

## Files changed

- `envs/ant_envs/build_vsim.py` — reverse joint declaration order (DOF-order fix).
- `transformer_rl/models.py` — `TransformerMaskedNorm` model; `log_std = mask_dof * log_std_param` in both builders.
- `transformer_rl/train_utils.py` — register `transformer_masked_a2c_logstd`.
- `configs/ppo_ant_adaptive.yaml`, `configs/ppo_ant_dynamic_p1.yaml` — `model.name: transformer_masked_a2c_logstd`.

## Open questions

1. Should entropy/neglogp be masked to active dims so the readout matches classic
   (`≈11.35`)? Currently cosmetic-only; left as-is.
2. Multi-morph codesign: do we need per-morph advantage/reward normalization once
   morphologies with very different dynamics are mixed?
3. Any adaptive checkpoints trained before the `build_vsim` fix are on scrambled
   obs and should be discarded — confirm none are being resumed.
