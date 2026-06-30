# Reading the codesign metrics

How to read the TensorBoard metrics for single-network codesign (`CodesignAgent`,
`configs/ppo_ant_codesign_single.yaml`) and how to debug the algorithm from them.

Terms (limb, generator, GenAct/GenCrit, marginal value, R, resample window, …) are defined in
[`transformer_rl/CONTEXT.md`](../transformer_rl/CONTEXT.md). The algorithm itself is in
[`temp/codesign_single_network_plan.md`](../temp/codesign_single_network_plan.md). This doc is
only about *what the numbers mean and what to do when they look wrong*.

---

## 1. What happens each window (recap)

One control net and one morphology generator share a trunk. Training alternates:

- **Per step** (body fixed for the whole window): plain combined PPO on control
  (ContAct + V0.98). The generator heads get no gradient. → `control/*`, `rewards/*`, `losses/*`.
- **At each resample** (window boundary, `resample_interval` episodes): `_resample_update` runs
  once, then a new body is sampled and the gym is rebuilt. In one update it:
  1. fits **GenCrit/V1.0** to body quality `R` (on rollout states *and* designed prefixes),
  2. updates **GenAct** — PPO on the marginal-Shapley advantage, or BC toward the built body
     during **pretrain**,
  3. **clones** control (β·KL[ContAct] + λ·MSE[ContCrit]) so the shared-trunk step doesn't drift
     the controller.

  → `build/*`, `gen/*`, `gencrit/*`, `quality/*`, `clone/*` (logged **once per window**, so these
  curves are sparse compared to the per-epoch control metrics).

**Pretrain → RL handoff.** The first `n_pretrain` windows warm the generator up around the base
morph (`gen/fraction` ramps 0→1; a fraction of envs are base±flip *draws* instead of generator
samples). Once `gen/fraction == 1` the generator drives every body and the marginal-value RL signal
(`gen/marg/*`) turns on.

**`R` (body quality).** The generator's reward: the body's true mean completed-episode return over
the window (γ=1), scaled by the reward shaper (`scale_value`, currently 0.01) so it sits in the
same O(1) units the control critic fits. Every `quality/*` and `gencrit/*` number is in these units.

**Half-step note.** `build/*` describe the body just sampled for the **next** window; the learning
and `quality/*` metrics describe the window that just **ended** (aligned with its `R`). They are one
window out of phase — expected, not a bug.

---

## 2. Metric reference

Subsystem-keyed. `<slot>` ∈ {F, FR, R, BR, B, BL, L, FL} (limb compass slots); `<k>` = a limb count.

### `control/` — controller training health (every epoch, all bodies)
| metric | meaning | healthy | bad → likely cause |
|---|---|---|---|
| `control/sigma_{mean,min,max}` | action std = exp(log_std) | mean ~0.3–1.0, slowly falling | `sigma_min`→0 = log_std collapse (dead exploration) |
| `control/action_sat` | frac of active mean-action dims pinned at the tanh rail (\|μ\|>0.99) | < ~0.3 | →1 = saturated policy, often with collapsed sigma |
| `control/grad_norm` | control update grad norm, pre-clip | stable, O(1–10) | spikes/blowup = LR or clone too aggressive |
| `control/adv_{mean,std}` | raw advantage scale, pre-normalization | std O(1), mean ~0 | std→0 = critic explains nothing / no learning signal |

### `build/` — the body the generator produces (per window)
| metric | meaning | healthy | bad → likely cause |
|---|---|---|---|
| `build/limbcount` | mean #limbs in the generator's **raw sample** (its intent) | climbs base→optimum, then steady | flat at base = generator not learning; →1 or →8 stuck = collapse |
| `build/limbcount_realized` | mean #limbs actually **built/run** (post-ramp); `R` is measured on these | ≈ `limbcount` once ramp off | persistent gap after RL onset = ramp logic bug |
| `build/limbcount_base` | mean #limbs of the base±flip **draws** (pretrain only, ~flat ≈3.2) | flat reference line | (reference only) |
| `build/limbcount_var` | variance of generated limb count across envs | > 0 | →0 = **mode collapse** (every env builds one body) |
| `build/n_distinct` | # distinct bodies sampled this window (RL only) | ≳ 5 | < 5 = low diversity; also voids `gencrit/value_rank_corr` |
| `build/p/<slot>` | per-limb built on-rate | base limbs ~1; useful limbs climb | a useful slot stuck at 0 = generator won't add it |

### `gen/` — GenAct (generator actor) learning (per window)
| metric | meaning | healthy | bad → likely cause |
|---|---|---|---|
| `gen/actor_loss` | GenAct PPO loss (BC NLL during pretrain) | finite, no blowup | NaN/blowup = advantage or LR problem |
| `gen/entropy` | on/stop policy entropy | drops gradually toward 0 | →0 *fast* = premature collapse; pinned high = no learning |
| `gen/grad_norm` | generator update grad norm, pre-clip | stable | spikes = unstable generator update |
| `gen/fraction` | ramp progress (pretrain→RL) | 0→1 over `n_pretrain`, then 1 | (schedule) |
| `gen/marg/<slot>` | **marginal value** v(prefix+limb)−v(prefix) for that limb (RL only) | useful limbs **>0**, detrimental **<0**, consistent across windows | all ≈0 = dead GenCrit; sign-flipping = noisy value |

### `gencrit/` — GenCrit/V1.0 value-head fit + calibration (per window)
| metric | meaning | healthy | bad → likely cause |
|---|---|---|---|
| `gencrit/loss_prefix` | designed-prefix value fit, **scale-free** MSE/Var(R) (= frac. variance unexplained) | falls well below 1 | ~1 or rising = GenCrit not fitting the prefixes |
| `gencrit/loss_rollout` | rollout-state value fit, MSE/Var(R) | falls below 1 | ~1 = GenCrit not fitting live states |
| `gencrit/value_rank_corr` | Spearman of v(full) vs per-body **mean** `R`, over distinct bodies (denoised, diversity-robust) | → +1 | ≤0 = value mis-ranks bodies; **NaN** = <5 bodies or constant v |
| `gencrit/value_ev` | denoised per-body explained variance of v vs `R` | → +1 | ≈0 = value ≈ constant (dead head); <0 = worse than mean |

### `quality/` — body-quality outcome (the optimization target, per window)
| metric | meaning | healthy | bad → likely cause |
|---|---|---|---|
| `quality/R_mean` | mean body return `R` (scaled) over the window | rises as bodies improve | flat/falling = bodies not improving |
| `quality/R_std` | spread of `R` across bodies | > 0 while exploring; shrinks as generator converges | →0 early = no body diversity to learn from |
| `quality/by_limbcount/<k>` | mean `R` of bodies with exactly `<k>` limbs | monotone-ish in k (no limb cost ⇒ more limbs earn ≥) | non-monotone = controller can't yet exploit extra limbs |

### `clone/` — control preservation at resample (per window)
| metric | meaning | healthy | bad → likely cause |
|---|---|---|---|
| `clone/actor_kl` | KL[ContAct_old ‖ ContAct] after the generator update | small (~0.01–0.3) | large = shared-trunk update is drifting control; raise β |
| `clone/critic_mse` | MSE(ContCrit, ContCrit_old) | small | large = V0.98 drifting; raise λ |

### Untouched rl_games defaults
`rewards/*` (mean reward — overall control skill across all bodies), `episode_lengths/*`,
`losses/{a_loss,c_loss,entropy,bounds_loss}`, `info/*` (lr, kl, e_clip), `diagnostics/*`.

---

## 3. Debugging playbook

Symptom → metrics that confirm it → cause/fix. Work top-down: a dead GenCrit (1) makes everything
below it meaningless, so rule it out first.

### 1. Dead GenCrit / value head
**Looks like:** `gen/marg/*` all ≈0 or noise · `gencrit/value_rank_corr` NaN or ≤0 ·
`gencrit/value_ev` ≈0 · `gencrit/loss_prefix` not falling.
**Means:** the body-quality value is (near-)constant across bodies, so marginal advantages are
noise and the generator gets no usable gradient. (This was the original bool-dtype design-mode bug:
the design pass was body-invariant.)
**Check/fix:** confirm v(full) actually varies across bodies (it should track `quality/R_std`).
If `loss_prefix` is stuck near 1, raise `gencrit_coef` or generator `epochs`; if it fits
(`loss_prefix`→0) but `value_ev`≈0, the design pass isn't seeing the morphology — suspect the
encode/mode path, not the optimizer.

### 2. Mode / diversity collapse
**Looks like:** `build/limbcount_var`→0 · `build/n_distinct`<5 · `gen/entropy`→0 fast ·
`build/limbcount` pinned at 1 or 8 · `gencrit/value_rank_corr` goes NaN (too few bodies).
**Means:** the generator collapsed to one body; no exploration ⇒ no signal to improve.
**Fix:** raise generator `entropy_coef`; lower generator LR / `clip`; lengthen pretrain
(`n_pretrain`) so it doesn't commit early.

### 3. Control craters at resample
**Looks like:** `rewards/step` drops at window boundaries (saw-tooth) · `clone/actor_kl` and/or
`clone/critic_mse` spike at those boundaries · sometimes `control/grad_norm` spikes.
**Means:** the shared-trunk generator update is dragging the controller off its policy.
**Fix:** raise β (`beta`, actor KL clone) and/or λ (`lam`, critic MSE clone); reduce generator
`epochs`/LR so the joint step perturbs the trunk less.

### 4. Noisy / sign-flipping marginals
**Looks like:** `gen/marg/<slot>` flips sign window-to-window · `gencrit/value_rank_corr` jittery ·
`quality/R_std` large · `gencrit/loss_rollout` high.
**Means:** `R` is too noisy per window (few completed episodes) or GenCrit underfits, so marginals
are unreliable even though the head isn't dead.
**Fix:** raise `resample_interval` (more episodes per window → lower-variance `R`); raise
`gencrit_coef`/`epochs` for a tighter fit.

### 5. Generator stuck at base
**Looks like:** `build/limbcount` flat at ~base after `gen/fraction`=1 · `gen/marg/*`≈0 (but
GenCrit is alive per §1) · `gen/entropy` pinned high.
**Means:** advantages are real but tiny — body-quality differences are too small to move the policy,
or entropy regularization dominates.
**Check/fix:** confirm `quality/by_limbcount/<k>` actually rewards more limbs; if differences are
genuinely small, that's the landscape. Otherwise lower generator `entropy_coef`, or check `R`
scaling (§6).

### 6. R-scale wrong
**Looks like:** `quality/R_mean` huge or tiny vs control's `losses/c_loss` scale · `gencrit/loss_*`
behaving oddly (note these are already Var(R)-normalized, so a *scale* error shows up in the raw
`quality/R_*` magnitudes and in marginal sizes, not in the normalized loss).
**Means:** `R` isn't in the control critic's units, so the value fit and marginals are mis-scaled
relative to the clone terms.
**Fix:** the agent scales `R` by `reward_shaper.scale_value`; make sure that matches the control
reward shaping. `quality/R_mean` should land in roughly the same O(1) range as the per-step control
returns.

### Quick "is it healthy?" checklist
- `gencrit/value_rank_corr` > 0 and not NaN (value ranks bodies correctly, enough diversity)
- `gen/marg/*` for known-good limbs > 0 and stable in sign
- `build/limbcount` moving toward the optimum, `build/limbcount_var` > 0 (still exploring)
- `clone/*` small (control held), `rewards/step` not saw-toothing at resamples
- `quality/by_limbcount` rewards the direction the generator is moving
