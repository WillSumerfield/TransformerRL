# Reading the codesign metrics

How to read the TensorBoard metrics for single-network codesign (`CodesignAgent`,
`configs/ppo_ant_codesign_single.yaml`) and how to debug the algorithm from them.

Terms (limb, generator, GenAct/GenCrit, marginal value, R, resample window, …) are defined in
[`transformer_rl/CONTEXT.md`](../../transformer_rl/CONTEXT.md). The algorithm itself is in
[`temp/codesign_single_network_plan.md`](../../temp/codesign_single_network_plan.md). This doc is
only about *what the numbers mean and what to do when they look wrong*.

---

## 1. What happens each window (recap)

One control net and one morphology generator share a trunk. Training alternates:

- **Per step** (body fixed for the whole window): plain combined PPO on control
  (ContAct + V0.98). The generator heads get no gradient. → `control/*`, `rewards/*`, `losses/*`.
  Of these, **`control/r_step`** (mean raw reward per env-step over the epoch) is the per-epoch
  *performance* signal. Prefer it to `rewards/*`, which is rl_games' `game_rewards` ring buffer over
  the last 100 finished episodes — that series only moves when episodes end, and at
  `resample_interval: 1` every env truncates at once, at the instant of the resample.
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
| `control/grad_norm` | control update grad norm, **pre-clip** (see note below) | any magnitude OK *if losses fall*; large is common & benign | large **and** rising `c_loss`/`gencrit/loss_*` = real divergence |
| `control/adv_{mean,std}` | raw advantage scale, pre-normalization | std O(1), mean ~0 | std→0 = critic explains nothing / no learning signal |

> **Reading grad norms (`control/grad_norm`, `gen/grad_norm`).** Both log the **pre-clip** total
> norm (the value `clip_grad_norm_` returns), while the step is clipped to the `grad_norm` config
> (0.5–2.0). Once `‖g‖ ≫ clip` the update is just `clip · g/‖g‖` — **pure direction, magnitude pinned
> at the clip** — so a norm of 1500 and a norm of 15 take the *same-size* step. The metric measures
> raw signal magnitude, **not** how far the weights moved; a large value is not instability. It even
> *correlates with better `R`*: strong trials have large, informative gradients (clipping keeps them
> safe), while weak trials have small ones (dead GenCrit, collapsed policy — little signal). The
> resample step is a joint shared-trunk update (GenCrit MSE→`R` + control clone + GenAct) over few
> samples, so `gen/grad_norm` genuinely spikes at window boundaries. Two consequences: (1) when
> you're always clipped, the **clip is your real step-size**, not `learning_rate` (they're partly
> redundant knobs); (2) judge health by the **losses**, not the norm — large norm + *falling*
> `c_loss`/`gencrit/loss_*` + `value_rank_corr`>0 is fine; large norm + *rising* losses is a blowup.

### `build/` — the body the generator produces (per window)
| metric | meaning | healthy | bad → likely cause |
|---|---|---|---|
| `build/limbcount` | mean #limbs in the generator's **raw sample** (its intent) | climbs base→optimum, then steady | flat at base = generator not learning; →1 or →8 stuck = collapse |
| `build/limbcount_realized` | mean #limbs actually **built/run** (post-ramp); `R` is measured on these | ≈ `limbcount` once ramp off | persistent gap after RL onset = ramp logic bug |
| `build/limbcount_base` | mean #limbs of the base±flip **draws** (pretrain only, ~flat ≈3.2) | flat reference line | (reference only) |
| `build/limbcount_var` | variance of generated limb count across envs | > 0 | →0 = **mode collapse** (every env builds one body) |
| `build/n_distinct` | # distinct **typed** bodies sampled this window (RL only) | ≳ 5 | < 5 voids `gencrit/value_rank_corr`. **Not a collapse detector** — see `n_modes` |
| `build/n_modes` | **the diversity headline** (RL only): effective # distinct designs = Hill(q=1) over single-linkage `d_struct` clusters at τ=1, on the **subtype-collapsed** skeleton | > 1, ideally ≳ 2 | **1.0 = one design = full collapse**, by construction |
| `build/div_struct` | mean pairwise `d_struct` over the sampled skeletons (RL only) — threshold-free companion | > 0 | 0 = every draw identical. High here with `n_modes`≈1 = jitter around one design, not branching |
| `build/p/<slot>` | per-limb built on-rate | base limbs ~1; useful limbs climb | a useful slot stuck at 0 = generator won't add it |

### `gen/` — GenAct (generator actor) learning (per window)
| metric | meaning | healthy | bad → likely cause |
|---|---|---|---|
| `gen/actor_loss` | GenAct PPO loss (BC NLL during pretrain) | finite, no blowup | NaN/blowup = advantage or LR problem |
| `gen/entropy` | on/stop policy entropy | drops gradually toward 0 | →0 *fast* = premature collapse; pinned high = no learning |
| `gen/grad_norm` | generator update grad norm, **pre-clip** (see note below) | large/spiky is **expected** (joint resample step, few samples) | large **and** `gencrit/loss_*` not falling = unstable fit |
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
| `quality/Window_Rew_Mean` | mean reward per env-**step** over the window (scaled), every step counted — unfinished episode included | rises with `R_mean` | diverging from `R_mean` = the two weightings disagree (see below) |
| `quality/Window_Rew_Std` | spread of that per-step mean across bodies | as `R_std` | as `R_std` |
| `quality/by_limbcount/<k>` | mean `R` of bodies with exactly `<k>` limbs | monotone-ish in k (no limb cost ⇒ more limbs earn ≥) | non-monotone = controller can't yet exploit extra limbs |

**`R_mean` vs `Window_Rew_Mean`.** Both average over the same 4096 bodies; they differ in what they
average *within* one body. `R` is the mean over that env's **completed** episodes — equal weight per
episode, and the episode still running at the window boundary is thrown away. `W` is the env's
**total** reward over the window — equal weight per step, nothing discarded. The gap is not
cosmetic: episode length ramps across a window (measured 65 → 849 steps on a screen trial), so `R`
equal-weights a handful of short post-rebuild failures against one long good run, while `W` lets the
long run dominate in proportion to how long it lasted. `R` is the training target because GenCrit
regresses a per-episode quantity; `W` is a scoring/diagnostic metric only.

Both are invariant to `horizon_length`. `W` is divided by the window's accumulated step count
rather than left as a window total, because the window is `ceil(interval × max_episode_length ÷
horizon_length) × horizon_length` steps — 1000 / 1008 / 1024 at h = 8 / 16 / 32 — and a total would
hand h=32 a free 2.4%, which is disqualifying for a study that sweeps `h`. The divisor is counted in
`env_step`, not derived from the config, so it stays correct on a checkpoint resume that lands
mid-window.

Note the scales differ: `R` is a per-*episode* return (order 10 on the ant) while `W` is a per-*step*
mean (order 0.01). Neither is comparable to the other in absolute terms — only their trends are.

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
**Looks like:** `build/n_modes`→1 · `build/div_struct`→0 · `build/limbcount_var`→0 ·
`gen/entropy`→0 fast · `build/limbcount` pinned at 1 or 8 · `gencrit/value_rank_corr` goes NaN.
**Read `div_struct` for severity, `n_modes` for presence.** `n_modes` bottoms out at 1 for *any* tightly-clustered population, so it says *that* something collapsed, not *how far*: a 20× range of real spread maps into `[1.0, 3.0]`. It is also density-confounded (16× more samples from an identical distribution drops it 5×; `div_struct` moves 1%), so never compare it across populations of different size. `div_struct` is threshold-free and linear in structural spread — measured reference points: 1.74 / 0.92 / 0.37 for populations clustered ever tighter around one design, `0.0` only when every draw is identical. A run at `n_modes`≈6 with `div_struct`≈8 has **not** collapsed; it concentrated enough to percolate the 1-edit graph.
**Neither, rather than `n_distinct`:** `n_distinct` counts *typed* designs, and per the
`free_entropy` finding the skeleton commits while the subtype axis stays free — so subtype jitter
alone pins it near the sample size **through a total skeleton collapse** (measured: never below
286/4096 across an entire 20-trial study). `build/body_diversity` fails the same way for the same
reason. Only `n_modes` (collapsed skeleton) actually bottoms out at 1.
**Means:** the generator collapsed to one body; no exploration ⇒ no signal to improve.
**Fix:** raise generator `entropy_coef`; lower generator LR / `clip`; lengthen pretrain
(`n_pretrain`) so it doesn't commit early.

### 3. Control craters at resample
**Looks like:** `rewards/step` drops at window boundaries (saw-tooth) · `clone/actor_kl` and/or
`clone/critic_mse` spike at those boundaries. (A `control/grad_norm` spike alone is *not* a symptom —
it's clipped away; see the grad-norm note in §2. Trust the reward saw-tooth + clone spikes.)
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
