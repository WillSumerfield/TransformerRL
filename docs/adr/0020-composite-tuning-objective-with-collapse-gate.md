---
status: accepted
---

# The tuning objective is window-indexed body quality, gated on a saturating diversity floor

## Context & Decision

Tuning `ppo_codesign_single` on the ant is a first broad pass — **no hyperparameter in that config has been jointly tuned**, and the staged `3a`/`3b`/`3c` winners baked into it came from greedy narrow sweeps that could not see interactions. Sweeping that space needs an objective, and the previous one (`rewards/iter`, recovered level) has two problems here.

First, `rewards/iter` is **epoch-indexed**, so a fixed-length tail lands at an arbitrary phase within a resample window once `horizon_length` or `resample_interval` varies (window length in epochs is `max_episode_length ÷ horizon_length`). That forbids sweeping the timing knobs, which this pass wants. `quality/R_mean` is logged once per window and is phase-aligned by construction.

Second, neither metric penalises generator collapse, and `R_mean` **structurally rewards it**: `R_i` is the mean return of env *i* over the window and `R_mean` averages over the sampled body population, so a more diverse population necessarily contains more bad bodies and scores lower. Maximising `R_mean` alone is partly an instruction to stop exploring.

The objective is therefore:

```
score = mean(quality/R_mean, last 5 windows) × min(1, mean(build/n_modes, whole series) / D_floor)
```

Four choices inside that, each load-bearing:

- **A saturating floor, not a product.** `reward × D^alpha` requires a trade rate between reward and diversity that nothing justifies. A floor asserts only "below `D_floor` is degenerate"; above it, ranking is pure reward. It also defuses the tautology that a diversity term tuned alongside `generator.entropy_coef` would otherwise create — above the floor, extra entropy buys exactly zero score and still costs reward.
- **The gate spans the whole RL phase, not the reward tail.** A generator that explored and then committed to the best body it found is succeeding, and its late-run diversity is low for the right reason. The pathology is *never having searched*, which is a whole-phase question. The two halves answer different questions, so their spans deliberately do not align.
- **`build/n_modes`, not `build/n_distinct`.** `n_distinct` counts exact **typed** designs, and per the `free_entropy` finding the skeleton commits while the subtype axis stays free — so subtype jitter alone keeps it near the sample size under total skeleton collapse. Measured across 20 trials × 23 RL windows of the FD/FK study its global minimum was 286/4096 and its per-trial means 2841–4006: it never moved, because it cannot. `n_modes` is `Metrics.md`'s diversity headline — a Hill number over single-linkage `d_struct` clusters at τ=1 on the **subtype-collapsed** skeleton — where `1.0` is full collapse by definition, so the floor is interpretable without calibration. It is added to `_log_diversity` and logged **RL-only**, mirroring `n_distinct`, so "whole series" means "all RL windows" with no tuner-side arithmetic even when `n_pretrain` is swept.
- **`learning_rate` is kept as a swept parameter despite being near-inert.** `_WarmupThenAdaptiveScheduler` uses it as the ramp peak for `warmup_epochs` (64) epochs and then hands to `AdaptiveScheduler(kl_threshold)`, a ×1.5/÷1.5 walk that erases a 15× range in ~7 steps. `kl_threshold` is the real controller. It is swept anyway, deliberately, on the view that the first window may set the basin.

## Considered Options

- **Keep `rewards/iter` + `recovered` and freeze the timing knobs.** Rejected: it has ~2.4× the across-config discrimination of `R_mean`, which matters near a noise floor, but it costs `horizon_length` and `resample_interval` — and this pass is the one that should look at them. `final_frac` does not rescue it: `tune.py:136-139` subtracts `n_pretrain` (a **window** count) from `n` (an **epoch** count), a cap only meaningful for a window-indexed series.
- **Score `R_mean` alone and inspect diversity by hand afterwards.** Rejected: it leaves the sampler free to find a degenerate corner and only reveals it after the budget is spent. The floor costs nothing when nothing goes wrong.
- **Gate on `build/body_diversity` (`N_body_skel`).** Rejected on the doc's own warning — it inflates when independent limbs flip without breaking the common core, which is exactly what `n_modes` fixes. It is also computed from the generator's own per-step entropies, making it the most direct restatement of `entropy_coef`.
- **Gate on `build/mean_limb_diversity`.** Rejected: empty slots contribute `exp H = 1`, so configs emitting more limbs score mechanically higher and the gate becomes partly a limb-count gate.
- **Multi-objective Pareto (NSGA-II) over reward and diversity.** Rejected as disproportionate: `direction=`, `best_trial`, the top-k centre and the `best_params` export in `tune.py` all assume one scalar.
- **A hard gate (`reward` if `D ≥ D_floor` else `0.0`).** Rejected: the cliff gives TPE no gradient near the boundary and scores a trial that missed by 1% identically to full collapse.

## Consequences

- **Studies scored this way are not comparable to any earlier study.** The FD/FK study's `best_params.yaml` was already never adopted and never confirmed; it is superseded rather than compared against.
- **The gate is expected to be inert.** That is the intent — it is insurance for the low-`generator.entropy_coef` and low-`n_pretrain` corners this sweep visits for the first time, not a second objective. If it fires often, `D_floor` is wrong, not the configs.
- `D_floor` is set **by eye after the calibration wave**, which is a decision made under results. Accepted because `n_modes` has an absolute meaning (`1.0` = collapse) that bounds how far the choice can drift.
- **`_log_diversity` gains an O(M²) pairwise step per window.** `_dedup` runs first and subtype collapse is what makes it bite, so the cost is seconds against a 1.5–3 h trial — but it is CPU work in the training loop, and ADR-0017 leaves per-trial CPU unbounded across 8 concurrent slots.
- `quality/R_mean` carries `_r_scale`, so its absolute values (~30–70) are on a different scale from `rewards/iter` (~3500). Ranking is unaffected; quoted numbers are not interchangeable.
- The across-config CV of `R_mean` tail-5 in the FD/FK study was **17%**, inside ADR-0018's measured seed-noise band for `R_mean` (9–48%). That study varied only five aux parameters so a narrow spread is expected, but the 16-parameter sweep must still be read against its own calibration wave before its winner means anything.
