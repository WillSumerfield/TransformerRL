---
status: accepted
---

# Optuna tuning prunes on a TB-polled EMA and scores on recovered-level reward

## Context & Decision

`scripts/tune.py` ran each trial as a black box — launch the trainer subprocess, wait, read the final **max-over-training** `morph_reward/mean` from TensorBoard. To prune sub-average trials early we add mid-run reporting **without instrumenting the trainer**: a poller thread in `tune.py` re-reads the growing TB event file (~15–30 s), reports an **EMA (α≈0.3) of current `morph_reward/mean`** at each integer **epoch** to the Optuna trial, and `MedianPruner(n_startup_trials=4, n_warmup_steps=200, interval_steps=25)` kills trials below the same-epoch median; on `should_prune()` the poller terminates the subprocess and raises `TrialPruned`. We also **change the objective** for completed trials to the **mean of `morph_reward/mean` over the final ~50 epochs** ("recovered-level"). Both behaviours live behind a `pruner.enabled` flag in the tune config and are **disabled whenever `resample_interval` is a tuned parameter** (the full-ant sweep).

## Considered Options

- **Instrument the trainer to call `trial.report` directly** (pass study URL + trial number into the subprocess). Rejected: couples the trainer to Optuna and needs an rl_games epoch hook. TB-polling leaves the trainer untouched and the event file already exists; the ~15–30 s detection lag is negligible.
- **Keep max-over-training as the objective.** Rejected: it rewards a transient pre-resample peak. The deployment run resamples morphologies repeatedly (~4× at 1500 epochs; the 500-epoch proxy sees one, ~epoch 313), so a spike-then-collapse hyperparameter set scores well on a max but transfers badly. Recovered-level rewards stable post-resample reward, which is what transfers; the prune EMA is collapse-sensitive and coherent with it.
- **Epoch-indexed vs cycle-indexed reporting.** Epoch-indexed is only fair when every trial dips at the same epoch — true only when `resample_interval`, `horizon_length`, `max_episode_length` are all fixed (the PPG sweep). A cycle-indexed signal (compare per-resample-cycle peaks) would generalise to a tuned `resample_interval`, but PPG's proxy spans only ~1.6 cycles, so cycle-indexing would yield a single prune checkpoint at ~epoch 313 and forfeit the cheap pre-resample kills. Rejected in favour of the simpler `enabled` toggle: keep epoch-indexing where it is valid (fixed timing) and turn pruning off where it is not (tuned timing).
- **Aggressive pruning (ASHA / low warmup).** Rejected for a ~30-trial, single-seed, collapse-prone setting; conservative MedianPruner with a 200-epoch warmup avoids cutting slow-but-stable winners. Pruning is a wall-clock saver, not a smarter sampler.

## Consequences

- "Best params" is now defined by recovered-level mean, not peak — re-running an old study under the new objective can pick a different winner; historic `best_params.yaml` from prior (max-objective) studies are not comparable.
- Pruning is **common-mode-safe only because resample timing is fixed across trials.** This is enforced operationally: the `enabled` flag must be off for any sweep that tunes `resample_interval` (or `horizon_length` / `max_episode_length`). The recovered-level objective is phase-confounded under tuned timing for the same reason, so a cycle-length sweep also falls back to the original max objective.
- Pruned trials are `PRUNED` (last reported EMA as value), not `failed`; the existing crash→retry path does not fire on a prune. TPE still consumes pruned trials' last value, so the budget is not wasted.
- α≈0.3 ⇒ EMA half-life ~2 epochs, effective window ~6 epochs: it damps single-epoch noise but tracks real trends (including the resample dip, which cancels in the same-epoch median). It deliberately does not smooth over the ~100-epoch recovery ramp — doing so would lag genuine collapse.
- No mid-run hyperparameter adaptation (e.g. raising critic-lr from critic-loss) — that is Population-Based Training, explicitly out of scope; TPE still only correlates input HPs → final score.
