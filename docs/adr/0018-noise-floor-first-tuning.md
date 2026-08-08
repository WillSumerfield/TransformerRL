---
status: accepted
---

# Tuning measures its noise floor before it spends budget, and picks a winner by cluster centre

## Context & Decision

Seed-only variance at a fixed config was measured across five run families: CV **11 / 37 / 39 / 52 / 67 %** on `rewards/iter` (15 / 29 / 9 / 9 / 48 % on `quality/R_mean`). The last completed study's spread *across hyperparameters* was CV **33%** on `rewards/iter` and 14% on `R_mean`.

For both metrics the hyperparameter spread is **at or below the seed noise**. That study was ranking seeds as much as configs, and neither its winner nor its parameter importances mean what they appear to. Concretely, `phase3` seed 44 finished at 880 against seed 43's 4163 on *identical* hyperparameters.

The methodology therefore changes:

1. **Per-trial seeds are randomised**, drawn from a study-level RNG seeded by `sampler_seed` (so the study as a whole stays reproducible) and recorded as `trial.set_user_attr("seed", s)`.
2. **A calibration wave runs first**: the anchor config at one seed per slot, ~2 h, ~4% of a sweep. It measures the noise floor of *this exact study*, which every downstream number depends on — top-k size, trial count, whether a thin screen is affordable, whether pruning is safe, and whether memory creeps across ~32 rebuilds.
3. **The winner is the centre of the top-k cluster** — per-parameter median for continuous, mode for categorical — exported to `best_params.yaml`, with the raw argmax kept alongside in `best_params_argmax.yaml`.
4. **Per-parameter top-k spread is reported** as the fraction of the search range the cluster covers (IQR-based; modal share for categoricals), and flagged when wide.
5. **A confirmation wave closes the sweep**: the exported winner at one seed per slot, its band compared against the calibration anchor's.
6. **Pruning stays off** until a calibration wave shows mid-run rank predicts final rank.

Waves are ordinary trials, enqueued into the search study and tagged `wave`, not run in a side database.

## Considered Options

- **Keep one fixed seed for every trial (ADR-0009's premise).** Rejected: it makes the study reproducible while confounding the hyperparameter effect with a single draw. TPE then fits seed luck as if it were signal, and the fit is confidently wrong rather than noisily right.
- **Average k seeds per trial.** Rejected at this budget: it multiplies the cost of every observation by k, so a fixed budget buys 1/k as many distinct configs — the opposite of what a search below its noise floor needs. Randomised seeds plus a top-k cluster centre de-noises at the aggregate level for the same spend, and the calibration wave measures the floor directly rather than paying for it on every trial.
- **Switch the objective to `quality/R_mean` because it is 2.6× quieter.** Rejected: it also discriminates proportionally less (14% vs 33% across hyperparameters). The *ratio* is what matters and it is unchanged, so this buys nothing.
- **Keep `study.best_trial` as the winner.** Rejected: argmax selects the luckiest draw, and that bias grows as the sampler concentrates — late trials are drawn from a tighter region, so the max is increasingly a seed outcome. It is still reported, because a large centre-vs-argmax gap is itself a diagnostic.
- **Run the calibration wave in a separate study database.** Rejected: keeping the runs in the search study means they are scored by exactly the same path as every other trial and need no special case in resume or reporting. The cost is accepted deliberately — N observations stacked on one coordinate skew the sampler's density model, which is why a wave is an explicit `--calibrate N` and not automatic.
- **Enable `MedianPruner` now to buy throughput.** Rejected until measured: under SNR < 1 it preferentially kills *unlucky seeds* rather than bad configs, and it thins the trial density that a top-k winner depends on, since pruned trials yield no final score.

## Consequences

- **Supersedes the single-seed premise of ADR-0009.** Its recovered-level objective stands; its rejection of cycle-indexed reporting is **stale** — that reasoning assumed a 500-epoch proxy spanning ~1.6 resample cycles, whereas the current config is 2000 epochs at `resample_interval: 1`, i.e. ~32 windows, where window-indexed reporting has ~32 checkpoints and is phase-aligned by construction. If pruning is ever re-enabled, `interval_steps: 25` against 62-epoch windows also lands checks at arbitrary phase within a window and must be aligned to window boundaries.
- **A calibration wave consumes `n_trials` budget**, since its runs are trials in the study. `n_trials` must be raised by the wave size, or the sweep is short by that many.
- **A winner is not final until a confirmation wave has run it.** The sweep's own top score is an in-sample maximum over a noisy objective, so it is biased upward by construction; only a fresh set of seeds tests it.
- **An 8-sample wave fixes the noise floor's order of magnitude, not its value.** SE(CV) ≈ CV/√(2(n−1)) is ~27% of the CV at n = 8, so a measured 39% is pinned to roughly ±10 points. That is enough to decide screen-vs-dense; it is not enough to quote.
- A wide top-k spread on a parameter means it is **undetermined**, not that the centre is optimal — the reported centre is then a midpoint of the search range.
- The coordinate-wise centre is only meaningful over **one** blob. If the top-k is bimodal in a parameter, the centre lands between two clusters and assembles a config no trial ever ran; the spread diagnostics exist to make that visible, and the confirmation wave to catch it.
- OOM trials are scored `0.0` and tagged, so they steer the sampler but are excluded from the winner, the top-k and the plots — they are not measurements.
