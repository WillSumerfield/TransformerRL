---
status: accepted
---

# Cross-method comparison runs on one interface, one frame budget, and one method-agnostic metric

## Context & Decision

[ADR-0021](0021-paper-experiment-metric-protocol.md) fixes the measurement contract for experiments
1–3, all of which are one algorithm ablated against itself. [Experiment
5](../experiments/baselines.md) is not: it compares our method against other codesign algorithms,
several of which have no generator, no value head over designs, and no window axis. Four of ADR-0021's
five metrics are undefined for them, and its x-axis does not exist. This ADR is that experiment's
protocol.

Three decisions, fixed before the rival implementations land so that each one has a target to land
against rather than defining its own.

**1. Every condition is a `codesigner` `Algorithm`, and every task a `codesigner` `Task`.** Rivals are
not ported into this repo and our method is not ported out. All implement `run()`, `is_finished()`,
`control_policy()` and `morphology_generator()` against the same `Task` and `ModuleLibrary`, so
simulator, grammar, reward and observation layout are identical *by construction* rather than by
matching. `codesigner` is the single source of truth for both axes. The control-only reference
condition needs no special mechanism — the interface's D18 already states that a fixed generator is how
a control-only baseline is expressed.

The task axis spans **multiple sources**, deliberately including the benchmarks the rivals introduced
(`components/tasks/bodygen/`, ported from BodyGen, ICLR 2025) and a manipulation family
(`components/tasks/adroit/`, `grasp.py`) that our grammar and hyperparameters were not developed on. A
method compared to rivals only on the task it was tuned for measures task fit, not method quality.

**2. The budget is equal environment frames, matched by configuration.** The interface exposes no
frame count: `Progress` carries `step`, `reward`, `best_reward`, `wall_time` and `morphologies`, and
each algorithm's `is_finished()` counts its own native unit — `evolutionary.py:179` counts generations,
ours counts resample windows. Frames per unit is nevertheless deterministic in this stack (fixed
environment count, fixed episode length, fixed horizon), so each algorithm's native stopping count is
derived arithmetically from one shared frame budget `F`. **Every `Algorithm` entering this comparison
must document its frames-per-unit**; without it, its budget cannot be matched and it cannot be in the
grid.

`Progress.wall_time` is recorded and published alongside. Equal frames is the right rule for a
sample-efficiency claim, but it advantages a method whose search rides on rollouts the controller
already collects over one that pays for separate fitness evaluations, and that asymmetry belongs in
the paper rather than buried in the budget.

**3. The decision metric is specialized return, normalized per task by the fixed-body condition.**
Specialized return is the one ADR-0021 metric that transfers: it needs nothing but a committed body —
strip the scaffolding, warm-start control, fine-tune on that body alone, measure — and every method in
the grid outputs one. Sharing a decision metric with the ablations is what lets the paper's sections be
read against each other.

Raw returns across ten tasks from three sources have unrelated scales, so each cell is divided by the
task's fixed-body score. Every entry then reads as *how much better than this task's reference
morphology*, which is what a codesign paper claims. Aggregate with the per-task table and a **median**
across tasks.

## Alternatives considered

- **Citing rivals' published numbers.** Rejected: different simulators, bodies, rewards and action
  spaces. Placing those numbers in a table beside ours would not be a comparison, and the `codesigner`
  interface exists precisely so it does not have to be one. Published results still appear in prose, as
  context for a rival's relative standing — see the reporting rule in Consequences.
- **Porting our method onto a rival's benchmark stack.** Rejected once the BodyGen *tasks* were ported
  into `codesigner`: the benchmark is obtainable without leaving the interface, so the comparison keeps
  one simulator, and every rival gets its own benchmark represented without a second stack to maintain.
- **Equal wall-clock.** Rejected as the decision axis: it is not the standard for a sample-efficiency
  claim, and it favours whichever method the current implementation happens to have optimized. Reported,
  not decided on.
- **Mean of per-task ratios.** Rejected: one task with a weak fixed-body denominator dominates the
  aggregate. Median across tasks, with the full table always shown.
- **Normalizing by the best algorithm per task.** Rejected: the reference then moves with the result, so
  a task where no method beats the reference body reads the same as one where every method does. The
  fixed-body denominator makes "morphology search did not pay for itself here" visible as a ratio at or
  below 1.0.
- **Dropping tasks where morphology search does not help.** Rejected explicitly. Those tasks are the
  experiment's most informative cells and the ones a reader most needs.
- **Reusing ADR-0021's metric 1 as a cross-method curve.** Rejected: "return on its own bodies" means
  different things for a learned generator, a GA population and a fixed body, so the curves do not share
  an axis even when they share units.

## Consequences

- **This ADR does not govern [experiment 4](../experiments/attention.md).** That experiment is also
  off ADR-0021 — `resample_interval: 0` on one fixed morphology, so it has no generator either — but it
  is a single-algorithm study with its own five measurements, defined in its own doc. Two experiments
  sit outside ADR-0021 for opposite reasons and they do not share a protocol.
- **Exploration partially transfers, for free.** `Progress.morphologies` is the interface's
  population report, so breadth and travel are computable for any algorithm holding a population — no
  per-algorithm instrumentation and no `gen_pop/*.npz`. Undefined for the fixed-body condition and for a
  deterministic search, which report `NaN` rather than zero.
- **`specialize.py` must be algorithm-agnostic.** It needs only
  `morphology_generator().generate(1, deterministic=True)` and the task; any branching on which
  algorithm produced the checkpoint is a bug, and the interface guarantees it is unnecessary.
- **`scrape.py` reads `Progress` records for this series, not TensorBoard.** Rivals write none of
  `quality/*`, `build/*`, `gencrit/*` or `clone/*`, and the tolerance belongs in the loader.
- **The frame budget must be asserted, not assumed.** It is set by arithmetic that nothing checks at
  runtime, so the stored per-cell budget is compared across the algorithm axis before any figure is
  drawn. A silently mismatched budget invalidates the whole experiment and is its most likely failure
  mode.
- **Tuning budget is part of the protocol, not an implementation detail.** Our hyperparameters were
  screened on `ant` (ADR-0020). Each rival gets an equal, stated per-task-family tuning budget, or the
  grid measures tuning effort. The amount is undecided and blocks the first real wave.
- **A rival underperforming its published relative standing is reported as such.** The discrepancy is
  stated next to the number rather than presented as a clean loss. A weak port is worse than no port,
  and this is the rule that keeps it from being worse than nothing.
- **The grid is 6 algorithms × 10 tasks × 8 seeds = 480 cells**, plus a specialization pass each —
  roughly a week of wall-clock on 8 slots for training alone. Tiering into a headline suite and an
  appendix is expected; the levers and the undecided choice are in
  [the build plan](../../temp/experiment_plan_baselines.md).
