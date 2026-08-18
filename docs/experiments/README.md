# Paper experiments

The series. Each experiment asks one question about the codesign algorithm. Experiments **1–3** ask it
by switching one thing off and measuring the same five things — the protocol is fixed in
[ADR-0021](../adr/0021-paper-experiment-metric-protocol.md), formulas in
[Metrics.md](../reference/Metrics.md), terms in [experiments/CONTEXT.md](../../experiments/CONTEXT.md).
Experiments 4 and 5 have no generator to measure and carry their own measurements; see below.

Shared across **1–4**: **8 seeds** (42–49) per condition, artifacts gitignored, curves read against
the study's noise floor per [ADR-0018](../adr/0018-noise-floor-first-tuning.md), and runs going out on
the 8 MIG slots in **mixed waves** — every wave carries a spread of conditions, so no condition is
confounded with time or machine state over a multi-hour session. Experiments 1–3 additionally share
the budget **48 windows** (8 pretrain, 40 RL) — 3024 epochs at the shipped 63 epochs/window, derived
rather than rounded, so no window is left unclosed and unlogged; experiment 4 runs 3000 epochs with
no windows at all, and experiment 5 is budgeted in environment frames per
[ADR-0022](../adr/0022-cross-method-comparison-protocol.md).

| # | Experiment | Question | Status |
|---|---|---|---|
| 1 | [Shared backbone](backbone.md) | Does a shared trunk transfer representation in a way a distillation channel cannot? | **specified** |
| 2 | [Auxiliary prediction](aux.md) | Does predicting forward dynamics and kinematics build a representation that makes control transfer between bodies? | **specified** |
| 3 | [Control clone](clone.md) | Do the clone KL/MSE terms preserve control across a resample, or anchor it to bodies that no longer exist? | **specified** |
| 4 | [Control attention](attention.md) | Is cross-token information actually used by a per-token control policy? | **specified** — off-protocol |
| 5 | [Outside baselines](baselines.md) | Does the method beat other codesign algorithms, on tasks it was not built for? | **specified** — [ADR-0022](../adr/0022-cross-method-comparison-protocol.md), blocked on rival ports |

Two experiments sit outside ADR-0021, for opposite reasons. **Experiment 4** deliberately has no
generator — one fixed body, no resampling — so metrics 3–5 are undefined and metric 1's
`quality/R_mean` is never written; it defines its own five measurements in its doc. **Experiment 5**'s
*rivals* have no generator, and it has no window axis; its protocol is
[ADR-0022](../adr/0022-cross-method-comparison-protocol.md). The two do **not** share a protocol.

Experiment 5 runs entirely through the sibling **`codesigner`** package: every condition is an
`Algorithm` and every task a `Task`, so all six algorithms search the same `ModuleLibrary` in the same
simulator with the same reward. Its task axis spans three sources — ours (`ant`, `grasp`), BodyGen's
four planar locomotion tasks, and four Adroit manipulation tasks. All ten tasks exist; three of the six
algorithms (BodyGen, NGE, RoboGrammar) do not yet, which is what blocks it.

## Run order

**2, 3, 4 in parallel → Phase5 port → 1**, then 5 when the rival algorithms land.

Experiments 2, 3 and 4 all run on `HEAD` with no new arms — 2 and 3 are pure `--set` overrides, 4 adds
only an attention mask — so they run first and shake the harness out on cheap runs. Experiment 3 must
precede experiment 1's real waves because it decides how experiment 1's `single` arm is configured:
[backbone.md](backbone.md) defines that arm as "the shared backbone *and its mitigation*", and if
experiment 3 retires the clone, experiment 1 would otherwise be testing a component already removed
from the algorithm.

Experiment 4's runs are the cheapest per wave in the series — no generator update, no 16-epoch
resample loop, no aux heads, and no post-training ladder or specialization passes.

**Naming:** two experiments are named for their mechanism rather than the paper's shorthand, because
the shorthand names things this codebase does not have.

- Experiment 2 is *auxiliary prediction*, not *JEPA*. `config.jepa` is the masked-token I-JEPA head,
  which is disabled and not under test; what is ablated is FD (whose `latent` variant is JEPA-*like*)
  plus FK (which is not JEPA at all). See
  [the head glossary](../../transformer_rl/CONTEXT.md#auxiliary-prediction-heads).
- Experiment 3 is the *control clone*, not *PPG*. Codesign control is combined PPO on the shared
  trunk ([ADR-0013](../adr/0013-codesign-single-network-merged-gencrit.md) superseding
  [ADR-0008](../adr/0008-codesign-via-ppg-three-nets.md)); the PPG shape survives only as the
  resample update's aux phase over stored rollout states. What is ablated is `generator.beta` and
  `generator.lam`.

## The five metrics, in one line each

These are experiments 1–3's. Experiment 4 defines its own five (return curve, asymptotic return,
sample efficiency, gait diagnostics, attention structure); experiment 5 keeps only metric 2, which is
the one that needs nothing but a committed body.

1. **Return curve** — `quality/R_mean` per window. Joint body×control; the training-dynamics curve.
2. **Specialized return** — committed body, scaffolding stripped, control fine-tuned 250 epochs on
   it alone. What the design is worth. **The default decision metric** — except in experiment 3,
   where the fine-tune repairs the very damage under test, so metric 1 decides and metric 2 serves as
   the collapse check.
3. **Control-generalization curve** — return vs perturbation distance on the spread ladder. How far
   outside its own distribution control stays valid.
4. **GenCrit excess bias** — predicted vs actual return over the same ladder. Whether the
   generator's judgement generalizes with control's competence.
5. **Exploration** — breadth (`build/n_modes`) *and* travel (energy distance) *and* mode coverage.

Numbering here is presentation order; ADR-0021 numbers them by the order they were designed
(1 return, 2 control-gen, 3 gencrit, 4 exploration, 5 specialized).
