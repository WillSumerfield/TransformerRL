# Credit assignment and response-probe experiments

Status: exploratory line closed after diagnostic and feasibility work. The code is
kept as a reproducible checkpoint, not promoted to the default codesign objective.

## Question

The experiments asked whether the generator's decisions could receive more useful
credit than a scalar completed-body return or a sequential GenCrit prefix delta.
The related response-probe work asked whether contextual morphology representations
contained predictive information about the physical response of a module that could
justify a learned auxiliary signal.

These were two related but distinct questions:

1. Can credit be mapped to the physical module created by each generator action?
2. Does contextual body information improve prediction of subsequent dynamics enough
   to be worth integrating into training?

## Approaches tried

### Generator-credit instrumentation

`generator_credit.py` reconstructs the autoregressive construction trace and maps each
active decision to its limb, depth, module token, and controller DOF. It records
GenCrit prefix deltas, PPO advantages, telescoping residuals, within-body variation,
and context dependence by subtype, depth, parent, and body size.

This established an important accounting invariant: the sequential credit can be
checked against the change in the prefix value, and the generator action can be
identified with the physical module it creates. It is diagnostic instrumentation;
the basic logging path does not by itself change the training objective.

### Contextual spatial credit

`spatial_credit.py` adds an optional post-Transformer per-module head:

```text
V_spatial = V_global(CLS) + sum(module_credit_i)
```

Tree propagation, matched structural counterfactual pairs, shuffled controls, and
body-mean centering were added to distinguish genuine topology/context effects from
an easier change in the marginal credit distribution. The implementation supports
diagnostics and optional auxiliary/gen-act experiments, but the default config keeps
the feature disabled.

### Return-target and adaptation comparisons

The generator can be evaluated against training-time body return or post-adaptation
return. The accompanying scripts compare whether using post-adaptation return
changes the effector-credit bias, morphology complexity, diversity, and final
adapted performance.

### Static response probes

The response-probe path captures short state/action/contact transitions at resample
boundaries without changing PPO. Offline models then compare interaction-only,
metadata, pre-attention, post-attention, local learned, fresh full-body, and
checkpoint-initialized representations. Splits are by body identity, and results
are reported per body rather than per transition.

### Baseline harness

The harness work makes native SoftwarePackage baseline algorithms resumable,
scrapable, and comparable under the same task and frame-budget protocol. This is
infrastructure for the outside-baseline experiment; it is not evidence that the
baseline comparison has been completed.

## What happened

The results were useful as diagnostics but did not provide a sufficiently clean
positive case for integration.

### Credit signals were weak and sensitive to the target/control

The available pilot credit artifacts show low action-level association with body
return. In representative window-2 artifacts, correlations between module-level
signals and body return were roughly 0.06–0.18, and the sign/magnitude of prefix
credit changed across baseline, body-mean, aligned, and shuffled conditions. The
tree-propagated signal was often more variable than the direct spatial signal. These
are pilot diagnostics, not a final multi-seed estimate, but they do not justify
claiming that one credit definition is clearly correct.

The main positive result was bookkeeping: the physical mapping, telescoping checks,
counterfactual pairing, and artifact format make the question measurable. They did
not establish that the new signal improves codesign.

### Contextual response prediction was feasible, but not decisive

For the recorded window-2 seed-42 probe, the normalized test errors were:

| Model | Test body-normalized MSE |
|---|---:|
| interaction only | 0.2542 |
| interaction + metadata | 0.2290 |
| interaction + metadata + pre-attention tokens | 0.2272 |
| interaction + metadata + contextual tokens | 0.2516 |

The contextual model's static-swap diagnostic increased matched-row error by 2.19,
which suggests the representation contains information relevant to the prediction.
However, the contextual model did not beat the metadata or pre-attention baselines;
their bootstrap intervals against the contextual model included zero. The offline
learned-representation probe likewise found metadata/local baselines competitive or
better than fresh or checkpoint-initialized full-body encoders.

The interpretation is therefore “feasible observational signal, no demonstrated
training benefit,” not “context is useless.” The probe was observational, used one
controller run for the reported example, and did not test a structural holdout or a
causal intervention.

### Outside-baseline work reached protocol/infrastructure stage

The baseline harness now handles native runners, checkpoint detection, reranking,
and scraping. The full comparison remains a future experiment because several rival
algorithms still need to be ported to the shared SoftwarePackage interface. No
cross-method performance claim should be inferred from this checkpoint.

## Why the line was stopped

We stopped before integrating the spatial head or response predictor into the
default objective because:

- the credit correlations were weak and unstable across plausible targets and
  controls;
- the contextual response probe did not beat simpler metadata/pre-attention
  baselines on the available test bodies;
- the most interesting positive signal was diagnostic (static swaps), not causal;
- a convincing claim would require a larger pre-registered multi-seed study,
  structural holdouts, and a training-level ablation;
- the added objectives would introduce more hyperparameters before the basic signal
  had cleared a useful gate.

This is a stopping decision about research priority, not a claim that the mechanisms
are disproven.

## What was learned

- Preserve the generator-to-module mapping and telescoping checks; they are useful
  audit infrastructure even without a new objective.
- Always compare contextual signals against metadata and pre-attention controls.
- Body-level splits and body-averaged metrics are necessary; transition-level rows
  would overstate confidence.
- A static-swap effect can show that a representation carries information, but it
  does not show that using that representation improves co-design.
- Post-adaptation return is a meaningful alternative target, but changing the target
  changes the scientific question and needs its own controlled comparison.
- The baseline protocol must be completed before making claims against outside
  methods; shared task, simulator, and frame budget are more important than matching
  published numbers.

## Reproduction pointers

- Core diagnostics: `transformer_rl/generator_credit.py`,
  `transformer_rl/spatial_credit.py`
- Training integration and artifact logging: `transformer_rl/codesign_agent.py`
- Static probe guide: `docs/guides/static_response_probe.md`
- Learned-response guide: `docs/guides/learned_response_representation.md`
- Offline analysis: `scripts/analyze_generator_credit.py`,
  `scripts/analyze_spatial_credit.py`, and the `compare_*` scripts
- Unit tests: `tests/test_*credit.py`, `tests/test_response_probe.py`,
  `tests/test_learned_response.py`
- Outside-method protocol: `docs/experiments/baselines.md`
