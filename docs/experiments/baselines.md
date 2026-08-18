# Experiment 5 — Comparison against other codesign methods

Slug `baselines`. **Off-protocol**: [ADR-0021](../adr/0021-paper-experiment-metric-protocol.md) assumes
one algorithm ablated against itself. Protocol here is
[ADR-0022](../adr/0022-cross-method-comparison-protocol.md).

## Question

Experiments 1–4 ask what inside our algorithm is load-bearing. This one asks the only question a
reader outside the project cares about: **does it beat the alternatives, on tasks it was not built
for?**

The second clause is the whole design. A codesign method tuned on one locomotion task and compared to
rivals on that same task tells you almost nothing — the comparison measures which method the task
suits. So the grid spans tasks from **multiple sources**, including the benchmarks the rivals
introduced, and every task, every module library, and every algorithm is obtained through the same
`codesigner` package. Nothing is ported into our stack and nothing of ours is ported out: rivals and
ours implement the same `Algorithm` interface against the same `Task` and `ModuleLibrary`, so
simulator, grammar, reward and observation layout are identical by construction rather than by
careful matching.

## Conditions

The algorithm axis. Each is a `codesigner` `Algorithm` — `run()`, `is_finished()`, `control_policy()`,
`morphology_generator()`.

| algorithm | what it is | in `codesigner` |
|---|---|---|
| `ours` | shared-trunk transformer codesign; one `run()` = one generator window | `transformer_rl/algorithm.py:45` |
| `evolutionary` | GA over the body population, fitness from the shared controller; one `run()` = one generation | `components/algorithms/evolutionary.py` |
| `bodygen` | learned generative morphology policy trained by policy gradient — the closest rival | pending |
| `nge` | Neural Graph Evolution: graph mutation with a learned controller | pending |
| `robogrammar` | graph grammar plus heuristic search; **no learned generator** | pending |
| `fixed_body` | the task's reference morphology, our transformer controller, no morphology search | expressible now |

`fixed_body` is not an afterthought — it is the **normalization reference** (see below) and it is
what the interface's D18 means by "a **fixed** generator is how a control-only baseline is
expressed". It is also experiment 4's `full` arm on the ant task, so that condition arrives already
run.

`robogrammar` has no learned generator and no learned controller in the same sense as the others; it
is included precisely because it is the most different, and it is the condition most likely to expose
whether learning the design distribution buys anything over searching it.

## Tasks

The task axis, all from `codesigner.components.tasks`, spanning three sources and two problem
families:

| source | tasks | family |
|---|---|---|
| ours | `ant`, `grasp` | locomotion (free root), manipulation (fixed mount) |
| BodyGen (ICLR 2025) | `hopper`, `walker`, `gap`, `swimmer` | planar locomotion |
| Adroit / gymnasium-robotics | `door`, `hammer`, `pen`, `relocate` | dexterous manipulation |

Including BodyGen's four is deliberate and asymmetric in the rivals' favour: they are the benchmark
BodyGen introduced, ported here from the paper. Winning on them is worth more than winning on `ant`,
and losing on them is the more honest result to have to report.

The manipulation half matters for a different reason: our generator, its grammar and every
hyperparameter we tuned were developed on locomotion. `grasp` and the Adroit four are where the method
either generalizes past the family it was built on or does not.

## Held fixed

Per (algorithm × task): the `Task`, the `ModuleLibrary`, the reward, the observation layout, the
environment count, and the **environment-frame budget**. Seeds 42–49.

**Budget matching is by equal environment frames**, configured analytically rather than measured. The
interface does not expose a frame count — `Progress` carries `step`, `reward`, `best_reward`,
`wall_time` and `morphologies`, and each algorithm's `is_finished()` counts its own native unit
(`evolutionary.py:179` counts generations, ours counts windows). Frame consumption per unit is
deterministic here — fixed environment count, fixed episode length, fixed horizon — so each
algorithm's native stopping count is set from the shared frame budget by arithmetic. Details and the
per-algorithm formulas are in [ADR-0022](../adr/0022-cross-method-comparison-protocol.md).

`Progress.wall_time` is recorded and reported alongside, so the compute asymmetry between a method
whose search is nearly free and one that pays for separate fitness evaluations is visible in the
paper rather than hidden by the frame budget.

## Measurements and decision metric

Most of ADR-0021 does not survive contact with a method that has no generator. What does:

| ADR-0021 metric | transfers? | why |
|---|---|---|
| 1 — return curve | **no** | "return on its own bodies" means different things for a learned generator, a GA population and a fixed body; the curves are not on comparable axes |
| 2 — control-generalization | **no** | needs a generator distribution to perturb along |
| 3 — GenCrit excess bias | **no** | needs GenCrit; only `ours` has one |
| 4 — exploration | **partly** | breadth and travel need only a population, and `Progress.morphologies` reports one for **every** algorithm through the interface. Undefined for `fixed_body` and `robogrammar`'s deterministic search |
| 5 — specialized return | **yes** | it needs nothing but a committed body: strip the scaffolding, warm-start control, fine-tune 250 epochs on that body, measure |

**Decision metric: specialized return, per (algorithm × task)** — the series default, and the only
ADR-0021 metric that is method-agnostic. Every method in the grid outputs a morphology, and
specializing a controller onto it is the same operation regardless of how the morphology was found.
That the cross-method experiment and the ablations share a decision metric is what lets the paper's
sections be read against each other.

**Aggregation across tasks: normalize each task by `fixed_body`.** Raw returns are incomparable across
ten tasks from three sources with unrelated reward scales. Every task has a reference morphology and
therefore a `fixed_body` number, so per-task scores are reported as a ratio to it — each entry reads
as "how much better than the task's human-designed body", which is the quantity a codesign paper is
actually claiming. Aggregate with the per-task table plus a median across tasks; **never a mean of
ratios across tasks**, which a single small-denominator task can dominate.

Also reported per cell: exploration (where defined), wall-clock, and the committed morphology itself.

## Expected results and falsifier

**Expected.** `ours > bodygen > nge ≈ evolutionary > robogrammar > fixed_body` on the locomotion half,
with the margin over `bodygen` narrowest on BodyGen's own four tasks. On the manipulation half the
ordering is genuinely unknown — that is why they are in the grid.

**Falsified if `fixed_body` is competitive.** If specializing a controller onto the reference
morphology matches what any search finds, morphology search is not paying for itself on that task, and
the task belongs in the paper as a negative result rather than being dropped. This is the outcome to
guard against dropping quietly, and it is the reason `fixed_body` is the normalizer: at a ratio of
1.0, the claim disappears, visibly, in the same column as everything else.

**Interpretive hazards:**

- **A weak port is worse than no port.** Every rival's number is only as good as its implementation
  against the `Algorithm` interface. Any rival whose result falls below its published relative
  standing must be reported with that discrepancy stated, not presented as a clean loss.
- **Hyperparameter asymmetry.** Our hyperparameters were tuned on `ant` (ADR-0020's screen). Unless
  each rival gets a comparable tuning budget per task family, the grid measures tuning effort as much
  as method quality. Whatever budget is given must be equal and stated.
- **Equal frames is not equal compute.** It is the standard for a sample-efficiency claim and it is
  the choice made here, but a method whose search rides on rollouts control already collects is
  advantaged on this axis for reasons unrelated to search quality. Hence the wall-clock column.
- **`fixed_body`'s controller is ours.** The reference condition uses our transformer control policy,
  so it is not "the published baseline for this task" — it is "the task's reference body, controlled as
  well as we can control it". That makes it a *strong* normalizer, which is the conservative direction.

## Where it lands

The results section's centrepiece, after the ablations. Experiments 1–4 explain the method; this one
is the reason to care. Its grid, not its narrative, is the contribution: a multi-source task suite and
several codesign algorithms behind one interface is a comparison nobody has been able to run, and the
`codesigner` package is what makes it runnable.

Written up ahead of being runnable: three of the six algorithms are not yet implemented in
`codesigner`. The protocol is fixed now so that each one lands against a target rather than defining
its own.
