---
status: accepted
---

# Codesign benchmark methodology

Codesign will be compared against four required comparators: a fixed-base-morph control, a uniform-action generation policy, faithful Neural Graph Evolution (NGE), and faithful BodyGen. NGE retains NerveNet++, parent-to-child policy sharing, population selection, Graph Mutation with Uncertainty, and all four graph mutation classes; BodyGen retains MoSAT, TopoPE, its topology and attribute design stages, enhanced temporal credit assignment, and method-native value networks. The two controls use the codesign control stack with learned generator updates disabled.

Every method uses the same locomotion task, the full typed morphology grammar, and the `[1,4,6]` base morph as its initial morphology. The base morph provides morphology only: every controller uses its method-native random initialization under matched training-seed identifiers, with no uncounted pretrained controller. NGE policy inheritance begins within the measured run and all inherited-policy training consumes its shared environment-step budget. The shared grammar has eight fixed limb slots, each absent or containing one to three typed effectors followed by a typed cap; NGE's graph mutations and BodyGen's design actions are constrained to this grammar. Training follows ADR-0015's fair-comparison axis: equal total physics env-steps, parallel-environment cap, and reporting seeds `[42, 43, 44, 45, 46]`; methods may use a narrower vector width when temporal depth is algorithmically required, but never exceed the shared cap. Runtime and memory are measured outputs rather than budget inputs. The step counter includes every physics transition consumed before the final checkpoint, whether used for controller updates, morphology fitness or selection, GM-UC labels, or BodyGen design evaluation. Simulator-free design-network computation does not add environment steps. Faithful native architectures are not resized to match parameter counts; trainable parameters, wall time, peak host and device memory, and environment-step throughput are reported as efficiency outcomes.

BodyGen retains the upstream unified trajectory: five simulator-free topology
waves, one simulator-free attribute wave, and the resulting body's complete
control episode. Topology decisions are made once for every node present at
the start of a wave and applied from that snapshot; new nodes first act in the
next wave. Root Add fills the first empty limb slot in numeric order `1…8`,
terminal root children may be deleted, at least one effector is retained, and
chains remain at most three effectors long. The upstream continuous MuJoCo
attribute action is replaced by one simultaneous absolute categorical choice
of the three shared effector types and four terminal cap types. This is the
minimum method adaptation that exposes the same complete typed grammar to
every comparator without adding a BodyGen-only limb-slot policy head.

BodyGen also retains six independent MoSAT trunks—topology, attribute, and
control actors plus their three critics—rather than sharing this repository's
CoDesign network. TopoPE uses collision-free little-endian base-nine root
paths that inherently fit the upstream-sized 256-entry table. Enhanced
temporal credit assignment uses undiscounted remaining episode return for both
design stages and GAE for control, with one observation normaliser shared by
every trunk and separate design/control return normalisation. Node log
probabilities sum to the corresponding body-transition log probability.

Algorithm fidelity also applies to training objectives. All methods receive the same locomotion task reward, but NGE retains its native species-fitness, selection, and GM-UC process; BodyGen retains its native policy and value objectives; and codesign retains its existing objectives. The common benchmark score is external to optimisation: monitoring cannot update or select the policy, while tuning and final reporting use their designated disjoint seeds, so reporting samples cannot leak into optimisation.

NGE species selection uses a separate complete-episode pass at every morphology update because the 4,096-environment PPO layout supplies sample count through width rather than enough temporal depth to finish a horizon-length episode. That pass is part of training, uses frozen stochastic native controllers, and is fully charged to the same environment-step budget. Its fixed rollout is longer than the task horizon, its parallel width remains below the shared cap, and every species is ranked by exactly the first completed raw return from each evaluation lane. Later auto-reset activity remains charged but receives no ranking weight, avoiding both partial-return fitness and unequal episode counts. The default schedule reserves one of the upstream twenty 2,048-transition-per-species batches for this pass: nineteen PPO batches plus selection therefore preserve the same 40,960 transitions per species and the same exact generation boundary. At that boundary, survivors retain their controller, value baseline, observation statistics, and current adaptive policy learning rate; children inherit the same state from their parent. Optimizer moments are rebuilt as in the upstream worker lifecycle, but the adapted rate is not reset to its initial value.

BodyGen reproduces the upstream twenty logical collection streams and
50,000-transition minimum batch. Every stream collects complete unified
trajectories until reaching `floor(50,000/20)=2,500` stored MDP transitions.
VSim streams advance in synchronous waves; every physical transition,
including work performed by lanes waiting for another trajectory to finish, is
charged. Checkpoints separately audit retained trajectory physics,
synchronisation waste, and any exact-budget discarded trajectory tail. At the
shared physics-step boundary, an incomplete PPO collection is discarded rather
than changing BodyGen's native batch rule; `final.pth` is written only after the
environment closes at exactly the required count. The six design transitions
per trajectory remain PPO data but are simulator-free and therefore do not
increment the shared physics counter.

Evaluation uses each method's native morphology–controller pairs from its final checkpoint at the exact shared environment-step budget. This final-budget checkpoint is the sole source of the headline result; rl_games' method-specific best checkpoint and other temporally selected checkpoints cannot replace it. The primary score—and the held-out tuning objective—is probability-weighted expected deterministic control return under the method's final native output distribution: stochastic design-policy samples for codesign and BodyGen, uniform valid-action samples for the uniform control, uniform samples over NGE's final surviving species, and the single fixed pair for the fixed control. Equal-weight unique-body return, Top-1-of-M, Top-K-of-M, nominal-task stability, convergence, diversity, runtime, and memory are secondary outcomes; learned-policy greedy decoding is diagnostic only because random generation order means it need not be the global distribution mode. Nominal fall rate and episode length are called stability, not robustness; robustness claims require a separate perturbation or generalisation suite and are outside this benchmark.

VSim reports termination or truncation one simulator call after the actual
terminal transition. On that notification call the lane has already reset, so
its returned reward is the next reset state's healthy reward rather than part
of the completed episode. Both `scripts/eval.py` and the benchmark evaluator
finish from the previously accumulated transitions, exclude that reset reward
and extra length, and retain the notification call only where training-step
accounting must charge its physics. Evaluation CSVs generated before this
correction include `+2` reward and `+1` length per episode under the current
task defaults and must be regenerated rather than mixed with corrected
artifacts. This incompatible correction increments the saved benchmark
`protocol_version` from 1 to 2.

The final checkpoint preserves the complete frozen state needed to reproduce that native distribution. Codesign and BodyGen retain their design and control policies; NGE retains every surviving species with its graph, controller, and population metadata; fixed and uniform retain their controller and morphology-source state. The evaluator samples native pairs afresh from this state using recorded reporting seeds instead of embedding a preselected pair bank or reducing a method to one champion.

Every training checkpoint is also a complete resume boundary: it stores model and optimiser states, RNG states, schedulers, the authoritative environment-step counter, and all method-native search state. NGE includes its population and GM-UC training data; BodyGen includes its design-policy state. Resume preserves the original run identity and continues the remaining shared budget rather than beginning a new run or resetting the counter.

Evaluation uses common random numbers without coupling morphology distributions. The protocol lists literal morphology-sampling, rollout, and diversity seeds; these integers are used directly without hashing or derivation. Methods sample their native pairs independently, but corresponding jobs receive the same configured rollout seed and therefore matched initial-condition randomness wherever the simulator permits it.

Statistical summaries treat the five independently trained reporting seeds as the primary replication units. The paper reports every seed-level estimate, the across-seed mean with a 95% bootstrap confidence interval, and paired seed-level differences between methods. Native-pair and episode samples estimate each trained method's output distribution but are not promoted to independent training replicates.

The paper evaluation preset uses 128 native pairs per method and reporting seed, 32 deterministic episodes per pair, `K=10`, and 4,096 design-only samples for distribution and diversity estimates. Cross-method diversity is reported as (1) the unique fraction in those samples, (2) empirical entropy and its effective-body-count transform from sample frequencies, and (3) mean normalized typed-token distance between sampled bodies in the canonical grammar encoding. Codesign committance remains a method-specific diagnostic rather than a shared diversity result. Codesign, faithful NGE, and faithful BodyGen each receive exactly 30 complete candidate configurations, with every configuration run at the same full proxy environment-step budget on the same three fixed tuning seeds: 90 proxy runs per method. Early pruning and discretionary early stopping are disabled. Candidates are ranked by their mean primary benchmark score across the three tuning seeds, which are disjoint from reporting seeds `[42, 43, 44, 45, 46]`. The fixed and uniform controls inherit the selected codesign controller settings.

Training reward curves remain diagnostics because their episode sampling differs
between algorithms. BodyGen writes its rolling complete-trajectory return to
`rewards/step` and `rewards/time` after each synchronous collection wave, while
`rewards/iter` remains once per native PPO update; multiple waves do not invent
policy iterations while the policy is unchanged. For readable matched-step
progress, the shared fixed-pair
rollout core may optionally run at configured checkpoint boundaries and log the
probability-weighted complete-episode return as `rewards/step_eval` against the
checkpoint's charged training environment-step count. This monitoring is
disabled by default, uses seeds separate from final reporting, does not modify
the optimiser or training-step counter, and cannot be used to choose a
headline checkpoint. NGE exposes a simple `every_generations` schedule; the
BodyGen equivalent is scheduled by native PPO update. The standalone
checkpoint evaluator writes the same tag for every method. BodyGen's upstream
greedy/mean native evaluation is also optional and disabled by default. When
enabled it may write a diagnostic `best.pth`, but it cannot change training,
select the paper result, or replace `final.pth`.

Each adaptive method stores one concrete runnable configuration in
`configs/benchmarks/<method>.yaml` and its ranges in
`configs/benchmarks/tune_<method>.yaml`. Named conditional constraints in the
search contract are also hard validation rules in the method loader, so a
direct run cannot bypass sample-count, complete-episode, batch-divisibility, or
exact-budget requirements. Infeasible proposals are rejected before simulation
and do not count among the thirty completed candidates.

Benchmarking lives beside, not inside, the current codesign implementation: a separate benchmark package owns the shared evaluator and plainly named method modules. After initial parity was demonstrated, `scripts/eval.py` and the shared evaluator were changed together to exclude VSim's delayed reset notification from episode reward and length. `configs/benchmarks/benchmark.yaml` is the single readable authority for evaluation requirements, seeds, artifacts, logging, and the currently selected method; each native trainer has a directly runnable method YAML whose stamped budget and seed are checked against those requirements. Configuration inheritance and dynamic adapter loading are deliberately deferred until multiple concrete implementations show they are needed. The launcher validates the resolved configuration and saves it with the result. TensorBoard is the durable local record for every method, with W&B as an optional second sink for the same canonical metrics, resolved configuration, and run identity. W&B is disabled by default, enabled only through explicit YAML, lazily imported, and supports online and offline modes; credentials alone never activate it, and requesting an unavailable or invalid integration produces a clear error. Comparable values use a strict shared `benchmark/...` metric namespace in TensorBoard, W&B, and `summary.csv`; method-native training and diagnostic values cannot substitute for shared metrics.

Each benchmark invocation writes one self-contained comparison artifact: `manifest.yaml` records the protocol version, git commit, resolved method configs, checkpoint identities, seeds, and budgets; `summary.csv` contains one canonical metric row per method, training seed, and checkpoint; and `pairs/<method_run_epoch>.npz` retains sampled typed morphologies, native controller identities, sampling weights, and per-episode rollout outcomes for each row. The CLI follows `scripts/eval.py` by accepting positional runs and final-or-numeric epochs, printing a side-by-side table, and writing the combined CSV. A positional `METHOD@CHECKPOINTS=RUN` selector supports convergence comparisons when native counters differ—for example, CoDesign epochs against NGE generations—while the saved environment-step count remains the alignment axis. TensorBoard and W&B mirror summaries but do not replace these files as the reproducible result.
A BodyGen numeric selector such as `bodygen@100,200=RUN` names saved native PPO
updates; `bodygen=RUN` selects `checkpoints/final.pth`.

Generated state is separated from source: benchmark training runs live at `runs/benchmarks/<method>/<run-id>/s<seed>/`, tuning studies at `logs/tune/benchmarks/<method>/`, and cross-method comparison bundles at `evals/benchmarks/<evaluation-id>/`.

Implementation is staged behind contract gates: first the shared protocol and copied evaluator, then codesign parity with the existing evaluator, then fixed and uniform controls, then faithful NGE, and finally faithful BodyGen. A later stage does not weaken or replace an earlier method; all five methods remain part of the paper suite.

Each gate runs a shared benchmark contract suite covering grammar validity properties, deterministic seed replay, exact global environment-step accounting, save/resume equivalence, adapter schema conformance, and validation of all local evaluation artifacts. The codesign stage additionally reproduces the existing `scripts/eval.py` body and rollout results within an explicit numerical tolerance before the companion evaluator becomes authoritative.

Faithful NGE and BodyGen are ports from pinned commits of their official implementations, not paper-only reconstructions or external simulator jobs. Licensed attribution and upstream provenance are retained, and every behaviourally relevant adaptation is documented. Their native algorithms and architectures are translated onto this repository's vlearn/VSim environment interface and shared typed grammar; neither their original MuJoCo backends nor Isaac Sim is part of the benchmark.

The label `faithful` is gated by a component-by-component fidelity checklist mapping the paper and pinned upstream implementation to local code and tests. NGE and BodyGen each keep `ADAPTATIONS.md` plus a machine-readable upstream commit and licence record in their method package. Each adaptation entry records the upstream symbol, local counterpart, rationale, expected behavioural effect, fidelity status, and validating test. Evaluation manifests include hashes of these provenance records. A material unresolved omission changes the reported label to `adapted NGE` or `adapted BodyGen` until it is resolved.

## Consequences

- Claims about NGE and BodyGen refer to full-method adaptations, not hybrids using the codesign controller.
- The shared parallel-environment requirement is a positive upper bound:
  paper-compliant checkpoints satisfy `0 < peak_parallel_envs <= 4096`, not
  equality with 4,096.
- All paper comparisons can be regenerated from one training/evaluation contract without method-specific metric definitions.
- Changing the suite, design-space boundary, budget axis, native-pair rule, primary score, or paper preset after experiments begin requires rerunning affected methods.
