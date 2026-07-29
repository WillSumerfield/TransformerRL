# Benchmarks

The deliberately small implementation of ADR-0016. Benchmark settings live in
`configs/benchmarks/benchmark.yaml`; `scripts/benchmark_eval.py` is the CLI.

## Read the code in this order

1. `scripts/benchmark_eval.py` — the familiar multi-run CLI and job loop.
2. `evaluate.py` — the complete sample → VSim rollout → score → save flow.
3. `codesign.py` — CoDesign plus the saved-controller loading shared with its controls.
4. `fixed_body.py` — the fixed-base-morph control; one short specialization.
5. `uniform_action.py` — the uniform grammar-action control; one short specialization.
6. `nge/graph.py` then `nge/population.py` — NGE's four mutations and selection loop.
7. `nge/nervenet.py` then `nge/gm_uc.py` — NerveNet++ and uncertainty pruning.
8. `nge/training.py` then `nge/method.py` — measured training and evaluation routing.
9. `metrics.py` — exact definitions of every shared reported metric.
10. `data.py` — the two arrays-based data containers exchanged between them.

No dynamic adapter framework or YAML inheritance is used. When another method arrives, it should
first be added as one plainly named module with the same small set of methods used in
`evaluate.py`. Introduce a formal abstraction only if repeated implementations demonstrate a real
need for one.

## Language

**Evaluation pairs**:
The typed arrays `counts[M,8]`, `eff_sub[M,8,max_len]`, and `cap_sub[M,8]`, plus one native
controller identity and sampling weight per row. Repeated rows are intentional: their frequency
already represents the method's output probability, so stochastic draws use equal row weights.

**Evaluation seed plan**:
The literal `evaluation.seeds` integers for morphology sampling, rollout, and diversity sampling.
The evaluator records and uses exactly these values—there is no hashing or derived root seed.
`--seed N` sets all three to `N` for a direct, easily understood comparison.

**Matched progress comparison**:
A shared benchmark evaluation of method-native checkpoints that consumed the same number of
training environment steps. `METHOD@CHECKPOINTS=RUN` lets one invocation select different native
checkpoint counters, such as CoDesign epochs and NGE generations, while retaining one evaluator and
one comparison artifact.
_Avoid_: overlaying method-specific training reward tags as if they had identical sampling semantics

**Training progress evaluation**:
An optional deterministic, complete-episode call to the shared fixed-pair
rollout core, logged as `rewards/step_eval` against charged training environment
steps. It is disabled by default, has its own non-reporting seeds, never changes
the optimiser or charged step counter, and cannot replace the final paper
evaluation. NGE schedules it with
`training_evaluation.every_generations`; checkpoint-based evaluation writes the
same tag for every method.
_Avoid_: interpreting a peak in rolling `rewards/step` as a benchmark score,
charging monitoring transitions as learning data, selecting the headline
checkpoint from the monitoring curve

**Paper run requirements**:
The three saved-training facts checked before a result may be labelled paper-compliant: total
physics environment steps, maximum/controller-rollout environment width, and training seed. They do
not configure or start training. Development runs may bypass these checks explicitly.

**Benchmark artifact**:
A no-clobber comparison directory containing `manifest.yaml`, one multi-row `summary.csv`,
`pairs/<method_run_epoch>.npz` raw files, and per-job TensorBoard events. Each NPZ retains every
episode rather than only averages. Optional W&B mirrors the same metrics.

**CoDesign benchmark method**:
`codesign.py` reuses the saved run config and proven policy loader. It performs native stochastic
generator sampling, installs those bodies in `AntCodesignEnv` on VSim, restores the raw observation
tail, and selects deterministic actions. `checkpoint: final` accepts only the final epoch snapshot;
the rl_games bare best checkpoint is never a fallback.

**Fixed-body benchmark method**:
`fixed_body.py` reuses CoDesign's saved-controller loader and deterministic control step. Its native
distribution is the single `[1,4,6]` base morph, repeated only to obtain parallel rollout estimates
and to flow through the exact same metric/artifact path. Morphology and diversity seeds have no
effect, diversity correctly has effective body count one, and the environment stays on its
canonical initial build rather than invoking a rebuild.

**Uniform-action benchmark method**:
`uniform_action.py` reuses the same controller loader and changes only
`CodesignMethod.sampling_mode` from `stochastic` to `uniform`. This invokes the generator's existing
grammar-masked MDP with zero action logits: every valid action is equally likely at each decision.
It is not a uniform distribution over completed bodies. Repeated sampled bodies retain their
frequency as probability mass, exactly like CoDesign samples.

**NGE benchmark method**:
`nge/` is a package because the full published method has four independently
auditable parts; it is not a generic baseline framework. `training.py` trains
one recurrent NerveNet++ controller per species in population-grouped VSim
environments, charges every transition to one global counter, evolves only at
complete generation boundaries, and saves the population plus GM-UC and RNG
state. `method.py` samples final survivors uniformly and routes each evaluation
row to its own controller. Read `nge/ADAPTATIONS.md` for the component-by-
component paper/upstream mapping before changing it.

**NGE selection evaluation**:
The paid, method-native complete-episode pass that supplies raw-return fitness
for species selection at each morphology update. Every species contributes
exactly one first completed episode per configured evaluation lane; later
auto-reset activity is charged but has no ranking weight.
_Avoid_: partial-return fitness, unequal episode counts, held-out benchmark evaluation

**NGE rollout return estimate**:
The short-PPO-rollout diagnostic logged as
`rewards/rollout_return_estimate`; it is not species fitness.
_Avoid_: fitness estimate, selection reward

**NGE rolling training return**:
`rewards/step`, the rl_games-compatible rolling mean over the latest 100
completed training or selection episodes. A point after selection contains
more complete selection returns but can still contain older and short-rollout
completions. It is a training-health signal, not the shared comparison curve.
_Avoid_: treating a post-selection peak as `rewards/step_eval`

**NGE step audit**:
The controller and selection transition counters whose sum is the authoritative
training environment-step count. All three are checkpointed and logged.
_Avoid_: free selection evaluation, PPO-only step count

**Benchmark search contract**:
The method-specific `tune_<method>.yaml` ranges and named feasibility
conditions paired with one runnable `<method>.yaml`; the method validator
enforces the conditions again after resolution. NGE uses this split now and
BodyGen must adopt it with its faithful port.
_Avoid_: mixing ranges into runnable configs, tuner-only validation

## Check the benchmark contracts

In a new shell, load the project environment and then run all three small
benchmark test groups:

```bash
source scripts/activate_uv.sh
python -m unittest discover -s tests -p 'test_benchmark*.py'
```

## Implementation history

- Stage 1 originally introduced generic adapters, recursive config overlays, and separate runner,
  logging, artifact, rollout, seed, and type modules.
- Before adding baselines, that framework was collapsed into the small files above so the complete
  execution path can be audited directly. Final-checkpoint selection, step/seed checks, raw
  episodes, canonical metrics, TensorBoard/W&B, and CoDesign parity tests were retained.
- The first control baseline added one inherited training config, one training entry point, and
  `fixed_body.py`. It reuses the complete selected CoDesign controller training stack with
  `resample_interval=0`; therefore generator updates and morphology rebuilds never occur.
- The second control added the equally small `uniform_action.py` adapter and a
  `UniformActionAgent`. Training begins on the shared base morph, then replaces each morphology
  window with `net.sample(..., mode="uniform")`. Controller PPO/AdamW/FD/FK training is unchanged;
  GenAct, GenCrit, and the generator's control-cloning update are not run.
- The NGE stage is deliberately larger than either control because it retains a
  population of native controllers, four graph mutations, GM-UC, recurrent PPO,
  genealogy, and complete resume state. Its package contains no reusable
  adapter hierarchy; every file corresponds to one named NGE component.
