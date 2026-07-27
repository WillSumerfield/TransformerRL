# Benchmarks

The deliberately small implementation of ADR-0016. Benchmark settings live in
`configs/benchmarks/benchmark.yaml`; `scripts/benchmark_eval.py` is the CLI.

## Read the code in this order

1. `scripts/benchmark_eval.py` — the familiar multi-run CLI and job loop.
2. `evaluate.py` — the complete sample → VSim rollout → score → save flow.
3. `codesign.py` — CoDesign plus the saved-controller loading shared with its controls.
4. `fixed_body.py` — the fixed-base-morph control; one short specialization.
5. `uniform_action.py` — the uniform grammar-action control; one short specialization.
6. `metrics.py` — exact definitions of every shared reported metric.
7. `data.py` — the two arrays-based data containers exchanged between them.

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

**Paper run requirements**:
The three saved-training facts checked before a result may be labelled paper-compliant: total
physics environment steps, parallel environment count, and training seed. They do not configure or
start training. Development runs may bypass these checks explicitly.

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
