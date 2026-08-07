# Context Map

Research repo for **codesign**: jointly optimizing a robot's *morphology* (currently an ant) and a transformer *control* policy. The transformer must generalize across morphologies so the controller keeps working as the body changes during codesign. Attention is the bridge between the control policy and the generative morphology policy — both now realized as a single shared-trunk codesign network.

## Structure

```
/
├── CONTEXT-MAP.md                    ← this file; shared kernel below
├── docs/
│   ├── adr/                          ← system-wide decisions
│   ├── reference/                    ← lookup docs
│   │   ├── transformer_architecture.md (living doc: full obs→action flow, shapes, why)
│   │   ├── Metrics.md                (eval.py metric reference: meaning/formula/reading)
│   │   ├── codesign_metrics.md       (codesign TB metrics + debugging playbook)
│   │   └── vsim_geometry_api.md
│   ├── guides/                       ← how-to / cost playbooks
│   │   ├── group_count_throughput.md (group-count-independent env throughput)
│   │   ├── morphology_resampling_cost.md (rebuild cost + resample cadence for the full ant)
│   │   └── deterministic_embedding.md (matmul mode-embedding: determinism + speed)
│   └── troubleshooting/              ← crashes / gotchas / bug records
│       ├── determinism_bf16_nan.md  (why --seed drops deterministic algos)
│       ├── resample_rebuild_crash.md (intermittent gym-rebuild race)
│       └── adaptive_ant_fixes.md
├── task_envs/                         ← Morphology context
│   ├── CONTEXT.md
│   ├── multigroup_environment.py
│   ├── codesign_environment.py       (CodesignEnvironmentGpu: shared base, one group per morphology)
│   ├── build_vsim.py                 (programmatic vsim per limb subset; generic over ModuleLibrary)
│   ├── modular_libraries/            (ModuleLibrary: per-robot-family module vocabulary + root mounting; ADR-0016)
│   ├── ant_envs/
│   │   ├── ant_codesign.py           (codesign env; generator-driven bodies per resample)
│   │   └── assets/
│   └── dexman_envs/
│       └── grasp_codesign.py         (object-grasping codesign env; hand = world-mounted root)
├── transformer_rl/                   ← Control context
│   ├── CONTEXT.md
│   ├── architectures.py              (LimbTransformer, MultiMorphLimbTransformer)
│   ├── tokenize.py                   (obs → root/module tokens; codesign uniform module tokens)
│   ├── models.py                     (rl_games model/network builders)
│   ├── rollout.py                    (test-mode rollout engine; ADR-0007)
│   ├── logging_agent.py
│   └── train_utils.py
├── scripts/                          ← Training context
│   ├── CONTEXT.md
│   ├── train_ant_*.py
│   └── tune.py                       (Optuna sweep)
├── configs/                          ← Training context (rl_games yaml)
│   ├── ppo_ant*.yaml
│   └── tune_config.yaml
├── experiments/                      ← Analysis context
│   ├── CONTEXT.md
│   ├── phase_comparison.py           (cross-phase harness; ADR-0015)
│   └── diversity_p5.py               (population diversity estimators)
├── notebooks/                        ← Analysis context
├── data/                             ← Analysis context (figures, .npz)
├── logs/  runs/  videos/             (run artifacts, gitignored/untracked)
```

## Contexts

- [CoDesigner](../SoftwarePackage/CONTEXT.md) *(upstream package)* — Task / ModuleLibrary / Algorithm interfaces, Module, Orientation, Morphology, attachment slot
- [Morphology](./task_envs/CONTEXT.md) — the ant body design space: vsim physics builds, the morphology set, active/inactive DOFs, the DOF mask
- [Control](./transformer_rl/CONTEXT.md) — the transformer policy that controls any morphology: tokenization, limb encoding, token masking, rl_games integration
- [Training](./scripts/CONTEXT.md) — PPO training, Optuna tuning, play/render orchestration
- [Analysis](./experiments/CONTEXT.md) — attention studies over trained policies

## Relationships

- **Morphology → Control**: Morphology emits the **893-D codesign** observation (`AntCodesignEnv`, variable-length `module_lengths` + 32 DOF mask, layout from `tdims`); Control tokenizes it and reads the DOF mask to decide which limb/module tokens exist.
- **Control → Training**: Control registers networks/models with rl_games under names Training selects via config `model.name` / `network.name`.
- **Training → Analysis**: Training produces checkpoints; Analysis loads them to collect attention.
- **Shared kernel** (below): Robot, Morphology, Limb, Module, Root, DOF, DOF mask, active/inactive, EnvironmentGroup, codesign, Task — defined once here, used identically across all contexts.

## Upstream — CoDesigner

The **CoDesigner** package ([`../SoftwarePackage`](../SoftwarePackage/CONTEXT.md), installed editable as `codesigner`) owns the `Task` / `ModuleLibrary` / `Algorithm` interfaces this repo implements, and its [CONTEXT.md](../SoftwarePackage/CONTEXT.md) is the source of truth for their vocabulary. Migration in flight: our tasks (`task_envs/`) and module libraries (`task_envs/modular_libraries/`) move **into** the package; our codesign algorithm (the shared-trunk transformer + PPG agent) **stays here** and plugs in as one `Algorithm`.

## Upstream — vlearn / VSim

This repo runs on the **vlearn** RL framework and the **VSim** GPU simulator, vendored as a sibling checkout at [`../vlearn-main`](../vlearn-main). Coupling: every env/script inserts `../vlearn-main/train` on `sys.path` and `import vlearn` (`vlearn.spaces`, `vlearn.torch_utils`); `CodesignEnvironmentGpu` and `AntEnv` subclass `EnvironmentGpu` (via `envs.ant_environment_gpu.AntEnvironmentGpu`).

Reference these when working on env physics or the group/motor/sensor plumbing:

- **Glossaries** (same CONTEXT.md style): [`../vlearn-main/train/CONTEXT.md`](../vlearn-main/train/CONTEXT.md) (VLearn framework — `EnvironmentGpu`) and [`../vlearn-main/docs/api/CONTEXT.md`](../vlearn-main/docs/api/CONTEXT.md) (VSim simulator).
- **API reference** (markdown): [`../vlearn-main/docs/api/index.md`](../vlearn-main/docs/api/index.md). Most-used here:
  - [`environments.md`](../vlearn-main/docs/api/environments.md) — `EnvironmentGroup`/`EnvironmentDef` (our one-group-per-morphology unit; see ADR 0001)
  - [`training.md`](../vlearn-main/docs/api/training.md) — `EnvironmentGpu` RL base class
  - [`control.md`](../vlearn-main/docs/api/control.md) — motors / joint commands · [`sensors.md`](../vlearn-main/docs/api/sensors.md) — force sensors · [`gpu_arrays.md`](../vlearn-main/docs/api/gpu_arrays.md) — batched motor/sensor buffers
- **Built HTML** (browsable): `../vlearn-main/vlearn-docs/html/index.html`; type stubs in `../vlearn-main/docs/stubs/`.

Our local [`docs/reference/vsim_geometry_api.md`](./docs/reference/vsim_geometry_api.md) covers the geometry subset `build_vsim.py` uses.

## Shared Language

**Robot**:
The generic body being built and controlled — a **root** plus repeating **limbs**. "Robot" is the primary word in code and docs; **"ant" is reserved for env identity only** (the `Ant*` classes, env keys, `.vsim` assets, `ppo_ant*.yaml` configs, `train_ant_*.py` scripts). The ant is the current (only) robot instance. See [ADR-0014](docs/adr/0014-generalized-construction-vocabulary.md).

**Codesign**:
Jointly optimizing the robot's morphology and its transformer controller in one loop. **Implemented** as a single shared-trunk network — a **GenAct/GenCrit** morphology generator + **ContAct/ContCrit** controller — in `AntCodesignEnv`: the generator emits a body per resample window, the controller earns the reward that trains both. See the Control glossary's *Codesign heads / tokens*.

**Morphology** (morph):
A specific robot body — which limbs exist, where, and each limb's ordered modules. Emitted by the algorithm's generator; each maps to one vsim build / EnvironmentGroup. The Task **resamples** its set mid-training, one full sim rebuild per draw (see the Morphology glossary and [ADR-0005](docs/adr/0005-runtime-morphology-resampling-via-gym-rebuild.md)). "Morph" is an accepted shorthand.

**Limb** (was leg):
One repeating appendage — a chain of **modules** attached to the root. Up to 8 limbs, placed at multiples of 45° around the ant's root. Spans **1–4 modules per limb** (variable-length; ADR-0014 collapse). Adding/removing a limb — or a module — adds/removes tokens (the source of the architecture's count-invariance).
_Avoid_: leg (retired), "structural unit"/"part-token" (retired generic glosses)

**Module**:
The physical body-part a generator **token** realizes; one actuated module = one token = one DOF. Its **module type** (semantic kind) is an actuated **effector**, a passive **link**, or a terminal **cap** — only effector is built (types = Phase 5). The unit the generator minimizes (**Phase 8a, Limb Costs**). Module-token layout + the *token-role* vs *module-type* axes live in [Control](./transformer_rl/CONTEXT.md).
_Avoid_: segment (retired for module); conflating **module type** (effector/link/cap — semantic) with **token role** (root/start/module — structural; see Control)

**Root** (was torso):
The single non-repeating body token = the **CLS** aggregator; its encoder output feeds the value heads. The ant's root is its central torso body (the sole surviving use of "torso": a physical-build name).

**DOF**:
One actuated joint = one actuated module. **Up to 4 per limb, 32 max** (variable-length limbs, padded). The unit of action and of the DOF mask.
_Avoid_: joint (informal only)

**Active / Inactive**:
A limb or DOF is *active* if it exists in the current morphology, *inactive* if it's a padded-out slot (8 limbs / **32 DOFs** max). Inactive actions are zeroed; inactive DOF values are 0.

**Stable morphology**:
A morphology that is dynamically viable as a walker: **≥3 limbs**, **no circular gap between adjacent limbs > 135°**, and **≥2 limbs of length ≥2 modules** (a body standing only on 1-module stubs cannot walk). The admission rule for the **generalization suite**. The generator is *not* constrained to stable bodies (it has only a ≥1-limb guard) — stability filters what we *evaluate on*, not what it may *emit*.

**Warmup** (pretrain):
The first `n_pretrain` generator windows, in which the generator is trained by **supervised** behavior-cloning — *not* RL. Its imitation target is a **warmup teacher** draw; the entropy bonus does not apply here (it is an RL-only term), since BC is a max-likelihood fit and the teacher, not an entropy bonus, is the intended source of body diversity. GenCrit (V1.0) still fits the *built* body → R throughout, because R is measured on the body that actually ran. Warmup's job is to **install the generator's prior**; RL is the phase that moves off it.

**Warmup teacher**:
The sampler that draws bodies during warmup — it *defines the generator's post-warmup prior*, because the generator is behavior-cloned onto it. Two teachers exist (`generator.teacher`): **flip** (the seed body ± per-token noise — an edit-distance ball, so mass at radius *r* decays like `flip^r` and the cheap edits are degenerate: sprouting a limb costs one flip, a *useful-length* limb costs three) and **parts** (seed-relative parts-copy: each slot copies a limb template from the seed, then takes a per-limb length offset; unstable draws are rejection-resampled but kept with probability `prob_invalid`). The teacher shapes *geometry*; entropy supplies *pressure* — see the Phase-8b note in the escalation plan.

**DOF mask**:
The `{0,1}` vector marking which DOFs are active — **32-bit** (all obs offsets derive from the net's `tdims`, not hardcoded). Written once at allocation, constant per env. Read by the tokenizer and policy via a `> 0` test. Code identifier `dof_mask`.
_Avoid_: limb_mask (old code identifier; per-DOF not per-limb), bare "mask"

**EnvironmentGroup** (group):
vlearn's unit of one vsim build shared by a batch of envs. The repo uses one group per morphology, since real limb removal needs a distinct vsim per body.

**Task**:
A VSim scene plus the objective that scores it — defined by the [CoDesigner](../SoftwarePackage/CONTEXT.md) package, whose `Task` interface our envs are migrating onto. Task **absorbs** what this repo calls an env: it owns reward/termination, root pose, scene construction, and optionally the observation, over a package-owned backend (gym, groups, buffers, module layout). It owns neither the module library nor the initial body — both come from the Algorithm. Current tasks: **Locomotion** (`AntCodesignEnv`, forward velocity) and **Grasp** (`GraspCodesignEnv`); **knob-rotation** reserved. Class/key renames onto task names are deferred to Phase 7.
_Avoid_: env, environment (retiring — Task absorbs them)
