# Context Map

Research repo for **codesign**: jointly optimizing a robot's *morphology* (currently an ant) and a transformer *control* policy. The transformer must generalize across morphologies so the controller keeps working as the body changes during codesign. Attention is the bridge between the control policy and the generative morphology policy — both now realized as a single shared-trunk codesign network.

## Structure

```
/
├── CONTEXT-MAP.md                    ← this file; shared kernel below
├── docs/
│   ├── adr/                          ← system-wide decisions
│   ├── experiments/                  ← the paper's experiments (README.md = index + one doc per experiment)
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
├── transformer_rl/                   ← Control context
│   ├── CONTEXT.md
│   ├── architectures.py              (LimbTransformer, MultiMorphLimbTransformer)
│   ├── tokenize.py                   (obs → root/module tokens, off the Task's published layout)
│   ├── morphology.py                 (seed body, generator designs → Morphology, body sampling)
│   ├── runtime.py                    (the run's one ModuleLibrary + seed body + obs layout)
│   ├── models.py                     (rl_games model/network builders)
│   ├── algorithm.py                  (CodesignAlgorithm: the agent as a package Algorithm)
│   ├── artifacts.py                  (the trunk's two heads as ControlPolicy/MorphologyGenerator)
│   ├── rollout.py                    (test-mode rollout engine; ADR-0007)
│   ├── logging_agent.py
│   └── train_utils.py
├── scripts/                          ← Training context
│   ├── CONTEXT.md
│   ├── train_*.py
│   ├── optimize_codesign.py          (the same run, driven by codesigner.optimize)
│   ├── evaluate_codesign.py          (score a checkpoint on named bodies)
│   └── tune.py                       (Optuna sweep)
├── configs/                          ← Training context (rl_games yaml)
│   ├── ppo_*.yaml
│   └── tune_config.yaml
├── experiments/                      ← Analysis context
│   ├── CONTEXT.md
│   ├── harness/                      (shared measurement layer: diversity, committance, policy)
│   ├── joint_optimization/           (2D toy: designer/predictor coupling; standalone, no repo deps)
│   └── <slug>.py                     (one data-gathering script per paper experiment)
├── notebooks/                        ← Analysis context (one <slug>.ipynb per experiment)
├── data/                             ← Analysis context (figures, .npz)
├── logs/  runs/  videos/             (run artifacts, gitignored/untracked)
```

## Contexts

- [CoDesigner](../SoftwarePackage/CONTEXT.md) *(upstream package)* — Task / ModuleLibrary / Algorithm interfaces, Module, Orientation, Morphology, attachment slot
- [Control](./transformer_rl/CONTEXT.md) — the transformer policy that controls any morphology: tokenization, limb encoding, token masking, rl_games integration
- [Training](./scripts/CONTEXT.md) — PPO training, Optuna tuning, play/render orchestration
- [Analysis](./experiments/CONTEXT.md) — attention studies over trained policies

## Relationships

- **CoDesigner → Control**: the package's `Ant` Task emits the **893-D codesign** observation (variable-length `module_lengths` + 32 DOF mask) and **publishes its layout** via `Task.obs_layout()`; Control tokenizes off that layout and reads the DOF mask to decide which limb/module tokens exist. Nothing on this side re-derives where the blocks are.
- **Control → Training**: Control registers networks/models with rl_games under names Training selects via config `model.name` / `network.name`.
- **Training → Analysis**: Training produces checkpoints; Analysis loads them to collect attention.
- **Shared kernel** (below): Robot, Morphology, Limb, Module, Root, DOF, DOF mask, active/inactive, EnvironmentGroup, codesign, Task — defined once here, used identically across all contexts.

## Upstream — CoDesigner

The **CoDesigner** package ([`../SoftwarePackage`](../SoftwarePackage/CONTEXT.md), installed editable as `codesigner`) owns the `Task` / `ModuleLibrary` / `Algorithm` interfaces this repo implements, and its [CONTEXT.md](../SoftwarePackage/CONTEXT.md) is the source of truth for their vocabulary. **The tasks and module libraries now live there**; this repo is a consumer. Our codesign algorithm — the shared-trunk transformer + PPG agent — **stays here** and plugs in as one `Algorithm`.

A run names its library once, in the config's `env:` block — `module_library: simple` (3 effectors, 4 caps) or `basic` (the original ant's `swing`/`knee`/`bare` and nothing else, for runs that want no subtype choice to make). `run_training` constructs exactly one and hands it to the Task at `setup()` and to the network through `transformer_rl/runtime.py`, which exists because rl_games builds the network from a config dict and gives it no env to ask. The network reads both per-type vocabulary sizes off the library and never assumes either fills `subtype_width`.

**Two entry points, one agent.** `scripts/train_codesign_single.py` runs the agent directly under rl_games' `Runner` — the day-to-day path, with play, video and the follow camera. `scripts/optimize_codesign.py` runs the same agent under `codesigner.optimize`, which calls `CodesignAlgorithm.run()` once per **resample window** and fires a progress tick and a checkpoint after each. Both drive `LoggingA2CAgent._train_iter`, a single copy of rl_games' `train()` reshaped as a generator, so the two paths cannot drift. `scripts/evaluate_codesign.py` scores a checkpoint on bodies you name, rebuilding the library from the checkpoint's own provenance.

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
The generic body being built and controlled — a **root** plus repeating **limbs**. "Robot" is the primary word in code and docs; **"ant" is reserved for env identity only** (the `Ant*` classes, `.vsim` assets, and the `ppo_ant*.yaml` task-leaf configs). The ant is the current (only) robot instance. See [ADR-0014](docs/adr/0014-generalized-construction-vocabulary.md).

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
A VSim scene plus the objective that scores it — defined by the [CoDesigner](../SoftwarePackage/CONTEXT.md) package, whose `Task` interface our envs are migrating onto. Task **absorbs** what this repo calls an env: it owns reward/termination, root pose, scene construction, and optionally the observation, over a package-owned backend (gym, groups, buffers, module layout). It owns neither the module library nor the initial body — both come from the Algorithm. Six ship: **Ant** (locomotion on a plane, forward velocity) and five world-mounted manipulation tasks — **Grasp** (mounts and observes a target but scores nothing, the deliberate witness for the mounted path) plus the Adroit ports **Door**, **Hammer**, **Pen**, **Relocate**, whose hand is *also* a codesigned morphology, not a fixed Shadow Hand. A run names its task **once**, in the config's `env:` block (`task: ant`), resolved through the package's `tasks.REGISTRY` exactly as `module_library` is — a training script names an *algorithm*, never a task. Class/key renames onto task names are a Phase 9 loose end.
_Avoid_: env, environment (retiring — Task absorbs them); treating "ant" as a synonym for "task"

**Root axis**:
An actuated joint by which a Task **mounts its robot to the world** — Grasp's two prismatic approach axes, Door's four wrist DOFs, Relocate's six. A Task property, not a designed one: the generator never emits root axes, they are fixed per Task and constant per env (never padded, always active). Ranges 0–6 across the shipped tasks; `Ant` is free-floating and has **none**. They widen both the action space and the root observation block (`root_dim = 13 + 3·n_root_axes`, carrying pos/vel/last-action per axis), which is why the root token has always read them even though nothing acted on them. How they become actions: [Control](./transformer_rl/CONTEXT.md).
_Avoid_: calling them DOFs without qualification — a **DOF** in this repo is a module's actuated joint and the unit of the DOF mask; a root axis is neither masked nor designed.

**Checkpoint provenance**:
The facts a checkpoint carries so it can be read without the run that wrote it: the **module library** (registry key, construction args, and its ordered vocabulary), the **observation layout**, and the **task**, by the same registry key a config names it with. Each answers a different silent failure — a reordered vocabulary misreads every module token, a wrong layout misreads every offset, and a wrong task reads everything correctly while scoring a different question. The task key is the only one the layout cannot stand in for, because two tasks can agree on every offset. Verified on load, never merely recorded; a mismatch refuses rather than warns.
_Avoid_: treating it as resume state — provenance is what a *reader* needs, and an algorithm's own resume file is a separate, larger thing.

**Task observation field**:
An observation field an objective needs that a **body alone does not carry** — Grasp's target pose, Door's hinge angle, Hammer's nail position. Declared like any other field, by name and width, in the group whose repeat unit fits it (every one of them is **global** today); the package derives where it lands. There is no separate "extra" region and no scalar width: a task's fields sit in the global region beside the robot's own root state, and the policy takes that whole region as the root token's content, one opaque scene vector it does not decompose ([ADR-0019](docs/adr/0019-task-adaptation-on-the-root-token.md)).
_Avoid_: "extra block" / `extra_obs_width` — both named a region that no longer exists. Which fields a *task* contributed is not a distinction any reader acts on.
