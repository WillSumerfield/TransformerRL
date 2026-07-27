# Context Map

Research repo for **codesign**: jointly optimizing a robot's *morphology* (currently an ant) and a transformer *control* policy. The transformer must generalize across morphologies so the controller keeps working as the body changes during codesign. Attention is the bridge between the control policy and the generative morphology policy — both now realized as a single shared-trunk codesign network.

## Structure

```
/
├── CONTEXT-MAP.md                    ← this file; shared kernel below
├── docs/
│   ├── adr/                          ← system-wide decisions
│   ├── paper/                        ← related-work and paper-facing evidence
│   ├── transformer_architecture.md   (living doc: full obs→action flow, shapes, why)
│   ├── adaptive_ant_fixes.md
│   ├── group_count_throughput.md     (playbook: group-count-independent env throughput)
│   ├── morphology_resampling_cost.md (playbook: rebuild cost + resample cadence for the full ant)
│   └── vsim_geometry_api.md
├── envs/                             ← Morphology context
│   ├── CONTEXT.md
│   ├── multigroup_environment.py
│   └── ant_envs/
│       ├── ant.py                    (classic 4-limb ant)
│       ├── ant_multimorph.py         (multi-morphology base; full ant = all 131 stable)
│       ├── ant_codesign.py           (codesign env; generator-driven bodies per resample)
│       ├── build_vsim.py             (programmatic vsim per limb subset)
│       └── assets/
├── transformer_rl/                   ← Control context
│   ├── CONTEXT.md
│   ├── architectures.py              (LimbTransformer, MultiMorphLimbTransformer)
│   ├── tokenize.py                   (obs → root/module tokens; codesign uniform module tokens)
│   ├── models.py                     (rl_games model/network builders)
│   ├── codesign_agent.py
│   ├── codesign_player.py
│   ├── logging_agent.py
│   ├── policy_switch.py
│   └── train_utils.py
├── benchmarks/                       ← Benchmark context
│   ├── CONTEXT.md
│   ├── codesign.py                   (CoDesign + shared controller loading)
│   ├── fixed_body.py                 (fixed-base-morph control)
│   ├── evaluate.py                   (shared rollout/artifact flow)
│   ├── metrics.py
│   └── data.py
├── scripts/                          ← Training context
│   ├── CONTEXT.md
│   ├── train_ant_*.py
│   ├── benchmark_eval.py             (ADR-0016 method-agnostic evaluator)
│   ├── eval.py                       (legacy CoDesign evaluator)
│   ├── activate_uv.sh                (manual UV/VLearn activation)
│   └── tune.py                       (Optuna sweep)
├── configs/                          ← Training context (rl_games yaml)
│   ├── benchmarks/                   (shared benchmark protocol)
│   ├── ppo_ant*.yaml
│   └── tune_config.yaml
├── tests/                             ← Benchmark contract tests
├── experiments/                      ← Analysis context
│   ├── CONTEXT.md
│   └── attention_over_time.py
├── notebooks/                        ← Analysis context
├── data/                             ← Analysis context (figures, .npz)
├── logs/  runs/  videos/             (run artifacts, gitignored/untracked)
```

## Contexts

- [Morphology](./envs/CONTEXT.md) — the ant body design space: vsim physics builds, the morphology set, active/inactive DOFs, the DOF mask
- [Control](./transformer_rl/CONTEXT.md) — the transformer policy that controls any morphology: tokenization, limb encoding, token masking, rl_games integration
- [Benchmarks](./benchmarks/CONTEXT.md) — readable method modules and the shared native-pair evaluation, metric, seed, and artifact contract
- [Training](./scripts/CONTEXT.md) — PPO training, Optuna tuning, play/render orchestration
- [Analysis](./experiments/CONTEXT.md) — attention studies over trained policies

## Relationships

- **Morphology → Control**: Morphology emits the observation — **139-D baseline** (107 physical + 8 hip_lengths + 8 ankle_lengths + 16 DOF mask) or **219-D codesign** (variable-length `module_lengths` + 32 DOF mask, layout from `tdims`); Control tokenizes it and reads the DOF mask to decide which limb/module tokens exist.
- **Control → Training**: Control registers networks/models with rl_games under names Training selects via config `model.name` / `network.name`.
- **Control/Morphology → Benchmarks**: plainly named method modules load native controllers, sample typed bodies, and install them through the common VSim morphology interface; `benchmarks/evaluate.py` owns comparison semantics.
- **Training → Analysis**: Training produces checkpoints; Analysis loads them to collect attention.
- **Shared kernel** (below): Robot, Morphology, Limb, Module, Root, DOF, DOF mask, active/inactive, EnvironmentGroup, codesign, Task — defined once here, used identically across all contexts.

## Upstream — vlearn / VSim

This repo runs on the **vlearn** RL framework and the **VSim** GPU simulator, vendored as a sibling checkout at [`../vlearn-main`](../vlearn-main). Coupling: every env/script inserts `../vlearn-main/train` on `sys.path` and `import vlearn` (`vlearn.spaces`, `vlearn.torch_utils`); `AntMultiMorphEnv` and `AntEnv` subclass `EnvironmentGpu` (via `envs.ant_environment_gpu.AntEnvironmentGpu`).

Reference these when working on env physics or the group/motor/sensor plumbing:

- **Glossaries** (same CONTEXT.md style): [`../vlearn-main/train/CONTEXT.md`](../vlearn-main/train/CONTEXT.md) (VLearn framework — `EnvironmentGpu`) and [`../vlearn-main/docs/api/CONTEXT.md`](../vlearn-main/docs/api/CONTEXT.md) (VSim simulator).
- **API reference** (markdown): [`../vlearn-main/docs/api/index.md`](../vlearn-main/docs/api/index.md). Most-used here:
  - [`environments.md`](../vlearn-main/docs/api/environments.md) — `EnvironmentGroup`/`EnvironmentDef` (our one-group-per-morphology unit; see ADR 0001)
  - [`training.md`](../vlearn-main/docs/api/training.md) — `EnvironmentGpu` RL base class
  - [`control.md`](../vlearn-main/docs/api/control.md) — motors / joint commands · [`sensors.md`](../vlearn-main/docs/api/sensors.md) — force sensors · [`gpu_arrays.md`](../vlearn-main/docs/api/gpu_arrays.md) — batched motor/sensor buffers
- **Built HTML** (browsable): `../vlearn-main/vlearn-docs/html/index.html`; type stubs in `../vlearn-main/docs/stubs/`.

Our local [`docs/vsim_geometry_api.md`](./docs/vsim_geometry_api.md) covers the geometry subset `build_vsim.py` uses.

## Shared Language

**Robot**:
The generic body being built and controlled — a **root** plus repeating **limbs**. "Robot" is the primary word in code and docs; **"ant" is reserved for env identity only** (the `Ant*` classes, env keys, `.vsim` assets, `ppo_ant*.yaml` configs, `train_ant_*.py` scripts). The ant is the current (only) robot instance. See [ADR-0014](docs/adr/0014-generalized-construction-vocabulary.md).

**Codesign**:
Jointly optimizing the robot's morphology and its transformer controller in one loop. **Implemented** as a single shared-trunk network — a **GenAct/GenCrit** morphology generator + **ContAct/ContCrit** controller — in `AntCodesignEnv`: the generator emits a body per resample window, the controller earns the reward that trains both. See the Control glossary's *Codesign heads / tokens*.
_Avoid_: using "codesign" for the multi-morphology env `AntMultiMorphEnv` (it does no codesign — it trains a controller across a *fixed* morphology set; see Morphology context)

**Morphology** (morph):
A specific robot body — which limbs exist, where, and (full ant) each limb's module lengths (today two modules per limb: the ant's hip- and ankle-segment lengths). Either drawn from a fixed enumerated set (classic ant) or sampled with continuous lengths (full ant); each maps to one vsim build / EnvironmentGroup. The full ant **resamples** its set mid-training, one full sim rebuild per draw (see the Morphology glossary and [ADR-0005](docs/adr/0005-runtime-morphology-resampling-via-gym-rebuild.md)). "Morph" is an accepted shorthand.

**Limb** (was leg):
One repeating appendage — a chain of **modules** attached to the root. Up to 8 limbs, placed at multiples of 45° around the ant's root. The **codesign** env spans **1–4 modules per limb** (variable-length; ADR-0014 collapse); the classic/multimorph ant stays at exactly 2 actuated modules (**effectors**) → 2 DOFs. Adding/removing a limb — or a module — adds/removes tokens (the source of the architecture's count-invariance).
_Avoid_: leg (retired), "structural unit"/"part-token" (retired generic glosses)

**Module**:
The physical body-part a generator **token** realizes; one actuated module = one token = one DOF. Its **module type** (semantic kind) is an actuated **effector**, a passive **link**, or a terminal **cap** — only effector is built (types = Phase 5). The unit the generator minimizes (**Phase 8a, Limb Costs**). Module-token layout + the *token-role* vs *module-type* axes live in [Control](./transformer_rl/CONTEXT.md).
_Avoid_: segment (retired for module); conflating **module type** (effector/link/cap — semantic) with **token role** (root/start/module — structural; see Control)

**Root** (was torso):
The single non-repeating body token = the **CLS** aggregator; its encoder output feeds the value heads. The ant's root is its central torso body (the sole surviving use of "torso": a physical-build name).

**DOF**:
One actuated joint = one actuated module. Classic/multimorph ant: 2 per limb, 16 max. **Codesign: up to 4 per limb, 32 max** (variable-length limbs). The unit of action and of the DOF mask.
_Avoid_: joint (informal only)

**Active / Inactive**:
A limb or DOF is *active* if it exists in the current morphology, *inactive* if it's a padded-out slot (padded to 8 limbs / **16 DOFs baseline, 32 DOFs codesign**). Inactive actions are zeroed; inactive DOF values are 0.

**Stable morphology**:
A morphology that is dynamically viable as a walker: **≥3 limbs**, **no circular gap between adjacent limbs > 135°**, and **≥2 limbs of length ≥2 modules** (a body standing only on 1-module stubs cannot walk). The length clause is *vacuous* for the classic/multimorph ant — every limb there is exactly 2 modules — so the enumerated 131-morph set is unchanged by it; it only bites on the **variable-length** codesign morphologies (phase-1 onward), where it is the admission rule for the **generalization suite**. Presence-only enumeration: `_stable_morphologies()`. The generator is *not* constrained to stable bodies (it has only a ≥1-limb guard) — stability filters what we *evaluate on*, not what it may *emit*.

**Warmup** (pretrain):
The first `n_pretrain` generator windows, in which the generator is trained by **supervised** behavior-cloning — *not* RL. Its imitation target is a **warmup teacher** draw; the entropy bonus does not apply here (it is an RL-only term), since BC is a max-likelihood fit and the teacher, not an entropy bonus, is the intended source of body diversity. GenCrit (V1.0) still fits the *built* body → R throughout, because R is measured on the body that actually ran. Warmup's job is to **install the generator's prior**; RL is the phase that moves off it.

**Warmup teacher**:
The sampler that draws bodies during warmup — it *defines the generator's post-warmup prior*, because the generator is behavior-cloned onto it. Two teachers exist (`generator.teacher`): **flip** (the seed body ± per-token noise — an edit-distance ball, so mass at radius *r* decays like `flip^r` and the cheap edits are degenerate: sprouting a limb costs one flip, a *useful-length* limb costs three) and **parts** (seed-relative parts-copy: each slot copies a limb template from the seed, then takes a per-limb length offset; unstable draws are rejection-resampled but kept with probability `prob_invalid`). The teacher shapes *geometry*; entropy supplies *pressure* — see the Phase-8b note in the escalation plan.

**DOF mask**:
The `{0,1}` vector marking which DOFs are active — **16-bit at obs `[123:139]` (baseline); 32-bit in codesign** (all obs offsets derive from the net's `tdims`, not hardcoded). Written once at allocation, constant per env. Read by the tokenizer and policy via a `> 0` test. Code identifier `dof_mask`.
_Avoid_: limb_mask (old code identifier; per-DOF not per-limb), bare "mask"

**EnvironmentGroup** (group):
vlearn's unit of one vsim build shared by a batch of envs. The repo uses one group per morphology, since real limb removal needs a distinct vsim per body.

**Task**:
The objective/reward a robot is optimized for, independent of the robot. The current task is **Locomotion** (forward velocity); **cube-pickup** and **knob-rotation** are reserved (Phase 7). An env instantiates one Task on one robot. Env class/key renames by task are deferred to Phase 7.
