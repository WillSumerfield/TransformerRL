# Context Map

Research repo for **codesign**: jointly optimizing a robot's *morphology* (currently an ant) and a transformer *control* policy. The transformer must generalize across morphologies so the controller keeps working as the body changes during codesign. Attention is the intended future bridge between the control policy and a (planned) generative morphology policy.

## Structure

```
/
├── CONTEXT-MAP.md                    ← this file; shared kernel below
├── docs/
│   ├── adr/                          ← system-wide decisions
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
│   ├── tokenize.py                   (obs → root/effector tokens)
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
│   └── attention_over_time.py
├── notebooks/                        ← Analysis context
├── data/                             ← Analysis context (figures, .npz)
├── logs/  runs/  videos/             (run artifacts, gitignored/untracked)
```

## Contexts

- [Morphology](./envs/CONTEXT.md) — the ant body design space: vsim physics builds, the morphology set, active/inactive DOFs, the DOF mask
- [Control](./transformer_rl/CONTEXT.md) — the transformer policy that controls any morphology: tokenization, leg encoding, token masking, rl_games integration
- [Training](./scripts/CONTEXT.md) — PPO training, Optuna tuning, play/render orchestration
- [Analysis](./experiments/CONTEXT.md) — attention studies over trained policies

## Relationships

- **Morphology → Control**: Morphology emits a 139-D observation (107 physical + 8 hip_lengths + 8 ankle_lengths + 16 DOF mask); Control tokenizes it and reads the DOF mask to decide which limb tokens exist.
- **Control → Training**: Control registers networks/models with rl_games under names Training selects via config `model.name` / `network.name`.
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
Jointly optimizing the robot's morphology and its transformer controller in one loop. The repo's end goal — **not yet implemented**; a generative morphology policy is planned to pair with the control policy. Reserve this word for that future loop; the present envs only *train a controller to generalize across* a fixed morphology set, which is the prerequisite.
_Avoid_: using "codesign" for the multi-morphology env `AntMultiMorphEnv` (it does no codesign; see Morphology context)

**Morphology** (morph):
A specific robot body — which limbs exist, where, and (full ant) each limb's module lengths (today two modules per limb: the ant's hip- and ankle-segment lengths). Either drawn from a fixed enumerated set (classic ant) or sampled with continuous lengths (full ant); each maps to one vsim build / EnvironmentGroup. The full ant **resamples** its set mid-training, one full sim rebuild per draw (see the Morphology glossary and [ADR-0005](docs/adr/0005-runtime-morphology-resampling-via-gym-rebuild.md)). "Morph" is an accepted shorthand.

**Limb** (was leg):
One repeating appendage — a chain of **modules** attached to the root. Up to 8 limbs, placed at multiples of 45° around the ant's root. Today each limb has exactly 2 actuated modules (**effectors**) → 2 DOFs. Adding/removing a limb adds/removes tokens (the source of the architecture's count-invariance).
_Avoid_: leg (retired), "structural unit"/"part-token" (retired generic glosses)

**Module**:
The physical body-part a generator **token** realizes: an actuated segment (**effector**), a passive segment (**link**), or a terminal (**cap**). The unit the generator minimizes (Phase 3). Detailed token-type vocabulary lives in [Control](./transformer_rl/CONTEXT.md).
_Avoid_: segment (retired for module)

**Root** (was torso):
The single non-repeating body token = the **CLS** aggregator; its encoder output feeds the value heads. The ant's root is its central torso body (the sole surviving use of "torso": a physical-build name).

**DOF**:
One actuated joint. 2 per limb today (the ant's two effectors), 16 max. The unit of action and of the DOF mask.
_Avoid_: joint (informal only)

**Active / Inactive**:
A limb or DOF is *active* if it exists in the current morphology, *inactive* if it's a padded-out slot (always padded to 8 limbs / 16 DOFs). Inactive actions are zeroed; inactive DOF values are 0.

**Stable morphology**:
A morphology that is dynamically viable as a walker: ≥3 limbs and no circular gap between adjacent limbs > 135°.

**DOF mask**:
The 16-bit `{0,1}` vector (obs `[123:139]`) marking which DOFs are active. Written once at allocation, constant per env. Read by the tokenizer and policy via a `> 0` test. Code identifier `dof_mask`.
_Avoid_: limb_mask (old code identifier; per-DOF not per-limb), bare "mask"

**EnvironmentGroup** (group):
vlearn's unit of one vsim build shared by a batch of envs. The repo uses one group per morphology, since real limb removal needs a distinct vsim per body.

**Task**:
The objective/reward a robot is optimized for, independent of the robot. The current task is **Locomotion** (forward velocity); **cube-pickup** and **knob-rotation** are reserved (Phase 7). An env instantiates one Task on one robot. Env class/key renames by task are deferred to Phase 7.
