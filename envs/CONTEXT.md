# Morphology

The robot body design space and its physics: which robot bodies exist, how they're built in vsim, and how a body's active DOFs are exposed to the controller. Shared terms (robot, morphology, limb, module, DOF, DOF mask, active/inactive, root, stable morphology, EnvironmentGroup, codesign, task) live in the [Context Map](../CONTEXT-MAP.md).

Phase 1 mostly retired the pre-generalization names: it collapsed lengths into **`module_lengths`** and renamed the vsim links/joints (`aux_n`/`leg_n` → `mod_{n}_{i}`, `hip_n`/`ankle_n` → `joint_{n}_{d}`). `legs` / `hip_lengths` / `ankle_lengths` survive as **back-compat shims** on `Morphology`; the `"torso"` root-body lookup is the one physical-build name that stays — see [ADR-0014](../docs/adr/0014-generalized-construction-vocabulary.md). (These changes are on the **codesign build**; classic/multimorph ant paths are untouched.)

## Language

**vsim**:
The GPU physics simulator this repo runs on (replaces an earlier MuJoCo stack). Bodies are described by vsim XML; one build per morphology.

**Morphology set**:
The list of morphologies a single env instance spans — one EnvironmentGroup per entry. For the full ant it is *sampled*: one body per env (limb count uniform in 3–8, topology uniform within that count, per-limb module lengths uniform in range), reproducible from the run seed. For the classic ant and other fixed-topology envs it is the enumerated topologies at default lengths.

**Morphology** (Morphology dataclass):
Represents one specific body. Phase-1 data model = **`module_lengths`** (dict `limb → [len_pos1, …]`, one entry per module — variable length). Built via `from_counts({limb → k})` (default-length k-chain: pos1 hip-range, pos2+ ankle-range) or `bodies_from_counts(grid)` (torch-free per-env parse + ≥1-module / range guards); `from_legs` = the length-2 default. `legs` / `hip_lengths` / `ankle_lengths` survive as **back-compat shims** (ADR-0014).

**Effector module** (ant: swing / knee by depth):
An actuated module — one segment plus its joint (**one DOF**). A codesign limb is a chain of **1–4** effector modules along a radial direction; the classic/multimorph ant is exactly 2. By within-limb **depth**: depth 0 = **swing** (the old *hip*: aux link, root → first joint), depth 1+ = **knee** (the old *ankle*). The force/contact sensor rides the **terminal** module only. `hip`/`ankle`/`swing`/`knee` are ant-specific physical-build names.

**Module length**:
Length of one actuated module, fed into that module's token (`module_len`). Range by within-limb **depth**: depth 0 (swing/hip) HIP_RANGE `[0.14, 0.42]`, depth 1+ (knee/ankle) ANKLE_RANGE `[0.32, 0.95]` → a length-2 default chain reproduces the current ant exactly. Stored raw (RMS-normalized by the policy input normalizer); codesign packs these depth-major in the 219-D obs. (Baseline ant kept two separate obs blocks: **proximal-effector length** `hip_lengths` obs `[107:115]`, **distal-effector length** `ankle_lengths` obs `[115:123]`.)
_Avoid_: hip/ankle/aux segment, upper/lower leg (physical-build names only)

**Classic ant**:
`AntEnv` — the fixed 4-limb baseline (limbs at 45/135/225/315°). 59-D obs, 8-D actions. The parity target everything else is checked against.

**Multi-morphology ant**:
The base env that spans a morphology set; 139-D obs, 16-D actions, always padded to 8 limbs / 16 DOFs. Parameterized by its morphology set.
_Avoid_: calling `AntMultiMorphEnv` the "codesign ant" — it does no codesign (fixed morphology set); the codesign env is `AntCodesignEnv` (below). Class `AntMultiMorphEnv`, env key `ant-multimorph-env`.

**Codesign ant** (codesign env):
`AntCodesignEnv` — the generator-driven env: a fresh body set per **resample window**, built from the generator's **module counts** (`set_next(counts)` → `bodies_from_counts`). Phase 1: **219-D** obs, **32-D** actions, up to 8 limbs × 4 modules; pre-`set_next` base = `[1,4,6]` @ count-2. The one env carrying the 32-DOF variable-length layout (baseline / full ant untouched). Class `AntCodesignEnv`, env key `ant-codesign-env`.

**Full ant**:
The multi-morphology ant over a sampled morphology set — one variable-length body per env (N = num_envs). The hard variant. `scripts/train_ant_full.py` + `configs/ppo_ant_full.yaml` (`env.sample_morphs`); geometry source: `ant_8leg.vsim`. Seeded sampling controls reproducibility.
_Avoid_: dynamic ant (legacy name from the old MuJoCo joint-masking approach).

**Morphology resampling**:
Replacing the full ant's sampled bodies with a fresh seeded draw partway through training, so the controller keeps meeting unseen morphologies. Because vsim bakes link geometry at finalize, it requires a full in-process sim rebuild (not in-place mutation), so it fires on a cadence rather than continuously. Reproducible from the run seed. Cost and cadence trade-off: [docs/guides/morphology_resampling_cost.md](../docs/guides/morphology_resampling_cost.md).

