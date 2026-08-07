# Morphology

The robot body design space and its physics: which robot bodies exist, how they're built in vsim, and how a body's active DOFs are exposed to the controller. Shared terms (robot, morphology, limb, module, DOF, DOF mask, active/inactive, root, stable morphology, EnvironmentGroup, codesign, task) live in the [Context Map](../CONTEXT-MAP.md).

Phase 1 mostly retired the pre-generalization names: it collapsed lengths into **`module_lengths`** and renamed the vsim links/joints (`aux_n`/`leg_n` → `mod_{n}_{i}`, `hip_n`/`ankle_n` → `joint_{n}_{d}`). `legs` / `hip_lengths` / `ankle_lengths` survive as **back-compat shims** on `Morphology`; the `"torso"` root-body lookup is the one physical-build name that stays — see [ADR-0014](../docs/adr/0014-generalized-construction-vocabulary.md).

## Language

**vsim**:
The GPU physics simulator this repo runs on (replaces an earlier MuJoCo stack). Bodies are described by vsim XML; one build per morphology.

**Morphology set**:
The list of morphologies a single env instance spans — one EnvironmentGroup per entry. *Generator-driven*: a fresh body set per resample window (see Morphology resampling, below), staged by the algorithm.

**Morphology** (Morphology dataclass):
Represents one specific body. Phase-1 data model = **`module_lengths`** (dict `limb → [len_pos1, …]`, one entry per module — variable length). Built via `from_counts({limb → k})` (default-length k-chain: pos1 hip-range, pos2+ ankle-range) or `bodies_from_counts(grid)` (torch-free per-env parse + ≥1-module / range guards); `from_legs` = the length-2 default. `legs` / `hip_lengths` / `ankle_lengths` survive as **back-compat shims** (ADR-0014).

**Effector module** (ant: swing / knee by depth):
An actuated module — one segment plus its joint (**one DOF**). A limb is a chain of **1–4** effector modules along a radial direction. By within-limb **depth**: depth 0 = **swing** (the old *hip*: aux link, root → first joint), depth 1+ = **knee** (the old *ankle*). The force/contact sensor rides the **terminal** module only. `hip`/`ankle`/`swing`/`knee` are ant-specific physical-build names.

**Module length**:
Length of one actuated module, fed into that module's token (`module_len`). Range by within-limb **depth**: depth 0 (swing/hip) HIP_RANGE `[0.14, 0.42]`, depth 1+ (knee/ankle) ANKLE_RANGE `[0.32, 0.95]` → a length-2 default chain reproduces the current ant exactly. Stored raw (RMS-normalized by the policy input normalizer); codesign packs these depth-major within the 893-D obs (see `_LEN_DIM` in `codesign_environment.py`).
_Avoid_: hip/ankle/aux segment, upper/lower leg (physical-build names only)

**Codesign ant** (codesign env):
`AntCodesignEnv` — the generator-driven env: a fresh body set per **resample window**, built from the generator's **module counts** (`set_next(counts)` → `bodies_from_counts`). **893-D** obs, **32-D** actions, up to 8 limbs × 4 modules; pre-`set_next` base = `[1,4,6]` @ count-2. The one env carrying the 32-DOF variable-length layout. Class `AntCodesignEnv`, env key `ant-codesign-env`.
_Avoid_: dynamic ant (legacy name from the old MuJoCo joint-masking approach); "full ant"/`AntMultiMorphEnv` (retired — that fixed-morphology-set env and `scripts/train_ant_full.py` were removed; `AntCodesignEnv` is the only multi-morphology ant env now, and it does codesign).

**Morphology resampling**:
Replacing the codesign env's sampled bodies with a fresh seeded draw partway through training, so the controller keeps meeting unseen morphologies. Because vsim bakes link geometry at finalize, it requires a full in-process sim rebuild (not in-place mutation), so it fires on a cadence rather than continuously. Reproducible from the run seed. Cost and cadence trade-off: [docs/guides/morphology_resampling_cost.md](../docs/guides/morphology_resampling_cost.md).

**ModuleLibrary** (`task_envs/modular_libraries/`; interface defined by [CoDesigner](../../SoftwarePackage/CONTEXT.md), [ADR-0016](../docs/adr/0016-modulelibrary-abstraction.md)):
Abstract base owning the **module subtype vocabulary** for a robot family (which effector/cap subtypes exist and their physical specifics — axis, limits, shape, mass) plus the **geometry** that realizes a morphology as a body (the ant's 8-slot ring placement, cap framing, torso). Concrete subclasses implement the subtype defs (class-level: fixed per robot family). Owns **no** scene or mounting concern: `root_axes` belongs to the Task, and `base_morphology` to the Algorithm. Selected in config under `env: module_library: <registry-name>` + `module_library_kwargs`.
_Avoid_: modlib (accepted shorthand in code only)

**root_axes**:
How the robot's root is fixed to the world — a **Task** property, since the scene decides whether the robot walks or is bolted above a table. `None` = free-floating base (the ant's unactuated 6-DOF root). `[]` = world-mounted, fixed. A non-empty subset of `x`/`y`/`z` = world-mounted with one actuated prismatic joint per named axis (e.g. `['y', 'x']` for the grasp hand's vertical + lateral movement). Read generically as extra fields on the root token (position/velocity/last-action per active axis) — not folded into the effector-module DOF system. Known before setup, so the Task's action width is fixed by it.

**Naming protocol** (Task ↔ ModuleLibrary):
The Task recovers each body's structure — slot/depth per DOF, per-module link and parent link, sensor-to-limb mapping, module lengths — from the names its ModuleLibrary emits. Rather than documenting a string convention and hoping, the package hands the library **emitters** keyed on `(slot, depth)`: they mint the names, impose declaration order, and record the structure the Task needs. Geometry stays library-owned — the emitters know names, not where a slot points. A DOF or sensor VSim reports that no emitter registered is warned about at setup.

