---
status: partially superseded
---

> **Root mounting has moved.** Clause 1 below (`root_axes` as a ModuleLibrary constructor param) is
> superseded: `root_axes` is now a **Task** property. Once the module library is supplied by the
> Algorithm rather than chosen by the env, a library instance can no longer encode the scene's
> mounting, and the Task must know its own action width before any library arrives. Clause 2 (the
> subtype vocabulary) stands. Full split recorded in
> [CoDesigner ADR-0001](../../../SoftwarePackage/docs/adr/0001-three-interface-split.md).

# ModuleLibrary: per-robot-family module vocabulary + root mounting, selected via config

## Context & Decision

`AntCodesignEnv` and `GraspCodesignEnv` (task_envs/ant_envs/ant_codesign.py,
task_envs/dexman_envs/grasp_codesign.py) duplicate an identical `_BASE_MORPHOLOGY`, and both build
off a root ("torso") that is a passive free-floating base — fine for a walker, wrong for the grasp
env's hand, which needs to be world-mounted with its own actuated movement DOFs (e.g. vertical +
lateral) and have those DOFs' position/velocity/last-action readable in the obs. Meanwhile every
per-subtype physical detail an effector/cap can have (joint axis, angle limits, geometry, mass,
proximal-cylinder-or-not — `build_vsim.py`'s `_ANKLE_AXIS`/`_effector_axis_limits`/`_CAP_GEOM`/
`_EFF_PROXIMAL_CYL`) and even *which subtypes exist* (`transformer_rl/vocab.py`'s `EFF_SWING`/
`EFF_KNEE`/`EFF_TWIST`/`CAP_BARE`/`CAP_FOOT`/`CAP_PAD`/`CAP_BALL`, `canonical_eff`, `CANON_CAP`) are
today global constants, documented in `transformer_rl/CONTEXT.md` as "the single source of truth
... shared by model/agent/builder." That stops being true once a second robot family (the grasp
hand) may need its own subtype vocabulary.

**`ModuleLibrary`** (abstract base, `task_envs/modular_libraries/`) is introduced to own exactly two
things:

1. **Root mounting + movement.** Constructor param `root_axes: list[Literal['x','y','z']] | None`:
   - `None` — free-floating base (today's ant behavior; unchanged).
   - `[]` — world-mounted, fixed (no root movement DOFs at all).
   - non-empty — world-mounted with one actuated prismatic joint per named axis (+ a range per
     axis), e.g. `['y', 'x']` for vertical + lateral hand movement.
   This is an **instance**-level choice (same concrete class, different `root_axes` per env).
   Root-axis position/velocity/last-action are read generically as extra fields on the **root
   token** itself (not as fake module slots in the existing 32-DOF padded system). `root_axes` is
   fixed per env/config, never resampled at runtime (unlike limb count), so the root block's width
   is computed ONCE at construction from the actual configured axes (`13 + 3*len(root_axes or [])`)
   -- no padding to a hypothetical max of 3, no mask field. Every downstream obs offset in
   `codesign_environment.py` becomes an instance attribute computed in `__init__`, not a module-level
   constant.

2. **Module subtype vocabulary.** Abstract methods/properties a concrete subclass **must** implement:
   effector subtype defs (axis, limits, shape, mass, proximal-cylinder-or-not) and cap subtype defs
   (geometry, mass), plus `canonical_eff(depth0)` and a canonical cap — i.e. everything `vocab.py`
   and `build_vsim.py` currently hardcode as globals for effector/cap *identity*, not just geometry.
   This is a **class**-level choice: which subtypes exist, and what each one physically is, belongs
   to the robot family, not the env instance.

**Explicitly NOT ModuleLibrary-owned** (stays generic/shared, at least for now):
- `base_morphology` (which limbs exist, at which positions) — stays per-env-class: each env class
  defines a `_BASE_MORPHOLOGY` default, now overridable via `env: base_morphology:` config (a
  `Morphology.from_design`-shaped dict), resolved in `run_training` right after `module_library`
  (requires an explicit `module_library` in config too — no silent cross-fallback). A ModuleLibrary
  does not decide limb layout, it's just the vocabulary the config's dict is built against.
- Root/torso link shape+size and the limb-attachment ring (8-slot radius/yaw) — stays a shared
  default in `build_vsim.py`. The hand keeps the sphere torso shape for now; only its mounting and
  movement change.
- Token **category** ids (`CAT_ROOT/CAT_START/CAT_EFFECTOR/CAT_CAP`) and the subtype one-hot
  **width** (`N_SUB=4`) — stay shared/global in `transformer_rl/vocab.py`. Only which subtypes
  *fill* that width, and what they mean, is per-ModuleLibrary.

**Same concrete class, different instances.** Since Ant and Grasp use identical subtypes today, one
concrete `ModuleLibrary` subclass (name TBD at implementation time) is instantiated twice:
`root_axes=None` for the ant, `root_axes=['y', 'x']` (or similar) for the hand. A second concrete
subclass is only needed once a robot family needs genuinely different subtypes.

**Config wiring.** `env:` gets a `module_library` (registry name) + `module_library_kwargs` block,
resolved once in `transformer_rl/train_utils.py::run_training` via a small name→class registry in
`task_envs/modular_libraries/__init__.py`:

```yaml
env:
  module_library: ant_default
  module_library_kwargs:
    root_axes: null
```

`run_training` builds the instance and passes `module_library=<instance>` into `env_class(**env_kwargs)`.
Per-ModuleLibrary subtype counts needed by the network/generator (constrained-decoder masking in
`architectures.py`, the BC teacher in `codesign_agent.py`) are threaded through the **existing**
manual-duplication pattern already used for `n_limbs`/`max_limb_length` (declared once under `env:`
for the env, once under `network.transformer:` for the model — these are already hand-kept-in-sync,
not auto-derived from a single source).

The two now-dead `env:` keys `sample_morphs`/`base_legs` (removed from configs this branch) are
confirmed unreferenced anywhere in current Python — that removal was legitimate cleanup, not
something this ADR restores.

## Considered Options

- **Subtype identity stays global, only geometry becomes per-ModuleLibrary.** Rejected: the user's
  stated intent is that "the children will be defining what modules are available" — a future robot
  family needs its own subtypes, not just its own numbers for the ant's subtypes. Accepted the
  larger ripple into `codesign_agent.py` / `tokenize.py` / `architectures.py`'s constrained decoder
  as the cost of that generality.
- **ModuleLibrary also owns `base_morphology`.** Rejected: limb layout is env-specific task design
  (how many fingers, where), not a property of the module vocabulary; keeping it on the env class
  avoids conflating "what modules can exist" with "which ones this body uses."
- **ModuleLibrary also owns root/torso shape + limb-ring geometry.** Rejected for now: no concrete
  need yet (hand keeps the sphere torso), and it's a separable axis of generality that can be added
  later without disturbing this ADR's scope.
- **Root axes reuse the effector-module slot machinery** (fake modules in the 32-slot padded
  system). Rejected: the root is already a distinct CLS token with its own state; extending it
  directly is simpler than routing through DOF-slot gather/scatter built for repeating limbs.
- **Dotted import path instead of a name registry for config selection.** Rejected: no other part of
  the repo selects classes from yaml by import path; a small registry matches the existing
  hardcoded-import-per-script convention more closely while still being config-selectable.

## Consequences

- `transformer_rl/vocab.py` loses `EFF_SWING/EFF_KNEE/EFF_TWIST/N_EFF`,
  `CAP_BARE/CAP_FOOT/CAP_PAD/CAP_BALL/N_CAP`, `EFF_NAMES/CAP_NAMES`, `canonical_eff`, `CANON_CAP` —
  these move to the concrete ModuleLibrary. It keeps `CAT_*`, `N_CAT`, `N_SUB`, `GEN_EFF/GEN_CAP/N_GEN_CAT`.
- `build_vsim.py` becomes a generic skeleton (link/joint/actuator XML assembly, DOF/sensor ordering)
  parameterized by whichever ModuleLibrary instance it's given for every per-subtype physical number.
- `codesign_agent.py` (BC teacher, canonical-morph logic) and `architectures.py` (constrained-decoder
  category/subtype masking) must read subtype counts/identity from the active ModuleLibrary rather
  than importing fixed constants.
- `tokenize.py`'s footprint is small (`CAP_BARE`, `N_SUB` only) — it slices by the shared structural
  width, not subtype identity, so it changes least.
- `CONTEXT-MAP.md` and `transformer_rl/CONTEXT.md`'s "Module type" paragraph need rewriting: the
  claim that `vocab.py` is *the* single source of truth for subtype ids is no longer true — it's the
  source of truth for category/width only, per-ModuleLibrary for identity.
