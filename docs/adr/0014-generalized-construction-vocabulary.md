---
status: accepted
---

# Generalized construction vocabulary: robot / limb / module / effector / root, ant is only the env identity

## Context & Decision

The repo's language and code were ant-specific: **ant**, **leg**, **hip**, **ankle**, **torso**.
Upcoming phases make all of these wrong — multi-segment limbs (Phase 1), typed modules (Phase 5:
Effector / Link / Cap / Connector), and non-locomotion, non-ant robots + explicit tasks (Phase 7).
We generalize the construction vocabulary now, before those phases bake the old terms deeper.

Resolved vocabulary (primary term ← what it replaces):

- **Robot** ← the generic body being built/controlled. **"ant" is reserved for env identity only**:
  the `Ant*` class names, env keys, `.vsim` assets, `ppo_ant*.yaml` configs, `train_ant_*.py`
  scripts. Inside code, a generic reference to "the ant" becomes "the robot".
- **Limb** ← **leg**. The repeating chain unit. "leg" is retired from vocabulary and code. The
  generic glosses **"structural unit"** and **"part-token"** are retired too (→ limb / token).
- **Token** / **Token type**. The generator emits tokens; each token has a **type**. Type names are
  **reserved now**, only some instantiated:
  - **Effector** — an actuated module (today's hip *and* ankle both become effectors, ordered by
    index outward from the root).
  - **Link** — a passive (non-actuated) module. Reserved, Phase 5.
  - **Cap** — a terminal token; **stop** = a morphology-less Cap. Reserved, Phase 5.
  - **Connector** — a semantic pre-marker before each link/effector; **start** = a special
    Connector. Reserved, Phase 5.
- **Module** ← the physical body-part a token realizes (Effector/Link/Cap → a module; Connector →
  none). The generator minimizes module count (Phase 3). **"Segment" is retired for "module".**
- **Root** ← **torso**. The single non-repeating body token = the **CLS** aggregator feeding the
  value heads. "torso" is retired everywhere including code.
- **Task** ← the objective/reward, robot-independent. The current task is **Locomotion**;
  cube-pickup and knob-rotation are reserved (Phase 7). An env = a Task on a robot.

**Unchanged (already generic):** DOF, morphology, codesign, active/inactive, stable morphology,
EnvironmentGroup, and the PPG / codesign-head vocabulary (ContAct / ContCrit / GenAct / GenCrit,
live/design mode, marginal-value advantage).

**Code rename scope in the accompanying commits:** `leg→limb` (`LegTransformer→LimbTransformer`,
`MultiMorphLegTransformer→MultiMorphLimbTransformer`, registration key
`multimorph_leg_transformer→multimorph_limb_transformer`, `n_legs→n_limbs`, leg encoding→limb
encoding, and the configs that reference the key), `torso→root` (token/state), `ant→robot`
(non-identity references), and the **transformer-facing** token layer hip/ankle→**effector** (token
+ type embedding `torso/hip/ankle → root/effector`).

**Deferred to Phase 1:** collapsing `Morphology.hip_lengths`/`ankle_lengths` (two `dict{leg→len}`)
into a variable-length **`module_lengths`**, and renaming the **physics-build identifiers** in
`build_vsim.py` (`aux_n`, `leg_n`, `hip_n`/`ankle_n` joints, motors, sensors) + the obs-scatter
indexing. These stay as ant physical-build joint names until Phase 1 makes limbs actually
multi-segment — that is the change that needs the variable-length data model, so it lands there.

## Considered Options

- **Neutral "segment + index" vocabulary** instead of the typed Effector/Link/Cap axis. Rejected:
  Phase 5 needs the typed axis regardless; a token's position in the limb already distinguishes
  stacked effectors, so a separate "segment" word is redundant — retired for "module".
- **Keep leg/torso as ant anatomy, generalize only the architecture layer** (dual generic/instance
  naming). Rejected: Phase 7's non-ant robot makes leg/torso wrong anyway; one word (limb/root)
  everywhere is simpler than maintaining a generic-vs-ant-instance split for each term.
- **Collapse the length dicts + rename physics-XML identifiers now.** Rejected: that is a Phase-1
  data-model change (variable-length limbs) with no current payoff and real bug risk (touches the
  `.vsim` build and obs scatter). Deferred to the phase that needs it.
- **Restructure env names by task now** (`ant-locomotion-codesign`). Rejected: premature before a
  second task exists to validate the scheme; deferred to Phase 7.

## Consequences

- Repo-wide identifier rename. Old registration keys (`multimorph_leg_transformer`) change, so
  configs referencing them are updated in the same commit; external configs/checkpoints keyed on the
  old names must update.
- Glossaries (`CONTEXT-MAP.md` + the four `CONTEXT.md`) are rewritten to the new vocabulary.
  hip/ankle survive only as **ant physical-build joint names**, noted as such.
- Effector/Link/Cap/Connector are **reserved** terms; the glossary marks the unbuilt ones "Phase 5,
  not built" so it never claims code that doesn't exist yet.
- **Phase 1 inherits** the `hip_lengths`/`ankle_lengths → module_lengths` collapse and the
  physics-XML rename as part of building multi-segment limbs.
