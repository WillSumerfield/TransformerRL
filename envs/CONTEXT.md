# Morphology

The ant body design space and its physics: which ant bodies exist, how they're built in vsim, and how a body's active DOFs are exposed to the controller. Shared terms (morphology, leg, DOF, DOF mask, active/inactive, stable morphology, EnvironmentGroup, codesign) live in the [Context Map](../CONTEXT-MAP.md).

## Language

**vsim**:
The GPU physics simulator this repo runs on (replaces an earlier MuJoCo stack). Bodies are described by vsim XML; one build per morphology.

**Morphology set**:
The enumerated list of morphologies a single env instance spans — one EnvironmentGroup per entry. Choosing the set defines the variant (adaptive vs full).

**Classic ant**:
`AntEnv` — the fixed 4-leg baseline (legs at 45/135/225/315°). 59-D obs, 8-D actions. The parity target everything else is checked against.

**Multi-morphology ant**:
The base env that spans a morphology set; 123-D obs, 16-D actions, always padded to 8 legs / 16 DOFs. Parameterized by its morphology set.
_Avoid_: codesign ant (the env does no codesign — "codesign" is reserved for the future loop). Class `AntMultiMorphEnv`, env key `ant-multimorph-env`.

**Full ant**:
The multi-morphology ant over the full set of all 131 stable morphologies. The hard variant; what the `_p1` run trains.
_Avoid_: dynamic ant (legacy name from the old MuJoCo joint-masking approach). This run is `scripts/train_ant_full.py` + `configs/ppo_ant_full.yaml`; the 8-leg source-geometry asset is `ant_8leg.vsim`.

**Adaptive ant**:
`AntAdaptiveEnv` — the multi-morphology ant restricted to the 46 stable 3–4-leg morphologies. The easier curriculum subset of the full ant.
