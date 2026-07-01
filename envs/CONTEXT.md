# Morphology

The robot body design space and its physics: which robot bodies exist, how they're built in vsim, and how a body's active DOFs are exposed to the controller. Shared terms (robot, morphology, limb, module, DOF, DOF mask, active/inactive, root, stable morphology, EnvironmentGroup, codesign, task) live in the [Context Map](../CONTEXT-MAP.md).

The physics build is the one place the pre-generalization names survive: `build_vsim.py` and the `Morphology` dataclass fields (`legs`, `hip_lengths`, `ankle_lengths`) plus the vsim link/joint names (`torso`, `aux_n`, `leg_n`, `hip_n`, `ankle_n`) stay until Phase 1 collapses lengths into `module_lengths` and renames them — see [ADR-0014](../docs/adr/0014-generalized-construction-vocabulary.md).

## Language

**vsim**:
The GPU physics simulator this repo runs on (replaces an earlier MuJoCo stack). Bodies are described by vsim XML; one build per morphology.

**Morphology set**:
The list of morphologies a single env instance spans — one EnvironmentGroup per entry. For the full ant it is *sampled*: one body per env (limb count uniform in 3–8, topology uniform within that count, per-limb module lengths uniform in range), reproducible from the run seed. For the classic ant and other fixed-topology envs it is the enumerated topologies at default lengths.

**Morphology** (Morphology dataclass):
Represents one specific body: `legs` (frozenset of active limb indices 1–8), `hip_lengths` (dict limb→float, proximal-effector segment length per active limb), `ankle_lengths` (dict limb→float, distal-effector segment length per active limb). The `legs`/`hip_lengths`/`ankle_lengths` field names and the fixed two-module structure are the deferred data-model: Phase 1 collapses them into a variable-length `module_lengths` (ADR-0014).

**Effector module** (ant: hip / ankle):
An actuated module — one segment plus its joint. Today's ant limb is exactly 2 effectors: a **proximal** effector (the old *hip*: aux link, from the root to the middle joint) and a **distal** effector (the old *ankle*: from the middle joint to the foot/force sensor). "hip"/"ankle" survive only as the ant's physical-build joint/link names.

**Proximal-effector length** (obs `[107:115]`, ant `hip_lengths`):
Length of a limb's proximal effector. Range [0.5×, 1.5×] default (0.1414–0.4243). One scalar per limb slot (0 for inactive), stored raw (RMS-normalized by the policy input normalizer). Fed to that effector's token.
_Avoid_: hip segment, upper leg, aux (physical-build names only)

**Distal-effector length** (obs `[115:123]`, ant `ankle_lengths`):
Length of a limb's distal effector. Range [0.5×, 1.5×] default (0.3163–0.9488). One scalar per limb slot, stored raw. Fed to that effector's token.
_Avoid_: ankle segment, lower leg, leg link (physical-build names only)

**Classic ant**:
`AntEnv` — the fixed 4-limb baseline (limbs at 45/135/225/315°). 59-D obs, 8-D actions. The parity target everything else is checked against.

**Multi-morphology ant**:
The base env that spans a morphology set; 139-D obs, 16-D actions, always padded to 8 limbs / 16 DOFs. Parameterized by its morphology set.
_Avoid_: codesign ant (the env does no codesign — "codesign" is reserved for the future loop). Class `AntMultiMorphEnv`, env key `ant-multimorph-env`.

**Full ant**:
The multi-morphology ant over a sampled morphology set — one variable-length body per env (N = num_envs). The hard variant. `scripts/train_ant_full.py` + `configs/ppo_ant_full.yaml` (`env.sample_morphs`); geometry source: `ant_8leg.vsim`. Seeded sampling controls reproducibility.
_Avoid_: dynamic ant (legacy name from the old MuJoCo joint-masking approach).

**Morphology resampling**:
Replacing the full ant's sampled bodies with a fresh seeded draw partway through training, so the controller keeps meeting unseen morphologies. Because vsim bakes link geometry at finalize, it requires a full in-process sim rebuild (not in-place mutation), so it fires on a cadence rather than continuously. Reproducible from the run seed. Cost and cadence trade-off: [docs/morphology_resampling_cost.md](../docs/morphology_resampling_cost.md).

**Morphology split**:
A stratified partition of a morphology set into *train morphologies* and *test morphologies*, derived deterministically from the config seed. Stratified by limb count — each limb-count stratum is split independently at `train_pct` so both halves see proportional coverage. Requires a seed; errors if `train_pct < 1.0` and no seed is set.
_Parameters_: `train_pct: float` (fraction kept for training, default `1.0` = no split), `test_set: bool` (use test morphologies instead of train).

**Train morphologies**:
The `floor(|stratum| * train_pct)` morphologies per limb-count stratum used for training when a split is active. The default (no split) is the full morphology set.

**Test morphologies**:
The held-out morphologies not seen during training. Selected via `--test-set` in `play` or `test` mode to evaluate out-of-distribution generalization. Not applied in `random` mode (that mode bypasses the split block entirely).
