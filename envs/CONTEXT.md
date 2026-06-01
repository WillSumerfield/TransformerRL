# Morphology

The ant body design space and its physics: which ant bodies exist, how they're built in vsim, and how a body's active DOFs are exposed to the controller. Shared terms (morphology, leg, DOF, DOF mask, active/inactive, stable morphology, EnvironmentGroup, codesign) live in the [Context Map](../CONTEXT-MAP.md).

## Language

**vsim**:
The GPU physics simulator this repo runs on (replaces an earlier MuJoCo stack). Bodies are described by vsim XML; one build per morphology.

**Morphology set**:
The list of morphologies a single env instance spans — one EnvironmentGroup per entry. For the full ant it is *sampled*: one body per env (leg count uniform in 3–8, topology uniform within that count, per-leg hip/ankle lengths uniform in range), reproducible from the run seed. For the adaptive ant and other fixed-topology envs it is the enumerated topologies at default lengths.

**Morphology** (Morphology dataclass):
Represents one specific body: `legs` (frozenset of active leg indices 1–8), `hip_lengths` (dict leg→float, hip segment length per active leg), `ankle_lengths` (dict leg→float, ankle segment length per active leg). Replaces the bare `frozenset` used when lengths were fixed.

**Hip segment**:
The upper leg link (aux link) connecting the hip joint to the ankle joint. Length ranges [0.5×, 1.5×] the default (0.1414–0.4243). One scalar per leg, stored in `Morphology.hip_lengths`. In obs: `obs[107:115]`, one slot per leg slot (0 for inactive), stored raw (RMS-normalized by the policy input normalizer). Fed to the leg's hip token.
_Avoid_: upper leg, aux

**Ankle segment**:
The lower leg link connecting the ankle joint to the foot (force sensor). Length ranges [0.5×, 1.5×] the default (0.3163–0.9488). One scalar per leg, stored in `Morphology.ankle_lengths`. In obs: `obs[115:123]`, stored raw (RMS-normalized by the policy input normalizer). Fed to the leg's ankle token.
_Avoid_: lower leg, leg link

**Classic ant**:
`AntEnv` — the fixed 4-leg baseline (legs at 45/135/225/315°). 59-D obs, 8-D actions. The parity target everything else is checked against.

**Multi-morphology ant**:
The base env that spans a morphology set; 139-D obs, 16-D actions, always padded to 8 legs / 16 DOFs. Parameterized by its morphology set.
_Avoid_: codesign ant (the env does no codesign — "codesign" is reserved for the future loop). Class `AntMultiMorphEnv`, env key `ant-multimorph-env`.

**Full ant**:
The multi-morphology ant over a sampled morphology set — one variable-length body per env (N = num_envs). The hard variant. `scripts/train_ant_full.py` + `configs/ppo_ant_full.yaml` (`env.sample_morphs`); geometry source: `ant_8leg.vsim`. Seeded sampling controls reproducibility.
_Avoid_: dynamic ant (legacy name from the old MuJoCo joint-masking approach).

**Adaptive ant**:
`AntAdaptiveEnv` — the multi-morphology ant restricted to the 46 stable 3–4-leg morphologies. The easier curriculum subset of the full ant.

**Morphology resampling**:
Replacing the full ant's sampled bodies with a fresh seeded draw partway through training, so the controller keeps meeting unseen morphologies. Because vsim bakes link geometry at finalize, it requires a full in-process sim rebuild (not in-place mutation), so it fires on a cadence rather than continuously. Reproducible from the run seed. Cost and cadence trade-off: [docs/morphology_resampling_cost.md](../docs/morphology_resampling_cost.md).

**Morphology split**:
A stratified partition of a morphology set into *train morphologies* and *test morphologies*, derived deterministically from the config seed. Stratified by leg count — each leg-count stratum is split independently at `train_pct` so both halves see proportional coverage. Requires a seed; errors if `train_pct < 1.0` and no seed is set.
_Parameters_: `train_pct: float` (fraction kept for training, default `1.0` = no split), `test_set: bool` (use test morphologies instead of train).

**Train morphologies**:
The `floor(|stratum| * train_pct)` morphologies per leg-count stratum used for training when a split is active. The default (no split) is the full morphology set.

**Test morphologies**:
The held-out morphologies not seen during training. Selected via `--test-set` in `play` or `test` mode to evaluate out-of-distribution generalization. Not applied in `random` mode (that mode bypasses the split block entirely).
