# Control

The transformer policy that controls any morphology. It tokenizes the observation into per-body-part tokens, reads the DOF mask to know which legs exist, and emits an action per active DOF. Shared terms (leg, DOF, DOF mask, active/inactive, morphology) live in the [Context Map](../CONTEXT-MAP.md). Full input→output flow, shapes, and the rl_games wrapper: [docs/transformer_architecture.md](../docs/transformer_architecture.md).

The architecture is **env-agnostic by design**; the ant is its current (only) instance. Terms below are stated generically, with the ant instance noted.

## Language

**Part-token** (generic) / **torso·hip·ankle token** (ant instance):
One transformer input vector per body part. The general pattern: one **root token** for the body, plus one part-token per actuated segment of each repeating **structural unit**. Ant instance: a torso (root) token, plus a hip and ankle token per leg (1 + 2·n_legs total). See **Token** below for the ant specifics.

**Root token** (generic) / **torso token** (ant instance):
The single non-repeating body token. Always active, never masked, attends to all parts; its encoder output feeds the value head (CLS-style whole-body aggregator).

**Structural unit** (generic) / **leg** (ant instance):
The repeating body element that tokens are grouped by. One unit contributes one part-token per actuated segment (ant: a leg → hip + ankle). Adding/removing a unit adds/removes tokens — the source of the architecture's count-invariance.

**Leg transformer**:
The architecture: a transformer encoder over body-part tokens, shared across morphologies because legs are tokens rather than fixed input slots. `LegTransformer(n_legs)`. The 8-leg / 3-layer instance is its multi-morphology config.
_Avoid_: dynamic leg transformer. The 8-leg factory is `MultiMorphLegTransformer`; registration key `multimorph_leg_transformer`.

**Token**:
One transformer input vector for a body part: one **torso token**, plus a **hip token** and **ankle token** per leg (1 + 2·n_legs total). Inactive-leg tokens are zeroed and excluded from attention. Token dims include leg geometry: hip token is 6D (adds hip segment length), ankle token is 12D (adds ankle segment length).

**Leg encoding**:
The sin/cos of a leg's physical placement angle, concatenated into that leg's hip and ankle token features. Encodes *where* the leg is on the body; zeroed for inactive legs.
_Avoid_: positional embedding (that's the separate learned scheme below)

**Positional embedding**:
A learned `nn.Embedding` added to each token by slot index (leg number; a leg's hip and ankle share an index). Distinct from leg encoding — this is a learned per-slot vector, not body geometry.

**Type embedding**:
A learned embedding marking a token's kind — torso / hip / ankle (3 types).

**Token mask**:
The attention-level masking of inactive legs: their token embeddings are zeroed and they're set as padding keys (`src_key_padding_mask`) so active tokens never attend to them. Distinct from the DOF mask (the raw input vector it derives from).

**Masked-norm model**:
The rl_games model wrapper that runs the stock input normalizer but restores the raw `{0,1}` DOF mask afterward, so normalization can't collapse the constant mask. Registered as `transformer_masked_a2c_logstd`. (See `docs/adaptive_ant_fixes.md` for why.)

## Generation (morphology generator)

The control-side realization of the generative morphology policy (see [Codesign](../CONTEXT-MAP.md)). Vocabulary for the generator that emits a body and is trained by **policy-gradient** on the same reward the controller earns. Architecture/wiring/schedule details live in the Phase-2 plan + [ADR-0010](../docs/adr/0010-codesign-generator-unconditional-bandit.md), not here.

**Morphology generator** (generator):
An **unconditional** policy (no observation input) that emits the ant's designed morphology, trained by **policy-gradient** (not value-ascent) on the **control reward** — a body is good if the controller earns high return on it. v1 is a per-leg-**presence** bandit: **8 learnable Bernoulli logits + 1 scalar return baseline**, no trunk, no input. Acts **once per resample window**; its body never enters the control's per-step action stream — it is applied to the env at the window's rebuild. Emits morphology, not per-DOF actions.
_Avoid_: conflating with the control **actor** (emits per-DOF actions). Also _avoid_ the retired framings: a **state-conditioned transformer** generator, a **PPG aux value head**, or a shared **V1.0 critic head** — the unconditional bandit self-baselines with its own scalar (see ADR-0010). The state-conditioned ("next-best-token") generator is the deferred future design.

**Resample window** (window):
The span of control episodes a single generated body set is held fixed (`resample_interval` episodes), bracketed by full gym rebuilds. One generator decision + one generator update per window.

**Base morph**:
The deterministic body the generator is warmed up around (`[1,4,6]` — a 3-leg ant). The pretrain phase centers the generated distribution on base ± small per-leg flip noise, then hands over to return-driven generation that climbs toward the optimum.

**Full token / Attribute token / Designed token** (nested views of a part-token):
- **Full token** — every feature the control policy consumes: physical state (pos, vel, sensors, last action) + leg encoding + morphology lengths. What the **leg transformer** reads.
- **Attribute token** — the morphology-defining subset only: **presence**, hip/ankle segment **length**, leg **angle**. The body properties that *could* be designed; excludes physical state. `Attribute ⊆ Full`.
- **Designed token** — the attributes a given codesign run actually generates and optimizes. The v1 ant designs **presence only**; segment lengths and angle stay fixed (deferred), so `Designed ⊊ Attribute` for v1. The framework permits any subset. `Designed ⊆ Attribute ⊆ Full`.

**Presence** (p):
A leg's existence, emitted by the generator as a per-leg **Bernoulli action** (presence logits → sampled `{0,1}` body). The sampled body is **built** for the resample window and its **discrete** presence is fed into obs (no continuous/differentiable signal — the value-ascent design that needed continuous `p` is retired). The generator's policy-gradient log-prob is the sum of per-leg Bernoulli log-probs.
