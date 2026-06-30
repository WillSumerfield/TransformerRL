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

**Dual-network PPG**:
The default PPG control agent: a **policy net** and a separate **value net** with disjoint weights (≈2× params). The value net does all RL value math; the policy net carries an **aux value head** trained only in the aux phase to distill value representations into its trunk. The `ppg_continuous` baseline.

**Single-network PPG** (shared trunk):
One network with a policy head and a value head on a **shared trunk**, recovering dual-net's 2× memory. Policy/value interference is managed by **gradient detach**, not by splitting the net: in the policy phase the value gradient is detached at the trunk (value head trains, trunk doesn't move from it); in the aux phase the value gradient flows through the whole net. No aux head — the value head *is* both critic and distill target. Selected by `ppg.shared_trunk`.
_Avoid_: "merged net" / "joint net" — say single-network or shared-trunk.

## Generation (morphology generator)

The control-side realization of the generative morphology policy (see [Codesign](../CONTEXT-MAP.md)). Vocabulary for the generator that emits a body and is trained by **policy-gradient** on the same reward the controller earns. Architecture/wiring/schedule details live in the Phase-3 plan + [ADR-0012](../docs/adr/0012-codesign-generator-sequential-token-ppg.md), not here.

**Morphology generator** (generator):
A **sequential, token-at-a-time** policy that emits the ant's designed morphology one slot at a time, trained by **policy-gradient** on the **control reward** — a body is good if the controller earns high return on it. It reuses `MultiMorphLegTransformer`; each leg slot is decided in **randomized order**, conditioned on the already-committed tokens. It is a **PPG** (policy head + aux value head on a shared trunk; the aux value is distilled per **phase** toward the real critic). Acts **once per resample window**; its body never enters the control's per-step action stream — it is applied to the env at the window's rebuild. Emits morphology, not per-DOF actions.
_Avoid_: conflating with the control **actor** (emits per-DOF actions). Also _avoid_ the retired framings: the **unconditional Bernoulli bandit** (ADR-0010, superseded) and the value-ascent generator.

**Generation MDP**:
The small MDP the generator solves: **state** = the committed token prefix, **action** = the next token, **reward** = the scalar body quality `R` paid **only at the terminal token**, γ=1. Slots are decided in random order. The generator is an actor-critic on this MDP.

**Token** (generator) — **limb token** / **stop token**:
The per-slot decision the generator emits: **limb** (the slot's leg exists) or **stop** (the slot is off). The body's **presence** is the set of limb decisions.

**Marginal-value advantage**:
A committed token's advantage = `v(prefix+token) − v(prefix)`, the token's marginal contribution to body quality. Because order is randomized and every prefix regresses to the same `R`, it is a Shapley-style estimate; being a **difference** of body-conditioned values, it avoids the body-agnostic-baseline trap (see ADR-0012).

**Control V1.0 head** (body-quality critic):
A second value head on the combined-PPO control net: γ=1, **truncation→0**, **time-aware** (reads a normalized time-remaining feature, V1.0-head-only). Trained on real returns and **isolated from the actor's advantages** (the actor uses V0.98). Produces the per-env body quality `R_i = V1.0(s0_i)` that the generator's aux value head distills.
_Avoid_: feeding the time feature into the shared obs (that would make the actor time-aware).

**Phase**:
One **resample window**, viewed as a PPG phase: the boundary at which the generator's aux value head is distilled toward the control V1.0 critic and the generator policy is updated.

**Resample window** (window):
The span of control episodes a single generated body set is held fixed (`resample_interval` episodes), bracketed by full gym rebuilds. One generator decision + one generator update per window (one **phase**).

**Base morph**:
The deterministic body the generator is warmed up around (`[1,4,6]` — a 3-leg ant). The pretrain phase centers the generated distribution on base ± small per-leg flip noise, then hands over to return-driven generation that climbs toward the optimum.

**Full token / Attribute token / Designed token** (nested views of a part-token):
- **Full token** — every feature the control policy consumes: physical state (pos, vel, sensors, last action) + leg encoding + morphology lengths. What the **leg transformer** reads.
- **Attribute token** — the morphology-defining subset only: **presence**, hip/ankle segment **length**, leg **angle**. The body properties that *could* be designed; excludes physical state. `Attribute ⊆ Full`.
- **Designed token** — the attributes a given codesign run actually generates and optimizes. The v1 ant designs **presence only**; segment lengths and angle stay fixed (deferred), so `Designed ⊊ Attribute` for v1. The framework permits any subset. `Designed ⊆ Attribute ⊆ Full`.

**Presence** (p):
A leg's existence, emitted by the generator as a per-slot **limb/stop token** decided in random order. The completed body is **built** for the resample window and its **discrete** presence is fed into obs. A **≥1-leg guard** masks the stop token on the forced last slot. (Retired: the per-leg Bernoulli-bandit and the continuous-`p` value-ascent framings.)
