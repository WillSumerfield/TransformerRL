# Control

The transformer policy that controls any robot. It tokenizes the observation into per-body-part tokens, reads the DOF mask to know which limbs exist, and emits an action per active DOF. Shared terms (robot, limb, module, DOF, DOF mask, active/inactive, root, morphology) live in the [Context Map](../CONTEXT-MAP.md). Full input→output flow, shapes, and the rl_games wrapper: [docs/transformer_architecture.md](../docs/transformer_architecture.md).

The architecture is **robot-agnostic by design**; the ant is its current (only) instance. Terms below are stated generically, with the ant instance noted.

## Language

**Token**:
One transformer input vector per body part. The general pattern: one **root token** for the body, plus one token per actuated module of each repeating **limb**. Ant instance: a root token, plus a proximal- and distal-**effector** token per limb (1 + 2·n_limbs total). Inactive-limb tokens are zeroed and excluded from attention. Effector token dims include limb geometry: the proximal-effector token is 6D (adds proximal length), the distal-effector token is 12D (adds distal length).
_Avoid_: part-token (retired generic gloss)

**Root token** (ant: the central torso body):
The single non-repeating body token. Always active, never masked, attends to all parts; its encoder output feeds the value head (CLS-style whole-body aggregator). Reused as the **CLS token** in codesign.
_Avoid_: torso (survives only as the ant's physical-build link name)

**Limb**:
The repeating body element that tokens are grouped by. One limb contributes one token per actuated module (ant: proximal + distal effector). Adding/removing a limb adds/removes tokens — the source of the architecture's count-invariance. In the generator a limb is one start token + its content tokens; a limb is *present* iff its start token has ≥1 committed content token, independent of module count — so a limb count never double-counts the two effectors. The env/physics **build** API keeps "leg" as an ant physical-build name (ADR-0014); the model/metric layer says "limb".
_Avoid_: leg, "structural unit" (retired)

**Limb transformer**:
The architecture: a transformer encoder over body-part tokens, shared across morphologies because limbs are tokens rather than fixed input slots. `LimbTransformer(n_limbs)`. The 8-limb / 3-layer instance is its multi-morphology config.
_Avoid_: leg transformer, dynamic leg transformer. The 8-limb factory is `MultiMorphLimbTransformer`; registration key `multimorph_limb_transformer`.

**Token types** (reserved — Phase 5, only Effector built):
The kind of module a token carries. **Effector** — an actuated module (today's two per limb). **Link** — a passive module (Phase 5). **Cap** — a terminal token; **stop** = a morphology-less Cap (Phase 5). **Connector** — a semantic pre-marker before each link/effector; **start** = a special Connector (Phase 5). Only Effector (and the implicit start/stop) exists before Phase 5; the rest are reserved vocabulary. See [ADR-0014](../docs/adr/0014-generalized-construction-vocabulary.md).

**Limb encoding**:
The sin/cos of a limb's physical placement angle, concatenated into that limb's effector token features. Encodes *where* the limb is on the body; zeroed for inactive limbs.
_Avoid_: leg encoding; positional embedding (that's the separate learned scheme below)

**Positional embedding**:
A learned `nn.Embedding` added to each token by slot index (limb number; a limb's two effector tokens share an index). Distinct from limb encoding — this is a learned per-slot vector, not body geometry.

**Type embedding**:
A learned embedding marking a token's kind — currently **root / effector** (the two built types). Reserved to extend to link / cap / connector at Phase 5.

**Token mask**:
The attention-level masking of inactive limbs: their token embeddings are zeroed and they're set as padding keys (`src_key_padding_mask`) so active tokens never attend to them. Distinct from the DOF mask (the raw input vector it derives from).

**Masked-norm model**:
The rl_games model wrapper that runs the stock input normalizer but restores the raw `{0,1}` DOF mask afterward, so normalization can't collapse the constant mask. Registered as `transformer_masked_a2c_logstd`. (See `docs/adaptive_ant_fixes.md` for why.)

**Dual-network PPG**:
The default PPG control agent: a **policy net** and a separate **value net** with disjoint weights (≈2× params). The value net does all RL value math; the policy net carries an **aux value head** trained only in the aux phase to distill value representations into its trunk. The `ppg_continuous` baseline.

**Single-network PPG** (shared trunk):
One network with a policy head and a value head on a **shared trunk**, recovering dual-net's 2× memory. Policy/value interference is managed by **gradient detach**, not by splitting the net: in the policy phase the value gradient is detached at the trunk (value head trains, trunk doesn't move from it); in the aux phase the value gradient flows through the whole net. No aux head — the value head *is* both critic and distill target. Selected by `ppg.shared_trunk`.
_Avoid_: "merged net" / "joint net" — say single-network or shared-trunk.

## Codesign heads (single-network)

Generalized roles, used when control and generator live on **one network** as four heads. _Actor_ = any head that emits **actions**; _critic_ = any head that **predicts value**. Both the control policy and the generator have an actor and a critic.

**ContAct** — control **actor**: emits per-DOF **actions** for the current body. Trained per rollout (PPO).
**ContCrit** — control **critic**: the **V0.98** value head driving ContAct's advantages (γ=0.98). Trained per rollout.
**GenAct** — generator **actor**: emits **limb/stop** morphology tokens (sequential, random order). Trained per **resample**.
**GenCrit** — generator **critic** = **the V1.0 body-quality head, merged**. One value head, evaluable on **live** full-state tokens *and* on **partial designed-token prefixes**. Yields the marginal-value advantage `V1.0(prefix+token) − V1.0(prefix)`. Trained per **resample** (needs true returns).

The merge: GenCrit and the old separate V1.0 head are now **one function**, not a distill pair. It's fit on two data sources toward the same body-quality target — rollout states (per-step return-to-go) and generation-token prefixes (toward the body's realized `R`). At resample the trunk learns **only** the generator side (GenAct + GenCrit/V1.0); **both** control heads are held by a clone term — **KL[ContAct_old, ContAct]** for the actor, **MSE(ContCrit, ContCrit_old)** for the critic. Per-step, control trains as **plain combined PPO** (trunk moves freely).

## Codesign tokens (single-network)

The merged net's limb-token vocabulary, spanning both reading **modes** (the net reads a real body vs a blueprint).

**Live token** / **live mode**:
A content token (effector) carrying **physical state** (pos/vel/sensors/last-action) — what the net reads during control rollout. _Reading the net in **live mode**_ = scoring a real, running body.

**Designed token** / **design mode**:
A content token carrying **morphology only** (module length + limb encoding), **no physical state** — what the net reads during the generation pass. _Reading the net in **design mode**_ = scoring a blueprint (possibly partial).

**Mode one-hot**:
A per-content-token field marking its mode: **live / committed / stop**. **committed** = an on (limb) slot in design mode; **stop** = a slot decided off; **live** = a real limb in rollout. Replaces overloading zeros (a zeroed token used to mean *inactive limb*, which collided with *designed* and *off*).

**Start token**:
A **persistent** per-slot anchor (one per limb slot), distinct from the content token. Two jobs: (1) in design mode it's where **GenAct** reads the on/stop decision for that slot; (2) it survives into **live** mode to tell **ContAct** which limb slots exist (the attachment point — basis for the deferred multi-module "generate-from an on-token" extension). It is **never replaced**; the slot's *content* token is what turns committed/live/stop. The start token is the pre-Phase-5 seed of the reserved **Connector** type.

**CLS token** (= the **root** token, reused):
Global-obs aggregator feeding the value heads (V0.98 + V1.0/GenCrit). `v(prefix)` is its design-mode readout over committed tokens (pending slots are simply **absent/masked**, not tokens). Its *content* differs by mode — root state embed in **live** mode, a learned `cls_design` parameter in **design** mode (generation has no root state) — but the same trunk and value heads read its output in both.

## Generation (morphology generator)

The control-side realization of the generative morphology policy (see [Codesign](../CONTEXT-MAP.md)). Vocabulary for the generator that emits a body and is trained by **policy-gradient** on the same reward the controller earns. Architecture/wiring/schedule details live in `temp/codesign_single_network_plan.md` and the **Codesign heads/tokens** sections above (the single-network design), not here. How to read the run's TensorBoard metrics + debug the algorithm from them: [`docs/codesign_metrics.md`](../docs/codesign_metrics.md).

**Morphology generator** (generator):
A **sequential, token-at-a-time** policy that emits the robot's designed morphology one slot at a time, trained by **policy-gradient** on the **control reward** — a body is good if the controller earns high return on it. It shares the **control trunk** (single network: GenAct + GenCrit heads alongside ContAct + ContCrit); each limb slot is decided in **randomized order**, conditioned on the already-committed tokens. Acts **once per resample window**; its body never enters the control's per-step action stream — it is applied to the env at the window's rebuild. Emits morphology, not per-DOF actions.
_Avoid_: conflating with the control **actor** (emits per-DOF actions). Also _avoid_ the retired framings: the **unconditional Bernoulli bandit** (ADR-0010), the value-ascent generator, and the **separate-net SeqGenerator** with a distill-pair aux value (ADR-0012 — superseded by the single-network merge).

**Generation MDP**:
The small MDP the generator solves: **state** = the committed token prefix, **action** = the next token, **reward** = the scalar body quality `R`, γ=1. GenCrit/V1.0 regresses **every** prefix toward `R`, and a token's advantage is the marginal difference of consecutive prefix values (below). Slots are decided in random order. The generator is an actor-critic (GenAct/GenCrit) on this MDP.

**Token** (generator) — **limb token** / **stop token**:
The per-slot decision the generator emits: **limb** (the slot's limb exists) or **stop** (the slot is off). The body's **presence** is the set of limb decisions. (stop is the pre-Phase-5 seed of the reserved **Cap** type.)

**Marginal-value advantage**:
A committed token's advantage = `v(prefix+token) − v(prefix)`, the token's marginal contribution to body quality. Because order is randomized and every prefix regresses to the same `R`, it is a Shapley-style estimate; being a **difference** of body-conditioned values, it avoids the body-agnostic-baseline trap (see ADR-0012).

**Body quality** `R_i`:
The generator's per-body reward: the body's **true mean completed-episode return** over the window (γ=1), accumulated in `env_step` — the *same scalar* for every state in the episode. (Retired: `R_i = V1.0(s0_i)`, the old separate time-aware V1.0-head estimate — superseded once GenCrit/V1.0 merged and the time feature was dropped; see **GenCrit** above.)

**Phase**:
One **resample window**: the boundary at which GenCrit/V1.0 is fit (rollout states + designed prefixes → `R_i`), GenAct is updated (PPO, or BC in pretrain), and **control is cloned** (β·KL + λ·MSE) so the shared-trunk update holds control still. (Not a PPG distill phase — there is no separate aux head.)

**Resample window** (window):
The span of control episodes a single generated body set is held fixed (`resample_interval` episodes), bracketed by full gym rebuilds. One generator decision + one generator update per window (one **phase**).

**Base morph**:
The deterministic body the generator is warmed up around (`[1,4,6]` — a 3-limb ant). The pretrain phase centers the generated distribution on base ± small per-limb flip noise, then hands over to return-driven generation that climbs toward the optimum.

**Full token / Attribute token / Designed token** (nested views of a token):
- **Full token** — every feature the control policy consumes: physical state (pos, vel, sensors, last action) + limb encoding + morphology lengths. What the **limb transformer** reads.
- **Attribute token** — the morphology-defining subset only: **presence**, module **length**, limb **angle**. The body properties that *could* be designed; excludes physical state. `Attribute ⊆ Full`.
- **Designed token** — the attributes a given codesign run actually generates and optimizes. The v1 ant designs **presence only**; module lengths and angle stay fixed (deferred), so `Designed ⊊ Attribute` for v1. The framework permits any subset. `Designed ⊆ Attribute ⊆ Full`.

**Presence** (p):
A limb's existence, emitted by the generator as a per-slot **limb/stop token** decided in random order. The completed body is **built** for the resample window and its **discrete** presence is fed into obs. A **≥1-limb guard** masks the stop token on the forced last slot. (Retired: the per-limb Bernoulli-bandit and the continuous-`p` value-ascent framings.)
