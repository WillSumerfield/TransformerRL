# Control

The transformer policy that controls any robot. It tokenizes the observation into per-body-part tokens, reads the DOF mask to know which limbs exist, and emits an action per active DOF. Shared terms (robot, limb, module, DOF, DOF mask, active/inactive, root, morphology) live in the [Context Map](../CONTEXT-MAP.md). Full input→output flow, shapes, and the rl_games wrapper: [docs/reference/transformer_architecture.md](../docs/reference/transformer_architecture.md).

The architecture is **robot-agnostic by design**; the ant is its current (only) instance. Terms below are stated generically, with the ant instance noted.

## Language

**Token**:
One transformer input vector per body part. The general pattern: one **root token** for the body, plus one **module token** per actuated module of each repeating **limb**. Codesign (Phase 1, variable-length): a root/CLS token + `n_limbs` **start tokens** + up to `max_limb_length` module tokens per limb = **41 tokens** at 8 limbs × max_len 4 (`1 + 8 + 32`). Inactive module slots become STOP tokens (state zeroed); inactive-limb tokens are excluded from attention. (Baseline ant kept the older split: a proximal-effector 6D + distal-effector 12D token per limb, `1 + 2·n_limbs` total.)
_Avoid_: part-token (retired); proximal/distal-**effector token** (superseded by the uniform module token in codesign)

**Module token**:
The codesign content token — one per actuated module, a uniform **12-D** vector `[pos, vel, last_action, sin, cos, module_len, cfrc(6)]` (`MODULE_DIM`, `tokenize_modules`). Contact (`cfrc`) rides **only** on a limb's **terminal** module. Same schema at every within-limb depth (no proximal/distal split) — depth is carried by the separate **depth embedding**, not by the token dims. Distinct from the generator's **grow/stop decision** (below).

**Root token** (ant: the central torso body):
The single non-repeating body token. Always active, never masked, attends to all parts; its encoder output feeds the value head (CLS-style whole-body aggregator). Reused as the **CLS token** in codesign. It is also where **both** task-varying quantities live — the whole **global region** of the obs enters its content, [task observation fields](../CONTEXT-MAP.md) and all, and the [root-axis actions](#) leave from its output — so the root token, not the module tokens, is what makes the policy task-general.
_Avoid_: torso (survives only as the ant's physical-build link name)

**Root-axis head**:
The policy output for a Task's [root axes](../CONTEXT-MAP.md) — a multi-output head on the **root token**, whose values are concatenated **after** the module actions to match the env's DOF order (`[padded module DOFs] ++ [root axes]`). Deliberately **not** one token per root axis: keeping them off the token stream is what leaves the generator's design MDP defined over module slots alone, since a root axis is fixed by the Task and never designed. Unmasked (root axes are always active), live-mode only, and invisible to the FD/FK aux heads. **Not constructed at all** when a Task has no root axes, which is what keeps the Ant network parameter-identical to its pre-task-general self. See [ADR-0019](../docs/adr/0019-task-adaptation-on-the-root-token.md).
_Avoid_: "wrist head" (task-specific — the same head serves Grasp's prismatic mount and Ant's absence of one)

**Limb**:
The repeating body element that tokens are grouped by. One limb contributes one token per actuated module (ant: proximal + distal effector). Adding/removing a limb adds/removes tokens — the source of the architecture's count-invariance. In the generator a limb is one start token + its content tokens; a limb is *present* iff its start token has ≥1 committed content token, independent of module count — so a limb count never double-counts the two effectors. The env/physics **build** API keeps "leg" as an ant physical-build name (ADR-0014); the model/metric layer says "limb".
_Avoid_: leg, "structural unit" (retired)

**Limb transformer**:
The architecture: a transformer encoder over body-part tokens, shared across morphologies because limbs are tokens rather than fixed input slots. `LimbTransformer(n_limbs)`. The 8-limb / 3-layer instance is its multi-morphology config.
_Avoid_: leg transformer, dynamic leg transformer. The 8-limb factory is `MultiMorphLimbTransformer`; registration key `multimorph_limb_transformer`.

**Token role** (built — the `type_emb` axis):
A token's **structural** slot in the sequence: **root** (the CLS aggregator), **start** (a limb's persistent anchor), or **module** (an actuated-module content token). The 3-row `type_emb` (`architectures.py`). Distinct from **module type** below; Phase 5 refines the `module` role into module types.

**Module type** (Effector + Cap **built** in 5a stages 1+2, grill 2026-07-20; Connector/Link still reserved):
The **semantic** kind a module token carries — now a **per-module designed field** the generator emits (GenAct action ≡ *"emit a module of type T"*). The vocabulary mirrors a **real modular-robot hardware library** (the eventual real task assembles bodies from such physical modules — that grounding is the *why*; the Phase-5 types are generic **placeholders** until the real catalog lands). **Effector** — an actuated (single-DOF) module (a pivot/motor); its **type** = a generator-chosen **joint kind**, defined position-independently as a **local joint axis + limits + fixed default length** (the build rotates the local axis into world per limb slot). Replaces today's depth-forced swing/knee assignment. **Cap** — a **terminal** module, realized as the **type field on the terminal (`depth=count`) slot** and forced there by the grammar at the deepest slot: **bare** = a zero-morphology cap that just ends a limb (== the old `stop`, contact stays on the last effector), vs **morphology** caps (feet/pads/…) built as a **passive terminal body** carrying the terminal contact. Caps stay **passive — no actions, no DOF** (present-but-actionless live token, so effector↔DOF 1:1 is preserved). **Connector** (Phase 5b, **designed 2026-07-21, not built**) — the **interface between modules**, and itself **a module**: not a marker or a field on another token, but an ordinary occupant of a depth slot that the generator emits in its **own step** and that therefore earns its **own marginal-value advantage**. Its type is **primarily a geometric mounting offset** — a rigid re-posing of the child relative to the parent (`offset` = pure sideways translation, `roll` = rotation about the limb's own axis, which leaves the centerline alone but re-aims every downstream joint axis, `bend` = an angular kink that redirects the limb) — plus a **compatibility mask** (which module type may attach next). One connector sits at **every** junction (**mandatory**, no identity/null kind, no connector→connector chaining), which is why bodies from 5b onward are **not comparable** to earlier phases. **Passive — no DOF, no action**, so effector↔DOF stays 1:1. **Bodyless in 5b** (a massless bracket folded into the child's mount at build); becomes an **independent physical module** with mass/collision in **5c** — superseding ADR-0014's `Connector → none`. **start** = a special (base) Connector, and it still serves that role, so a limb's depth-0 module is an effector rather than a designed connector. **Link** — a passive extension module (no motor); **deferred out of Phase 5** to a later **"Links & Parameterized Modules"** phase (which also makes per-module params — link length, motor range — generator-designed). 5a stages 1+2 **built** Effector + Cap as typed, generator-chosen fields; **Connector** (5b stages 3–4) and **Link** (Phase 10) remain reserved. `transformer_rl/vocab.py` is the single source of truth only for the **structural** axis — the category ids (`CAT_ROOT/CAT_START/CAT_EFFECTOR/CAT_CAP`). The subtype one-hot **width** is not there: it is `ModuleLibrary.subtype_width`, published as `Task.obs_layout()["n_sub"]` and carried as `net.n_sub`, because the obs subtype block is sized by the library and a constant beside it could only disagree. *Which* subtypes exist and what each one physically is (identity, not just geometry) is **per-ModuleLibrary** since [ADR-0016](../docs/adr/0016-modulelibrary-abstraction.md) — a ModuleLibrary's "subtype 1 effector" need not mean the same thing as another's. Cap *shapes* are **provisional placeholders** pending the deferred geometry grill. See [ADR-0014](../docs/adr/0014-generalized-construction-vocabulary.md).
_Avoid_: conflating **module type** (effector/link/cap — semantic) with **token role** (root/start/module — structural).

**Limb encoding**:
The sin/cos of a limb's physical placement angle, concatenated into that limb's effector token features. Encodes *where* the limb is on the body; zeroed for inactive limbs.
_Avoid_: leg encoding; positional embedding (that's the separate learned scheme below)

**Positional embedding** (`pos_emb`):
A learned `nn.Embedding` added to each token by **limb-slot** index (limb number; all of a limb's module tokens share the one slot index). A learned per-slot vector. Distinct from **limb encoding** (body geometry) and from **depth embedding** (within-limb depth).

**Depth embedding** (`depth_emb`):
A learned `nn.Embedding` added to each module token by its **within-limb depth** (`0..max_limb_length−1`) — *which* module along the chain. New in Phase 1; disambiguates the up-to-4 module tokens that share a limb slot. Ant instance: depth 0 = **swing** joint (hip axis), depth 1+ = **knee** (ankle axis) — *swing/knee are ant-specific*; the general concept is just depth. Phase 3a reserves a rotary alternative — see **Depth rotary embedding** below.

**Type embedding** (`type_emb` / `type_oh`):
Marks a token's **category** — pre-Phase-5 the 3 rows **root / start / module** (see *Token role*). Phase 5 splits `module` into **effector / cap** (category one-hot `{root, start, effector, cap}`, `N_CAT` 3→4; 5b adds **connector** as a fifth, `N_CAT` 4→5, with the subtype width unchanged at 4; +link later; a **pad** slot deeper than a limb's cap carries an ALL-ZERO category one-hot — a free null kind) and adds a **separate `subtype` one-hot** for the within-category kind (**shared index**, width = max per category = 4: effector `0..2` = swing/knee/twist, cap `0..3` = bare/foot/pad/ball; root/start/pad = null). Category carries the shared-structure signal; subtype picks the specific kind. Both are **concatenated** into the token content (`embed_module` input = `[physical, category_oh, subtype_oh, mode_oh]`), not a flat single one-hot and not additive.

**Token mask**:
The attention-level masking of inactive limbs: their token embeddings are zeroed and they're set as padding keys (`src_key_padding_mask`) so active tokens never attend to them. Distinct from the DOF mask (the raw input vector it derives from).

**Structural block** (the obs fields the layout flags `structural`):
The part of obs that is deliberately **not normalized**: the `{0,1}` **DOF mask**, the per-slot **`is_cap` flag**, the **subtype one-hot**, and **`has_sensor`**. Every channel is exactly 0 or 1 and is read back with a `> 0` threshold, which normalization would break — a channel that happens to be constant over a window collapses to ~0 and would be misread as absent. Constant per body, written once at env `allocate_buffers`. The token **category** is *derived* from it, not stored: effector ⟺ mask, cap ⟺ `is_cap`, pad ⟺ neither. The package flags each field individually (see its [CONTEXT.md](../../SoftwarePackage/CONTEXT.md)); `models._raw_tail` rebuilds the span from the flags and **refuses** if they are not contiguous, since restoring one slice is what keeps it cheap. Module **length** is *not* here — constant per body, but a measurement, so it normalizes.
_Avoid_: "raw tail" (it was a tail only while these fields happened to sit at the end of the buffer).

**Masked-norm model**:
The rl_games model wrapper that runs the stock input normalizer but restores the **structural block** (above) afterward, so normalization can't collapse the constant mask/type channels. Registered as `transformer_masked_a2c_logstd`. (See `docs/troubleshooting/adaptive_ant_fixes.md` for why.)

**Dual-network PPG**:
The default PPG control agent: a **policy net** and a separate **value net** with disjoint weights (≈2× params). The value net does all RL value math; the policy net carries an **aux value head** trained only in the aux phase to distill value representations into its trunk. The `ppg_continuous` baseline.

**Single-network PPG** (shared trunk):
One network with a policy head and a value head on a **shared trunk**, recovering dual-net's 2× memory. Policy/value interference is managed by **gradient detach**, not by splitting the net: in the policy phase the value gradient is detached at the trunk (value head trains, trunk doesn't move from it); in the aux phase the value gradient flows through the whole net. No aux head — the value head *is* both critic and distill target. Selected by `ppg.shared_trunk`.
_Avoid_: "merged net" / "joint net" — say single-network or shared-trunk.

## Codesign heads (single-network)

Generalized roles, used when control and generator live on **one network** as four heads. _Actor_ = any head that emits **actions**; _critic_ = any head that **predicts value**. Both the control policy and the generator have an actor and a critic.

**ContAct** — control **actor**: emits per-DOF **actions** for the current body. Trained per rollout (PPO).
**ContCrit** — control **critic**: the **V0.98** value head driving ContAct's advantages (γ=0.98). Trained per rollout.
**GenAct** — generator **actor**: emits per-tip **grow/stop** decisions along the limb frontier (sequential, random tip order), designing each limb's **module count**. Trained per **resample**.
**GenCrit** — generator **critic** = **the V1.0 body-quality head, merged**. One value head, evaluable on **live** full-state tokens *and* on **partial designed-token prefixes**. Yields the marginal-value advantage `V1.0(prefix+token) − V1.0(prefix)`. Trained per **resample** (needs true returns).

The merge: GenCrit and the old separate V1.0 head are now **one function**, not a distill pair. It's fit on two data sources toward the same body-quality target — rollout states (per-step return-to-go) and generation-token prefixes (toward the body's realized `R`). At resample the trunk learns **only** the generator side (GenAct + GenCrit/V1.0); **both** control heads are held by a clone term — **KL[ContAct_old, ContAct]** for the actor, **MSE(ContCrit, ContCrit_old)** for the critic. Per-step, control trains as **plain combined PPO** (trunk moves freely).

## Auxiliary prediction heads

Three separate mechanisms that shape the trunk's representation with a signal denser than PPO's
advantage. All independently config-gated; only the first two are on. Ablated together as
[experiment 2](../docs/experiments/aux.md).

**Forward-dynamics head** (FD):
Predicts the **next** physical state from the current one. Two targets, chosen by `fd_variant`:
`raw` regresses the next observation directly, `latent` predicts the content-only **embedding** of
the next state against a stop-grad target anchored by `embed_module`. `latent` is the shipped
variant and is JEPA-*like* — it predicts the network's own representation rather than the world —
which is why "the JEPA head" informally names it. Fused into the PPO loss; no extra forward pass.

**Forward-kinematics head** (FK):
Predicts each module's **torso-frame pose** (position + rot6D + velocity) at the current state,
against the **truly composed** target from the morphology. Not self-supervised and not JEPA at all —
ground-truth regression. Its job is to force the encoder to represent where a body's parts actually
are, which design mode then reads as tokens. Fused into the PPO loss.
_Avoid_: reading FK as evidence about transfer without checking it had signal to give — at a **rest
pose** the target is nearly a deterministic function of the morphology, which design mode already
sees, so a flat `gen/fk` makes an ablation uninterpretable.

**Masked-token JEPA** (`config.jepa`):
The literal I-JEPA: mask a random subset of tokens (CLS + present modules), swap in a learned
`[MASK]` latent, and predict the stop-grad unmasked hidden states through a BYOL-style predictor.
Same-step, not predictive over time. **Currently disabled** and unvalidated; carries its own
backward pass and a representation-anchor term in the resample update.
_Avoid_: saying "JEPA" for FD. In this codebase `jepa` names **this** head; FD's latent variant is
JEPA-like but is a different mechanism with a different gate.

## Codesign tokens (single-network)

The merged net's limb-token vocabulary, spanning both reading **modes** (the net reads a real body vs a blueprint).

**Live token** / **live mode**:
A content token (effector) carrying **physical state** (pos/vel/sensors/last-action) — what the net reads during control rollout. _Reading the net in **live mode**_ = scoring a real, running body.

**Designed token** / **design mode**:
A content token carrying **morphology only** (module length + limb encoding), **no physical state** — what the net reads during the generation pass. _Reading the net in **design mode**_ = scoring a blueprint (possibly partial).

**Mode one-hot**:
A per-content-token field marking **state availability**, not token kind (the kind is the category one-hot since Phase 5): **live** = present with physical state (control rollout), **committed** = present without physical state (design pass), **pad** = a slot deeper than the limb's cap, absent in both passes. Replaces overloading zeros (a zeroed token used to mean *inactive limb*, which collided with *designed* and *off*). Phase 5 note: the live pass no longer zeroes non-effector module embeddings — that would erase a cap's type and contact — so live and design mode now embed module tokens identically.

**Start token**:
A **persistent** per-slot anchor (one per limb slot), distinct from the content token. Two jobs: (1) in design mode it's where **GenAct** reads the on/stop decision for that slot; (2) it survives into **live** mode to tell **ContAct** which limb slots exist (the attachment point — basis for the deferred multi-module "generate-from an on-token" extension). It is **never replaced**; the slot's *content* token is what turns committed/live/stop. The start token is the pre-Phase-5 seed of the reserved **Connector** type.

**CLS token** (= the **root** token, reused):
Global-obs aggregator feeding the value heads (V0.98 + V1.0/GenCrit). `v(prefix)` is its design-mode readout over committed tokens (pending slots are simply **absent/masked**, not tokens). Its *content* differs by mode — root state embed in **live** mode, a learned `cls_design` parameter in **design** mode (generation has no root state) — but the same trunk and value heads read its output in both.

**Attention scope** (`network.transformer.attn_scope`):
Which tokens a token may attend to on the **control** encode path (`_encode_codesign`) — `full` (every token, the shipped network), `self_cls` (itself and the CLS/root token: own state plus torso, no inter-limb information), `self` (itself alone). A boolean mask only: all three are **parameter-identical**, and with `n_layers: 1` there is exactly one round of token mixing to switch off. Exists for [experiment 4](../docs/experiments/attention.md).
_Avoid_: wiring it into `_encode_design`. The generator is autoregressive over its generation prefix, so restricting its attention breaks generation outright. _Avoid_ also: replacing the `full` path with an all-True mask instead of `None` — an explicit mask changes the SDPA kernel path, so the control arm would stop being bit-identical to the shipped network.

## Generation (morphology generator)

The control-side realization of the generative morphology policy (see [Codesign](../CONTEXT-MAP.md)). Vocabulary for the generator that emits a body and is trained by **policy-gradient** on the same reward the controller earns. Architecture/wiring/schedule details live in `temp/codesign_single_network_plan.md` and the **Codesign heads/tokens** sections above (the single-network design), not here. How to read the run's TensorBoard metrics + debug the algorithm from them: [`docs/reference/codesign_metrics.md`](../docs/reference/codesign_metrics.md).

**Morphology generator** (generator):
A **sequential, token-at-a-time** policy that emits the robot's designed morphology one slot at a time, trained by **policy-gradient** on the **control reward** — a body is good if the controller earns high return on it. It shares the **control trunk** (single network: GenAct + GenCrit heads alongside ContAct + ContCrit); each limb slot is decided in **randomized order**, conditioned on the already-committed tokens. Acts **once per resample window**; its body never enters the control's per-step action stream — it is applied to the env at the window's rebuild. Emits morphology, not per-DOF actions.
_Avoid_: conflating with the control **actor** (emits per-DOF actions). Also _avoid_ the retired framings: the **unconditional Bernoulli bandit** (ADR-0010), the value-ascent generator, and the **separate-net SeqGenerator** with a distill-pair aux value (ADR-0012 — superseded by the single-network merge).

**Generation MDP**:
The small MDP the generator solves: **state** = the committed token prefix, **action** = the next token, **reward** = the scalar body quality `R`, γ=1. GenCrit/V1.0 regresses **every** prefix toward `R`, and a token's advantage is the marginal difference of consecutive prefix values (below). Slots are decided in random order. The generator is an actor-critic (GenAct/GenCrit) on this MDP.

**Grow / stop decision** (generator):
The per-step decision the generator emits at a **growable tip**. Pre-Phase-5: a binary **grow** (add one module) vs **stop** (end the limb). Phase 5 generalizes it to **emit a module type** via a **factored** decision: a **category** choice `{effector, cap}` (positionally grammar-masked), then a **subtype** choice whose logits are **masked to that category's kinds** (effector → swing/knee/twist; cap → stop/foot/pad/ball). An **effector** = grow (which joint kind), a **cap** = terminate (which end kind); **stop** = the **bare cap**. Joint log-prob = `logp(category) + logp(subtype | category)`. Each limb's **effector count** = how many effectors before a cap; **presence** is derived (present iff ≥1 effector — a bare cap at depth 0 = absent, and morphology caps are grammar-masked at depth 0). Phase 5b adds **connector** as a third emittable category, alternating with effectors/caps at every junction, so the decision reads *"emit a module of type T"* over `{effector, connector, cap}`. Distinct from the control **module token** (the obs token).

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
- **Designed token** — the attributes a given codesign run actually generates and optimizes. Phase 1 designs each limb's **module count** (variable length via count; presence emergent as count > 0); per-module continuous length and angle stay fixed (deferred), so `Designed ⊊ Attribute`. The framework permits any subset. `Designed ⊆ Attribute ⊆ Full`.

**Presence** (p):
A limb's existence — now a **derived** property (present iff its **module count** > 0), not a directly-emitted decision. The generator grows limbs from a frontier; a limb that stops at depth 0 is absent. A **≥1-limb guard** masks stop on the last all-empty tip. (Retired: presence as a per-slot bit; the per-limb Bernoulli-bandit and continuous-`p` value-ascent framings.)

**Module count** (counts):
The per-limb integer `0..max_limb_length` the generator designs (`from_counts`, `bodies_from_counts`); env `set_next` takes counts `(N, n_limbs)`. The Phase-1 replacement for the presence bit — **presence** derives from it (count > 0).

**Frontier / growable tip**:
The generation MDP's state = the set of still-growable limb tips. Each step picks a **random** growable tip and emits **grow/stop**; a tip drops when it stops or hits `max_limb_length`. Random tip order preserves the Shapley marginal-value credit. Fixed `n_limbs·max_limb_length` (=32) steps, no-op-masked when no tip is growable.

**Canonical slot** (depth-major):
The single token/DOF ordering `slot(n,d) = (d−1)·n_limbs + (n−1)` — depth-major over limb `n`, depth `d`. Module tokens are emitted in this order, which **is** the env action/DOF order, so the old `nat_to_dof` remap is gone. (vsim assigns DOF order per-limb depth-ascending regardless of XML order, so the scatter **queries** joint names `joint_{n}_{d}`.)

**`tdims`**:
The Task's own `obs_layout()`, carried on the net — the single source of truth for obs offsets/sizes, published by the package rather than re-derived here. Counts (`n_modules`, `n_slots`, `n_sub`, `n_root_axes`, `obs_total`) sit at the top level; fields live in one of two groups, `global` (one per env) and `module` (one per padded slot, `dim` being a *single* slot's width), each entry `{off, dim, structural}`. `models.py`, the tokenizer and the codesign agent all read it **by field name**, never by a remembered offset.

## Reserved (upcoming phases)

**Aux head** (Phase 2): a self-supervised prediction head on the shared control trunk, fused into the PPO loss (rl_games `get_aux_loss` hook — 0 extra trunk passes; disabled ⇒ baseline-identical). Two instances:
- **Forward Dynamics (FD)** — *temporal*: each active module token predicts its **own next-timestep parent-relative state** (rel-pos+rot+vel+cfrc, 21) from its post-trunk hidden + its **own** sampled action. Normalized-space MSE.
- **Forward Kinematics (FK)** — *spatial, same-timestep*: each active module token predicts its **own pose fully in the torso's frame** (pos + rot6D + vel, 15). Pos/rot are the torso-relative composition; vel is the torso-riding-observer velocity `R_root⁻¹(v−v_root−ω_root×(p−p_root))`. All are exact **pure limb-chain compositions** of the parent-relative obs (root terms cancel / are implicit in the chain, so CLS is inert). Verified against sim to ~1e-8. Target composed agent-side from the obs, per-depth normalized.

(A JEPA / masked-latent aux is committed but being reframed toward FD/FK — see the stale-marked `temp/Phase2_JEPA_plan.md`; its `[MASK]`/predictor/repr-anchor terms are deliberately **not** glossary vocabulary until Phase 2 settles.)

**6D rotation** (Phase 2): obs rotation representation, quaternion → 6D.

**Depth rotary embedding** (Phase 3a): a reserved alternative to the additive `depth_emb` — rotates attention Q/K by within-limb-depth phase instead of adding a learned per-depth vector. `pos_emb` (limb slot) stays additive alongside it; limb identity isn't a distance axis the way depth is.

**z_model** (Phase 3c, **not built**): a reserved knob to decouple the FD-latent target size from `d_model`. Clarification (2026-07-19): today's FD-latent "JEPA" target is **not an encoded latent** — no trunk pass on `next_obs`; it's the **pre-trunk input embedding** `next_phys @ embed_module.weight[:, :MODULE_DIM].T` (stop-grad), so target-space **==** `embed_module`'s own output space (`d_model`) **by construction** — that identity is why there's no config separating them, and why `embed_module` (RL-pinned) needs no separate anti-collapse anchor. `z_model` would break that identity via a dedicated **frozen/random** projection (RND logic: a fixed random map can't collapse), earning its keep only for a target dim ≠ `d_model` or a collapse guarantee independent of `embed_module`. The classic I-JEPA "run the target encoder on the target" was the `H(next_obs)` post-trunk variant — **rejected** for cost (0-extra-pass).

**Action field / actioned copy** (Phase 4, wiring locked 2026-07-19): sequential action decode — actions commit **depth-batched, proximal→distal** (all limbs' depth-*d* joints together, deepest last), each joint conditioning on the **committed shallower actions** (within-timestep cross-limb coordination). Carried by an **action field** on the module token — a current-action channel + a committed-validity bit **concatenated into `embed_module`'s input** (like `mode_oh`, computed from committed actions, *not* part of the obs/`tokenize` vector); the 2 dims are **gated off** when `sequential_actions=false` so the off-run is byte-identical to the parallel baseline (checkpoint-compatible with root). Decode lives in the **model-wrapper `forward`** (sampling seam): rollout = `max_limb_length` sequential passes (fill the just-committed level's field, re-encode), `V(s)` read on the all-blank **pass 0**; train recomputes `mu` **teacher-forced** on the stored sampled actions (so `mu_d` matches rollout → PPO ratio exact at step 0). neglogp/entropy unchanged (global state-independent `log_std` ⇒ the per-dim sum equals the parallel Gaussian). Single-pass **training** (`seq_train_mode: unroll`) duplicates each module token into an **actionless copy** (its output is where `mu` is read — the not-yet-acted rollout decode) + an **actioned copy** (a key/value carrying the committed action for deeper joints, depths `<max_limb_length−1` only), masked so each joint is attended exactly once (no leak/double-count); `multipass` instead replays the rollout's passes (correctness **oracle**). Numerically **exact at `n_layers=1`** (the two modes are identical there; Phase 6 layers make `unroll` an approximation). **FD/FK aux** read the hidden at each token's **about-to-act** state (the actionless copy / pass *d*), the same place `mu` is read — 0 extra trunk passes in unroll; committed shallower actions in that context are inert to the heads. Gated by `network.transformer.sequential_actions` (off == parallel actor) + `config.seq_train_mode`. See `transformer_rl/sequential_actions.py`. **Status (2026-07-20): built + numerically verified (oracle-exact at `n_layers=1`) but PARKED — too slow / memory-inefficient to run this phase; implementation lives on branch `phase-4`, deliberately NOT merged to root.**
_Avoid_: "separate action token" (retired — it's a field on the existing token; the actionless/actioned **copies** are a training-only unroll device, not persistent tokens).

**Constrained decoder** (Phase 5, built): a per-tip valid-next mask on GenAct logits, masking the generator's illegal type emissions. Because the action is **factored** (see *Grow / stop decision*), masking happens at **two levels**: the **category** logits are masked positionally, and the **subtype** logits are then masked to the sampled category's kinds (plus the depth-0 cap restriction below). General mechanism supports **prev-type → valid-next** conditioning (needed for Stage-3 connector compatibility), but at this stage the rules are **purely positional** (depth-keyed): depth 0 → effectors + bare cap only (morphology caps masked; ≥1-limb guard also masks bare on the last empty tip); depth `1..max_len−2` → all effectors + all caps; deepest slot (`max_len−1`) → caps only (effectors masked → **max `max_len−1` effectors per limb**, the terminal slot is always a cap). Applied identically in `sample()` and `gen_replay()` for PPO-ratio correctness. Phase 5b makes the rules **structural** as well as positional: connectors and non-connectors must strictly alternate, so the valid-next set depends on whether the tip's previous emission was a connector — the first real use of the **prev-type** conditioning the mechanism was built for.

**Starting-position head** (Phase 7b): a generator head predicting a valid, good start pose (per-effector start value + CLS start-offset).

**Elite morphologies** (Phase 8b): saved high-quality bodies retained (e.g. reserved envs) to promote population diversity.
