# Control

The transformer policy that controls any morphology. It tokenizes the observation into per-body-part tokens, reads the DOF mask to know which legs exist, and emits an action per active DOF. Shared terms (leg, DOF, DOF mask, active/inactive, morphology) live in the [Context Map](../CONTEXT-MAP.md).

## Language

**Leg transformer**:
The architecture: a transformer encoder over body-part tokens, shared across morphologies because legs are tokens rather than fixed input slots. `LegTransformer(n_legs)`. The 8-leg / 3-layer instance is its multi-morphology config.
_Avoid_: dynamic leg transformer. The 8-leg factory is `MultiMorphLegTransformer`; registration key `multimorph_leg_transformer`.

**Token**:
One transformer input vector for a body part: one **torso token**, plus a **hip token** and **ankle token** per leg (1 + 2·n_legs total). Inactive-leg tokens are zeroed and excluded from attention.

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
