# Transformer Architecture

This is the policy network that decides what the robot does each timestep. It's a
**transformer that treats every body part as a token** — so instead of one fat
observation vector going into an MLP, the torso and each joint each become their own
little input, and the network uses attention to let them talk to each other before
deciding how to move. Actions come out per joint; a single value estimate comes out of
the torso token.

### The task we're demonstrating it on

Everything here runs on **ant locomotion**: a simulated many-legged "ant" that has to
learn to walk. The twist that motivates the whole design is that the ant's *body keeps
changing* — it can have anywhere from 3 to 8 legs, placed at different angles around the
torso, with different segment lengths. We want **one** controller that can walk *any* of
these bodies, not a fresh network retrained per shape. (Down the line this feeds
**codesign** — evolving the body and the brain together; see the [Context Map](../CONTEXT-MAP.md).)

### Why a transformer is the right tool here

A plain MLP bakes in a fixed input size, so a 4-leg ant and a 6-leg ant would need
different networks. Tokens sidestep that: each leg is just another token, so the *same*
weights handle any number of legs in any arrangement. That's the core trick — the rest
of this doc is how we make it actually work (and §1 unpacks why this matters beyond
convenience).

The doc goes top-down: the big picture first, then data shapes, the augmentation we add,
and finally the rl_games plumbing. It's **env-agnostic by design** — the ant is the
running example, not the only thing this could control. Wherever you see torso/hip/ankle,
leg, or 16 DOFs, that's the ant *instance* of a general pattern (one root token +
repeated structural units → part-tokens → attention → per-part outputs).

Shared terms (leg, DOF, DOF mask, active/inactive, morphology) live in the
[Context Map](../CONTEXT-MAP.md); architecture vocabulary (part-token, root token,
leg encoding, token mask) lives in the [Control glossary](../transformer_rl/CONTEXT.md).

---

## 1. Why a transformer

**Headline: attention is the bridge to codesign.** The end goal is a loop that jointly
optimizes *morphology* and *control* (see [Codesign](../CONTEXT-MAP.md)). A transformer
control policy is the chosen interface for that future generative morphology policy:
both speak in per-part tokens and attention, so a morphology generator can eventually
attend to / be conditioned on the same token representation the controller uses. The
architecture is built now to make that coupling natural later.

**The enabling property: morphology-invariance.** Because legs are *tokens*, not fixed
input slots, one shared policy + value function spans every morphology — any leg count,
any placement — without ragged tensors and without retraining per body:

- **Count-invariance**: a body with fewer legs simply has fewer (active) tokens. The
  weights (`embed_hip`, `embed_ankle`, `joint_head`, attention) are shared across all
  legs, so adding/removing a leg adds/removes a token rather than changing the network.
- **Permutation-equivariance**: legs are interchangeable up to their geometry; identity
  is carried by *embeddings and a geometric encoding* (§5), not by position in a flat
  vector.

This is what lets the controller keep working as the body changes during codesign — the
prerequisite that the rest of the project is built on.

> Implementation note: today the shape is **padded** to a fixed 8 legs / 16 DOFs with a
> DOF mask marking which are real (see [ADR-0002](adr/0002-fixed-8-leg-padding-and-dof-mask.md)).
> Padding + masking is how count-invariance is realized in practice without ragged
> batches; the transformer never attends to padded tokens (§5).

---

## 2. High-level flow

```
             env obs  (B, 139)
             │
             ▼
  ┌────────────────────────────────────────────────────────┐
  │ rl_games wrapper · TransformerMaskedNorm               │
  │ normalize obs, but keep the raw {0,1} DOF-mask tail    │
  └────────────────────────────────────────────────────────┘
             │  normalized obs (B, 139)
             ▼
  ┌────────────────────────────────────────────────────────┐
  │ LegTransformer · architectures.py                      │
  │                                                        │
  │ tokenize → embed → +type/+pos → mask → encoder         │
  └────────────────────────────────────────────────────────┘
             │
       ┌─────┴─────┐
       ▼           ▼
   joint_head   value_head
       │           │
       ▼           ▼
   mu (B,16)    value (B,1)
       │
       ▼
  ┌────────────────────────────────────────────────────────┐
  │ builder · log_std = mask_dof · log_std_param   (B, 16) │
  └────────────────────────────────────────────────────────┘
       │
       ▼
   action distribution  N(mu, σ = exp(log_std))
```

`B` = batch (num envs). The wrapper and builder are rl_games glue (§8); the
`LegTransformer` (§3–§7) is the pure architecture.

---

## 3. The input contract: the observation

The transformer's input is the env observation. This doc takes the obs as a **given
contract** — how the env assembles it (DOF scatter, segment lengths, the mask) is
Morphology's concern; see [envs/CONTEXT.md](../envs/CONTEXT.md).

**Ant instance — 139-D, 8-leg padded layout:**

```
[0:11]    torso        y(1) + quat(4) + lin vel(3) + ang vel(3)
[11:27]   dof_pos      16 DOF slots (hip,ankle interleaved per leg; inactive = 0)
[27:43]   dof_vel      16 DOF slots
[43:59]   last_act     16 DOF slots (previous action)
[59:107]  sensors      8 legs × 6 force-sensor channels (on the ankle)
[107:123] lengths      8 hip + 8 ankle segment lengths (raw; RMS-normalized; inactive = 0)
[123:139] dof_mask     16 bits, 1 = active DOF, 0 = inactive   ← read via (>0)
```

The 4-leg classic ant is the **smaller instance** of the same contract: 59-D obs, 8
DOF slots, no length block, mask `[51:59]`. Same code path (`tokenize_4`), just smaller
constants.

| | dof slots | sensors | lengths | obs dim | mask |
|---|---|---|---|---|---|
| 4-leg (classic base) | 8 | 4×6 | — | **59** | `[51:59]` |
| 8-leg (multi-morphology) | 16 | 8×6 | 16 | **139** | `[123:139]` |

The **DOF mask** is the crux of the whole design: written once at allocation, constant
per env, and read *only* through a `> 0` boolean test by both the tokenizer and the
policy to decide which parts exist.

---

## 4. Tokenization — obs → part-tokens

`tokenize.py` slices the flat obs into one token per body part. **General pattern:** the
root gets one token; each repeated structural unit (a leg) contributes one token per
actuated segment (hip, ankle). **Ant instance:** `1 + 2·n_legs` tokens.

```
   obs (B,139)
        │   regroup fields by body part (tokenize.py)
        ▼
 ┌───────────────┬─────────────────────────┬─────────────────────────┐
 ▼               ▼                         ▼
 torso token     hip token × n_legs        ankle token × n_legs
 (B, 11)         (B, n_legs, 6)            (B, n_legs, 12)
 torso obs       dof_pos · vel · act       dof_pos · vel · act
                 · sin · cos · hip_len      · 6 force sensors
                                            · sin · cos · ankle_len
```

**Per-token features (ant instance, 8-leg):**

| token | dim | contents |
|---|---|---|
| torso (root) | 11 | y, quat, lin vel, ang vel |
| hip × 8 | **6** | dof_pos, dof_vel, last_act, **sin, cos** (leg encoding), hip length |
| ankle × 8 | **12** | dof_pos, dof_vel, **6 force-sensor channels**, last_act, **sin, cos**, ankle length |

4-leg instance: hip = 5, ankle = 11 (no length feature).

`tokenize_*` returns `(torso, hip_tokens, ankle_tokens, active_mask)`:

```
torso        (B, 11)
hip_tokens   (B, n_legs, hip_dim)
ankle_tokens (B, n_legs, ankle_dim)
active_mask  (B, 2·n_legs)        # 1.0 = active DOF, derived from dof_mask>0
```

`active_mask` is built `[all hips | all ankles]` (natural order, §7), from the hip/ankle
bits of `dof_mask`. A leg is "active" iff its hip bit is set.

### Leg encoding (geometric augmentation)

Each leg's **physical placement angle** is baked into its hip and ankle tokens as
`(sin θ, cos θ)`. This is data *augmentation that injects body geometry*: it tells the
policy *where on the body* a leg sits, which a flat obs would not convey once legs are
interchangeable tokens.

```
ant: leg n placed at angle (n-1)·45°   →  8 legs around the torso
                       0°
                  315° ┐ ┌ 45°
                       │ │
              270° ────┼─┼──── 90°       enc[i] = (sin θ_i, cos θ_i)
                       │ │                zeroed for inactive legs
                  225° ┘ └ 135°
                      180°
```

Inactive legs get their `(sin, cos)` zeroed, so a padded slot carries no spurious
geometry. Distinct from the **positional embedding** (§5): leg encoding is fixed body
*geometry* in the token features; the positional embedding is a *learned* per-slot
vector.

---

## 5. Embeddings + active masking

Each token is linearly projected to `d_model`, then two learned embeddings are added,
then inactive tokens are killed before attention.

```
 torso (B,11)──embed_torso─┐
 hip   (B,8,6)──embed_hip──┤ →  x (B, N, d_model)     N = 1 + 2·n_legs (=17 for ant)
 ankle (B,8,12)─embed_ankle┘

 x += type_emb[type_ids]      # 3 types: torso / hip / ankle
 x += pos_emb [pos_ids]       # slot index 0..n_legs; a leg's hip & ankle share an index

 x *= token_mask              # zero embeddings of inactive tokens (kills them as queries)
```

**Embeddings (learned, augmentation of identity):**
- **type embedding** — `nn.Embedding(3, d)`: marks each token's kind (root / hip / ankle).
- **positional embedding** — `nn.Embedding(1+n_legs, d)`: a learned per-slot vector;
  hip and ankle of the same leg share the slot index, so they are tied together.

**Token mask (attention-level masking of inactive legs)** — two coupled mechanisms:
1. `x *= token_mask` zeroes inactive token embeddings (they carry no signal as queries).
2. `src_key_padding_mask = ~active` tells the encoder those tokens are **padding keys**,
   so active tokens never attend to them. The root (torso) is always unmasked.

```
 token_mask = [1 | active hips | active ankles]   shape (B, N, 1)
 pad_mask   = [0 | ~active_hips | ~active_ankles]  shape (B, N)   (True = ignore as key)
```

After the encoder, `x *= token_mask` is applied **again** to zero inactive *outputs* —
this cuts the gradient flowing back through the transformer for padded slots.

---

## 6. Transformer encoder

A stock `nn.TransformerEncoder` of `n_layers` identical layers:

```
TransformerEncoderLayer(
    d_model, nhead=n_heads, dim_feedforward=ffn,
    dropout=0.0, activation="gelu",
    batch_first=True, norm_first=True,        # pre-norm
)
```

- **Pre-norm** (`norm_first=True`) for training stability at depth.
- **GELU** feed-forward, **no dropout** (on-policy RL; exploration comes from the action σ).
- **`enable_nested_tensor=False`** — required because we pass `src_key_padding_mask`
  with our own masking semantics.

Input/output shape is unchanged: `(B, N, d_model)`.

---

## 7. Output heads

Two heads read the encoder output `x (B, N, d_model)`.

### Action head (per part)

```
 joints      = x[:, 1:, :]                 # drop root → (B, 2·n_legs, d)
 a_nat       = tanh(joint_head(joints))    # (B, 2·n_legs)  one scalar per DOF, in [-1,1]
 a_nat      *= active_mask                  # zero inactive DOFs
 mu          = a_nat[:, nat_to_dof]         # reorder natural → DOF order  → (B, 2·n_legs)
```

- **One shared `joint_head`** (`Linear(d,1)`) applied to every leg token — weight sharing
  is what makes the policy permutation-equivariant across legs.
- **Zero-initialized** (`weight=0, bias=0`): the policy starts emitting `mu≈0`, a standard
  RL stability trick so early actions are small.
- **`tanh`** bounds each action to `[-1, 1]`.

**Natural vs DOF order (the `nat_to_dof` scatter).** The transformer lays its outputs out
in *natural* order — all hips, then all ankles — but the env wants them interleaved in
*DOF* order (hip, ankle, hip, ankle …). A fixed index map reshuffles them:

```
 natural (transformer):  [ h0 h1 h2 | a0 a1 a2 ]    all hips, then all ankles
 DOF     (env expects):  [ h0 a0 h1 a1 h2 a2 ]      hip,ankle per leg

 nat_to_dof[i] = i//2 + L·(i%2)        (L = n_legs; example below uses L = 3)
   DOF slot   :  0  1  2  3  4  5
   pulls nat  :  0  3  1  4  2  5
```

### Value head

```
 value = value_head(x[:, 0, :])    # root (torso) token → (B, 1)
```

The **root token acts as a CLS-style aggregator**: it attends to all active part-tokens
(and is never masked), so its encoder output is a whole-body summary — the natural place
to read a single state value.

---

## 8. rl_games integration (the wrapper layer)

The pure `LegTransformer` emits `(mu, value)`. Two rl_games pieces complete the
obs→action path. The *why* behind the masking choices is recorded in
[`adaptive_ant_fixes.md`](adaptive_ant_fixes.md) and
[ADR-0002](adr/0002-fixed-8-leg-padding-and-dof-mask.md) — summarized here, not duplicated.

### Network builder — `log_std` gating (`models.py`)

The policy is a diagonal Gaussian; the builder supplies `log_std` and packages
`(mu, log_std, value)`:

```
 log_std = mask_dof · log_std_param      # active: learned; inactive: 0 (σ=1)
```

`log_std_param` is a single learned vector (length 16 / 8). Multiplying by `mask_dof`
does two jobs: inactive dims get **σ=1** (a moderate value that keeps rl_games'
`policy_kl` well-conditioned — tiny σ poisons the adaptive-LR controller), and inactive
entries receive **zero gradient** (the `∂/∂log_std_param` is gated to active dims).
Registered networks: `leg_transformer` (4-leg), `multimorph_leg_transformer` (8-leg).

### Masked-norm model — `TransformerMaskedNorm` (`models.py`)

Input normalization that runs the stock `RunningMeanStd` but **restores the raw `{0,1}`
DOF mask** afterward:

```
 normed = super().norm_obs(obs)
 normed[..., -mask_dim:] = obs[..., -mask_dim:]   # keep raw mask tail
```

Without this, the constant mask's running mean converges to `1.0`, the fp32 cast rounds
it to exactly `1.0`, and `(x-mean)=0` collapses every active bit to 0 — flipping all legs
to "inactive" mid-training (a hard entropy cliff). Registered as
`transformer_masked_a2c_logstd`, selected via `model.name` in the configs.

---

## 9. Shapes reference

Ant 8-leg instance, `B` = batch, `d` = `d_model`, `N = 1 + 2·n_legs = 17`.

```
 obs                         (B, 139)
 ── tokenize ──
 torso                       (B, 11)
 hip_tokens                  (B, 8, 6)
 ankle_tokens                (B, 8, 12)
 active_mask                 (B, 16)        natural order [hips | ankles]
 ── embed + tag ──
 x                           (B, 17, d)
 token_mask                  (B, 17, 1)
 pad_mask                    (B, 17)        bool, True = ignore key
 ── encoder ──
 x                           (B, 17, d)     shape-preserving
 ── heads ──
 mu                          (B, 16)        DOF order [h0 a0 h1 a1 …]
 value                       (B, 1)
 ── builder ──
 log_std                     (B, 16)
```

---

## 10. Current configuration

Architecture defaults live in `LegTransformer.__init__`; configs override the size knobs.
These are **tuned** (Optuna; see `scripts/tune.py`) and will drift — treat as a snapshot.

| knob | 4-leg (`ppo_ant`) | 8-leg full (`ppo_ant_full`) |
|---|---|---|
| `n_legs` | 4 | 8 |
| `d_model` | 160 | 160 |
| `n_heads` | 16 | 8 |
| `n_layers` | 1 | 1 |
| `ffn` | 640 | 640 |
| dropout | 0.0 | 0.0 |
| activation | gelu, pre-norm | gelu, pre-norm |

(`MultiMorphLegTransformer` defaults to `n_layers=3` in code, but the live config sets 1.)

---

## Changelog

- **2026-06-03** — Initial doc. Captures the 8-leg multi-morphology architecture with
  per-leg segment-length features (obs 139-D, mask `[123:139]`), σ=1 inactive log_std +
  gradient gating, and the masked-norm model. 4-leg classic/adaptive documented as the
  smaller instance.
