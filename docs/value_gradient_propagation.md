# Value-gradient propagation for codesign

Can the control critic's gradient `∂V/∂p` (per-leg presence) act as a training signal for a
morphology **generator** — emit a body, push it uphill on `V`? This study answers **yes on sign,
no on magnitude**: across 3 seeds the gradient is positive almost everywhere (no myopia), it
interpolates onto held-out bodies, and it survives composition through a soft generator network
(sign-alignment 0.94) — the failure mode that killed the earlier binary version. But grounded against
*realized return*, the gradient's **sign is right 99% of the time while its magnitude barely predicts
which leg actually helps** (Pearson +0.17), and only in bodies that already walk. Full figures and the
numeric pipeline live in [`notebooks/value_grad_prop.ipynb`](../notebooks/value_grad_prop.ipynb);
experiment scripts are `experiments/value_grad_prop.py` (Step 1), `value_grad_step2.py`,
`value_grad_step3.py`, `value_grad_phase2.py`, and `value_grad_ablation.py` (grounding). First run 2026-06.

## Background

The codesign loop we want is: a generator emits a body, the trained critic scores it, and we
backprop `∂V/∂(morph)` into the generator to make it emit better bodies. That only works if the
critic's morphology gradient has **sensible sign and magnitude** and **survives being composed with
a generator network**. This study tests both.

An earlier **binary** `{0,1}` presence version failed in Phase 2: a discrete generator's saturated
sigmoid collapsed the gradient to ~zero. This recast fixes that by construction — presence is a
**continuous probability** `p ∈ [0,1]`, so the critic natively consumes the soft output a generator
produces, with no discrete bottleneck.

### The sign convention (read this first)

The quantity is **raw sensitivity** `∂V/∂p` — *which way a leg should move* — **not** an attribution
of how much a leg currently contributes. The sign tracks the *desired* state regardless of the
current one: a **helpful** leg is **+** (grow it) whether currently on or off; a **harmful** leg is
**−** (shrink it) either way. A critical-but-currently-*off* leg should be the single **largest
positive** gradient ("add me"). This is exactly what gradient ascent consumes.

## What was tested

An **experiment-only** ant variant (`AntBinaryLegEnv`, not the production controller):

- **All 8 legs always DOF-mask-active.** The production net zeroes masked tokens, giving them
  *exactly zero* length gradient. To get any presence gradient every leg must stay active; presence
  is carried purely by the input `p`.
- **Continuous presence `p ∈ [0,1]`**, tied across a leg's hip and ankle slots (`∂V/∂p` = sum of
  the two slot gradients). `p` is a *probability*, not a length.
- **Stochastic build.** The body is built by `Bernoulli(p)` per leg (on → 1×, off → a hidden 0.05×
  stub), but the **obs reports `p`, not the sampled outcome**, so `V(p) ≈ E[return]` and `∂V/∂p` is
  in-distribution everywhere. Training draws `p` as `√U` for on-legs / `1−√U` for off-legs around a
  bias center `S`, whose equal mixture is marginally `Uniform(0,1)` → uniform coverage of the knob.
- **Held-out set (guard).** All 56 five-leg topologies + 3 curated Step-2 sets are stripped from the
  pick-pool and rejected-and-resampled if a Bernoulli draw lands on them — so Steps 2/3 can probe
  the gradient *off* the training set.
- Two metrics per leg: **`ḡ`** (∂V/∂p averaged over a rollout — the signal a generator sees) and
  **`g0`** (at the t=0 reset pose — the static prior). **3 seeds** (s42/s43/s44), report mean±std.

The four probes and the conditioning check:

| | question | how |
|---|---|---|
| **Step 1** (myopia) | is `∂V/∂p` positive at low `p`, so ascent can grow a near-off leg? | bin `ḡ` by `p` over the training population |
| **Step 2** (curated) | does the gradient match intuition on 3 held-out shapes, and agree with a trained twin? | deterministic builds vs a mirror/rotation twin |
| **Step 3** (interpolation) | does the gradient generalize to held-out 5-leg bodies? | overlay held-out `∂V/∂p`-vs-`p` on Step 1 |
| **Grounding** | does `V` and its gradient predict *realized return*? | toggle each leg of 16 base bodies, correlate `V`/`∂V/∂p` vs realized `R`/`ΔR` |
| **Phase 2** (conditioning) | does the sign survive composition through a generator net? | pretrain a scatter-gather transformer, compose in front of `V`, backprop to its input |

## Results

### Step 1 — myopia averted, but weakest where it's needed most

Premise sanity holds: mean episode reward rises monotonically with sampled on-leg count (s42:
2-leg 341 → 8-leg 4948; 5 absent = holdout). The headline `∂V/∂p`-vs-`p` curve (seed-avg):

| p-bin | 0.0–0.1 | 0.3–0.4 | 0.6–0.7 | 0.7–0.8 | 0.9–1.0 |
|---|---|---|---|---|---|
| mean `∂V/∂p` | +2.7 | +4.7 | +6.9 | +6.9 | +6.2 |
| frac > 0 | ~0.88 | ~0.90 | ~0.92 | ~0.93 | ~0.94 |

**Positive everywhere** (frac > 0 never drops below ~0.85, even at the lowest `p`) → **no myopia**:
ascent can discover and grow a near-off leg. But the **magnitude shape is the opposite of the
predicted "diminishing returns, largest at low `p`"** — it's *weakest* at low `p` (~+2.7) and
*strongest* mid/high (~+7), barely drooping at `p=1`. The signal is **smallest exactly where a
generator most needs it** (bootstrapping a leg from near-off): ascent is slow-to-start, not stuck.

Two supporting reads:

- **Split by body fullness:** the predicted diminishing-returns shape *does* exist, but only
  **conditionally**. In **lean bodies (≤3 legs)** `∂V/∂p` rises with `p` (+1.7 → +6.8); in **full
  bodies (≥6 legs)** it's high and *declines* toward `p=1` (+5.6 → +7.9 → +5.5, i.e. redundancy).
  The pooled curve averages these.
- **`g0` (static prior) vs `ḡ` (rollout):** weak agreement (corr 0.32, sign-agree 0.87). `g0` has
  **~4× the absolute spread** of `ḡ` (std 20.6 vs 5.4, range −61…+148) at the **same coefficient of
  variation** (≈1.0) — i.e. uniformly inflated and noisier in absolute terms, sign-flipping more.
  **Trust `ḡ`; the static reset-pose prior is a poor proxy.**

### Step 2 — corners are OOD; interior `p` recovers the expected signal, and twins agree

Two findings co-headline here.

**(a) The `p=0/1` corners are out-of-distribution.** Training never places off-legs at *exactly* 0,
so corner gradients are distorted. On `critical_missing` (held `{5,6,7}`), the prediction was that
the *off* front/right legs should be the largest positive ("add me"). At the corner they were
**flat** (1F +0.7, 8FL +3.6) while the on-cluster dominated (6BL +22.0). At **interior `p`
(0.75 on / 0.25 off)** the prediction is **restored** — front-leg "add me" gradients grow and
on-cluster "keep" gradients collapse:

| `critical_missing` leg | 1F | 4BR | 5B* | 6BL* | 7L* | 8FL |
|---|---|---|---|---|---|---|
| corner `p∈{0,1}` | +0.7 | +4.3 | +9.7 | +22.0 | +14.6 | +3.6 |
| interior 0.75/0.25 | +2.9 | +7.9 | **−3.4** | +12.0 | **+1.4** | **+9.2** |

(\* = currently-on leg.) **Read all curated gradients at interior `p`, not the corners.** Even at
interior `p` the recovery is partial: the *cluster-adjacent* front-left (8FL +9.2) and the *balancing*
back-right (4BR +7.9) become strongly positive, but the *isolated* pure-front leg (1F +2.9) stays weak —
its payoff is contingent on co-adding partners, which a first-order signal can't see. The Grounding
section shows this is real, not an artifact: in sparse bodies the gradient's magnitude doesn't track
which leg actually helps.

**(b) Held-out shapes agree with trained twins.** At interior `p`, the mirror twin (`redundancy`:
lone leg 2FR held +16.4 vs twin +18.2) and rotation twin (`symmetric`: 1F held +13.4 vs twin +12.4)
≈ coincide after remapping — topology-specific generalization holds. `critical_missing` is the
noisiest comparison only because it's a near-dead body (reward ~30 vs `symmetric` ~2200).

### Step 3 — the signal interpolates; adding a leg cleanly shrinks the rest

**(a) Held-out 5-leg interpolation.** The held-out 5-leg `∂V/∂p`-vs-`p` curve coincides with the
in-distribution Step-1 curve in sign and shape (slightly *higher* at low `p`):

| p-bin | low | mid | high |
|---|---|---|---|
| Step 1 (in-dist) | +2.7 | +6.9 | +6.2 |
| Step 3 (held-out 5-leg) | +4.0 | +6.9 | +6.7 |

→ **The codesign signal survives off the training set.**

**(b) Per-limb add/remove** (mean `∂V/∂p` over present limbs, 3 held-out 5-leg bases):

| base | base | +1 leg | −1 leg |
|---|---|---|---|
| {3,4,5,6,8} | +9.65 | **+7.71** | +7.44 |
| {1,3,4,6,8} | +8.53 | **+6.47** | +9.05 |
| {3,4,5,7,8} | +10.05 | **+8.64** | +9.41 |

**Adding a leg consistently lowers every remaining limb's marginal value** (all 3 bases) — clean
per-body diminishing returns, consistent with the fullness-split. **Removing is mixed/noisier**
(1 down, 2 up): no clean symmetric story.

### Grounding — great sign, poor magnitude, and only useful once the body walks

Steps 1–3 judged the gradient against *intuition*; this grounds it against *realized return*. For 16
in-distribution base bodies (4 each at 3/4/6/7 legs), we build each base plus its 8 single-leg
**toggles** (flip each leg on↔off) at interior `p`, and measure realized return `R`, value `V`, and
per-leg `∂V/∂p` (seed-avg over s42/43/44). Two tests:

- **Value calibration — strong (in rank).** `V` vs `R` over all 144 bodies: **Pearson +0.93,
  Spearman +0.93**. The critic orders bodies good→bad correctly. (Absolute `V` is *compressed* —
  range ~24–89 vs realized 24–5187 — because it predicts *discounted* return-to-go; calibration is in
  rank, not scale.)
- **Gradient grounding — right sign, wrong magnitude.** `∂V/∂p_L` vs the realized marginal value
  `ΔR_L = R(with L) − R(without L)`: **sign-agreement 0.99** (the gradient nearly always knows a leg
  is worth adding/keeping — and nearly every leg has positive realized value), but **Pearson only
  +0.17** — magnitude barely predicts *which* leg helps. The gradient is squashed into +2.7…+15 while
  realized leg values span −330…+2800 (~10×).

The decisive cut is **by body fullness**:

| base leg-count | 3 | 4 | 6 | 7 |
|---|---|---|---|---|
| Pearson(`∂V/∂p`, `ΔR`) | −0.01 | −0.05 | **+0.37** | **+0.35** |

**The magnitude is informative only in bodies that already walk.** In sparse/broken bodies (3–4 legs)
it carries *no* information about which leg matters — exactly the `critical_missing` regime. Adding is
somewhat predictable (r≈+0.30); removing is not (ρ≈+0.03). This is the contingency limitation made
concrete: from a bad body a generator gets a reliable "add legs" *direction* but no reliable guidance
on *which* — see the diagnosis below for why.

### Why the magnitude fails on poor morphs — a value-learning failure, not a calculus artifact

Two candidate causes for the sparse-body failure: **(A)** the gradient is a *local slope* that can't
see the discrete "tips it over the locomotion threshold" jump adding a leg causes, or **(B)** the
value function never learned which leg matters in the un-walkable region. Three checks (all on the
ablation data) say **(B), decisively**:

- **Not a calculus artifact.** Compare the gradient to the critic's *own finite-difference*
  `ΔV = V(with leg) − V(without leg)` — which actually crosses the toggle, no derivative. It fails
  *identically*: 3-leg grad −0.01 / `ΔV` +0.06; 4-leg both ≈ −0.05; both recover only at 6–7 legs.
  If (A) were the cause, `ΔV` would beat the gradient. It doesn't → `V` itself doesn't encode the
  per-leg structure; the gradient faithfully reports a `V` that's wrong there.
- **Not noise.** The sparse-body per-leg signal is large and seed-reproducible (gradient SNR ≈ 1.9,
  *higher* than full bodies' 0.8). `V` confidently assigns a per-leg ranking that just doesn't match
  reality — wrong, not random.
- **`V` starves the floor.** It compresses all 3-leg bodies into `V ∈ [35,39]` (realized returns in
  the hundreds), allocating almost no output range to the un-walkable region.

**Mechanism.** Below quorum a body can't walk *regardless of which legs it has*, so the training
return signal carries ~no information about per-leg importance there (every bad body returns ~floor).
`V` can't learn structure it never saw — it interpolates from functional bodies, getting the sign
right but the magnitudes arbitrary. Compounded by **credit assignment** (a broken body never *uses* a
leg on-policy) and **complementarity** (the true per-leg value depends on *absent* legs — not a
function of the current body at all). You can't differentiate your way to information the value
function never contained. The implication: the fix is **data/representation in the floor region**
(curriculum, oversampling near-threshold bodies, informative shaping below quorum) — *not* value
normalization (calibration rank is already fine) — or accept the gradient as a **sign-only** proposer.

### Phase 2 — the soft generator preserves the sign

A small transformer is pretrained (MSE 0.0005) to regress each leg's `p` from a **scattered** token
encoding (forcing attention to gather it), frozen, and composed in front of `V`; gradients backprop
to its input.

| metric | value | meaning |
|---|---|---|
| **sign alignment** | **0.938** | input-grad sign matches the direct `∂V/∂p` — the headline |
| magnitude corr | +0.554 | sign survives strongly; magnitude only moderately |
| chain norms (out→in) | 1935 → 415 → 5540 → 3342 | **no collapse** (ends above where it starts) |

**The conditioning check passes.** This is the direct contrast with the binary generator, whose
saturated sigmoid collapsed the gradient — the continuous-`p` recast removes that bottleneck by
construction. Sign (what ascent consumes) survives at 0.94; magnitude survives only moderately, as
expected.

## What we learned

- **The value gradient is a usable codesign signal on sign.** Positive almost everywhere (no
  myopia), interpolates to held-out bodies, survives a generator network (sign-align 0.94), and agrees
  with *realized* leg value 99% of the time on sign. The directional premise holds.
- **Magnitude is the soft spot — and grounding shows it's worse than "soft".** The gradient is
  *weakest at low `p`*, only *moderately* magnitude-correlated through the generator (corr 0.55), and
  against realized return its magnitude correlation is just **+0.17** — and **zero in sparse bodies**
  (3–4 legs), rising to ~+0.35 only once the body walks. So ascent gets a trustworthy "add legs"
  direction but, from a bad body, can't tell *which* leg matters — the contingency/quorum limitation.
- **Value is well-calibrated in rank but blind to per-leg structure at the floor.** `V` orders bodies
  well (Spearman +0.93) yet doesn't encode *which* leg matters in un-walkable bodies — its own
  finite-difference fails there just like the gradient, so it's a value-*learning* gap (the floor
  carries no per-leg training signal), not a derivative artifact or noise. The root of the
  magnitude-poor gradient.
- **Evaluate at interior `p`, never the `{0,1}` corners** — corners are OOD and flip/distort the
  expected importance ordering (Step 2).
- **Trust the rollout-averaged `ḡ`, not the static `g0`** — the reset-pose prior is a 4×-inflated,
  weakly-correlated proxy.
- **Diminishing returns is real but conditional** — it appears in full bodies and on the add-side of
  Step 3, not in the pooled low-`p` curve.

## Caveats

- **All-mask-active ≠ production controller.** The deployed mask-based net gives masked tokens zero
  gradient, so this mechanism can't be read off it directly; productionizing presence-gradients
  needs the real controller retrained all-active. *Headline caveat.*
- **Mild extrapolation only.** Steps 2/3 test held-out *topologies* and *leg count*; far-novel
  bodies, where `V` may be garbage, remain untested.
- **`critical_missing` twin is noisy** because the body barely moves (reward ~30) — its twin
  agreement is the weakest of the three and should not be over-read.
- **Grounding bodies are in-distribution.** The grounding test uses stable, non-held-out bases at the
  interior operating point, so its correlations are a *best case*; the gradient's magnitude on
  far-novel bodies is likely worse, not better.
- **Stochastic build → noisy PPO targets.** A fixed `p` builds a different Bernoulli body each draw;
  `V` learns the expectation but training sees higher-variance returns than the old deterministic
  regime.

## Next

The capstone the premise actually demands: a **gradient-ascent test** — freeze `V`, ascend the raw
`p` vector (and through the generator), and check it converges to known-good morphologies. Grounding
predicts it gets the *direction* right (add legs) but bootstraps slowly and may stall on *which* leg
from a sparse start. Two leads worth chasing first: (1) **feed the floor region information** — the
diagnosis shows `V` is blind to per-leg structure where bodies can't walk because the return signal
there is flat (curriculum / oversample near-threshold bodies / shaping that's informative below
quorum), then re-check grounding; (2) accept the gradient as a *sign-only* proposer and let
realized-return ablation resolve magnitude. Beyond presence: continuous **segment-length**
(`∂V/∂length`) and **leg-angle** (`∂V/∂θ`) gradients.
