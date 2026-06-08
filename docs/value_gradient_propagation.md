# Value-gradient propagation for codesign

Can the control critic's gradient `∂V/∂p` (per-leg presence) act as a training signal for a
morphology **generator** — emit a body, push it uphill on `V`? The short answer: **yes on sign and
shape, no on magnitude — and it depends sharply on how good the control policy/value function is.**

This is the third version of the experiment, and the headline is a **contrast between a poor control
and a good one**. The previous version (a *stochastic-build* env: body `Bernoulli(p)`, critic blind to
the realized body) produced a misleading signal — the diminishing-returns curve came out *inverted*,
the `p=0/1` corners looked out-of-distribution, and the value function was blind to per-leg structure
in un-walkable bodies. This version (a *deterministic-build* env: body = a clean stable center, `p` fed
to obs/value only) trains a much stronger policy (seed-avg return 3992 vs 2545 per body, see
[the variant comparison](#contrast-poor-vs-good-control)) and the signal cleans up: the
diminishing-returns curve is now **correct** (largest exactly where a generator must bootstrap a leg),
corners are well-behaved, and value is rank-calibrated to realized return. **Magnitude is still not a
reliable "which leg" signal** — but the *reason* changed, which is the most interesting finding (see
[the diagnosis](#why-magnitude-still-fails--but-differently-than-before)).

Full figures and the numeric pipeline live in
[`notebooks/value_grad_prop.ipynb`](../notebooks/value_grad_prop.ipynb); scripts are
`experiments/value_grad_prop.py` (Step 1), `value_grad_step2.py`, `value_grad_step3.py`,
`value_grad_phase2.py`, and `value_grad_ablation.py` (grounding). Previous (stochastic-build) results
are archived under `data/value_grad_prop/_archive_v2/`. Re-run on the `*_nobern` checkpoints 2026-06.

## Background

The codesign loop we want is: a generator emits a body, the trained critic scores it, and we backprop
`∂V/∂(morph)` into the generator to make it emit better bodies. That only works if the critic's
morphology gradient has **sensible sign and magnitude** and **survives composition with a generator
network**. This study tests both — and, across three env versions, shows that *all of it is gated by
the quality of the underlying control policy/value function*.

**Three versions, each fixing the last:**

1. **Binary `{0,1}` presence.** Failed in Phase 2: a discrete generator's saturated sigmoid collapsed
   the gradient to ~zero. (On clean bodies this policy is also the weakest — seed-avg return 869.)
2. **Continuous-`p`, stochastic build ("the previous experiment").** Presence is a probability
   `p ∈ [0,1]`; the body is built by `Bernoulli(p)` and the obs reports `p`, so `V(p) ≈ E[return]`.
   This fixed Phase 2 by removing the discrete bottleneck. **But the critic never sees the body it is
   actually controlling** (the policy acts under a body it can't observe), and `V` is smeared over a
   distribution of degraded bodies (with `√U` sampling, an "on" leg is physically built only ~⅔ of the
   time). The result is a **poor value function**: the codesign signal it produces is misleading
   (details throughout). Its low *logged* training reward (~320) was a metric artifact of averaging
   over junk bodies, but per-body it still walks (seed-avg return 2545).
3. **Continuous-`p`, deterministic build (this experiment).** The body is built **exactly = the stable
   bias center `S`**; `p` (a `√U` cloud around `S`) is fed to obs/value **only**, never to geometry. The
   policy always controls a real, clean stable morph → it trains to a much stronger policy (seed-avg
   return 3992) and `V(p)` learns the value of the center implied by `p`. This is the **good control**.

This document reports version 3 and contrasts it against version 2 throughout — *the same probes, run
on a poor vs a good control/value function.*

### The sign convention (read this first)

The quantity is **raw sensitivity** `∂V/∂p` — *which way a leg should move* — **not** an attribution of
how much a leg currently contributes. The sign tracks the *desired* state regardless of the current
one: a **helpful** leg is **+** (grow it) whether currently on or off; a **harmful** leg is **−**
(shrink it) either way. A critical-but-currently-*off* leg should be the largest **positive** gradient
("add me"). This is exactly what gradient ascent consumes.

## What was tested

An **experiment-only** ant variant (`AntBinaryLegEnv`, not the production controller):

- **All 8 legs always DOF-mask-active.** The production net zeroes masked tokens, giving them *exactly
  zero* length gradient. To get any presence gradient every leg must stay active; presence is carried
  purely by the input `p`.
- **Continuous presence `p ∈ [0,1]`**, tied across a leg's hip and ankle slots (`∂V/∂p` = sum of the
  two slot gradients). `p` is a *signal*, not a length.
- **Deterministic build (this version).** The body is built **exactly = the stable bias center `S`**
  (on-legs 1×, off-legs a hidden 0.05× stub). `p` is drawn `√U` for on-legs / `1−√U` for off-legs
  around `S` (equal mixture marginally `Uniform(0,1)` → uniform coverage of the knob) and fed to **obs
  and value only** — never to geometry. So the policy always controls a clean, fully-realized morph,
  and `V(p)` learns the value of the center `p` implies. *(Contrast: the previous version built
  `Bernoulli(p)` and the policy was blind to the realized body.)*
- **Held-out set.** All 56 five-leg topologies + 3 curated Step-2 sets are stripped from the pick-pool,
  so a center — hence a trained body — is never held-out, and Steps 2/3 can probe the gradient *off*
  the training set. (No Bernoulli draw to guard against anymore.)
- Two metrics per leg: **`ḡ`** (∂V/∂p averaged over a rollout — the signal a generator sees) and
  **`g0`** (at the t=0 reset pose — the static prior). **3 seeds** (s42/s43/s44), report mean±std.

The four probes and the conditioning check:

| | question | how |
|---|---|---|
| **Step 1** (myopia + shape) | is `∂V/∂p` positive at low `p`, and is it largest there (so ascent can grow a near-off leg, with diminishing returns)? | bin `ḡ` by `p` over the training population |
| **Step 2** (curated) | does the gradient match intuition on 3 held-out shapes and agree with a trained twin? | deterministic builds vs a mirror/rotation twin, at corner & interior `p` |
| **Step 3** (interpolation) | does the gradient generalize to held-out 5-leg bodies? | overlay held-out `∂V/∂p`-vs-`p` on Step 1 |
| **Grounding** | do `V` and its gradient predict *realized return*? | toggle each leg of 16 base bodies, correlate `V`/`∂V/∂p` vs realized `R`/`ΔR` |
| **Phase 2** (conditioning) | does the sign survive composition through a generator net? | pretrain a scatter-gather transformer, compose in front of `V`, backprop to its input |

## Results

### Step 1 — diminishing returns now correct, and no myopia

Premise sanity holds strongly: mean episode reward rises monotonically with center leg-count (3-leg
1230 → 4-leg 1891 → 6-leg 4992 → 7-leg 6656 → 8-leg 8193; 5 absent = holdout). The headline
`∂V/∂p`-vs-`p` curve (seed-pooled):

| p-bin | 0.0–0.1 | 0.2–0.3 | 0.4–0.5 | 0.6–0.7 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|
| **this (good control)** | **+10.2** | +10.6 | +8.6 | +7.2 | +4.5 | +3.3 |
| *prev (poor control)* | *+2.7* | *—* | *+7* | *+6.9* | *—* | *+6.2* |
| frac > 0 (this) | 0.99 | 0.99 | 0.99 | 0.98 | 0.92 | 0.87 |

**Positive everywhere at low `p` (frac > 0 ≈ 0.99) → no myopia**, and now the **magnitude has the
correct diminishing-returns shape**: *largest at low `p`* (+10.2, where a leg is near-off and most
worth adding) and decaying toward `p=1` (+3.3, redundancy). **This is the exact inverse of the previous
experiment**, where the signal was *weakest* at low `p` (+2.7) — smallest exactly where a generator
must bootstrap a leg. The fix came entirely from giving the critic clean, observable bodies to value.

Two supporting reads:

- **Split by body fullness:** both lean and full bodies are positive throughout; the diminishing-
  returns droop is steepest in **full bodies (≥6 legs)** (low `+13.3` → high `+3.0`, i.e. strong
  redundancy at the top), and gentle in **lean bodies (≤3 legs)** (`+7.6` → `+6.0`). The pooled
  low-`p`-high shape is dominated by the full-body curve. *(Previously this split was the only place
  diminishing returns appeared at all; now it is the pooled headline.)*
- **`g0` (static prior) vs `ḡ` (rollout):** weak agreement (corr 0.32, sign-agree 0.93). `g0` has
  **~1.9× the absolute spread** of `ḡ` (std 10.1 vs 5.4) — uniformly inflated and sign-flipping more.
  **Trust `ḡ`; the static reset-pose prior is a poor proxy.**

### Step 2 — corners are no longer OOD; per-leg twin agreement is noisy

This is where the *poor → good control* shift is most visible — and where the limits of the good
control's magnitude show up.

**(a) The `p=0/1` corners are well-behaved now.** The previous experiment's signature failure was that
training never placed off-legs at exactly 0, so corner gradients were distorted and flipped the
expected importance ordering — "evaluate at interior `p`, never the corners" was a headline caveat.
Under the good control this **largely dissolves**: on `critical_missing` (held `{5,6,7}`) the off
front/right legs that should say "add me" are already positive at the corner (1F +3.9, 2FR +6.3,
3R +6.4), and interior `p` only sharpens them (1F +6.2, 2FR +10.6, 3R +10.8). For the two *walkable*
curated pairs, the held body's per-leg pattern actually correlates with its trained twin **better at
the corner** than at the interior (redundancy +0.59, symmetric +0.73 at corner). The corner-OOD
artifact was a symptom of the poor value function, not a property of the probe.

**(b) But per-leg twin agreement is only moderate, and noisy.** Aligned held-vs-twin per-leg
correlations are middling on the walkable pairs (+0.59 / +0.73 at corner; weaker and even negative at
interior) and meaningless on `critical_missing` — that body barely moves (reward ~25 vs symmetric
~3200), so its gradient is noise (corr −0.55/−0.88). The takeaway is consistent with grounding below:
the **fine-grained per-leg ordering does not transfer cleanly** even under a good control — the
gradient is a reliable *sign/direction* signal, not a reliable per-leg *ranking*.

### Step 3 — the signal interpolates; adding a leg cleanly shrinks the rest

**(a) Held-out 5-leg interpolation.** The held-out 5-leg `∂V/∂p`-vs-`p` curve coincides with the
in-distribution Step-1 curve in sign and shape (slightly *higher*, as 5 legs sit between lean and
full):

| p-bin | low | mid | high |
|---|---|---|---|
| Step 1 (in-dist, pooled) | +10.2 | +8.6 | +3.3 |
| Step 3 (held-out 5-leg) | +12.3 | +10.4 | +5.6 |

→ **The codesign signal survives off the training set**, with the same diminishing-returns shape.

**(b) Per-limb add/remove** (mean `∂V/∂p` over present limbs, 3 held-out 5-leg bases):

| base | base | +1 leg | −1 leg |
|---|---|---|---|
| {3,4,5,6,8} | +7.95 | **+6.07** | +6.10 |
| {1,3,4,6,8} | +7.47 | **+6.10** | +7.21 |
| {3,4,5,7,8} | +8.01 | **+7.61** | +6.44 |

**Adding a leg consistently lowers every remaining limb's marginal value** (all 3 bases) — clean
per-body diminishing returns, consistent with the fullness split. **Removing is mixed** (no clean
symmetric "removal raises the rest" story — magnitudes drift without consistent direction), echoing the
per-leg ranking noise from Step 2.

### Grounding — value rank-calibrated, magnitude still unreliable

Steps 1–3 judged the gradient against *intuition*; this grounds it against *realized return*. For 16
in-distribution base bodies (4 each at 3/4/6/7 legs), we build each base plus its 8 single-leg
**toggles** (flip each leg on↔off) at interior `p`, and measure realized return `R`, value `V`, and
per-leg `∂V/∂p` (seed-avg over s42/43/44). Two tests:

- **Value calibration — strong (in rank).** `V` vs `R` over all 144 bodies: **Pearson +0.89,
  Spearman +0.87**. The critic orders bodies good→bad correctly. (Absolute `V` is still *compressed* —
  ~[105,136] vs realized 1252–6675 — because it predicts *discounted* return-to-go; calibration is in
  rank, not scale.)
- **Gradient grounding — right sign, unreliable magnitude.** `∂V/∂p_L` vs the realized marginal value
  `ΔR_L = R(with L) − R(without L)`: **sign-agreement 1.00** (every leg has positive realized value and
  the gradient is positive — so sign-agreement is real but partly trivial), but **Pearson −0.08** —
  magnitude does not predict *which* leg helps.

The by-fullness cut **inverts relative to the previous experiment**:

| base leg-count | 3 | 4 | 6 | 7 |
|---|---|---|---|---|
| Pearson(`∂V/∂p`, `ΔR`) — this | **+0.35** | −0.09 | −0.20 | +0.03 |
| Pearson — *prev (poor control)* | *−0.01* | *−0.05* | *+0.37* | *+0.35* |

The poor control got magnitude right only in *functional* bodies; the good control gets it (weakly)
right only in *sparse* bodies, and adds-side is the only consistent winner (add r=+0.45, remove
ρ=−0.17). Either way, **magnitude is not a dependable "which leg" signal** — but for a very different
reason, next.

### Why magnitude still fails — but differently than before

The previous experiment's diagnosis was a **value-learning failure at the floor**: in un-walkable
bodies the critic's value `V` had never learned which leg mattered, and the proof was that the critic's
*own finite-difference* `ΔV = V(with leg) − V(without leg)` failed *identically* to the gradient there
(both ≈ 0 in sparse bodies, both recovering only at 6–7 legs). You can't differentiate your way to
information `V` never contained.

**Under the good control, that diagnosis no longer holds.** Comparing the analytic gradient to the
critic's own finite-difference, by leg-count:

| legs | gradient `∂V/∂p` | finite-diff `ΔV` |
|---|---|---|
| 3 | +0.35 | +0.47 |
| 4 | −0.08 | −0.08 |
| 6 | **−0.20** | **+0.20** |
| 7 | **+0.03** | **+0.42** |

On **functional bodies (6–7 legs) the critic's `ΔV` is clearly positive** (+0.20, +0.42) — i.e. `V`
*does* now encode which leg matters — **yet the analytic gradient diverges from it** (−0.20, +0.03).
The good value function learned the per-leg structure (finite-diff recovers it), but its **local slope
in `p` no longer points the same way as a discrete leg toggle**. So the failure mode moved from "the
value function is blind" (poor control) to "the value function knows, but the *gradient* doesn't read
it out" (good control) — a gradient-vs-finite-difference divergence, plausibly because under the
deterministic build `∂V/∂p` is a sensitivity to the *obs signal* with the body held fixed, whereas a
real leg toggle changes the body. Practically: **on functional bodies, prefer the critic's
finite-difference `ΔV` over the analytic gradient for magnitude; use the gradient for sign/direction.**

### Phase 2 — the soft generator preserves the sign

A small transformer is pretrained (MSE 0.0005) to regress each leg's `p` from a **scattered** token
encoding (forcing attention to gather it), frozen, and composed in front of `V`; gradients backprop to
its input.

| metric | value | meaning |
|---|---|---|
| **sign alignment** | **0.953** | input-grad sign matches the direct `∂V/∂p` — the headline |
| magnitude corr | +0.586 | sign survives strongly; magnitude only moderately |
| chain norms (out→in) | 998 → 248 → 2641 → 1622 | **no collapse** (ends above where it starts) |

**The conditioning check passes**, as in version 2 — the continuous-`p` recast removes the binary
version's saturated-sigmoid collapse by construction, and the good control doesn't change that. Sign
(what ascent consumes) survives at 0.95; magnitude survives only moderately, as expected.

## Contrast: poor vs good control

The same probes, run on a value function learned over degraded/unobservable bodies (previous,
stochastic-build) vs clean/observable ones (this, deterministic-build):

| aspect | **poor control** (stochastic build) | **good control** (deterministic build) |
|---|---|---|
| what the critic values | `Bernoulli(p)` body, policy blind to it | clean center `S`, fully observed |
| per-body policy (seed-avg return) | 2545 | **3992** |
| Step 1 `∂V/∂p` vs `p` | **inverted** — weakest at low `p` (+2.7) | **correct** — largest at low `p` (+10.2) |
| `p=0/1` corners | OOD, flip the ordering | well-behaved (agree with twins) |
| value rank-calibration (Spearman V,R) | +0.93 | +0.87 |
| magnitude grounding (where it works) | functional bodies only (6–7: +0.35) | sparse bodies only (3: +0.35); add-side +0.45 |
| value at the floor | doesn't encode per-leg (`ΔV` fails w/ grad) | **does** encode it (`ΔV` +0.2/+0.4 on functional) |
| residual magnitude failure | value-learning gap at the floor | gradient ≠ finite-diff on functional bodies |
| sign (grounding / Phase 2) | 0.99 / 0.94 | 1.00 / 0.95 |

**The meta-finding:** the value gradient is only as good as the control policy/value function behind
it. A poorly-trained critic produces a *misleading* codesign signal (inverted shape, fake OOD corners,
floor blindness); a well-trained one produces the **right sign and shape** and a rank-calibrated value
— but a usable per-leg *magnitude* requires more than control quality alone.

## What we learned

- **The value gradient is a usable codesign signal on sign and shape — given a good control.** Positive
  almost everywhere (no myopia), correct diminishing-returns curve (largest at low `p`), interpolates
  to held-out bodies, and survives a generator network (sign-align 0.95). The directional premise
  holds, and holds *better* than under the poor control.
- **Control quality gates the signal.** Every qualitative failure of the previous experiment (inverted
  low-`p` shape, OOD corners, floor value-blindness) was an artifact of valuing degraded/unobservable
  bodies, and disappears under a policy that controls clean morphs. Diagnose the control before
  trusting the gradient.
- **Magnitude is still not a reliable "which leg" signal**, and now for a subtler reason: on functional
  bodies the good `V` *does* encode per-leg structure (its finite-difference recovers it, +0.2/+0.4),
  but the analytic `∂V/∂p` diverges from that finite-difference. Use the gradient for sign; use `ΔV`
  (or realized-return ablation) for magnitude.
- **Value is well-calibrated in rank** (Spearman +0.87) but scale-compressed (predicts discounted
  return-to-go), so read it as an ordering, not a return estimate.
- **Trust the rollout-averaged `ḡ`, not the static `g0`** — the reset-pose prior is a ~1.9×-inflated,
  weakly-correlated proxy.

## Caveats

- **All-mask-active ≠ production controller.** The deployed mask-based net gives masked tokens zero
  gradient, so this mechanism can't be read off it directly; productionizing presence-gradients needs
  the real controller retrained all-active. *Headline caveat.*
- **`p` is obs/value-only now.** `V(p)` is the value of the *center implied by* `p` with the body held
  fixed — not `E_Bernoulli[return]`. This is why `∂V/∂p` (obs-signal sensitivity) can diverge from a
  real leg toggle (`ΔV`) on functional bodies.
- **Per-leg ranking is noisy** even under the good control (Step 2 twin agreement, Step 3 removal,
  grounding magnitude) — the gradient is a direction, not a ranking.
- **`critical_missing` is a near-dead body** (reward ~25); its twin comparison is noise and should not
  be over-read.
- **Grounding bodies are in-distribution.** Stable, non-held-out bases at the interior operating point,
  so the correlations are a *best case*; magnitude on far-novel bodies is likely worse, not better.

## Next

The capstone the premise actually demands: a **gradient-ascent test** — freeze `V`, ascend the raw `p`
vector (and through the generator), and check it converges to known-good morphologies. With the good
control, Step 1 predicts ascent now bootstraps *fast* (the signal is largest at low `p`) and gets the
direction right; grounding predicts it may still mis-rank *which* leg from a given body. Two leads: (1)
for magnitude, **use the critic's finite-difference `ΔV` rather than the analytic gradient** on
functional bodies (it recovers per-leg structure the gradient misses), and probe why the two diverge;
(2) accept the gradient as a *sign/shape* proposer and let realized-return ablation resolve magnitude.
Beyond presence: continuous **segment-length** (`∂V/∂length`) and **leg-angle** (`∂V/∂θ`) gradients.
