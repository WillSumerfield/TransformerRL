---
status: superseded by ADR-0013
---

> **Revision:** the **separate-net + distill-pair** framing here is superseded by
> [ADR-0013](0013-codesign-single-network-merged-gencrit.md) — control and generator now share **one
> trunk**, and the generator's aux value is **merged** with the control V1.0 head into a single
> GenCrit head (no distill pair, no standalone time-aware V1.0 head; `R` is the true window return,
> not `V1.0(s0)`). The **sequential per-token generation, the marginal-value advantage, the ≥1-leg
> guard, and the pretrain BC ramp below all still stand** — ADR-0013 only changes the network
> topology and the value function.

# The codesign generator is a sequential per-token PPG with a marginal-value advantage (supersedes ADR-0010's generator)

## Context & Decision

[ADR-0010](0010-codesign-generator-unconditional-bandit.md) made the generator an **unconditional
8-Bernoulli presence bandit** to dodge the *body-agnostic-baseline trap* (a body-conditioned value
baseline makes the γ=1 advantages average to ~0, killing the selection gradient). That worked for
v1 (it converges to ≈all-8) but it cannot model leg interactions or grow multi-segment limbs, and
it is a dead end for the intended **general autoregressive** morphology design.

We **replace the generator** with a **sequential, token-at-a-time** policy. This env (max limb
length 1) is the degenerate warmup for the general design; the machinery validated here
(autoregressive generation, per-token marginal-value credit, the two-value-function scheme)
carries over to multi-segment limbs unchanged.

- **Generation is a small MDP.** State = the committed token prefix; action = the next token;
  reward is paid **only at the terminal token** as the scalar `R` (body quality); γ=1. Slots are
  decided **sequentially in randomized order**, each conditioned via attention on already-committed
  tokens — this conditioning is the gain over the independent bandit. Vocabulary per slot:
  **limb** (active) or **stop** (off). The generator reuses `MultiMorphLegTransformer`.
- **The generator is a PPG.** Its net carries a **policy head** (limb/stop) and an **auxiliary
  value head** `v(prefix)` on a shared trunk. The aux value head is **periodically (once per
  resample window = one phase) distilled toward the "real" value function** — which lives in the
  *control* net (below). For v1 the policy update and the value distillation happen jointly at the
  window boundary (the degenerate one-update-per-phase case).
- **The real critic is a new control V1.0 head.** The combined-PPO control net (ADR-0010) gains a
  **second value head**, V1.0: γ=1, **truncation→0** (the rl_games default here, since `time_outs`
  is not exposed), **time-aware** (a normalized time-remaining feature feeds *only* this head so
  the finite-horizon value is unbiased). It is trained on real returns every control epoch and is
  **isolated from the actor's advantages** (the actor still uses V0.98). The per-env body quality
  is `R_i = V1.0(s0_i)`, read at the window boundary and averaged over the window's reset states.
- **Marginal-value advantage.** The advantage of a committed token is
  `A = v(prefix+token) − v(prefix)`. Because order is randomized and every prefix regresses to the
  same `R`, `v(prefix)` → the expected `R` over completions, so the difference isolates the token's
  marginal contribution (a Shapley-style estimate via random orderings). This is a **difference of
  two body-conditioned values**, which is non-zero even though each term reads the body — that is
  how it escapes ADR-0010's baseline trap.
- **≥1-leg guard** via **stop-masking on the forced last slot**: when committing the final
  undecided slot with all prior decisions = stop, the stop action is masked so the policy must emit
  limb; the masked log-prob is consistent for the PPO ratio.
- **Pretrain** = teacher-forced per-step BC toward base±flip bodies (cross-entropy on limb/stop at
  each step); the value head distills `R` exactly as in RL; ramp base→generator over `N_pretrain`
  windows, then PPO-clip RL.

## Considered Options

- **Keep the unconditional Bernoulli bandit (ADR-0010).** Rejected: structurally cannot model
  leg interactions or multi-segment limbs, and is not a step toward the general design.
- **A single body-conditioned value as the generator baseline.** Rejected: the body-agnostic
  baseline trap — advantages average to ~0. The marginal-value *difference* is what avoids it.
- **Use the realized γ=1 window return as `R`.** Rejected: at `resample_interval=1` each per-env
  body sees ~1 completed episode/window, so the raw return is a single high-variance sample. The
  control V1.0 head pools all 4096 envs' single episodes into a learned, cross-body-denoised
  estimate.
- **Disjoint generator nets / full PPG phase separation now.** Deferred: at ~24 generator updates
  per run, joint policy+value updates per window suffice; true phase separation is a later
  refinement if trunk interference appears.
- **A second generator aux head (V0.98) now.** Deferred: noted as future work — a second aux head
  could enrich generator features and enable a head that predicts the best next token to add
  (generation guidance).
- **Add the time feature to the shared obs.** Rejected: that would make the control *actor*
  time-aware and change its behavior. The feature goes to the V1.0 head only.

## Consequences

- **Reverses ADR-0010's** "unconditional, no V1.0 head" generator decision. ADR-0010's control-side
  framing (combined PPO) stands; the body-agnostic-baseline analysis stands as the reason the
  *baseline* (not the advantage) must avoid body input.
- Control gains a second value head (`value_size=2`) and a second γ=1 returns scan. Both heads are
  produced in **one forward pass**; the extra GAE is a cheap buffer scan (no second rollout).
- The generator does **8 sequential passes per body per window** (negligible beside the gym
  rebuild). One generator update per window still (~24/run); the per-token structure yields 8·N
  (prefix, token, advantage, old-logp) samples per update.
- Old codesign checkpoints (bandit generator) will not load.
- **Future:** multi-segment limbs (the real autoregressive case), the generator V0.98 aux +
  best-next-token head, and true PPG phase separation.
