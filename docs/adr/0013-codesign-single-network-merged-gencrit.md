---
status: accepted
---

# Codesign is a single network: control + generator share one trunk, V1.0 merged into GenCrit (supersedes ADR-0012 and ADR-0008)

## Context & Decision

[ADR-0012](0012-codesign-generator-sequential-token-ppg.md) put the generator on **its own net** (a
policy head + an auxiliary value head `v(prefix)` distilled toward a **separate** control-side V1.0
head), while [ADR-0008](0008-codesign-via-ppg-three-nets.md) had **three disjoint nets**. Both are
replaced. The sequential per-token generator, the marginal-value advantage, the ≥1-leg guard, and
the pretrain BC ramp from ADR-0012 **carry over unchanged**; what changes is the *network topology*
and the *value function*.

We put control and the generator on **one `LegTransformer` trunk** under one optimizer, as **four
heads**:

- **ContAct** — control actor: per-DOF actions for the live body. Trained per rollout (PPO).
- **ContCrit** — control critic: the **V0.98** value driving ContAct's advantages (γ=0.98).
- **GenAct** — generator actor: sequential **limb/stop** tokens, randomized slot order. Trained per
  resample.
- **GenCrit** — generator critic **= the V1.0 body-quality head, merged**. Trained per resample.

**The merge.** GenCrit *is* V1.0 — one value head, not a distill pair. It is fit on **two data
sources toward the same body-quality target**: rollout states (per-step return-to-go) **and**
generation-token prefixes (toward the body's realized `R`). The same trunk+head reads it in **live
mode** (a real running body) and **design mode** (a partial designed prefix), yielding the
marginal-value advantage `v(prefix+token) − v(prefix)` directly — no separate aux value, no distill
lag. The standalone **time-aware V1.0 head** of ADR-0012 (`value_size==2`, its progress/time obs
dim) is **removed**: `R` is now the body's **true mean completed-episode return** over the window
(γ=1), pooled across envs, not `V1.0(s0)`.

**Schedule.** Per step: **plain combined PPO** on control (ContAct + ContCrit) — the trunk moves
freely, generator heads get no gradient. At each **resample** (window boundary): one joint
shared-trunk update fits GenCrit (rollout states + designed prefixes → `R`), updates GenAct (PPO on
the marginal-Shapley advantage, or BC in pretrain), and **clones control** — **β·KL[ContAct_old ‖
ContAct] + λ·MSE(ContCrit, ContCrit_old)** — so the joint step doesn't drift the controller. This is
**not** a PPG distill phase: there is no separate aux head to distill.

## Considered Options

- **Keep the separate generator net + distill pair (ADR-0012).** Rejected: ~2× params and a
  distill *lag* between the generator's aux value and the real V1.0. Merging makes one head both the
  critic and the target, and shares control/generation representations on one trunk.
- **Keep the standalone time-aware V1.0 head (`value_size==2`).** Rejected: redundant once GenCrit
  regresses both rollout states and prefixes to `R`; the extra progress obs dim complicated
  tokenization for no measured gain. Dropped, along with `R = V1.0(s0)`.
- **Let control train freely at resample (no clone).** Rejected: the shared-trunk generator step
  drags control off-policy (reward saw-tooths at window boundaries). The β/λ clone holds it; the
  two coefficients are tunable.
- **Three disjoint nets (ADR-0008).** Rejected: maximal params and zero shared representation
  between control and generation.

## Consequences

- **Reverses** ADR-0012's separate-net + distill-pair generator and ADR-0008's three-net codesign.
  The marginal-value-difference analysis (why the *baseline* must not read the body) still stands.
- One trunk instead of two/three — recovers the parameter and memory cost.
- The value head must be correct in **both** modes: a design-mode encode/mode bug silently
  body-invariants the prefix values and kills the marginal signal. This is the first symptom to
  rule out — see the debugging playbook in [`docs/reference/codesign_metrics.md`](../reference/codesign_metrics.md) §1.
- Old codesign checkpoints (separate-net generator, or `value_size==2` control) will not load.
- Enables the restructured `gencrit/` metrics and the `quality/R_mean` tuning objective
  (`configs/tune_config.yaml`).
- **Future:** multi-segment limbs (the general autoregressive case), a generator V0.98 aux head +
  best-next-token guidance.
