---
status: superseded by ADR-0012
---

> **Revision:** the generator design here (unconditional Bernoulli bandit) is superseded by
> [ADR-0012](0012-codesign-generator-sequential-token-ppg.md) — the generator is now a sequential
> per-token PPG with a marginal-value advantage. The **body-agnostic-baseline** analysis below
> still stands as the reason a *baseline* must not read the body; ADR-0012 avoids the trap by using
> a marginal-value *difference* as the advantage instead.

# The codesign generator is an unconditional Bernoulli bandit (supersedes ADR-0008's generator)

## Context & Decision

[ADR-0008](0008-codesign-via-ppg-three-nets.md) specified the morphology generator as a
**state-conditioned `LegTransformer`** with a presence policy head + a PPG **auxiliary value
head**, drawing advantages from a shared **two-head critic** (the V_1.0 head). Working through
Phase 2 we found that design has a structural flaw and is far more machinery than the v1 problem
needs. We **replace the generator** with an **unconditional per-leg-presence bandit**:

- **Generator = 8 learnable Bernoulli presence logits + 1 scalar return baseline.** No trunk, no
  observation input. An 8-arm independent-Bernoulli bandit.
- **Self-baselines.** Advantage = `window_return_i − V`, where `V` is the scalar (the optimal
  variance-reduction baseline = `E[return]`). **No V_1.0 head on the control/critic net** — the
  generator is fully decoupled from the control value architecture.
- **Acts once per resample window** at the existing `_maybe_resample` seam: sample presence
  per-env → set bodies (fixed **default** lengths, presence-only) → full gym `_rebuild()` → run
  window → reward = per-env **mean completed-episode return** (γ=1).
- **Control = combined classic PPO** for Phase-2 v1 (shared actor-critic, `LoggingA2CAgent`). The
  ADR-0008 PPG-control path is kept but the PPG-vs-combined A/B is **deferred**.
- **Pretrain/warmup** ported from `ValueGradGenerator` (base `[1,4,6]` ± per-leg flip, ramp
  base→generator) but the old `root_k` 0.9/0.1 margin and 3000-step continuous-`p` regression are
  replaced by **behavioral-cloning to the supplied bodies + an entropy bonus**.

## Considered Options

- **State-conditioned transformer generator + aux head + 2-head critic (ADR-0008).** Rejected:
  the body-quality signal collapses. If the V_1.0 baseline reads the chosen body (it reads obs,
  which encodes presence/lengths), per-step γ=1 advantages average to ~0 **regardless of body
  quality** — no selection gradient. The baseline must be body-agnostic; the cleanest guarantee
  of that is a generator with **no morphological input** at all.
- **Summed per-step control γ=1 advantages as the generator advantage.** Rejected: same ~0-signal
  trap, for the same reason.
- **Shared-trunk *unconditional* actor-critic (value shapes morphology via the trunk).**
  Rejected: with a constant input the trunk maps constant→constant, so there is no representation
  to share — the value→morphology coupling is notional and only couples a bias. Reverted to a
  plain scalar baseline.
- **Make the generator conditional now ("next-best-token" sequential design).** Deferred: it
  reopens the body-agnostic-baseline trap and is the intended *future* design where per-morphology
  value and real trunk feature-sharing become meaningful.

## Consequences

- The generator is tiny and **unit-testable in isolation** (feed synthetic returns; assert logits
  climb). It is a standalone **component** wired into the agent at `_maybe_resample`; the same
  component drops into `PPGAgent` unchanged if/when the PPG-control A/B is revisited.
- **Update cadence is rebuild-bound.** The generator gets exactly one (action, reward) tuple per
  env per window, so it can only update once per window. At `resample_interval=1` (~62 epochs/
  window) a 1500-epoch run yields only ~24 updates; Phase-2 raises **`max_epochs` to 3000** for a
  legible base→optimum curve. Rebuild overhead ≈ 6 min/run.
- **v1 expected behavior is collapse to ≈all-8** (no leg/energy cost yet). Costs and richer
  attributes (lengths, angle) come later. The independent-Bernoulli generator cannot model
  leg-interaction constraints; acceptable while the optimum is all-8, and the future conditional
  design handles interactions.
- ADR-0008's **control-side** decisions (PPG, disjoint nets, V_0.98 flow) remain valid and Phase-1
  `PPGAgent` is untouched — it is simply not exercised by Phase-2 v1.
- Bodies are presence-only at fixed default lengths. Sampling is **per-env** (`envs_per_morph=1`),
  so the env always builds `num_actors` groups (the full-ant path) — no dedup. This maximizes the
  bandit's raw sample count (one body per env) at the cost of a constant, expensive rebuild.
