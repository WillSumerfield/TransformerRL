---
status: superseded by ADR-0013
---

> **Superseded by [ADR-0013](0013-codesign-single-network-merged-gencrit.md):** the three-net
> codesign architecture is fully replaced by a **single shared trunk** (control + generator as four
> heads). The generator half was already superseded by
> [ADR-0010](0010-codesign-generator-unconditional-bandit.md) → ADR-0012; the **control** half here
> (PPG, disjoint nets *for codesign*) is now retired too — codesign control trains as combined PPO
> on the shared trunk. (Dual-network PPG survives only as a stand-alone control baseline, unrelated
> to codesign.)

# Codesign trains the morphology generator with classic PPG over three separate nets

## Context & Decision

We are replacing the morphology generator's **differentiable value-ascent** training (generator emits a continuous presence distribution, spliced into obs, trained by ascending the detached control critic via `∂V/∂obs`) with **classic policy-gradient**. The value-ascent implementation is preserved on the `ValueGradGenerator` branch. The codesign system becomes **three separate `LegTransformer` nets trained as standard PPG** (Cobbe et al. 2020, `temp/ppg.pdf`):

- **Actor net** — control policy head + PPG auxiliary value head (γ=0.98).
- **Generator net** — morphology (per-leg Bernoulli **presence**) policy head + PPG auxiliary value head (γ=1.0). State-conditioned (reads obs); acts **once per resample window** as a body→body transition policy. Its body is applied to the env via `set_next`/`resample` and **never** enters the control per-step action stream.
- **Critic net** — disjoint value net with **two heads**, `V_0.98` (actor advantages) and `V_1.0` (generator advantages).

The generator is credited per **window decision** at the window-start state `s₀`, with advantage = the **aggregated** per-step γ=1 advantages over the window. Pretrain-to-base-morph and a warmup freeze (critic stabilizes before the generator's PPG begins) are retained. Implementation is phased: **(1)** convert control to PPG and verify parity with current PPO on the all-active env, **(2)** add the generator, **(3)** tune. We author a **custom PPG agent** that reuses rl_games components (ExperienceBuffer, GAE, `common_losses`, normalizers, runner) — rl_games' single-policy `A2CAgent` cannot host two policy nets + PPG aux phases.

## Considered Options

- **Differentiable value-ascent (the prior design).** Rejected: a chain of hard failures — flat `V(p)` (unconditional broadcast), a silent torch.compile backward that zeroed `∂V/∂obs`, and irreversible sigmoid saturation. The signal stayed weak even after fixes.
- **Morph head on a shared trunk (intermediate proposal).** Rejected: re-couples the generator to PPO's trunk drift (the exact problem the embedding-decoupling removed), forcing a per-step pinning servo. PPG's disjoint nets solve interference without a servo.
- **Single shared critic (one γ).** Rejected: the generator needs γ=1 (average within-episode deviations over the full undiscounted episode return); the actor keeps γ=0.98. Different γ ⇒ different value functions ⇒ two value heads.

## Consequences

- **Reverses two prior decisions:** the value-gradient study's direct-gradient codesign loop, and (by re-introducing a separate full generator net) the unconditional/base-token generator. The value-ascent code lives on `ValueGradGenerator`; old codesign checkpoints will not load.
- The torch.compile obs-gradient hazard **disappears** — PPG/distillation are parameter-grad losses, so the eager-inner-net workaround is no longer needed.
- The control pipeline itself changes (shared actor-critic → PPG actor + disjoint critic); Phase 1 must confirm no regression before codesign is added.
- γ=1 is safe only because episodes truncate at `max_episode_length` and GAE resets at dones.
