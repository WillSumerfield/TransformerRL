# Experiment 4 — Attention over module tokens

Slug `attention`. **Off-protocol**: [ADR-0021](../adr/0021-paper-experiment-metric-protocol.md)
assumes a generator, and this experiment deliberately has none. Its measurements are defined here.

## Question

The control policy reads the body as a set of module tokens and emits one action per token
(`_action_mu`, `architectures.py:454`). Attention is what lets a limb's action depend on the *other*
limbs' states rather than only its own. **Is that cross-token information actually used, and is it
worth the mechanism?**

This is the one experiment in the series that is about the network rather than the algorithm, so it
is stripped to the network: **one fixed morphology, no generator, no resampling, no aux heads.** The
canonical base ant, all 4096 envs, for the whole run. Nothing varies except which tokens a token may
look at.

Answering it matters because everything else in the series assumes the token-set encoder is the right
substrate — experiment 1 argues about *sharing* a trunk, experiment 2 about what the trunk *holds*,
and both presuppose the trunk's attention is doing work. If a limb only ever needs its own state and
the torso's, the transformer is machinery in place of an MLP, and the paper should say so.

## Conditions

Three arms, distinguished **only** by the attention mask on the control encoder. Token layout is
`[CLS] [start × n_slots] [module × n_slots·max_depth]` — `n_tokens = 1 + n + n·max_len` and
`_content_start = 1 + n` (`architectures.py:265`). For the ant's `simple` library that is
**n_slots = 8, max_depth = 4**, so `[CLS] [start × 8] [module × 32]` = **41 tokens** with
`_content_start = 9`. CLS carries the torso observation (`embed_root`) and the start tokens are
batch-independent learned compass anchors carrying no state at all. The figure derives the layout
from the run's own library rather than restating it, since a library change moves both numbers.

| arm | a module token may attend to | what it has |
|---|---|---|
| `full` | every token | own state, torso, every other limb — the shipped network |
| `self_cls` | itself and CLS | own state and torso; **no inter-limb information** |
| `self` | itself only | own state alone — no torso either |

**All three are parameter-identical.** The ablation is a boolean mask handed to
`F.scaled_dot_product_attention`; every projection, the FFN, RoPE and the residual stream are
untouched, and `n_layers: 1` means there is exactly one round of token mixing to switch off. No other
architecture ablation in this series is free of a capacity confound, and this one is.

`self` is an **anchor, not a hypothesis.** An ant plainly needs torso orientation and velocity to
walk, so it is expected to collapse. Its job is to make `self_cls` legible: "limbs need the torso but
not each other" is a strong statement only when removing the torso is demonstrably catastrophic on
the same axes.

## Held fixed

One script, one config, one mask. Every hyperparameter, the seed body, 4096 actors, seeds 42–49.

Relative to the codesign runs, three things are switched off and must be off in **all three arms**:

| setting | value | why |
|---|---|---|
| `resample_interval` | 0 | `_maybe_resample` returns immediately; the env keeps its window-0 build, the canonical base morphology, for the whole run |
| `fd.enabled` / `fk.enabled` | false | FD and FK are per-step control-side terms fused into the PPO loss — they do **not** switch off with the generator. Experiment 2 owns that variable |
| generator | never fires | with no resample there is no generator update and no pretrain; `n_pretrain` is inert |

Two consequences of `interval = 0` that are silent and easy to miss:

- **`lr_warmup` does not apply.** It is gated on `interval` being nonzero (`codesign_agent.py:167`),
  so the scheduler stays rl_games' plain `AdaptiveScheduler`. Uniform across arms, so it does not
  confound the comparison, but this run is *not* on the same LR schedule as experiments 1–3, and any
  cross-experiment return comparison is invalid. The same is true of every metric-5 specialization
  pass.
- **`codesign_tokens` stays `true`.** The point is to test the encoder the rest of the paper uses.
  The legacy `ppo_ant.yaml` / `train_transformer.py` path is a *different* tokenization
  (`_encode_legacy`, `[CLS][eff0×4][eff1×4]`, `_content_start = 1`) and is not a substitute.

## Process

Three mixed waves of 8 seeds, 3000 epochs. No ladder pass, no specialization pass — with one fixed
body there is no design distribution to perturb and nothing to specialize onto.

Runs alongside experiments 2 and 3: it needs no Phase5 port and no new arms.

## Measurements and decision metric

ADR-0021's five metrics do not apply. Metrics 3, 4 and 5 need a generator; metric 1 reads
`quality/R_mean`, which is written by the generator log and never appears in these runs. This
experiment's measurements:

| # | measurement | source |
|---|---|---|
| A | **Return curve** — return vs epoch on the fixed base body | `control/r_step`, per epoch |
| B | **Asymptotic return** — mean over the final 200 epochs | derived from A |
| C | **Sample efficiency** — epochs to first reach a fixed return threshold, set from `self_cls`'s asymptote | derived from A |
| D | **Gait diagnostics** — `episode_lengths/iter` (falling over), `control/sigma_mean` (entropy collapse), `control/adv_std` | TB |
| E | **Attention structure** — attention mass from module queries split into self / CLS / start-anchors / other-modules, traced across checkpoints, with the state-averaged `n_tokens × n_tokens` map as supporting evidence. `full` arm only | eval-time diagnostic |

**Decision metric: B, asymptotic return**, mean over 8 seeds with 95% CI against the study's noise
floor per [ADR-0018](../adr/0018-noise-floor-first-tuning.md).

C is reported alongside and can dissociate from B: attention may reach the same gait faster without
reaching a better one. That is a real and different finding — "attention is an optimization
convenience, not a representational necessity" — and it is only visible if C is committed to in
advance.

**E is what makes a positive result interpretable.** If `full` beats `self_cls`, the learned map must
actually attend across limbs; a near-diagonal learned map paired with a return gap means the gap came
from something other than cross-token information and the result is not yet explained. It needs an
eval-time-only manual softmax path, since `F.scaled_dot_product_attention` does not return weights.

**The reported number is the mass split, not the map.** Attention mass is linear in the weights, so
cross-limb mass survives the state-averaging exactly (mean-of-mass = mass-of-mean). The map's
*pattern* does not: a limb attending to its contralateral partner at one gait phase and a different
limb at another averages into diffuse mass over both, and the pairing is gone. So the fraction is the
robust reading and the map is the evidence behind it.

## Expected results and falsifier

**Expected.** `full > self_cls ≫ self` on B. The mechanism: an ant gait is a phase relationship
between legs, and a limb whose action cannot depend on the other limbs' states must encode that
relationship implicitly through the torso token alone — a bottleneck of one token's worth of
bandwidth. The gap should be visible in D as well, as shorter episodes (falling) in `self_cls`.

**Falsified if `self_cls` matches `full`.** The single torso token would then carry enough for
coordinated gait, cross-limb attention would be unused, and the token-set encoder is doing the work
of a per-limb MLP with a global feature vector. Experiments 1–3 remain valid — they are about the
trunk's *content* and *coupling*, not its attention — but the paper's framing of the architecture has
to change, and a cheaper encoder becomes the honest default.

**Interpretive hazards:**

- **Generalization is not measured here, by construction.** One body means this experiment says
  nothing about whether attention helps control *transfer* between morphologies, which is the
  property experiment 2 cares about and the one attention over a variable module set is most plausibly
  for. A null here is a null *on a fixed body only*, and must be reported that way. The natural
  follow-up is the same three masks inside the full codesign loop.
- **Off-body-distribution absence cuts both ways.** With one morphology, the module tokens' body
  content is constant for the whole run, so `full`'s attention only ever has to solve one
  configuration. It may underperform its own potential relative to a run that must handle many
  bodies — which biases this experiment *against* attention. Conservative if `full` wins, a real
  problem only if `self_cls` does.
- **The base morphology is one sample.** The canonical base ant is a single point in design space,
  and its leg count and symmetry determine how much cross-limb coordination is worth. Re-running one
  arm pair on a second fixed body is the cheap robustness check.

## Where it lands

The architecture section's foundation, before experiments 1 and 2 in the paper even though it runs
alongside them: it establishes that the token-set encoder is the right substrate, which is what the
other two then argue about sharing and filling. Its off-protocol status is stated where it appears —
one body, no generator, its own five measurements.
