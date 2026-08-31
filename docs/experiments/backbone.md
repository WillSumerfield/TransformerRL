# Experiment 1 — Shared backbone

Slug `backbone`. Protocol: [ADR-0021](../adr/0021-paper-experiment-metric-protocol.md).
Split arm: [ADR-0016](../adr/0016-split-backbone-codesign-distillation.md).

## Question

Control and the generator share one transformer trunk. **What does that sharing actually buy, and
is it something a cheaper coupling could buy instead?**

The claim the architecture rests on is that a shared trunk performs *representation transfer*: the
encoder learns, from control's rollouts, what a body's morphology implies about how it moves, and
the generator reads that same encoder when it designs. If true, the generator inherits an
understanding of bodies it never had to learn from its own sparse signal — 4096 scalar returns per
window — and control in turn stays valid across a wider set of bodies, which is what permits the
search to be global rather than local.

That claim has a cheaper rival, and the experiment exists to separate them. A generator with its own
weights can be *taught* the control policy by distillation, acquiring its function without sharing a
parameter. If that works as well, the shared trunk is an implementation convenience and the paper
should say so. If it does not, then something about sharing — a single representation shaped by both
objectives at once, rather than one copied into another — is load-bearing, and that is a claim about
codesign architectures generally, not about this codebase.

So the experiment is not "does coupling help" (it obviously does) but **which coupling channel
carries the benefit**, which is why it has three arms and not two.

## Conditions

A ladder of decreasing coupling. Every arm has the same control net and the same task.

| arm | weights | channels control → generator |
|---|---|---|
| `single` | one trunk, one optimizer | everything — one representation shaped by both objectives |
| `split` | disjoint trunks **and** disjoint token embeddings | shadow ContAct'/ContCrit' distilled over a transition reservoir; shared normalizer and FK stats |
| `decoupled` | disjoint trunks and embeddings | **R only** — one scalar per env per window. Generator's sole obs-space signal is FK at rest poses, against ground truth |

The middle rung is what makes this diagnostic. `single` vs `decoupled` alone would show that
coupling helps but could not distinguish trunk-sharing from representation transfer, because `split`
gets the latter through a different mechanism while holding weights disjoint.

**"Shared backbone" here means backbone *plus its mitigation*.** The `single` arm runs generator
preservation (`gen_preserve_beta/lam` = 1.0, 256 bodies) — control PPO takes ~992 optimizer steps
per window against the generator's 368, and each moves nearly all of `_encode_design`, so without it
the generator is dragged by control. `split` and `decoupled` set `_genclone_on = False`; with
disjoint weights control cannot move their generators, and leaving it on would push gradient through
the *control* trunk to stabilise design outputs nothing reads. The mechanism therefore exists on
exactly one rung — correctly, since trunk-sharing is what creates the interference it fixes.

## Held fixed

Task (`ant_codesign`, `simple` module library), seed body, budget (48 windows = 1536 epochs,
`n_pretrain: 7`), `num_actors: 4096`, `horizon_length: 32`, seeds 42–49, and every shared
hyperparameter — `learning_rate`, `entropy_coef`, `gencrit_coef`, `grad_norm`, `e_clip`,
`critic_coef`, `weight_decay`, `lr_warmup`/`warmup_epochs`, the `generator` warmup block, and
`fd`/`fk` both enabled at the tuned coefficients (`fd.coef 0.0074`, `fk.coef 0.0034`).

Each arm's config is `extends: ppo_ant_codesign_tuned.yaml` plus `algo.name` plus its arm-unique
block only, so an arm cannot silently drift off-topology. Optimizer steps per window are already at
parity (~368 in the generator update for all three), so the epoch budget is a fair budget.

**Not held fixed, deliberately: parameter count.** `single` is 454,744 params; the disjoint arms are
~909,488, because a second trunk *is* the treatment. Matching downward was rejected in the split
config's own header — it would halve the control trunk, giving weak control → weak R → weak
generator signal, a worse confound than the one it fixes. The gap is conservative under the expected
result: `single` winning with half the parameters is a stronger finding, not a weaker one. Only if a
disjoint arm wins does capacity become a live alternative explanation, and then a `single_wide` arm
(`d_model=228`, `ffn=912`) is the follow-up.

## Process

1. Port the six `Phase5` commits onto the paper branch and re-smoke all three arms.
2. Launch three waves of 8 seeds — `single`, `split`, `decoupled` — 48 windows each, ~3 h/run.
3. Each run dumps its per-window generator population (48 npz) and saves a checkpoint at every
   window boundary, which is what puts one at each of the three ladder points: pretrain→RL boundary
   (w=8, epoch 504), mid-RL (w=28, epoch 1764), final (w=47, epoch 2961).
4. Run the spread ladder at each checkpoint of each run → metrics 3 and 4.
5. Run the specialization pass at each checkpoint of each run → metric 2. Committed body on all
   4096 envs, `resample_interval: 0`, aux off, 250 epochs.
6. Scrape, chart, write findings in the notebook.

Roughly 9 h of training waves plus 9 waves of specialization and the ladder passes.

## Measurements and decision metric

All five of ADR-0021, on the shared window axis with the pretrain→RL boundary ruled at w=8.

**Decision metric: specialized return at the final checkpoint** — mean over 8 seeds with a 95% CI,
read against the noise floor. This and not the return curve, because metric 1 is a joint
body×control score that charges an arm for having stayed general; specialized return gives every arm
the same chance to collapse onto its own design, so what is compared is the designs the methods
produced. Metric 1 is reported as the training-dynamics curve, and 3–5 as the mechanism.

**Reported together, not optional:** metric 3's valid width against metric 2. An arm that wins
metric 2 while its control-generalization curve collapses got a good design without getting good at
codesign — it found one body and specialised, which is the local-search story. The pair is the
finding; either alone is not.

## Expected results and falsifier

**Expected.** `single > split > decoupled` on specialized return, with the gap between `single` and
`split` smaller than between `split` and `decoupled` — distillation recovers much, but not all, of
what sharing gives. Metric 3 shows `single`'s control staying valid furthest out, and metric 5 shows
`decoupled` either collapsing early (no signal to search with) or wandering without improving.

**Falsified if** `split` matches `single` on specialized return *and* on metric 3's valid width. The
shared trunk would then be an implementation convenience: everything it provides is obtainable by
distilling into a separate network, and the architecture section of the paper has to be rewritten
around that.

**Also falsified if** `decoupled` matches both. That is the stronger negative — it would mean the
generator needs nothing from control beyond the return signal, and that the aux heads, the shared
representation and the distillation channel are all decoration.

**Two results that would be uninterpretable rather than informative:**

- `split` losing while its distillation coefficients are untuned. `distill_act`/`distill_crit` are
  the treatment strength — set them to 0 and `split` degenerates to roughly `decoupled` — so a
  `split` null at the wrong weight collapses the ladder back to the two-arm experiment. See open
  questions.
- `decoupled` losing on the strength of its FK signal. Its own design note records this: FK at a
  rest pose is close to a deterministic function of the morphology, which design mode already reads
  as tokens, so a `decoupled` null cannot separate "transfer doesn't help" from "FK had nothing to
  transfer". Watch `gen/fk` against `build/*` diversity before reading that arm.

## Where it lands

The architecture section's central justification. It is also the experiment the other three
ablations are scoped against: 2 (JEPA), 3 (PPG) and 4 (attention) all modify what the *shared* trunk
learns, so they presuppose this one's result. If the backbone turns out not to be load-bearing, the
later ablations are measuring a component that does not matter, and the series is re-ordered.
