# Experiment 2 — Auxiliary prediction heads

Slug `aux`. Protocol: [ADR-0021](../adr/0021-paper-experiment-metric-protocol.md).
Head definitions: [transformer_rl/CONTEXT.md](../../transformer_rl/CONTEXT.md#auxiliary-prediction-heads).

> **Naming.** The paper calls this the JEPA experiment; the code does not. In this repo `config.jepa`
> is the masked-token I-JEPA head, which is **disabled** and not under test. What is ablated here is
> **FD** (`fd_variant: latent`, JEPA-*like*: predicts its own embedding of the next state against a
> stop-grad target) plus **FK** (ground-truth torso-frame pose regression, not JEPA at all). The
> experiment is named for the mechanism, not the acronym.

## Question

PPO shapes the trunk with one scalar per step. The aux heads shape it with a dense, structured
target every step, for free — both are fused into the existing PPO loss and add zero forward passes.
**Does that denser signal build a representation that transfers between bodies, or does PPO's
advantage already contain everything the encoder needs?**

The claim is specific, and it is the claim experiment 1 depends on. A shared trunk is only worth
sharing if what it holds is *body-conditioned dynamics* — how this morphology moves — rather than
just "what action scores well right now". FD and FK are how that content is supposed to get in: FD
forces the encoder to carry enough about the current state to predict the next one, FK forces it to
carry where the body's parts physically are. If they matter, then the backbone's value in
experiment 1 is predictive structure, and the story is a coherent one about representation. If they
do not, then experiment 1's backbone is doing something else — shared gradients, or an optimization
effect — and the paper cannot claim the trunk carries a world model.

The sharper version, and the one metric 3 answers: aux training should widen the range of bodies
control remains valid on, because a representation organised around *how a body moves* generalizes
across bodies in a way one organised around *what scored well on this body* does not.

## Conditions

Two conditions, both on the `single` shared-trunk config, differing only by `--set`. Only `none`
is an arm of *this* study: the unmodified condition is the series' shared `baseline` study, trained
once and compared against by every ablation that is a delta off it.

| arm | `fd.enabled` | `fk.enabled` | note |
|---|---|---|---|
| `none` | false | false | `get_aux_loss` returns `None` — bit-identical to the phase-1 baseline |
| `aux` | true | true | the tuned config — the shared `baseline` study, not run again here |

FD and FK are ablated **together**, not factorially. The accepted cost is that a null result cannot
be attributed to one head — "prediction doesn't help" and "one helped, one hurt" are
indistinguishable. The follow-up if the result is null or small is the 2×2, not a reinterpretation
of this one.

`config.jepa` stays `false` in both arms.

## Held fixed

Everything in [backbone.md](backbone.md#held-fixed): task, seed body, 48 windows = 1536 epochs,
`n_pretrain: 7`, 4096 actors, seeds 42–49, and every shared hyperparameter. Both arms run
`train_codesign_single.py` on `ppo_ant_codesign_tuned.yaml`; the only difference is two `--set`
flags, so drift is structurally impossible.

The aux heads' **parameters exist in both arms** — they are allocated by the network builder and
simply never armed in `none`. Param counts are therefore identical and the `none` arm carries a
small number of dead weights. Deliberate: it keeps the comparison to the *loss*, not the
architecture.

## Process

Identical to experiment 1 minus the port. One wave of 8 seeds against the shared baseline, 48
windows (~3 h); per-window
population dumps; checkpoints at w=7/27/47; then ladder and specialization passes at each checkpoint
(6 specialization waves).

## Measurements and decision metric

All five of ADR-0021. **Decision metric: specialized return at the final checkpoint**, mean over 8
seeds with 95% CI against the noise floor — the series default.

**Metric 5 strips the aux heads from both arms.** This is the protocol's default and here it removes
this experiment's own treatment: both arms fine-tune under identical conditions and differ only in
which checkpoint they start from. That is the intended measurement — it asks what the aux training
left *durably in the representation*, which is the actual claim. The alternative, letting the aux
arm keep its heads through fine-tuning, was rejected: the arms would then differ during the
fine-tune as well as before it, so a gap could be 250 epochs of denser gradient rather than anything
the full run learned, and metric 5 would stop meaning the same thing here as in the other
experiments.

**Metric 3 is the mechanism reading and is reported with equal weight.** The claim is about transfer
between bodies, which is what the control-generalization curve measures directly; specialized return
only tells us whether the resulting design was better.

## Expected results and falsifier

**Expected.** `aux` shows a wider valid region on metric 3 — control stays within tolerance further
out along the perturbation ladder — and a modestly higher specialized return. Metric 1's gap is
expected to be smaller than metric 3's, because aux's benefit is generalization rather than
peak-on-current-body performance, and metric 1 measures the latter.

**Falsified if** `none` matches `aux` on metric 3's valid width. PPO's advantage would then already
carry everything the encoder needs, the dense signal buys nothing, and experiment 1's backbone
result — if positive — has to be explained by something other than a learned world model.

**Interpretive hazards, both to check before reading a null:**

- **FK may have had nothing to teach.** Its own design note records this: at a rest pose the
  torso-frame target is nearly a deterministic function of the morphology, which design mode already
  reads as tokens. Watch `gen/fk` against `build/*` diversity — a flat FK loss means the head was
  never doing work, which is a different finding from "kinematic grounding doesn't help".
- **Aux may help control while doing nothing for the generator.** FD and FK are computed on rollout
  states and shape the trunk that design mode reads, but only indirectly. If metric 3 improves while
  metric 4's excess bias is unchanged, the representation transferred to control and not to
  judgement — worth reporting as such rather than as a general win.

## Where it lands

Directly after experiment 1 in the architecture section: 1 establishes that the shared trunk
matters, 2 establishes *what it is holding*. A positive 1 with a null 2 is publishable but changes
the story from "the trunk learns a world model" to "the trunk couples the two learners", and the
paper's framing follows whichever way this lands.
