# Experiment 3 — Control clone at resample

Slug `clone`. Protocol: [ADR-0021](../adr/0021-paper-experiment-metric-protocol.md).

> **Naming.** The paper calls this the PPG-KL experiment; the code does not have PPG in it. Codesign
> control trains as **combined PPO** on the shared trunk —
> [ADR-0008](../adr/0008-codesign-via-ppg-three-nets.md) was superseded by ADR-0013, and dual-network
> PPG survives only as a stand-alone control baseline unrelated to codesign. What is ablated here is
> the **control clone** inside the resample update: `beta * KL[ContAct_old ‖ ContAct] +
> lam * MSE(ContCrit, ContCrit_old)` (`codesign_agent.py:684`). The only PPG *shape* left is
> structural — the resample update is an aux phase replaying stored rollout states.

## Question

The generator's resample update runs 16 epochs on the shared trunk, and nearly every one of its
gradients moves `_encode_design`, which the control heads read. So at every window boundary the
control policy is displaced — not by a policy step, but as collateral damage from sharing. The clone
is the correction: it pulls `ContAct` and `ContCrit` back toward their pre-update selves while the
generator learns.

**Does that correction preserve competence, or does it anchor control to bodies that no longer
exist?** Both readings are live and they oppose each other. The clone's target is the policy fitted
to the *previous* window's morphologies, which the resample has just replaced. Preserving it protects
against interference and simultaneously resists the adaptation the new bodies demand.

This is the cost side of experiment 1's argument. A shared trunk is only defensible if the
interference it creates is cheaper than the coupling it buys, and the clone is the mechanism that
sets that price. [backbone.md](backbone.md) commits to "shared backbone means backbone *and its
mitigation*" — this experiment is what licenses that sentence, or forces it to be rewritten.

The clone exists **only** in the `single` arm. `split` and `decoupled` have disjoint trunks, so there
is nothing to displace and nothing to clone. This experiment therefore has no cross-arm version.

## Conditions

Four arms, 2×2 over the two coefficients, all `train_codesign_single.py` on the shipped config.

| arm | `generator.beta` | `generator.lam` | preserves |
|---|---|---|---|
| `both` | 1.0 | 1.0 | actor and critic — the shipped config |
| `kl_only` | 1.0 | 0 | actor only |
| `mse_only` | 0 | 1.0 | critic only |
| `none` | 0 | 0 | nothing |

The two terms are separated because they fail differently. A displaced **actor** is a one-time policy
jump that PPO walks back within a few epochs. A displaced **critic** poisons the advantages for all
~992 control PPO steps of the following window, so its damage compounds rather than decays. That
asymmetry is a prediction, and the two-arm version cannot test it.

**The ablation is free and structurally clean.** `beta`/`lam` scale terms that are computed
unconditionally; setting them to zero changes the loss and nothing else. The rollout-minibatch
forward still runs (`v_roll`, the GenCrit rollout-state fit, needs it), so compute, sampling and
minibatch composition are identical across all four arms.

**The off arms measure the drift they are not correcting.** `kl` and `crit` are computed *before*
being scaled and logged unconditionally as `clone/actor_kl` and `clone/critic_mse`. The `none` arm
therefore reports, for free, exactly how far the trunk moves control per window with no correction
applied — the counterfactual that makes every other arm readable.

## Held fixed

Everything in [backbone.md](backbone.md#held-fixed), and more tightly than any other experiment in
the series: one script, one config, one flag pair. Param counts are identical by construction — the
clone adds no parameters.

`resample_interval: 1` in all arms. This matters for scope: generator drift per window is fixed at 16
epochs regardless of window length, while control's recovery budget scales with the window, so a
longer window would dilute the effect. **The result is specific to this interval** and should be
stated as such.

## Process

Four mixed waves of 8 seeds, 48 windows (~12 h); per-window population dumps; checkpoints at
w=7/27/47; then ladder and specialization passes at every checkpoint for all four arms.

**Ordering.** This experiment runs *before* experiment 1's real waves. It needs no Phase5 port, and
its outcome determines how experiment 1's `single` arm is configured — if `none` matches or beats
`both`, running `single` with the clone on would be testing a component this experiment already
retired. Series order is 2, 3, port, 1.

## Measurements and decision metric

All five of ADR-0021, plus one experiment-specific diagnostic.

**Decision metric: metric 1 (return curve) at the final window** — a deliberate deviation from the
series default, for this experiment only. Clone damage is accumulated training-dynamics cost, and
metric 1 is the training-dynamics curve. Metric 5 warm-starts a checkpoint and fine-tunes 250 epochs
on a single committed body with no resampling, which is precisely the regime that *repairs* clone
damage: both a displaced actor and a poisoned critic get clean single-body PPO to recover in. Metric
5 is expected to be flat across arms and is reported as the **collapse check** on metric 1 — metric
1 can be won by a run that locks onto one easy body, and metric 5 is what catches that.

A metric-5 null with a metric-1 gap is the *predicted* outcome, not a disappointment: it reads as
"the clone buys training efficiency, not final design quality," which is a real and reportable
result. Committing to metric 1 in advance is what makes that reading honest rather than post-hoc.

**Mechanism diagnostic — the boundary-recovery trace.** `control/r_step` — the mean raw reward per
env-step over an epoch — is folded on the 32-epoch window boundaries and averaged over the 41 RL
windows, giving the dip depth and recovery time per arm. Metric 1 is one point per window and cannot
see this at all.

The fold is **not** taken on `rewards/iter`, rl_games' own per-epoch series. That is
`game_rewards.get_mean()`, a ring buffer of the last 100 finished episodes pooled over every env, so
it only moves when episodes end — and at `resample_interval: 1` all 4096 envs truncate at once, at
the same instant as the resample. Folded, it produces a fourfold excursion whose shape `episode_lengths/iter`
reproduces exactly, i.e. episode-completion phase rather than re-adaptation, and the same excursion
would appear in a run that never resampled. `control/r_step` has none of that: no episode gating, no
buffer memory, and `num_actors × horizon_length` samples per point.

The dip has two components — *the bodies are new* and *the trunk moved* — and only the second is
under test. `clone/actor_kl` and `clone/critic_mse` separate them: they measure trunk-induced
displacement directly, independent of body novelty. Read the trace only alongside them.

## Expected results and falsifier

**Expected.** `both ≥ mse_only > kl_only > none` on metric 1, from the compounding argument: critic
preservation should matter more than actor preservation, because advantage bias persists through a
whole window while a policy jump decays. `clone/actor_kl` in the `none` arm should be large — the
trunk really does move control — and the boundary trace should show a deeper dip and slower recovery
as preservation is removed.

**Falsified if `none` matches `both`.** Two distinct causes, told apart by `clone/actor_kl` in the
`none` arm:

- **Small KL** — the trunk barely displaces control, there was nothing to preserve, and the clone is
  dead weight. A clean simplification: delete both terms from the algorithm.
- **Large KL, no return gap** — control absorbs the displacement on its own within a window, and
  47 boundaries of it never accumulate. The clone is unnecessary, not inert.

**Falsified in the opposite direction if `none` beats `both`.** The anchoring cost dominates: the
clone holds control to a policy fitted to bodies the resample already discarded. This is the
"just slows learning down" reading, and it forces experiment 1's `single` arm to drop the clone.

**Interpretive hazards:**

- **The clone constrains a sample, not the batch.** `kl`/`crit` are computed on `N` rollout states
  drawn from `HN`, with the code's own justification that "the clone is a soft regularizer, so a
  subset sample is fine". The treatment is therefore weaker than a full-batch constraint, and the
  ablation's effect size is bounded by that sampling — a small gap may mean the clone is weak as
  configured rather than unnecessary in principle.
- **The other direction is not preserved.** Control PPO takes ~992 steps per window against the
  generator's 368, so the generator's design encoding drifts far harder than control's does. The
  symmetric term (`gen_preserve_*`) is on `Phase5` and not on `HEAD`, so all four arms here suffer
  unmitigated generator drift. A positive result means "the *weaker* direction of interference
  already matters", which strengthens rather than weakens the finding — but the 2×2 against
  generator preservation is a separate, post-port experiment.
- **`clone/repr_anchor` is inert.** The third clone-adjacent term is gated behind `config.jepa`,
  which is off. It is logged and will read zero in all four arms; do not report it.

## Where it lands

Immediately after experiment 1 in the architecture section, as its cost accounting: 1 shows the
coupling pays, 3 prices the interference it creates and shows the correction is what makes it pay.
A negative result is publishable and simplifying — it removes two hyperparameters and one loss term
from the algorithm, and the paper says so.
