# Analysis

The paper's experiments and the shared measurement layer they run on. Owns `experiments/`, `notebooks/`, `docs/experiments/`, and the artifacts in `data/`. Shared terms live in the [Context Map](../CONTEXT-MAP.md); architecture terms (token, limb transformer) in [Control](../transformer_rl/CONTEXT.md).

## Paper metrics

The five measurements nearly every experiment reports. Each is a *comparison* between conditions,
never a single run's number.

All five presuppose a **generator**. Experiments 4 and 5 do not have one — experiment 4 by design
(`resample_interval: 0`, one fixed body), experiment 5 because its baselines lack one — so both define
their own measurements in their own docs and none of the terms below apply to them.
_Avoid_: putting a no-resample return on shared axes with a resampling one. They run on different LR
schedules (warmup is gated on a nonzero interval) and over different body distributions.

**Return curve**:
Task performance over time: mean true body return over the bodies of the window that just ended,
one point per [resample window](#). Measured on the generator's **own** bodies, so it is a joint
body×control score — a run that collapses onto one easy body can beat one that keeps exploring.
That is the intended headline ("which method ends up better"); the other metrics are what
decompose it.
_Avoid_: reading it as a control-quality curve.

**Specialized return**:
Best-case performance of the body a run committed to: control fine-tuned on that **one** body alone,
with the codesign scaffolding stripped (no resampling, no generator, no aux heads). Deliberately
collapses control from generalist to specialist, so every method is given the same chance to
specialize and **generalization stops reading as weak performance** in the
[return curve](#return-curve). Reported as markers on that curve, at the
[spread ladder](#spread-ladder)'s checkpoints.
_Avoid_: reading it as a control measure — it is a **body-quality** measure. Enough fine-tuning
drives every method's control to the same single-body policy by construction, so a null here means
the designs were comparable and says nothing about the policies that found them.

**Spread ladder**:
The one-parameter family of body distributions obtained by scaling the randomness of the
generator's own output, from fully committed to fully random. Its three landmarks are the existing
body sources: zero spread = the generator's **committed body**, default spread = the trained
distribution, maximum spread = a **random policy on the same grammar**. Levels are indexed by
**perturbation distance** — mean [`d_struct`](#tip-aligned-structural-distance) from the committed
body, in modules — so the axis is the same for every run, and its top end is set by the grammar
rather than by the run.
_Avoid_: reading a level as "bodies exactly this far out" — a level is a *distribution* whose mean
distance is that number.

**Control-generalization curve**:
Control performance as a function of [perturbation distance](#spread-ladder), with the generator's
default spread marked. How far outside its own distribution the control policy remains valid —
the difference between a method doing local and global optimization. At zero distance every body is
the same committed body, so the spread there is pure measurement noise and serves as the metric's
own noise floor.
_Avoid_: reading a flat curve as "control handles new body plans" without checking the **skeleton
share** of the distance — the [free vs committed axes](#free-vs-committed-axes) finding means a
ladder can move subtypes while leaving the body plan untouched.

**GenCrit excess bias**:
The paired metric: the **signed** gap between GenCrit's predicted return and the actual return of
the same bodies, over the same [perturbation distance](#spread-ladder), anchored at level 0.
Whether the *generator's* judgement, not just control's competence, survives outside the
distribution it trained on. Prediction flat while actual decays = over-optimism, the mechanism that
lets a generator wander into bodies that do not work; prediction falling faster than actual =
pessimism, which pins it to local search. Global optimization needs both this and the
[control-generalization curve](#control-generalization-curve) to hold up.
_Avoid_: reading it as an accuracy, or reporting a per-level correlation. At level 0 the bodies are
identical, so `r = 0` however good GenCrit is, and a per-level `r` curve climbs with distance purely
because the target's signal-to-noise improves — the reverse of the truth. Correlation is also blind
to bias by construction, and bias is the whole mechanism.

**Exploration curves**:
How much searching the generator actually does, over a run. Two independent components, both
required — [breadth](#effective-number-of-modes) (how many designs at once) and
[travel](#travel) (how far the distribution moves between windows). They come apart: a generator
holding one design that marches across design space explores while reading as fully collapsed, and
one holding the same three designs forever reads as healthy while exploring nothing.
_Avoid_: reporting breadth alone as "exploration".

**Boundary-recovery trace**:
Per-epoch reward (`control/r_step`) folded on the resample boundaries and averaged over all RL
windows, giving a dip depth and a recovery time. The only per-epoch series in the paper series —
every other measurement is per-window, so the transient is invisible in them. Built for
[experiment 3](../docs/experiments/clone.md).
_Avoid_: folding `rewards/iter` instead. That series is rl_games' `game_rewards` meter — a ring
buffer of the last 100 **finished episodes** pooled over all envs — so it moves only when episodes
end, and at `resample_interval: 1` every env truncates simultaneously at the same instant as the
resample. Its fold is a fourfold excursion that tracks `episode_lengths/iter` exactly and would look
the same in a run that never resamples: it measures which episodes happened to finish, not
re-adaptation. `control/r_step` is the mean raw reward per env-step over the epoch, unlagged and
sampled from every step of the rollout.
_Avoid_: reading the dip as interference. It mixes two causes — *the bodies are new* and *the trunk
moved under control* — and only [uncorrected clone drift](#uncorrected-clone-drift) isolates the
second. Also avoid deriving the fold origin from a `clone/*` scalar's step index without converting
it: those steps are **frames**, and the frame counter lags by one epoch (window *k* closes at the end
of epoch `63k` but is logged at frame `(63k−1)·num_actors·horizon_length`). The epoch spacing itself
is exact — `_steps_since_resample` resets to 0 at the boundary instead of subtracting, so the 8-step
overshoot per window never accumulates.

**Boundary checkpoint**:
A checkpoint saved at the end of epoch `k·epochs_per_window`, which the [spread ladder](#spread-ladder)
and [specialized return](#specialized-return) both read. It is written *after* that epoch's resample,
so it holds `gen_window = k`: the generator has had exactly *k* updates and the next window's bodies
are already installed. `save_frequency` is set to one window so every save is one of these
(`harness/launch.py`); the shipped config's 50 epochs would put each one 4–14 epochs into the
*following* window, one generator update short of its own label.
_Avoid_: warm-starting from one without overriding the epoch budget. `set_full_state_weights` restores
`epoch_num`, and training stops at `epoch_num >= max_epochs`, so a 250-epoch fine-tune off a
1536-epoch checkpoint exits after one epoch. Also avoid assuming `resample_interval: 0` keeps the
restored run on the committed body — restoring **re-installs the checkpoint's sampled population**
(`codesign_agent.py:993-996`), which is not the committed body and must be replaced explicitly.

**Uncorrected clone drift**:
`clone/actor_kl` and `clone/critic_mse` read off a run whose clone coefficients are **zero**. Both
terms are computed before being scaled by `beta`/`lam`, so switching the clone off leaves them fully
measured and merely unoptimized — the ablation reports its own counterfactual for free. This is what
makes an experiment-3 null attributable: small drift means there was nothing to preserve (the clone is
dead weight), large drift with no return gap means control absorbs it unaided.
_Avoid_: skipping these scalars for zero-coefficient arms — they are the arms where they mean the most.

## Diversity

Metrics comparing the *set* of morphologies a codesign run produces, on a shared representation: a morphology = 8 fixed compass **slots**, each holding a limb as a **distal→proximal** module sequence (index 0 = tip) or `∅`. Reused across phases as modules gain types/lengths. All are reported **within-run** (spread over the M bodies one converged generator samples) and **between-seed** (spread over each seed's **dominant body** = its argmax-likelihood design), on the same converged-body sample the eval-return pass draws.

**Morphology distance**:
A distance `d(A,B)` between two robot designs. Two instantiations feed the diversity metrics — composition and tip-aligned structural. Representation-agnostic so it survives added module types/lengths.

**Composition distance** (`d_comp`):
L1 between the robots' module-type **histograms** (bag-of-modules; position/limb-invariant). Raw counts headlined (size-sensitive — "more/bigger robot"), normalized frequency secondary (pure composition). Answers *what parts, how many*.

**Tip-aligned structural distance** (`d_struct`):
Slot-matched sum of per-limb **tip-anchored edit** distances — limbs aligned at the distal tip (index 0), length slack charged at the proximal end, so `E-C` and `E-E-C` are near. Absent limb = empty sequence, folding presence in. Answers *how are limbs shaped*.
_Avoid_: root-aligned / positional limb comparison (misranks unequal-length limbs — the reason tip-alignment exists).

**Effective number of modes** (`N_modes`):
Prevalence-weighted count of *distinct designs* a converged generator produces: Hill number (order q=1, perplexity) over `d_struct`-clusters (τ = 1 module) of the sampled bodies. `1.0` = single design (ES-like); `>1` = branching (EA-like). The Phase-8 diversity target. Between-seed variant adds **mode-overlap** (fraction of seed-pairs sharing a mode).
_Avoid_: generator-entropy as the diversity headline (inflates independent-component flipping without breaking the common core — see Phase 8).

## Travel

The complement to [Diversity](#diversity), which is a snapshot: how far the generator's body
distribution **moves** from one [resample window](#) to the next. Measured on the same
subtype-collapsed skeleton, and reported alongside the typed version — "skeleton travel stops while
subtype churn continues" is a finding, not a nuisance.

**Energy distance**:
The travel headline. A window-to-window divergence built from
[`d_struct`](#tip-aligned-structural-distance): the mean cross-window distance with **each window's
own breadth subtracted off**, so it is zero exactly when the two windows' distributions match and
positive in proportion to real movement, still in module units. Its same-distribution null is
measured rather than assumed — one window's sample split in half and compared against itself.
_Avoid_: plain mean cross-window distance as travel. For two *identical* distributions it equals
the within-window mean pairwise distance, so a wide static generator scores as fast-moving; breadth
and travel are inseparable in it.

**Mode coverage**:
Cumulative count of distinct [modes](#effective-number-of-modes) seen up to a given window: a greedy
cover, where a design opens a new mode when it is more than one module from every mode already
found, so a mode that shifts by one module is not counted as new. Its **slope** is the discovery
rate; still-climbing vs long-plateaued is the local-vs-global-search reading at a glance.
_Avoid_: reading a plateau as "the generator stopped moving" — the curve is monotone by
construction and only reports finding, not motion. Pair it with [energy distance](#energy-distance).
_Avoid_: pooling the windows and counting single-linkage clusters. Those components merge as points
accumulate, so the count can fall between windows and its slope is not a discovery rate.

## Committance

A **separate axis from [Diversity](#diversity)**: not *how many* distinct designs a generator produces, but *how committed vs. free* it is, and whether its limbs form a coordinated body plan. Measured by entropy-decomposition of the generator's own per-step distributions (analytic Rao–Blackwell), not by clustering samples — so it is confounded with total entropy and must not be read as diversity. Formulas + column names live in [docs/reference/Metrics.md](../docs/reference/Metrics.md).

**Redundancy** (`rho`):
Total correlation across the 8 limb slots, in perplexity units — how much more committed the joint body is than the product of its per-slot marginals. `rho ≈ 1` = eight **independent limb-lotteries** (no body plan; entropy coef bought jitter, not branching); `rho ≫ 1` = limbs co-vary into real correlated body plans.
_Avoid_: reading `rho` alone — a fully-committed generator has `rho = 1` **trivially** (no variance left to correlate), so it is confounded with total entropy; pair it with the effective-skeleton count.

**Free vs. committed axes**:
The **free-entropy finding** (from a since-retired study, cited by ADR-0020 and `Metrics.md`): under
training the **skeleton commits** (effective skeleton count → 1) while the **subtype axis stays free** (effective subtype count > 1). This split is why [Diversity](#diversity) is measured on the subtype-collapsed skeleton — the free subtype axis would otherwise inflate any count.

## Joint optimization

A **toy abstraction** of codesign on a 2D domain, in `experiments/joint_optimization/`: two
optimizers maximizing reward on a single shared landscape, one choosing a design and one choosing how to act
on it. Deliberately shares *no* vocabulary with the real system — the terms below are not the
generator, control, or GenCrit, and a finding here transfers only as far as the abstraction does.
The coupling is **emergent**, not imposed: nothing in the landscape links the two optimizers, but a
design is only worth what the other optimizer manages to achieve on it.

**Designer**:
The optimizer that chooses a design, maximizing the reward it observes. **Unconditional** — it sees
nothing and simply emits from a distribution, so its analogue of a radius is over its own output,
not over any input. The generator-analogue, but it emits a scalar rather than a morphology under a
grammar.
_Avoid_: generator (reserved for the morphology generator).

**Controller**:
The optimizer that chooses how to act on a design, maximizing the same reward. **Conditional** — its
output is a function of the design it is handed. There is exactly one controller serving every
design, which is what makes its [generalization](#generalization) a scarce resource: give each
design its own controller and generalization becomes free.
_Avoid_: control (reserved for the real policy), critic (it acts, it does not fit a return).

**Landscape**:
The single reward surface both optimizers maximize, over (design, action). Neither optimizer sees
it whole: the designer only ever sees a design's realized reward, and the controller only ever sees the
slice for the design in front of it.

**Marginal landscape**:
Design quality as actually observed — the landscape evaluated at whatever action the controller
produced. Equals the true per-design best only for a perfect controller; otherwise it is a
smeared, distorted version of it. **The designer never searches the true design landscape, only
this one**, which is the entire coupling.

**Spread**:
How much of the space an optimizer's outputs cover. Selects *which* region gets sampled, and so
which coarse structure of the landscape an optimizer can perceive at all.
_Avoid_: entropy (reserved for policy entropy in the real system).

**Exploration**:
How often an optimizer leaves a sample where it randomly landed rather than improving it. Gates
[generalization](#generalization) rather than sitting beside it: an optimizer that never improves
its samples has no use for a radius, so maximal exploration makes generalization inert. That
degeneracy is a finding, not an artifact.

**Generalization**:
How reliably an optimizer can improve a sample as a function of distance from what it knows. A
perfectly general optimizer improves any sample anywhere; a non-general one only improves samples
near its current centre. For the controller this is the capacity to act well on *unfamiliar
designs* — which is what makes designer spread and controller generalization compete directly: a
designer that ranges beyond its controller's radius gets its good designs scored badly.
_Avoid_: generalization gap (a train/test notion; this is a radius, not a gap).

**Climb**:
Improving a sample by ascending the hill it landed on, partially — the fraction of the way
governed by [generalization](#generalization) and distance. Deliberately *local*: a sample is
raised only to its own local peak, never teleported to the global optimum, so which hill
[spread](#spread) put it in still decides the outcome.

**Sampling ratio**:
How many times the controller updates per designer update. High ratio buys the controller time to
adapt to the designs in front of it before they are judged, so it can partially *substitute* for
controller generalization. Compared only under a fixed total evaluation budget — otherwise a high
ratio just buys more compute.

**Design fitness**:
What a design is judged on: its mean reward across the controller's whole adaptation window, not its
best single moment. Rewards designs the controller can exploit *reliably and soon*, and denies
credit for a lucky one-off.

**Paired cell comparison**:
Every configuration cell is run on the **same set of seed starting positions**, so any two cells are
compared as a *paired* difference rather than as two independent means. The variance of that
difference is far below what per-cell error bars imply -- x32.6 lower in Experiment 1 -- because the
shared start is the dominant noise term and cancels. This is what makes an 11x11 grid readable at a
few thousand seeds instead of tens of thousands. The factor is **recomputed per experiment**, never
carried over as a constant.
_Avoid_: reading a per-cell error bar as the resolution of a cell-to-cell difference, or comparing
cells across two experiments (they draw different start sets, so nothing is paired between them).

**Seed spread**:
The companion reading, and a *different question*: how much a single cell's outcome varies over the
starting positions -- whether that configuration is reliable or a lottery. A property of the config,
not of any comparison, so it is the wrong scale for "is this cell above its neighbour". Both are
reported; neither substitutes for the other.
_Avoid_: calling it an error bar on the surface height.

## Language

**Attention weight**:
A single entry of the encoder's attention tensor `(n_layers, n_heads, n_tokens, n_tokens)` — how much one token attends to another, collected per env step over rollout episodes.

**Attention over time**:
The time series of attention weights across an episode (and across seeds), saved as `attention_over_time_seed*.npz`. **Historical**: the collector was retired with the classic ant, so these are frozen artifacts read by `notebooks/attention_over_time.ipynb`, not something regenerable.

**Attention–reward correlation**:
Correlation between attention weights and reward, computed two ways: **episode-level** (one value per episode) and **step-level** (per env step). Rendered as `heatmap_corr_episode*` / `heatmap_corr_step*` and `scatter_attention_reward_*`.

**Value estimate**:
The critic's prediction at a state, stored in **raw-reward units** — i.e. denormalized (`value_mean_std`) and divided by the reward-shaper scale (`0.01`) so it is directly comparable to raw episode reward.
_Avoid_: comparing the network's raw normalized output to reward — wrong units.

**Aggregation granularity** (per-token vs grouped):
Plots come in `_tokens` and `_groups` families. `_tokens` keeps individual tokens; `_groups` pools them — but *how* it pools is intentionally not fixed (could be per limb, per token type, or per morphology depending on the view). Treat "group" as the coarser aggregation, defined per plot.

## Harness

`experiments/harness/` is the shared measurement layer — one implementation per metric, imported by
the per-experiment scripts, by `scripts/eval.py`, and (for the logged diversity/committance) by the
training agent itself:

- `diversity.py` — morphology distances (`d_comp`, `d_struct`) and `N_modes`.
- `committance.py` — typed population representations + the entropy-decomposition committance
  metrics (`rho`, `N_body`, `N_limb_mean`). Was `diversity_p5.py`.
- `policy.py` — checkpoint → `(net, obs normalizer)`. Was `ppg_parity._load_policy`.
- `slots.py` — what a slot is, how a process is pinned to one, why it died, and which checkpoint is
  newest. Shared with `scripts/tune.py`, which imports these names rather than keeping its own copy:
  two implementations of GPU pinning means two runs on one slice.
- `launch.py` — the run scheduler. Studies are **data** (`STUDIES`): an arm is a name plus the
  `--set` overrides that define it, taken from the build plans. Also owns the two derived budget
  numbers — `max_epochs` as a whole number of windows, and `save_frequency` as one window.
- `evalpass.py` — one eval pass: open a task, load a checkpoint's policy, install bodies, roll out at
  μ. Lifted out of `scripts/eval.py`, which imports it; the ladder and the specialization pass need
  that exact rollout, and it is the rollout that defines what "return" means for every metric.
- `specialize.py` — the [specialized return](#specialized-return): doctor a boundary checkpoint onto
  the [committed body](#committed-body), fine-tune 250 epochs with the scaffolding stripped, roll out.
- `scrape.py` — run dirs → one per-experiment rollup npz. The only place a scalar's TensorBoard step
  becomes a window index, and the only reader of a resumed run's two overlapping event files.

[Joint optimization](#joint-optimization) is **not** part of the harness — it is a self-contained toy
with no dependency on the repo's tasks, checkpoints or metrics.

The pre-CoDesigner-migration analysis scripts (the phase-comparison harness, `ppg_parity`, and the
one-off `free_entropy`/`commit_metrics`/`teacher_fidelity`/… studies) and their notebooks were
**deleted** with the move to the paper-experiment series; only the code above survived them.
