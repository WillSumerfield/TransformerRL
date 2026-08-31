# Metrics

Reference for every metric the project reports. Formulas + how to read the number live here; the
canonical *term* definitions live in the glossaries and are linked, not restated.

Every metric names its **provenance** — where the number comes from:

| Tag | Source |
|---|---|
| `CSV` | a column of `scripts/eval.py`'s wide row (one per run × epoch) |
| `TB` | a TensorBoard scalar the training agent logs |
| `HARNESS` | computed offline by `experiments/harness/` from a run's saved artifacts |

Sections up to [Committance](#committance) are all `CSV` — each names its column, and each is a
single number about a single run. The [paper metrics](#paper-metrics) at the end are the five
every experiment reports; they are curves compared *between conditions* and tag themselves.

## Setup

Eval decouples a **control policy** from a **body source** — see
[*Eval* and *Body source*](../../scripts/CONTEXT.md). Per (run, epoch) it creates a **fixed**
population of $B$ bodies (one body per env; `num-envs` sets $B$), then rolls out $K$ episodes
per body (`--episodes`, default 32) with the **deterministic** control policy (mean action
$\mu$, no sampling). Every body-level metric is a per-env statistic (EPM = 1, so per-env =
per-body), then reduced over the population.

Symbols used below: $B$ = bodies, $K$ = episodes/body, $R(b,k)$ = raw episode return of the
$k$-th episode on body $b$ (sum of unshaped reward to termination/truncation), $\bar R(b) =
\frac1K\sum_k R(b,k)$ = that body's mean return, $L$ = max episode length.

Three body sources feed the reward/diversity metrics (all walk the same grammar-masked
generation MDP; see [*Body source*](../../scripts/CONTEXT.md)): **gen** (`stochastic` draw),
**best** (`greedy`/argmax — the committed body), **random** (`uniform` — the diversity
reference). Diversity/committance are computed on the **gen** sample; reward is reported per
source.

---

## Reward

### Mean return

Average control reward per body, under each body source.

#### Meaning
Columns `gen_avg` / `best_avg` / `random_avg` — the same deterministic control policy scored on
gen, best, and random bodies respectively. `gen_avg` is in-distribution performance; `best_avg`
scores the generator's committed design; `random_avg` is the chance-body baseline.

#### Formula
$$\text{mean\_return} = \frac{1}{B}\sum_{b=1}^{B} \bar R(b), \qquad \bar R(b)=\frac1K\sum_{k=1}^{K} R(b,k)$$

#### Reading it
Higher = better. Raw-reward units (task-dependent, unbounded). Only comparable across runs at
equal `--episodes` and body population; the three columns are directly comparable to each other
because the control policy is held fixed across sources.

### Top-k / top-1 return

Best-body performance, ignoring the weak tail of the population.

#### Meaning
Columns `gen_topk_mean` (mean of the top-`--top-k` body returns) and `gen_top1` (single best
body), over the **gen** sample. Isolates "can the generator produce *some* great bodies" from
"is the whole distribution good" (`gen_avg`).

#### Formula
Let $\bar R_{(1)}\ge\bar R_{(2)}\ge\dots$ be the sorted per-body means. Then
$$\text{top-k} = \frac1k\sum_{i=1}^{k}\bar R_{(i)}, \qquad \text{top-1}=\bar R_{(1)}$$

#### Reading it
Higher = better. Always $\ge$ `gen_avg`. A large top-k − gen_avg gap ⇒ a broad, low-quality tail
(the generator hedges); a small gap ⇒ a tight, uniformly-good distribution.

### Gen-advantage-over-random

The gap between control's reward on the generator's bodies and on random ones — a *joint* signal
about the generator **and** control's generalization. See
[*Gen-advantage-over-random*](../../scripts/CONTEXT.md).

#### Meaning
Column `gen_advantage`. Mean control reward on the generator's bodies minus on random bodies,
**same** control policy. Co-adapted (control was trained on generator bodies), so it is not a
pure generator score: the gap has **two entangled drivers** — (a) the generator genuinely picks
better bodies, and (b) control is specialized to the generator's distribution and transfers
poorly to unfamiliar random bodies. Both push it up; this metric alone cannot separate them.

#### Formula
$$\text{gen\_advantage} = \texttt{gen\_avg} - \texttt{random\_avg}$$

#### Reading it
Higher = the generator's bodies beat chance **and/or** control has narrowed to them. $\approx 0$ ⇒
either the generator adds nothing over random morphologies, or control generalizes equally to
anything (a broadly-competent controller and a strong-but-narrow one can both sit here — disambiguate
with `gen_avg`/`random_avg` levels, not just their gap). Can be negative early in training.

---

## Robustness

### Fall rate

Fraction of episodes that ended by falling rather than timing out.

#### Meaning
Columns `gen_fall` / `best_fall`. Per body, the fraction of its $K$ episodes that **terminated**
(the ant fell / hit a termination condition) as opposed to **truncated** (reached the time
limit $L$), then averaged over bodies.

#### Formula
$$\text{fall\_rate} = \frac1B\sum_b \frac{1}{K}\sum_k \mathbb{1}[\text{episode }(b,k)\text{ terminated}]$$

#### Reading it
Range $[0,1]$, **lower = more stable**. 0 ⇒ every episode survived to the time limit; 1 ⇒
always falls. Complements reward: a body can score modestly yet never fall (stable but slow).

### Episode length

How long bodies stay alive per episode.

#### Meaning
Column `gen_ep_len`. Mean steps per episode over the gen population.

#### Formula
$$\text{ep\_len} = \frac1B\sum_b \frac1K\sum_k \text{len}(b,k)$$

#### Reading it
Higher = survives longer, capped at $L$ (the truncation length). Near $L$ with low `gen_fall` ⇒
bodies reliably run out the clock. Mostly redundant with fall rate for a fixed $L$; useful when
comparing termination causes.

---

## Calibration

### Value calibration

Does the control critic's value predict realized return, across bodies?

#### Meaning
Column `val_calib_r`. Pearson correlation between the critic's value at each episode's start
(`V0.98`, discounted-return units; see [*Value estimate*](../../experiments/CONTEXT.md)) and that
body's realized return, over the gen population.

#### Formula
$$\text{val\_calib\_r} = \operatorname{corr}_b\big(V_0(b),\ \bar R(b)\big)$$
where $V_0(b)$ is the mean start-state value over body $b$'s episodes.

#### Reading it
Range $[-1,1]$, higher = better-calibrated critic. $\approx 1$ ⇒ value ranks bodies correctly;
$\approx 0$ ⇒ value is blind to which body it is driving. `nan` if either series is constant.

### GenCrit calibration

Does the generator's *own* body-quality head predict how good a body will be?

#### Meaning
Column `gencrit_calib_r`. Pearson correlation between **GenCrit** (`V1.0` = the generator
critic's predicted body quality, `v_states[:, -1]` — value at the end of the generation
rollout) and the body's realized control return.

#### Formula
$$\text{gencrit\_calib\_r} = \operatorname{corr}_b\big(\text{GenCrit}(b),\ \bar R(b)\big)$$

#### Reading it
Range $[-1,1]$, higher = the generator can tell good bodies from bad *before* they are simulated
— the signal that lets it steer design. Low ⇒ the generator is picking bodies without an
accurate internal quality estimate.

---

## Diversity

Diversity is measured on the **subtype-collapsed skeleton** of the gen sample: the
[*free_entropy* finding](../../experiments/CONTEXT.md) is that the skeleton commits while the
subtype axis stays free, so a full-typed count would inflate on that free axis. Both metrics
below cluster/compare on [`d_struct`](../../experiments/CONTEXT.md).

### Effective number of modes

How many genuinely distinct designs the converged generator produces. See
[*Effective number of modes* (`N_modes`)](../../experiments/CONTEXT.md).

#### Meaning
Column `div_nmodes`. Prevalence-weighted count of distinct skeletons: a Hill number (order
$q=1$) over single-linkage `d_struct` clusters (radius $\tau=1$ module) of the sampled bodies.
This is the diversity **headline** — the distance-clustered replacement for the entropy counts
below, robust to independent limb-flipping.

#### Formula
$$\text{div\_nmodes} = \exp\!\Big(-\sum_{c} p_c \ln p_c\Big)$$
where $p_c$ is the sampled prevalence of cluster $c$ (clusters = designs within $\tau=1$ module
under `d_struct`).

#### Reading it
$\ge 1$. **1.0 = a single design** (ES-like collapse); **>1 = branching** into distinct body
plans (EA-like). Unlike `N_body_skel`, it does not inflate when independent components flip
without breaking the common core.

### Structural spread

Continuous companion to `div_nmodes` with no cluster threshold.

#### Meaning
Column `div_struct`. Mean pairwise `d_struct` over the sampled skeletons — average structural
edit distance between two designs the generator draws.

#### Formula
$$\text{div\_struct} = \frac{2}{M(M-1)}\sum_{i<j} d_{\text{struct}}(B_i, B_j)$$

#### Reading it
$\ge 0$, higher = more spread. **0 ⇒ every draw identical.** Threshold-free, so it moves under
small structural changes that `div_nmodes` (which needs a full cluster split) would not yet
register. Read alongside `div_nmodes`: high `div_struct` with `div_nmodes ≈ 1` ⇒ jitter around
one design, not real branching.

---

## Committance

How *committed* vs. *free* the generator is, and whether its limbs form a coordinated body plan
or independent lotteries. These are **not** diversity — they are entropy-decomposition
diagnostics and are confounded with total entropy (see the caveats). Computed analytically from
`net.sample`'s per-step entropies (Rao–Blackwell), not from clustering.

### Limb redundancy

Are the limbs coordinated into a body plan, or eight independent lotteries?

#### Meaning
Column `rho`. Total correlation across the 8 limb slots, in perplexity units: how much the joint
body distribution is more committed than the product of its per-slot marginals.

#### Formula
$$\rho = \exp(C), \qquad C = \sum_{n=1}^{8} H(L_n) - H(B)$$
$H(L_n)$ = entropy of slot $n$'s limb, $H(B)$ = Rao–Blackwell joint body entropy.

#### Reading it
$\rho \approx 1$ ⇒ **eight independent limb-lotteries**, no body plan (entropy coef buys jitter,
not branching). $\rho \gg 1$ ⇒ limbs co-vary into real correlated body plans. **Caveat:**
confounded with total entropy — a fully-committed generator has $\rho = 1$ trivially (no
variance left to correlate), so read it *with* `N_body_skel`, not alone.

### Effective skeleton count

Entropy-based effective number of skeletons — the metric `div_nmodes` supersedes as a diversity
headline, kept for `rho`'s interpretation.

#### Meaning
Column `N_body_skel`. $\exp$ of the Rao–Blackwell joint body entropy: the "number of skeletons"
implied by the generator's *entropy* (not by clustering actual samples).

#### Formula
$$\text{N\_body\_skel} = \exp\big(H(B)\big)$$

#### Reading it
$\ge 1$, higher = higher-entropy generator. **Do not use as the diversity headline** — it
inflates when independent limbs flip without breaking the common core (exactly what
`div_nmodes` fixes). It is the denominator of $\rho$, so it is retained for that reading.

### Within-slot limb freedom

How undecided each individual limb slot is, on average.

#### Meaning
Column `N_limb_mean`. Mean over the 8 slots of the effective number of limb designs at that
slot.

#### Formula
$$\text{N\_limb\_mean} = \frac{1}{8}\sum_{n=1}^{8} \exp\big(H(L_n)\big)$$

#### Reading it
$\ge 1$, higher = each slot is less committed. $\approx 1$ ⇒ every slot has essentially decided
its limb. Feeds $\rho$ (its slot-product is $\rho$'s numerator).

### Effective subtype count

How free the module **subtype** axis is (phase-5 typed modules only).

#### Meaning
Column `N_sub`. $\exp$ of the Rao–Blackwell subtype entropy — the effective number of subtype
configurations. Fixed at `1.0` for phases without a subtype axis.

#### Formula
$$\text{N\_sub} = \exp\big(H_{\text{sub}}\big)$$

#### Reading it
$\ge 1$, higher = subtype choice stays free. Per the `free_entropy` finding this typically
stays $>1$ even as the skeleton commits — which is *why* diversity is measured on the collapsed
skeleton, not here.

### Greedy distinct designs

How many distinct bodies survive greedy (argmax) decoding.

#### Meaning
Column `best_n_unique`. Number of unique typed designs in the **best** (greedy) sample — a crude
collapse check on the committed decode.

#### Formula
$$\text{best\_n\_unique} = \big|\{\, \text{argmax-decoded } B_i \,\}\big|$$

#### Reading it
$\ge 1$. **1 = the generator commits to a single body** under greedy decoding (full collapse);
larger ⇒ the argmax body still varies (residual per-draw structure). Counts *typed* designs, so
subtype variation inflates it relative to the skeleton diversity above.

---

## Paper metrics

The five measurements nearly every [paper experiment](../experiments/README.md) reports. Unlike
everything above, each is a **curve compared between conditions**, not a single run's number. Term
definitions live in [*Paper metrics*](../../experiments/CONTEXT.md); formulas are here.

**Shared conventions.** There are two x-axes, not one. Metrics 1 and 4 are per-window curves on the
**resample window index** $w = 0 \dots W$, every window plotted, with a rule marking the pretrain→RL
boundary at `n_pretrain`; metric 5's markers sit on that same axis at its three checkpoints. Metrics
2 and 3 are on the **perturbation distance** $k$ of the [spread ladder](#spread-ladder), one panel
per checkpoint. Each condition
is run at **8 seeds**; unless stated otherwise a curve is the across-seed mean with a 95% CI band,
read against the study's noise floor. Symbols from [Setup](#setup) carry over; $N$ = bodies per
sample (= `num_actors`).

**Series budget.** Every run is **48 windows**, $W = 47$: eight pretrain windows (0–7) and forty RL
windows (8–47). A window is
$\lceil$`resample_interval * max_episode_length / horizon_length`$\rceil = \lceil 1000/16 \rceil = 63$
epochs, so the budget is **3024 epochs** — derived, not rounded. Rounding to 3000 would close only 47
windows and leave the 48th unlogged, since every window's metrics are written by the resample that
closes it. The budget is fixed in **windows**, not frames, because the x-axis is the window index and
conditions must land on the same one. Roughly double the tuner's trial budget, so metric 4's
cumulative curves have a readable slope — 16 RL windows is enough for a trend, not for a plateau.

**Window boundaries are exact.** The resample fires when accumulated horizon steps reach
`resample_interval * max_episode_length` and then resets the counter to zero, so the per-window
overshoot (1008 steps against 1000) never accumulates: window $w$ closes at the end of epoch
$63(w+1)$. Window-cadence scalars are indexed by **frame**, and the frame counter lags one epoch, so
window $w$ appears at frame $(63(w+1) - 1) \cdot$ `num_actors` $\cdot$ `horizon_length`.

**Symbol clash:** $\tau$ is the mode-cluster radius (1 module) throughout this document, so the
ladder's temperature is written $T$.

### Return curve

Task performance over time — the headline "which method ends up better".

#### Meaning
`TB` — scalar `quality/R_mean`, one point per resample window. Mean true body return over the
population of the window that just ended, in shaped units (× the reward-shaper `scale_value`).

#### Formula
$$\text{quality/R\_mean}(w) = \frac{s}{N}\sum_{i=1}^{N} \bar R_w(b_i)$$
where $\bar R_w(b_i)$ is body $i$'s mean completed-episode return over window $w$ under the
*training* (sampled) control policy, and $s$ = reward-shaper scale.

`quality/Window_Rew_Mean` is logged beside it: the same population average over the window's mean
reward per env-**step**, counting every step rather than only completed episodes, and normalised by
the step count so it is invariant to `horizon_length`. It is a diagnostic, not part of this
protocol — see `codesign_metrics.md` for when the two diverge.

#### Reading it
Higher = better. **This is a joint body × control score**, not a control-quality curve: it is
measured on the generator's own bodies, so a run that collapses onto one easy body can beat one
that keeps exploring. That is deliberate — it is the honest "final performance" number, and the
other three metrics are what decompose it. The window *average* includes the post-resample
re-adaptation dip, so a method is charged for its own adaptation cost; for the PPG and
shared-backbone ablations that dip is signal, not nuisance.

### Specialized return

Best-case performance of the body a run actually committed to — the "what did this method deliver"
number that the return curve cannot give.

#### Meaning
`HARNESS` — the return reached by fine-tuning control on the committed body **alone**, after the
codesign scaffolding is stripped: no resampling, no generator, no aux heads. All `num_actors` envs
carry the same body (identical to ladder level 0), and the surviving network is ContAct + ContCrit.
Measured at the same three checkpoints as the [spread ladder](#spread-ladder) and plotted as markers
on the [return curve](#return-curve)'s axes.

Config: `resample_interval: 0`, `fd.enabled=false`, `fk.enabled=false`, warm-started from the
checkpoint, **250 epochs** (≈ 4 windows).

#### Formula
$$\text{spec}(c) = \frac{1}{N}\sum_{i=1}^{N} \bar R\big(\pi_c^{+250},\, B_{\text{greedy}}(c)\big)$$
for checkpoint $c$, where $\pi_c^{+250}$ is control fine-tuned 250 epochs from $c$ on the single body
$B_{\text{greedy}}(c)$, and $\bar R$ is mean completed-episode return over the final window.

#### Reading it
Higher = better. It exists so that **generalization does not read as weak performance**: an arm that
kept a broad control policy is otherwise charged, in the return curve, for never having specialized.
Here every arm is given the same chance to collapse onto its own choice, so the comparison is
between the *designs* the methods produced.

The 250-epoch budget is chosen to approach the fixed-body ceiling deliberately — the intent is to
collapse control from generalist to specialist. That makes this a **body-quality** measure, not a
control-quality one: with enough fine-tuning every arm's control converges on the same single-body
policy, so a null result here means the bodies were comparable, and says nothing about the control
policies that produced them.
_Avoid_: reading it as evidence about control. Pair it with the
[control-generalization curve](#control-generalization-curve), which measures the opposite property
on the same checkpoint.

For [experiment 2](../experiments/README.md) the strip removes that experiment's own treatment, so
its specialized return measures the **legacy** of aux training on the representation, not the aux
heads' continued action. That experiment's doc must say so.

### Spread ladder

The protocol metrics 2 and 3 are measured on — not itself a plotted number.

#### Meaning
`HARNESS` — a one-parameter family of body distributions obtained by dividing the generator's
masked logits by a temperature $T$ before sampling, at both the category and subtype heads. Its
three landmarks are exactly [eval.py's body sources](#setup): $T \to 0$ is **best** (`greedy`),
$T = 1$ is **gen** (`stochastic`), $T \to \infty$ flattens masked logits to uniform-over-valid,
which is **random** (`uniform`). The ladder is the existing three-point comparison filled in.

Levels are indexed by **perturbation distance** $k$ — mean `d_struct` from the committed body —
rather than by $T$, so the axis is identical across every run. For each integer
$k = 0, 1, \dots, k_{\max}$, $T_k$ is found by bisection:

#### Formula
$$T_k = \{\, T : \mathbb{E}_{B \sim P_T}\big[d_{\text{struct}}(B, B_{\text{greedy}})\big] = k \,\}$$
$$k_{\max} = \mathbb{E}_{B \sim P_\infty}\big[d_{\text{struct}}(B, B_{\text{greedy}})\big]$$

Bisection is sampling-only (no rollouts), so its cost is negligible.

#### Reading it
A level is a **distribution whose mean distance is $k$**, not "bodies exactly $k$ out". $k_{\max}$
is set by the grammar's uniform policy, so it is the *same for every condition and seed* — the
ladders are directly comparable end to end. Each level also reports its **skeleton share**,
$d_{\text{struct}}$ on the subtype-collapsed skeleton divided by typed $d_{\text{struct}}$: per the
`free_entropy` finding the cheap subtype axis moves first, so a level can accumulate distance
without ever changing the body plan. A flat control-generalization curve over levels with near-zero
skeleton share means the ladder never tested a new body plan.

### Control-generalization curve

How far outside its own distribution the control policy stays valid.

#### Meaning
`HARNESS` — mean return over the $N$ bodies of ladder level $k$, rolled out with the
**deterministic** policy ($\mu$) for $K$ episodes/body, at three checkpoints (pretrain→RL boundary,
mid-RL, final). The generator's default spread ($T = 1$) is marked with a dotted rule.

#### Formula
$$G(k) = \frac{1}{N}\sum_{i=1}^{N} \bar R(b_i), \qquad b_i \sim P_{T_k}$$
Bands are nested: inner = 95% CI of the per-seed $G(k)$ across the 8 seeds; outer = across-body
s.d. $\operatorname{sd}_i \bar R(b_i)$, seeds pooled.

#### Reading it
A curve that decays fast is a control policy valid only near its generator — the signature of a
method stuck in **local** optimization. A flat curve is **global** validity (check the skeleton
share first). Comparing the three checkpoints shows the width *narrowing* as the generator commits,
which catches the local trap forming rather than inferring it from an endpoint.

**Its own noise floor is free at $k = 0$:** every one of the $N$ bodies there is the identical
committed body, so the outer band at level 0 is pure episode noise and nothing else. All widening
above it at $k \ge 1$ is genuine body-to-body variation. If level 0's band is a large fraction of
level $k_{\max}$'s, $K$ is too low.

### GenCrit excess bias

The paired metric: does the generator's *judgement* survive outside its distribution?

#### Meaning
`HARNESS` — plotted as an **overlay**, not a separate panel: GenCrit's predicted return and the
actual return of the same bodies, both against perturbation distance, both in raw return units
(GenCrit divided by the reward-shaper scale). The actual line *is* the control-generalization
curve. The vertical gap between the lines is the bias; the reported number is that bias **anchored
at level 0**.

#### Formula
$$\text{bias}(k) = \frac{1}{N}\sum_i \Big(\tfrac{1}{s}\,\text{GenCrit}(b_i) - \bar R(b_i)\Big),
\qquad b_i \sim P_{T_k}$$
$$\text{excess bias}(k) = \text{bias}(k) - \text{bias}(0)$$

#### Reading it
GenCrit's line **staying flat while actual decays** is over-optimism about unfamiliar designs — the
mechanism that lets a generator wander into bodies that do not work. The line **falling faster than
actual** is pessimism about them, which pins the generator to local search. Either way the paper
claim is about the gap's *growth with distance*, which is what excess bias isolates.

**Why anchored, and why not a correlation.** GenCrit regresses `R` — returns collected under the
*sampled* policy, by *earlier and weaker* control policies — while the ladder rolls out the final
policy at $\mu$. Both mismatches push GenCrit toward apparent under-prediction, and neither is
constant across conditions: the $\mu$-vs-sampled gap scales with policy noise (which the PPG and
backbone ablations change), and the staleness gap scales with how fast control improved (which is
what experiment 1 measures). Level 0 is the generator's own mode, so $\text{bias}(0)$ *is* the
in-distribution offset and subtracting it removes both at once. The assumption — that the offset is
flat across levels — is checked once, on one seed, by running a ladder at both action modes.

A per-level **correlation** is the wrong statistic here and is not reported. Measured $r$ is
attenuated by roughly $\sqrt{\sigma^2_{\text{bodies}} / (\sigma^2_{\text{bodies}} +
\sigma^2_{\text{noise}}/K)}$; at level 0 the bodies are identical, so $\sigma^2_{\text{bodies}} = 0$
and $r = 0$ regardless of GenCrit's quality. A per-level $r$ curve therefore rises with distance
*purely because the target's signal-to-noise improves* — an artifact that reads as "GenCrit
generalizes better further out", the exact opposite of the truth. Correlation is also blind to
bias by construction, and bias is the whole mechanism. (The pooled `gencrit_calib_r`
[above](#gencrit-calibration) remains valid — pooling gives the return spread that a single level
lacks.)

### Travel: energy distance

How far the generator's distribution **moves** between windows.

#### Meaning
`HARNESS` — computed offline from the per-window population dump, on the subtype-collapsed
skeleton (typed version reported alongside). Mean cross-window `d_struct` with **each window's own
breadth subtracted off**.

#### Formula
$$E(w) = 2\,\mathbb{E}\big[d_{\text{struct}}(A,B)\big]
        - \mathbb{E}\big[d_{\text{struct}}(A,A')\big]
        - \mathbb{E}\big[d_{\text{struct}}(B,B')\big]$$
with $A, A' \sim P_w$ and $B, B' \sim P_{w-1}$ independent. The same-distribution null is measured,
not assumed: split $P_w$'s sample in half and evaluate $E$ between the halves.

#### Reading it
$0$ ⇔ the two windows' distributions match; positive in proportion to real movement, in module
units. Read **against the split-half null**, which is the sampling floor. Sustained $E$ near the
null with high `div_nmodes` is a generator holding the same designs forever — breadth without
exploration. Sustained $E$ well above the null with `div_nmodes ≈ 1` is sequential, ES-like
hill-climbing — exploration that the diversity headline alone would call total collapse.

_Avoid_: plain mean cross-window distance $\mathbb{E}[d_{\text{struct}}(A,B)]$ as travel. For two
*identical* distributions it equals the within-window mean pairwise distance, so a wide static
generator scores as fast-moving; breadth and travel are inseparable in it. The two subtracted terms
are exactly what fixes this.

### Mode coverage

Cumulative exploration: how many distinct designs have been found by window $w$.

#### Meaning
`HARNESS` — distinct [modes](#effective-number-of-modes) seen in windows $0 \dots w$, matched
across windows by single-linkage `d_struct` clustering at $\tau = 1$ over the *pooled* populations,
so a mode that shifts by one module is not counted as new.

#### Formula
$$C(w) = \Big|\,\text{clusters}_{\tau}\big(\textstyle\bigcup_{u \le w} P_u\big)\,\Big|$$

#### Reading it
Monotone non-decreasing; its **slope is the discovery rate**. Still climbing at the final window ⇒
the generator was still finding new designs when the budget ran out. A long plateau ⇒ search
finished, whether by converging or by getting stuck.

_Avoid_: reading a plateau as "the generator stopped moving" — the curve is monotone by
construction and reports finding, not motion. A generator cycling among already-seen designs
plateaus while travelling. Pair it with [energy distance](#travel-energy-distance).
