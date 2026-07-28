# Metrics

Reference for every metric `scripts/eval.py` emits (one wide CSV row per run × epoch; a
subset shows in the console table). Formulas + how to read the number live here; the
canonical *term* definitions live in the glossaries and are linked, not restated.

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
