# Analysis

Studies of what a trained leg transformer attends to, and how that relates to reward — the groundwork for using attention as a future bridge to a generative morphology policy. Owns `experiments/`, `notebooks/`, and the figures/`.npz` in `data/`. Shared terms live in the [Context Map](../CONTEXT-MAP.md); architecture terms (token, leg transformer) in [Control](../transformer_rl/CONTEXT.md).

## Language

**Attention weight**:
A single entry of the encoder's attention tensor `(n_layers, n_heads, n_tokens, n_tokens)` — how much one token attends to another, collected per env step over rollout episodes.

**Attention over time**:
The time series of attention weights across an episode (and across seeds), saved as `attention_over_time_seed*.npz`. The primary raw artifact.

**Attention–reward correlation**:
Correlation between attention weights and reward, computed two ways: **episode-level** (one value per episode) and **step-level** (per env step). Rendered as `heatmap_corr_episode*` / `heatmap_corr_step*` and `scatter_attention_reward_*`.

**Sample**:
One population draw of 4096 morphologies (`env.resample()` under `sample_morphs=True`), run until every env hits its **first** term/trunc — i.e. exactly one [Episode](../scripts/CONTEXT.md) per env. Between Samples the morph set is redrawn. The unit of the morph-value sweep: a study collects N Samples per checkpoint (default plan: 5 Samples × 3 checkpoints).
_Avoid_: calling a Sample an "episode" — an Episode is one env's term/trunc cycle; a Sample is one 4096-morph population pass.

**Morph-value sweep**:
The study that rolls trained checkpoints over **Samples** to relate morphology (leg count, per-leg hip/ankle segment lengths) to outcome (episode reward) and to the critic's per-step **value estimate**. Run via `test --data-type full`; writes one self-contained `.npz` per checkpoint to `data/morph_value_sweep/`, keyed by the checkpoint's run-dir name (or `--name`) so distinct runs don't collide. Consumed by `notebooks/morph_value_sweep.ipynb`.

**Value estimate**:
The critic's prediction at a state, stored in **raw-reward units** — i.e. denormalized (`value_mean_std`) and divided by the reward-shaper scale (`0.01`) so it is directly comparable to raw episode reward. The sweep correlates it against morphology factors and against episode reward.
_Avoid_: comparing the network's raw normalized output to reward — wrong units.

**Sweep trace** (`<run>_traces.npz`):
The single sweep artifact. Holds padded `[n_sample, n_env, max_len]` float16 arrays of per-step **value** and per-step **reward** (the latter kept to allow a future discounted return-to-go analysis), plus per-env metadata constant within an episode: leg count, 16 segment lengths, episode reward, episode length, term-vs-trunc, and value at t0. Self-contained — the notebook needs nothing else.

**Aggregation granularity** (per-token vs grouped):
Plots come in `_tokens` and `_groups` families. `_tokens` keeps individual tokens; `_groups` pools them — but *how* it pools is intentionally not fixed (could be per leg, per token type, or per morphology depending on the view). Treat "group" as the coarser aggregation, defined per plot.

**Leg-ablation sweep**:
The study that measures how well the critic's **value estimate** identifies which legs matter. For each leg count 3..8 it draws M random base morphs (topology + per-leg lengths, like a **Sample**'s `sample_morphologies` draw but one body at a time); each base morph is rolled out alongside its single-leg ablations to compare predicted vs realized **leg importance**. Run via a dedicated `experiments/` script, one checkpoint per run; consumed by its own notebook. Distinct from the **Morph-value sweep**, which relates *whole-body* morphology to value over random population draws and never removes a leg.
_Avoid_: calling its per-base-morph unit a "Sample" — a Sample is a 4096-morph population pass; this unit is one body plus its ablations (see **Ablation set**).

**Base morph** / **Ablation**:
A **base morph** is one sampled intact body (the [Token](../transformer_rl/CONTEXT.md) set the critic was trained on). An **ablation** of it removes exactly one leg — and with it that leg's hip+ankle tokens / 2 DOFs — keeping every other leg's position and lengths unchanged. A base morph with N legs yields N ablations. Ablations may be unstable or fall below the 3-leg training minimum (down to 2 legs); that degradation is the point, not a defect.

**Ablation set**:
The unit of the leg-ablation sweep: one base morph plus all N of its ablations, built as N+1 [EnvironmentGroups](../scripts/CONTEXT.md) in a single env (group 0 = intact, group i = leg-i-removed). Each group gets an equal env budget (`envs_per_group`, fixed across the study and capped so the 8-leg worst case of 9 groups fits the env ceiling). Rolled out until **every env has completed exactly K episodes** — K fixed per env so unstable bodies can't contribute extra short episodes and bias the per-group statistics. One ablation set is processed at a time; the body geometry differs per set, so each is a fresh env build.

**Value gradient** (presence gradient):
`∂V/∂p` — the critic's value sensitivity to a leg's **presence probability** `p`, the
candidate training signal for a future morphology generator (ascend `V`). The **raw
sensitivity** ("which way should this input move"), *not* an attribution of how much the
input contributes to the current value — the two have opposite signs and only the
sensitivity drives ascent. Studied in [the value-gradient propagation study](../docs/value_gradient_propagation.md).
_Avoid_: reading it as leg attribution/importance (that's **Leg importance**, a finite
difference, below).

**Presence probability** (`p`):
A leg's continuous probability of being built, in `[0,1]` — the morphology input the critic
sees in the [value-gradient propagation study](../docs/value_gradient_propagation.md) (replacing the earlier binary on/off). The body is built
by **Bernoulli-sampling** `p`, so `V(p)` is the *expected* return over the bodies `p`
induces and `∂V/∂p` (the **Value gradient**) is a smooth, in-distribution codesign signal.
Distinct from a leg's **segment length** (hip/ankle), which `p` here does not vary.

**Should-be sign**:
The convention that a value-gradient's sign tracks a leg's **desired** state, independent
of its current state: positive = should be on (grow it), negative = should be off (shrink
it). A helpful leg is positive whether currently on or off. Distinct from the
current-on/off state, which the gradient never reports.

**Leg importance** (predicted vs realized):
For a leg i within an **Ablation set**: **value-predicted importance** = `V(intact) − V(ablate_i)` and **realized importance** = `Return(intact) − Return(ablate_i)`, both joined within the set. The sweep's headline is how well predicted tracks realized — i.e. whether the critic anticipates which legs the body actually needs. Paired only within a base morph (the intact group is the reference); never compared across base morphs without re-pairing.
