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
