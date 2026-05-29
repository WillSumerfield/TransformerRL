# Analysis

Studies of what a trained leg transformer attends to, and how that relates to reward — the groundwork for using attention as a future bridge to a generative morphology policy. Owns `experiments/`, `notebooks/`, and the figures/`.npz` in `data/`. Shared terms live in the [Context Map](../CONTEXT-MAP.md); architecture terms (token, leg transformer) in [Control](../transformer_rl/CONTEXT.md).

## Language

**Attention weight**:
A single entry of the encoder's attention tensor `(n_layers, n_heads, n_tokens, n_tokens)` — how much one token attends to another, collected per env step over rollout episodes.

**Attention over time**:
The time series of attention weights across an episode (and across seeds), saved as `attention_over_time_seed*.npz`. The primary raw artifact.

**Attention–reward correlation**:
Correlation between attention weights and reward, computed two ways: **episode-level** (one value per episode) and **step-level** (per env step). Rendered as `heatmap_corr_episode*` / `heatmap_corr_step*` and `scatter_attention_reward_*`.

**Aggregation granularity** (per-token vs grouped):
Plots come in `_tokens` and `_groups` families. `_tokens` keeps individual tokens; `_groups` pools them — but *how* it pools is intentionally not fixed (could be per leg, per token type, or per morphology depending on the view). Treat "group" as the coarser aggregation, defined per plot.
