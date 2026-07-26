# Analysis

Studies of what a trained limb transformer attends to, and how that relates to reward — the groundwork for using attention as a future bridge to a generative morphology policy. Owns `experiments/`, `notebooks/`, and the figures/`.npz` in `data/`. Shared terms live in the [Context Map](../CONTEXT-MAP.md); architecture terms (token, limb transformer) in [Control](../transformer_rl/CONTEXT.md).

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

## Language

**Attention weight**:
A single entry of the encoder's attention tensor `(n_layers, n_heads, n_tokens, n_tokens)` — how much one token attends to another, collected per env step over rollout episodes.

**Attention over time**:
The time series of attention weights across an episode (and across seeds), saved as `attention_over_time_seed*.npz`. The primary raw artifact.

**Attention–reward correlation**:
Correlation between attention weights and reward, computed two ways: **episode-level** (one value per episode) and **step-level** (per env step). Rendered as `heatmap_corr_episode*` / `heatmap_corr_step*` and `scatter_attention_reward_*`.

**Value estimate**:
The critic's prediction at a state, stored in **raw-reward units** — i.e. denormalized (`value_mean_std`) and divided by the reward-shaper scale (`0.01`) so it is directly comparable to raw episode reward.
_Avoid_: comparing the network's raw normalized output to reward — wrong units.

**Aggregation granularity** (per-token vs grouped):
Plots come in `_tokens` and `_groups` families. `_tokens` keeps individual tokens; `_groups` pools them — but *how* it pools is intentionally not fixed (could be per limb, per token type, or per morphology depending on the view). Treat "group" as the coarser aggregation, defined per plot.
