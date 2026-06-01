# Shared base config for rl_games boilerplate

Each `configs/ppo_ant_*.yaml` was a full, self-contained rl_games config. In practice the four
configs were ~70% identical: the continuous-action space block, `separate`, and a wall of always-on
rl_games flags (`ppo`, `multi_gpu`, the normalization/precision toggles, `lr_schedule`/`schedule_type`,
`score_to_win`) were byte-for-byte the same in every file and never varied per experiment. That
boilerplate drowned the handful of keys a reader actually tunes, and some keys were worse than noise:
inert (`seq_length` — no RNN), redundant-and-crash-if-changed (`torso_dim`/`hip_dim`/`ankle_dim`,
which must match the obs/tokenizer layout the architecture already defaults to), or a structural
duplicate (`env_name`, which must equal the name the training script registers).

A config should contain only what a user would reasonably change to optimize the env/agent/training —
not static plumbing that crashes if touched.

## Decision

Split each config into **what varies** (stays in `ppo_ant_*.yaml`) and **shared structure** (moves to
`configs/defaults/base.yaml`), reassembled at load.

- **Base file.** `configs/defaults/base.yaml` holds the uniform rl_games boilerplate: `algo.name`
  (`a2c_continuous` everywhere), the `network.space.continuous` block, `network.separate`, and the
  `config` flags `ppo`, `multi_gpu`, `mixed_precision`, `normalize_input`, `normalize_value`,
  `value_bootstrap`, `normalize_advantage`, `clip_value`, `truncate_grads`, `lr_schedule`,
  `schedule_type`, `score_to_win`.
- **Merge.** `run_training` deep-merges the base **under** the loaded config — per-config values win on
  conflict — right after `yaml.safe_load`, before `runner.load`. A config can still override any
  default if it ever needs to diverge.
- **Identity injected, not configured.** The fields that name the run are removed from every config and
  written by `run_training` from its own arguments, so they can't drift from what the script registers:
  `config.env_name` ← `env_name`, `network.name` ← the `network=(name, builder)` arg, `model.name` ←
  the new `model=` arg (default `continuous_a2c_logstd`), and `config.name` (the experiment-family
  label that drives the `train_dir` subfolder) ← the new `name=` arg. A `None` builder in the
  `network` tuple means an rl_games built-in (e.g. `actor_critic`): the name is still injected but no
  custom network is registered. The training script is thus the single source of truth for which env,
  network, model, and run label a config maps to.
- **Per-token dims removed.** `torso_dim`/`hip_dim`/`ankle_dim` deleted from the `transformer` blocks;
  the architecture's defaults already supply the only values that match the fixed obs layout
  (11/5/11 classic, 11/6/12 multimorph).
- **`seq_length` removed.** Inert under feed-forward networks (`is_rnn()` is `False`); dropped rather
  than carried.

A runnable config now lists only: the `env` block, `seed`, `algo`/`model`/`network` names, the
architecture knobs (`d_model`, `n_heads`, `n_layers`, `ffn`, or the `mlp` block), and the PPO
hyperparameters actually swept (`learning_rate`, `kl_threshold`, `entropy_coef`, `e_clip`,
`critic_coef`, `grad_norm`, `gamma`, `tau`, rollout/optim sizes, run length, checkpoint cadence,
`resample_interval`).

## Alternatives considered

- **Keep configs self-contained.** Zero indirection, but the duplication invites drift (change one
  flag, forget the other three) and keeps crash-if-changed statics sitting next to real knobs, which is
  exactly the confusion this removes.
- **Strip boilerplate and rely on rl_games' own defaults.** Some of these keys *are* rl_games defaults,
  but not all, and the defaults are version-dependent and invisible. An explicit base file documents
  the intended values in one auditable place.
- **A subclass / code-level default dict instead of a YAML file.** Hides the values in Python and
  splits config across two media; a sibling YAML keeps all configuration in one format and diffable.

## Consequences

- **A config is no longer a complete rl_games config on its own** — it is only valid after the base
  merge in `run_training`. Anyone loading a config by another path must replicate the merge.
- **The tuner is unaffected.** `tune.py` writes a trial config and runs it via `--config`, which still
  passes through the base merge in `run_training`; it only ever writes the keys it sweeps.
- **Changing a shared default is now one edit** in `base.yaml` that reaches all configs, instead of
  four edits kept in sync by hand.
- **A per-config override silently wins over the base.** Intended (it's the escape hatch), but a
  divergence is less obvious than when every value was inline — the merge order (config over base) is
  the thing to remember.
