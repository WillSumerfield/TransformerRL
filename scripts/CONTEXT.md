# Training

PPO training, Optuna tuning, and play/render/test orchestration for the leg transformer over the ant envs. Owns `scripts/` and `configs/`. Each `train_ant_*.py` pairs with a `configs/ppo_*.yaml`; configs select the registered network/model by name and the env by `env_name`.

## Language

**Run mode**:
The first positional arg to any `train_ant_*.py`: `train` (default), `play`, `random`, or `test`. Headless defaults: `train` and `test` are headless; `play` opens a render window. `--video` is not supported in `train` mode.

**test mode**:
Headless evaluation mode. Runs a loaded checkpoint on the env for a fixed number of episodes per env slot, then prints a per-morph results table and saves a CSV, bar-chart PNG, and markdown summary to `results/` alongside the checkpoint directory. Requires a checkpoint. Default: 10 episodes per env slot.
_Avoid_: calling it "evaluation" (ambiguous with rl_games internal eval metrics).

**`--num-episodes`**:
In `test` mode: episodes per env slot to collect (default 10). In `play` mode: stops the player after `num_episodes × max_episode_length` total steps (default: runs until window closed). When `--video` is set in `play` or `random` mode: bounds recording duration (default 1 episode when unset).

**`morphology_set`**:
Optional parameter to `run_training`. When provided, enables `--train-pct`/`--test-set` CLI flags and handles morphology-count snapping of `num_actors`. Pass the full morphology list for the env; `run_training` computes the effective subset. Multi-morph scripts pass this; single-morph scripts (`mlp`, `transformer`) do not. `--test-set` works in all modes including `train`; training on the test split prints a reminder to omit `--test-set` when evaluating generalization.

**`--compare`**:
`test` mode flag. Requires `--train-pct < 1.0` and a seed. Runs the full morphology set in one env instance, then labels each morph's scores as train or test post-hoc. The Summary and By Leg Count sections of all outputs break down stats by split (train row, test row, global row). Produces a single chart with blues (train morphs, shaded by leg count) and oranges (test morphs, shaded by leg count). Incompatible with `--test-set`.

**Episode score**:
Cumulative raw reward over one episode (sum of `_rew_buf` across steps until termination or truncation). The unit of measurement in `test` mode results.

**`resample_interval`**:
Config knob (full ant only): episodes between morphology resamples. The training agent rebuilds the sim with a fresh sampled body set every `resample_interval` episodes; `0` (default elsewhere) disables it. The mechanism and cost live in the Morphology context — [Morphology resampling](../envs/CONTEXT.md) and [docs/morphology_resampling_cost.md](../docs/morphology_resampling_cost.md).
