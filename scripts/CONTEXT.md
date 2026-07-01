# Training

PPO training, Optuna tuning, and play/render/test orchestration for the limb transformer over the ant envs. Owns `scripts/` and `configs/`. Each `train_ant_*.py` pairs with a `configs/ppo_*.yaml`; configs select the registered network/model by name and the env by `env_name`.

## Language

**Run mode**:
The first positional arg to any `train_ant_*.py`: `train` (default), `play`, `random`, or `test`. Headless defaults: `train` and `test` are headless; `play` opens a render window. `--video` is not supported in `train` mode.

**Run name**:
The leaf label identifying a single training run, the last segment of its output dir (`runs/<env>/<model>/<run-name>`). Defaults to a timestamp; `--name` overrides it with a chosen label. In `train` mode a name that already exists errors out rather than clobbering the prior run.

**test mode**:
Headless evaluation mode. Requires a checkpoint. Owns its own rollout loop (reuses the rl_games player only to restore the checkpoint; see [ADR-0007](../docs/adr/0007-test-mode-owns-rollout-loop.md)). Two `--data-type`s: `summary` (default) runs a fixed number of episodes per env slot, then prints a per-morph results table and saves a CSV, bar-chart PNG, and markdown summary to `results/` alongside the checkpoint; `full` runs the [morph-value sweep](../experiments/CONTEXT.md) — one self-contained `.npz` (per-step value/reward traces + per-env morph features) to `data/morph_value_sweep/`, keyed by the checkpoint's run-dir name or `--name`.
_Avoid_: calling it "evaluation" (ambiguous with rl_games internal eval metrics).

**`--data-type`** / **`--num-samples`** (`test` only):
`--data-type summary|full` selects the per-morph score table vs the full per-step capture. `--num-samples N` is the number of fresh morphology draws (resample between [Samples](../experiments/CONTEXT.md)); default 1, hard-errors if `>1` unless the env has `sample_morphs=True`, and requires `--data-type full`. The sweep uses `--num-samples 5 --num-episodes 1 --data-type full`; run once per checkpoint, same `--seed`, to align the morph draws across models. The `.npz` is keyed by the checkpoint's run-dir name; pass `--name <label>` to override (the notebook's `STEMS` list these labels).

**`--num-episodes`**:
In `test` mode: episodes per env slot to collect (default 10). In `play` mode: stops the player after `num_episodes × max_episode_length` total steps (default: runs until window closed). When `--video` is set in `play` or `random` mode: bounds recording duration (default 1 episode when unset).

**`morphology_set`**:
Optional parameter to `run_training`. When provided, enables `--train-pct`/`--test-set` CLI flags and handles morphology-count snapping of `num_actors`. Pass the full morphology list for the env; `run_training` computes the effective subset. Multi-morph scripts pass this; single-morph scripts (`mlp`, `transformer`) do not. `--test-set` works in all modes including `train`; training on the test split prints a reminder to omit `--test-set` when evaluating generalization.

**`--compare`**:
`test` mode flag. Requires `--train-pct < 1.0` and a seed. Runs the full morphology set in one env instance, then labels each morph's scores as train or test post-hoc. The Summary and By Limb Count sections of all outputs break down stats by split (train row, test row, global row). Produces a single chart with blues (train morphs, shaded by limb count) and oranges (test morphs, shaded by limb count). Incompatible with `--test-set`.

**Follow camera**:
The viewer's camera controller in `play`/`random`. Has three viewing states: **auto-cycle** (default — hops to a random robot each episode), **manual-follow** (locked to one operator-chosen group+env, persists across episode resets), and **free-cam** (camera detached, driven by the renderer's built-in WASD/drag). Group = morphology (`EnvironmentGroup`), env = one robot instance within it. Auto-cycle and manual-follow are mutually exclusive; free-cam is an orthogonal overlay that restores the prior state on exit.
In both fixed states (auto-cycle and manual-follow) the operator can **orbit** (mouse motion rotates the viewpoint around the focused robot) and **zoom** (scroll wheel sets the focus distance); the chosen angle and distance persist as the focus hops between robots. Orbit/zoom have no effect in free-cam (the built-in controls own the camera there). While following, the cursor is pinned to the window (so the mouse can orbit), which makes the GUI panel unclickable — switch to free-cam to use it.
_Avoid_: calling free-cam "manual" — manual-follow still tracks a robot; free-cam tracks nothing. Orbit is not free-cam: orbit keeps the robot centred, free-cam does not.

**Episode score**:
Cumulative raw reward over one episode (sum of `_rew_buf` across steps until termination or truncation). The unit of measurement in `test` mode results.

**`resample_interval`**:
Config knob (full ant only): episodes between morphology resamples. The training agent rebuilds the sim with a fresh sampled body set every `resample_interval` episodes; `0` (default elsewhere) disables it. The mechanism and cost live in the Morphology context — [Morphology resampling](../envs/CONTEXT.md) and [docs/morphology_resampling_cost.md](../docs/morphology_resampling_cost.md).

**Proxy run** (tuning):
The short run each Optuna trial executes — `max_epochs=500`, checkpoint writes off (`configs/ppo_ant_ppg_tune.yaml`) — standing in for the 1500-epoch deployment run. A short-horizon signal: at 500 epochs the morphology [resample](../envs/CONTEXT.md) fires only once (~epoch 313, period ≈ `resample_interval × max_episode_length ÷ horizon_length`), versus ~4 resamples at deployment length. The tuned winner is meant to transfer to `ppo_ant_ppg.yaml`.

**Recovered-level score** (tuning objective):
What a completed trial is scored on: the mean of `morph_reward/mean` over the final ~50 epochs of the [proxy run](#) — performance *after* the resample dip has recovered, not a transient peak. Chosen over max-over-training because the deployment run resamples repeatedly: a hyperparameter set that spikes then collapses post-resample scores well on a max but transfers badly, so the objective rewards stable recovered reward instead.
_Avoid_: calling the tuning objective "max reward" — that was the old metric and rewards transient peaks the resampling punishes.

**Prune signal** (tuning):
The mid-run health value the tuner reports to Optuna for [MedianPruner](https://optuna.readthedocs.io). An EMA of current `morph_reward/mean`, collapse-sensitive by design (a stalled/collapsing trial's EMA falls, so it can be killed), as opposed to the recovered-level *score* it is later judged on. Distinct from the score but coherent with it — both reward current/recovered performance rather than peaks. The dip from a resample is common-mode (every trial dips at the same epoch, since resample timing is fixed across trials), so it cancels in the same-epoch median comparison. This holds only while resample timing is fixed, so pruning (and the recovered-level objective) is disabled for any sweep that tunes `resample_interval` — see [ADR-0009](../docs/adr/0009-tuning-pruning-and-recovered-level-objective.md).

**Base config** (`configs/defaults/base.yaml`):
Shared rl_games boilerplate deep-merged *under* every `ppo_*.yaml` at load (per-config values win on conflict). Holds the keys every config shares and nobody tunes per-experiment — `algo.name`, the continuous action space block, `separate`, and the always-on rl_games flags (`ppo`, `multi_gpu`, the normalization/precision toggles, `lr_schedule`/`schedule_type`, `score_to_win`, …). The run's *identity* fields are not in any yaml: `env_name`, `model.name`, `network.name`, and `config.name` (the experiment-family label, which drives the `train_dir` subfolder) are all injected by `run_training` from the training script's own args (`env_name=`, `model=`, `network=(name, builder)`, `name=`), so they can't drift from what's registered. A runnable config therefore lists only what it actually varies: the `env` block, `seed`, architecture dims, and PPO hyperparameters. See [ADR-0006](../docs/adr/0006-shared-base-config-for-rl_games-boilerplate.md).
_Invariant_: a config is **not** a complete rl_games config on its own; it's only valid after the base merge + identity injection in `run_training`.
_Avoid_: pasting boilerplate or identity fields back into individual configs to make them self-contained.
