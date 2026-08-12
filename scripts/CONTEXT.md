# Training

PPO training, Optuna tuning, play/render orchestration, and headless checkpoint evaluation for the limb transformer over any [Task](../CONTEXT-MAP.md). Owns `scripts/` and `configs/`. A **training script names an algorithm, never a task**: it pins the agent, network and model, and the config names the task (`env.task`) alongside the module library and the seed body. `--config` is therefore mandatory and has no default — a script that fell back to an ant config would be naming a task by omission. `eval.py` is standalone (not paired with a config — it reads each run's stamped `config.yaml`, which is also where it learns the run's task).

## Language

**Run mode**:
The first positional arg to any `train_*.py`: `train` (default), `play`, or `random`. Headless defaults: `train` is headless; `play` opens a render window. `--video` is not supported in `train` mode. In `play` the positional checkpoint arg accepts a directory (see [Controller](#) / [Policy switching](#)), not just a `.pth` file. Headless checkpoint evaluation is a separate concern owned by `eval.py` (not a run mode).

**Run name**:
The leaf label identifying a single training run, the last segment of its output dir (see [Family / experiment](#)). Defaults to a timestamp; `--name` overrides it with a chosen label. In `train` mode a name that already exists errors out rather than clobbering the prior run.

**`--num-episodes`**:
In `play` mode: stops the player after `num_episodes × max_episode_length` total steps (default: runs until window closed). When `--video` is set in `play` or `random` mode: bounds recording duration (default 1 episode when unset).

**Eval** (`scripts/eval.py`):
Standalone headless checkpoint evaluation, decoupling a **control policy** from a **body source** so the same policy is scored across body distributions. Loads a run's checkpoint (best by default, or `--epochs`), pairs it with each body source, and rolls out `--episodes` deterministic-`mu` episodes per body over a **fixed** body population (one body per env; drawn once, held — never resampled mid-run). Reports per-body reward (mean, top-k), robustness (fall rate, episode length), generator [diversity + committance](../experiments/CONTEXT.md), and value calibration. Runs are compared **side-by-side**: one wide CSV row per (run, epoch) written to `evals/` (git-ignored). Reuses the lightweight `experiments/` load+rollout path, not the rl_games runner.
_Avoid_: calling it a "run mode" — it is a separate script, deliberately not a `train_*.py` positional mode.

**Body source** (eval):
Where each evaluated body comes from, held independent of the control policy. Three, all unrolling the generator's grammar-masked MDP (`net.sample` mode): **general** — the trained generator's stochastic draw (in-distribution performance); **best morph** — greedy/argmax decode, the generator's *committed* body; **random** — uniform over the grammar-valid set, i.e. a random policy on the same MDP, doubling as the diversity reference and the baseline for **gen-advantage-over-random**.

**Gen-advantage-over-random**:
Mean control reward on the generator's bodies minus on random bodies, with the *same* control policy. The headline "did the generator learn to pick better bodies than chance" number. Co-adapted (control trained on generator bodies), so read as a paired comparison, not a pure generator score.

**Follow camera**:
The viewer's camera controller in `play`/`random`. Has three viewing states: **auto-cycle** (default — hops to a random robot each episode), **manual-follow** (locked to one operator-chosen group+env, persists across episode resets), and **free-cam** (camera detached, driven by the renderer's built-in WASD/drag). Group = morphology (`EnvironmentGroup`), env = one robot instance within it. Auto-cycle and manual-follow are mutually exclusive; free-cam is an orthogonal overlay that restores the prior state on exit.
In both fixed states (auto-cycle and manual-follow) the operator can **orbit** (mouse motion rotates the viewpoint around the focused robot) and **zoom** (scroll wheel sets the focus distance); the chosen angle and distance persist as the focus hops between robots. Orbit/zoom have no effect in free-cam (the built-in controls own the camera there). While following, the cursor is pinned to the window (so the mouse can orbit), which makes the GUI panel unclickable — switch to free-cam to use it.
The same panel also hosts the [policy-switching](#) controls (epoch dropdown, run dropdown, and the `R`/`T` epoch keys); the camera controller detects those changes and hands them to the player.
_Avoid_: calling free-cam "manual" — manual-follow still tracks a robot; free-cam tracks nothing. Orbit is not free-cam: orbit keeps the robot centred, free-cam does not.

**Controller**:
The trained checkpoint currently driving the robots in `play` — the control policy (and, in codesign envs, the body generator inside the same net). "Changing the controller" means loading different weights: a different [epoch](#) or a different run. Every controller change triggers a full env reset; in codesign envs that reset re-runs the new generator and rebuilds the sim to freshly-sampled bodies, so a controller change also changes the morphology distribution on screen.

**Policy switching** (`play`):
The `play`-mode feature for stepping through a run's checkpoints live. The positional arg is a directory, not a `.pth`:
- a **run dir** (contains `nn/`, e.g. `runs/ant_full/full_transformer/s42`) → one **epoch dropdown**;
- a **model dir** (a `<model>` dir of run folders, e.g. `runs/ant_full/full_transformer`) → an extra **run dropdown** plus the epoch dropdown.
The epoch dropdown lists a run's `nn/` checkpoints with **best** on top (the bare `<name>.pth`, rl_games' best-mean-reward save, identified via `config.name`) followed by the `last_<name>_ep_<N>_rew_<R>.pth` snapshots ascending by epoch; playback starts on **best**. `R`/`T` step the epoch backward/forward through the dropdown order (wrapping); the run dropdown has no hotkey and, when changed, is itself a [controller](#) change that repopulates the epoch dropdown and jumps to that run's best. The env/network are still built from the script default config (or `--config`) — the run dir supplies weights only. `play` also defaults to **64 envs** (overriding the config's `num_actors`, itself overridden by `--num_envs`) to keep the codesign rebuild-on-switch snappy.
_Avoid_: calling the run dropdown the "model" dropdown in prose — it selects **runs** within one `<model>` dir; all share one architecture, which is what makes live restore shape-safe.

**Episode score**:
Cumulative raw reward over one episode (sum of `_rew_buf` across steps until termination or truncation).

**`resample_interval`**:
Config knob (full ant only): episodes between morphology resamples. The training agent rebuilds the sim with a fresh sampled body set every `resample_interval` episodes; `0` (default elsewhere) disables it. The mechanism and cost live in the Morphology context — [Morphology resampling](../envs/CONTEXT.md) and [docs/guides/morphology_resampling_cost.md](../docs/guides/morphology_resampling_cost.md).

**Proxy run** (tuning):
The short run each Optuna trial executes, standing in for the longer deployment run. A short-horizon signal, and each study has its own; **fidelity is the number of [resample](../envs/CONTEXT.md) windows the proxy spans** (period ≈ `resample_interval × max_episode_length ÷ horizon_length`) — the fewer, the less the tuned winner transfers.
A proxy needs no config file of its own. Its length is a **frames budget** expressed as a derived `max_epochs = FRAMES ÷ (num_actors × horizon_length)`, so the study's `base_config` is the ordinary [task leaf](#) and the proxy/deployment difference lives in the tune config. Holding frames fixed rather than epochs is what makes `horizon_length` sweepable at all: it holds the window count constant too, so every trial spans the same number of windows however it is sliced.
_Avoid_: "checkpoints off" — checkpointing must stay **on**, or a [resume](#) is impossible (ADR-0017); the tuner scores from TB and simply never reads one.

**Slot** (tuning):
One unit of trial concurrency — a single pinned CUDA device running exactly one trial process at a time. The slot, not the card, is what the tuner's scheduler allocates and what bounds a trial's memory; a trial sees only its own slot and calls it `cuda:0`. How slots are supplied is a property of the machine, not of the term: the tuning server partitions its two cards by MIG into 8 instances (¼ card, 24 GB each), while a workstation offers one slot per whole GPU.
_Avoid_: equating a slot with a MIG instance (that is only how the server supplies them) or with a GPU (a partitioned card holds four); "worker" (the tuner has one scheduler, not eight).

**Screen** / **focus** (tuning):
The two stages of a sweep, run as two studies. A **screen** searches many parameters thinly to rank which ones matter; a **focus** sweep then searches only those, densely, at a longer [proxy](#). Staged narrow sweeps that freeze a winner per stage (the `3a`/`3b`/`3c` comments in the codesign configs) are the older, greedy alternative — they cannot see interactions between parameters fixed in different stages.
The handoff is by config, not by hand: the focus's `base_config` bakes in the screen's [sweep winner](#) for **all** screened parameters, so the focus's own [calibration wave](#) *is* the screen winner's [confirmation](#) at deployment length. A screen therefore needs no confirmation wave of its own. Which parameters carry forward is read from two reports together — fANOVA importance says a parameter **matters**, top-k spread says it was **determined** — and neither alone is sufficient.

**Sweep winner**:
The configuration a completed study exports to `best_params.yaml`. Taken as the **centre of the top-k cluster** (per-parameter median/mode), not the single best-scoring trial: a raw argmax selects the luckiest draw rather than the best region, and that bias grows as the sampler concentrates. Because each trial draws its own [trial seed](#), the cluster centre averages over seed noise as well as hyperparameter noise — which is what makes it trustworthy.
The centre is a *coordinate-wise* summary, so it is only meaningful when the top-k is one blob: over a bimodal cluster it can assemble a configuration no trial ever ran, and over a parameter the study failed to determine it reports a median just as confidently as over one the study nailed. Both are read off the reported **top-k spread** (per-parameter fraction of the search range the cluster covers; margin of a categorical mode). A winner is therefore not final until a [confirmation wave](#) has run it.
_Avoid_: equating the winner with `study.best_trial` — that is the argmax, reported alongside for comparison only.

**Noise floor** (tuning):
The spread in a study's objective attributable to the [trial seed](#) alone, at fixed hyperparameters — the resolution limit below which no hyperparameter difference can be detected. Measured, not assumed: across run families it ranges from CV 11% to 67% on `rewards/iter`, and a completed study whose across-hyperparameter spread sits at or below its noise floor has demonstrated nothing. Switching objective metric does not help — `quality/R_mean` is ~2.6× less noisy but discriminates proportionally less, leaving the ratio unchanged.
_Avoid_: treating a study's best trial as a result without knowing its noise floor.

**Calibration wave** (tuning):
The first wave of a sweep: one configuration run on every [slot](#) at once, differing only by [trial seed](#), to measure the [noise floor](#) of that exact study before spending the rest of the budget. Its output sets the top-k size, the trial count, and whether pruning and a [screen](#) stage are statistically affordable at all. Also the cheapest place to observe full-length memory growth, on the host as well as the device.
The wave's runs are ordinary trials of the study, enqueued ahead of the search rather than measured off to one side — so the anchor is scored in the same study everything else is scored in, at the cost of stacking that many observations on one coordinate of the sampler's density model. Being 8-ish samples, it fixes the noise floor's order of magnitude, not its value (a CV of 39% is pinned to roughly ±10 points) — which is all the decisions it feeds actually need.

**Confirmation wave** (tuning):
The wave that closes a sweep, as the [calibration wave](#) opens it: the exported [sweep winner](#) run at one seed per [slot](#), and its band compared against the calibration anchor's. The only step that tests the winner rather than describing it — necessary because the winner is a coordinate-wise construction that may correspond to no trial the study ever ran, and because a study whose spread sits at its [noise floor](#) can rank configurations confidently on nothing.

**Trial seed**:
The per-trial RNG seed, drawn fresh for every trial rather than fixed across the study, so trials are independent draws and neighbouring configurations do not share correlated noise. Drawn from a study-level RNG seeded by `sampler_seed` (the study stays reproducible as a whole) and recorded on the trial so any single run can be reproduced exactly.
_Invariant_: set through the trial config's `params.seed`, **never** the trainer's `--seed` flag — that flag additionally forces cuDNN-deterministic mode, a restricted cuBLAS workspace and single-threaded CPU, none of which a tuning run wants.

**Retry** / **resume** (tuning):
The two ways a dead trial gets another attempt, kept apart because they cost differently and mean differently. A **resume** relaunches from the trial's latest checkpoint and answers a [rebuild crash](../docs/troubleshooting/resample_rebuild_crash.md) — an environment fault that says nothing about the configuration — costing only the epochs since that checkpoint. A **retry** relaunches from scratch and costs the whole trial. A configuration that genuinely cannot run (OOM) gets neither: that is a real property of the hyperparameters, so it is scored and recorded, letting the sampler learn the region is unavailable. Checkpoints in a tuning run exist *only* to make resume possible — the tuner scores from TensorBoard and never reads one.
_Avoid_: "retry" for both. At ~32 rebuild windows per trial the crash reaches roughly one trial in five, so which of the two fires is the difference between a sweep losing minutes and losing hours.

**Recovered-level score** (tuning objective):
The reward half of what a completed trial is scored on: the mean of the objective metric over the final windows of the [proxy run](#) — performance *after* the resample dip has recovered, not a transient peak. Chosen over max-over-training because the deployment run resamples repeatedly: a hyperparameter set that spikes then collapses post-resample scores well on a max but transfers badly.
For codesign the metric is `quality/R_mean` over the last 5 windows. **Window-indexed, not epoch-indexed** — one point per window is phase-aligned by construction, which is what makes `horizon_length` and `resample_interval` legal sweep parameters; an epoch-indexed tail lands at an arbitrary phase within a window as soon as either moves. The full objective multiplies this by the [collapse gate](#) — see [ADR-0020](../docs/adr/0020-composite-tuning-objective-with-collapse-gate.md).
_Avoid_: calling the tuning objective "max reward" — that was the old metric and rewards transient peaks the resampling punishes.

**Collapse gate** (tuning objective):
The second half of the tuning objective: a **saturating floor** on generator diversity, `min(1, D / D_floor)`, multiplying the [recovered-level score](#). Above `D_floor` it is exactly 1, so ranking stays pure reward; below, it discounts linearly. It exists because `quality/R_mean` averages over the sampled body population and therefore *structurally rewards collapse* — a diverse population contains more bad bodies and scores lower — so maximising reward alone is partly an instruction to stop exploring.
Saturation, not a product, is the load-bearing choice: a product would need a reward-per-body trade rate nothing justifies, and would pay without limit for `generator.entropy_coef`, which the gate is otherwise nearly a restatement of. `D` is `build/n_modes` averaged over the **whole RL phase**, not the reward tail — late commitment is success, and the pathology being caught is *never having searched*.
_Invariant_: the gate is **insurance, expected to be inert**. If it fires routinely, `D_floor` is wrong — it is not a second objective.

**Diversity headline** (`build/n_modes`):
The generator-diversity number anything may be *gated or judged* on: a Hill number (q=1) over single-linkage `d_struct` clusters at radius τ=1 module, computed on the **subtype-collapsed skeleton**. `1.0` means a single design (full collapse) by definition, which is what makes a floor on it interpretable without calibration. Reported as `div_nmodes` by `eval.py`; logged per window (RL only) by the training agent.
_Avoid_: `build/n_distinct` — it counts exact **typed** designs, and per the `free_entropy` finding the skeleton commits while the subtype axis stays free, so subtype jitter alone pins it near the sample size even under total skeleton collapse (measured: never below 286/4096 across a whole study). Also _avoid_ `build/body_diversity` (`N_body_skel`) as a headline: it inflates when independent limbs flip without breaking the common core, and being built from the generator's own per-step entropies it mostly restates `entropy_coef`.

**Prune signal** (tuning):
The mid-run health value the tuner reports to Optuna for [MedianPruner](https://optuna.readthedocs.io). An EMA of current `morph_reward/mean`, collapse-sensitive by design (a stalled/collapsing trial's EMA falls, so it can be killed), as opposed to the recovered-level *score* it is later judged on. Distinct from the score but coherent with it — both reward current/recovered performance rather than peaks. The dip from a resample is common-mode (every trial dips at the same epoch, since resample timing is fixed across trials), so it cancels in the same-epoch median comparison. This holds only while resample timing is fixed, so pruning (and the recovered-level objective) is disabled for any sweep that tunes `resample_interval` — see [ADR-0009](../docs/adr/0009-tuning-pruning-and-recovered-level-objective.md). Fixed timing is necessary but not sufficient: pruning is also unsafe whenever mid-run rank is dominated by the [noise floor](#), since it then preferentially kills unlucky seeds rather than bad configurations, and pruned trials contribute no final score to the [sweep winner](#)'s cluster. Enabling it is therefore gated on the [calibration wave](#).

**Algorithm config** / **task leaf**:
The two-file split a runnable config is assembled from, via the single-parent `extends:` chain. An **algorithm config** (`ppo_codesign_single.yaml`) holds everything about *how* training works — agent, architecture dims, PPO and generator hyperparameters — and names **no task and no seed body**. A **task leaf** (`ppo_ant_codesign_single.yaml`) extends one and supplies the task-level facts: `env.task`, the `base_morphology` seed body, and any Task constructor knob. One leaf per (task, algorithm) pair actually run. The split exists because the seed body is a *task* fact, not an algorithm one — three limbs of swing/knee is an ant, and a hand wants something else — and leaving it in the algorithm file is how "ant" survives a refactor that was meant to remove it. A config splits only once there is task content to hoist; an algorithm whose leaf would be `env.task` alone stays one file until a second task needs it.
_Invariant_: an algorithm config is not runnable on its own; naming no task is the point, not an omission.
_Invariant_: anything reading a config **file** must resolve it (`_load_config`), never `yaml.safe_load` it — a raw read of a leaf sees `extends:` and the `env` block and nothing else. A config stamped into a run dir is already resolved and is read raw.
_Avoid_: adding a second task by copying an algorithm config — that forks the hyperparameters, which is exactly what `extends:` exists to prevent.

**Family** / **experiment** (run paths):
The two slugs a training script hands `run_training`, from which every output path is composed: `runs/{task}_{family}/{experiment}/{run-name}`, with `config.name` (the checkpoint stem) as `{task}_{experiment}`. The task comes from the config, the other two from the script, so a run's location states both halves of what produced it. Composition, not a literal path, is what lets a second task coexist without renaming anything: `ant` + `codesign` reproduces the historical `runs/ant_codesign/…` exactly, and `grasp` lands beside it.

**Base config** (`configs/defaults/base.yaml`):
Shared rl_games boilerplate deep-merged *under* every `ppo_*.yaml` at load (per-config values win on conflict). Holds the keys every config shares and nobody tunes per-experiment — `algo.name`, the continuous action space block, `separate`, and the always-on rl_games flags (`ppo`, `multi_gpu`, the normalization/precision toggles, `lr_schedule`/`schedule_type`, `score_to_win`, …). The run's *identity* fields are not in any yaml: `model.name`, `network.name` come from the training script's own args (`model=`, `network=(name, builder)`) so they can't drift from what's registered, while `env_name` and `config.name` are **composed** from the script's [family/experiment](#) slugs and the config's task. A runnable config therefore lists only what it actually varies: the `env` block, `seed`, architecture dims, and PPO hyperparameters. See [ADR-0006](../docs/adr/0006-shared-base-config-for-rl_games-boilerplate.md).
_Invariant_: a config is **not** a complete rl_games config on its own; it's only valid after the base merge + identity injection in `run_training`.
_Avoid_: pasting boilerplate or identity fields back into individual configs to make them self-contained.
