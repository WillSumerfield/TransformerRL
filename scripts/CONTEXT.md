# Training

PPO training, Optuna tuning, play/render orchestration, and headless checkpoint evaluation for the limb transformer over the ant envs. Owns `scripts/` and `configs/`. Each `train_ant_*.py` pairs with a `configs/ppo_*.yaml`; configs select the registered network/model by name and the env by `env_name`. `eval.py` is standalone (not paired with a config — it reads each run's stamped `config.yaml`).

## Language

**Run mode**:
The first positional arg to any `train_ant_*.py`: `train` (default), `play`, or `random`. Headless defaults: `train` is headless; `play` opens a render window. `--video` is not supported in `train` mode. In `play` the positional checkpoint arg accepts a directory (see [Controller](#) / [Policy switching](#)), not just a `.pth` file. Headless checkpoint evaluation is a separate concern owned by `eval.py` (not a run mode).

**Run name**:
The leaf label identifying a single training run, the last segment of its output dir (`runs/<env>/<model>/<run-name>`). Defaults to a timestamp; `--name` overrides it with a chosen label. In `train` mode a name that already exists errors out rather than clobbering the prior run.

**Shared worktree state**:
`alex/create_worktree.sh PATH BRANCH [START_POINT]` creates a Git worktree
and links its machine-local runtime state to the primary checkout. Generated
`runs`, `evals`, `logs`, `videos`, and `data`, plus `.venv`, `.envrc`, and
`TurboActivate.dat`, are shared. The personal `alex/commands.sh` is linked
inside the otherwise tracked `alex/` helper directory. Caches, generated
simulator assets, `uv.lock`, and arbitrary untracked source files remain
worktree-local. The primary
checkout owns the real shared directories, so removing an experimental
worktree cannot remove their contents.
For an existing worktree, run
`alex/share_worktree_state.sh --migrate-existing WORKTREE` after stopping
its trainers. It refuses conflicting files, copies non-conflicting artifacts
into the primary directories, and retains the original directories under the
ignored `.worktree-state-backup/`.
_Avoid_: automatically linking every ignored or untracked path; some are
branch-specific mutable state rather than shared experiment artifacts.

**`--num-episodes`**:
In `play` mode: stops the player after `num_episodes × max_episode_length` total steps (default: runs until window closed). When `--video` is set in `play` or `random` mode: bounds recording duration (default 1 episode when unset).

**Eval** (`scripts/eval.py`):
Standalone headless checkpoint evaluation, decoupling a **control policy** from a **body source** so the same policy is scored across body distributions. Loads a run's checkpoint (best by default, or `--epochs`), pairs it with each body source, and rolls out `--episodes` deterministic-`mu` episodes per body over a **fixed** body population (one body per env; drawn once, held — never resampled mid-run). Reports per-body reward (mean, top-k), stability (fall rate, episode length), generator [diversity + committance](../experiments/CONTEXT.md), and value calibration. Runs are compared **side-by-side**: one wide CSV row per (run, epoch) written to `evals/` (git-ignored). Reuses the lightweight `experiments/` load+rollout path, not the rl_games runner.
_Avoid_: calling it a "run mode" — it is a separate script, deliberately not a `train_ant_*.py` positional mode.

**Benchmark eval**:
The `scripts/benchmark_eval.py` companion to **Eval**, deliberately using the same interface shape: positional `RUN [RUN ...]`, `--epochs`, evaluation-size overrides, a side-by-side console table, and one multi-row CSV. It preserves every method's native morphology–controller pairs and adds paper checks plus raw episodes. Stage 1 implements codesign: it reads the single `configs/benchmarks/benchmark.yaml`, requires `--epochs final` by default, uses the literal configured morphology/rollout/diversity seeds, and writes one comparison directory. `--seed N` sets all three directly to `N`; `--preset smoke` selects the cheap preset; numeric epochs require an explicit development budget exemption when incomplete. Its probability-weighted expected return is also written as `rewards/step_eval`, with the selected checkpoint's charged training environment steps as the TensorBoard/W&B x-axis; this is the same tag used by optional NGE training-progress evaluation. Each sequential checkpoint closes its VSim environment after rollout before the next job creates the one permitted `GymSingleton`. When invoked directly with `.venv/bin/python`, the launcher restarts once with this virtualenv's CUDA 13 and VLearn native-library directories prepended to `LD_LIBRARY_PATH`, avoiding accidental resolution through an older Conda VLearn installation. It remains separate from Eval until GPU/VSim parity is demonstrated.
The fixed-base-morph stage adds `--method fixed_body` for an unprefixed run. Mixed comparisons use
explicit positional labels, for example `codesign=RUN_A fixed_body=RUN_B`, and still produce one
comparison artifact through the same evaluator.
The uniform-action stage adds `--method uniform_action`; all three can be compared in one invocation
with `codesign=RUN_A fixed_body=RUN_B uniform_action=RUN_C`.
For convergence comparisons, a positional label may include its method-native checkpoint list:
`codesign@200,400=RUN_A uniform_action@200,400=RUN_C nge@5,10=RUN_D`. The resulting rows retain
their authoritative training environment-step counts, so checkpoints are aligned by consumed
physics rather than by unlike epoch/generation numbers.

**Fixed-body training**:
`scripts/train_ant_fixed_body.py` pairs with `configs/ppo_ant_fixed_body.yaml`. The config inherits
the selected CoDesign controller settings and changes only the algorithm label plus
`resample_interval: 0`. The fixed algorithm uses the same CoDesign PPO/AdamW/warmup/FD/FK agent but
a normal fixed-body player; with no resample boundary, generator updates and morphology rebuilds
cannot occur. Training starts and remains on the `[1,4,6]` base morph.

**Uniform-action training**:
`scripts/train_ant_uniform_action.py` pairs with `configs/ppo_ant_uniform_action.yaml`. The config
inherits the complete selected CoDesign controller configuration and changes only the algorithm
label; it writes `resample_interval: 1` explicitly to prevent accidental fixed-body training.
Training begins on `[1,4,6]`, then `UniformActionAgent` installs a fresh
`net.sample(..., mode="uniform")` population at each normal morphology boundary. It runs controller
PPO/AdamW/FD/FK exactly as CoDesign does, but it does not train GenAct or GenCrit and does not run the
generator's control-cloning update.

**Benchmark implementation sequence**:
Contract-gated delivery in increasing integration complexity: shared protocol and evaluator, codesign parity, fixed and uniform controls, faithful NGE, then faithful BodyGen. Every completed stage remains in the final five-method suite.
_Avoid_: implementing all adapters before validating the shared comparison path

**Faithful benchmark port**:
A locally integrated adaptation from a pinned official NGE or BodyGen commit onto this repository's vlearn/VSim environment interface and typed grammar. The `faithful` label requires a component-by-component mapping to local code and tests. Upstream provenance, licensing, and every behaviourally relevant adaptation are recorded; the original simulator backend is not retained.
_Avoid_: paper-only reconstruction when official code exists, external MuJoCo jobs, calling VSim "Isaac Sim"

**Baseline adaptation log**:
A tracked `benchmarks/<method>/ADAPTATIONS.md` record of port changes that can affect algorithm behaviour or results, accompanied by a machine-readable upstream commit and licence record. Each entry identifies the upstream symbol and local counterpart, rationale, expected behavioural effect, fidelity status, and validating test. Evaluation manifests retain hashes of both records.
_Avoid_: logging formatting-only changes, silently omitting a native component

**Benchmark contract suite**:
The gate for each implementation stage: grammar validity, deterministic seed replay, exact environment-step accounting, raw artifact loading, and method parity. Codesign also has to match the existing evaluator within a documented numerical tolerance. Tests are grouped by the code a person audits: metrics, evaluation flow, and the CoDesign integration.
_Avoid_: relying only on smoke training or dashboard inspection

**Paired benchmark randomness**:
Explicit deterministic seeds control native morphology sampling, rollout and diversity sampling.
Methods retain different native body samples, while corresponding jobs use the same configured
rollout seed and simulator initial-condition randomness wherever supported.
_Avoid_: forcing matched morphologies, independent rollout randomness that needlessly inflates between-method variance

**Benchmark replication unit**:
One independently trained reporting seed. Paper summaries show all five seed-level estimates, their mean and 95% bootstrap confidence interval, plus paired seed-level method differences. Pair and episode samples describe one trained method's output distribution.
_Avoid_: treating sampled bodies or rollout episodes as additional independent training runs

**Benchmark logging**:
The shared experiment-tracking contract for every benchmark method. TensorBoard is the durable local record; W&B is a YAML-enabled, lazy-loaded optional second view with online and offline modes. It is disabled by default and credentials do not activate it. Comparable outputs use identical `benchmark/...` names in TensorBoard, W&B, and local summaries; algorithm internals use their explicit method namespace, such as `nge/...`.
_Avoid_: method-specific names for comparable outputs, W&B-only runs

**Benchmark output layout**:
Generated training runs use `runs/benchmarks/<method>/<run-id>/s<seed>/`; tuning studies use `logs/tune/benchmarks/<method>/`; cross-method comparison bundles use `evals/benchmarks/<evaluation-id>/`. The source `benchmarks/` package contains no generated state.
_Avoid_: writing results into source, scattering one comparison across method-specific eval directories

**Benchmark config**:
`configs/benchmarks/benchmark.yaml` is the readable authority for evaluation and paper-run eligibility checks; each native trainer has its own directly runnable method YAML, which is stamped into the run and checked against those requirements. Explicit CLI flags cover common development changes and every result saves its resolved evaluation config.
_Avoid_: YAML inheritance, unchecked budget drift, hiding protocol values in Python constants

**Benchmark tuning**:
Equal-budget hyperparameter selection for codesign, faithful NGE, and faithful BodyGen: exactly 30 complete candidate configurations × the same three fixed tuning seeds at the full proxy environment-step budget, or 90 proxy runs per adaptive method. Each adaptive method pairs a runnable `configs/benchmarks/<method>.yaml` with search ranges and named feasibility rules in `configs/benchmarks/tune_<method>.yaml`; the method validator checks resolved candidates again. Candidates are ranked by mean primary benchmark score; invalid proposals do not count, tuning seeds are disjoint from reporting seeds, and fixed/uniform controls inherit selected codesign settings.
_Avoid_: tuning on reporting seeds, tuning only codesign, early pruning, discretionary stopping

**Native benchmark training**:
Each benchmark method retains its published or existing optimisation machinery while receiving the same locomotion task reward. The final evaluator remains external to optimisation. Optional progress monitoring may reuse its rollout core with separate seeds, but cannot supply objectives, charged learning data, or a selected headline checkpoint to a method.
_Avoid_: optimising against progress evaluation, leaking final reporting samples into training

**Benchmark environment step**:
Any physics transition consumed before a method's final checkpoint, including controller learning, morphology fitness or selection, GM-UC labels, and BodyGen design evaluation. All such transitions draw from one shared per-run budget; simulator-free design-network computation does not.
_Avoid_: counting only policy-gradient batches, hiding morphology-search simulation in a separate allowance

**Benchmark resource parity**:
Methods receive equal physics environment-step budgets and the same maximum parallel-environment cap but retain their native architecture sizes. A method may use less width when its native algorithm requires temporal depth; trainable parameters, wall time, peak RAM/VRAM, and environment-step throughput are reported outcomes.
_Avoid_: resizing faithful methods to match codesign, substituting wall-clock matching for the primary budget

**Primary benchmark score**:
The final native-pair probability-weighted expected deterministic control return at the fixed evaluation budget. Repeated stochastic draws retain their natural frequency; hyperparameters are selected on this score.
_Avoid_: method-specific training reward, selecting hyperparameters by best-pair return

**Final-budget checkpoint**:
The complete frozen method state captured when a method reaches the exact shared physics environment-step budget. It is the only checkpoint used for the headline benchmark result, preventing temporal checkpoint selection from becoming a hidden method-specific advantage. Benchmark eval samples fresh native pairs from it under recorded reporting seeds; for NGE it includes the complete surviving species population, not just a champion.
_Avoid_: rl_games' bare `best` checkpoint, best-over-training as the primary paper result

**Benchmark resume boundary**:
A complete checkpoint of model and optimiser states, RNGs, schedulers, the authoritative environment-step counter, and method-native search state. Resume keeps the same run identity and continues the remaining budget; NGE includes population and GM-UC data, and BodyGen includes design-policy state.
_Avoid_: weights-only resume, resetting the budget counter after interruption

**Unique-body mean**:
The equal-weight mean deterministic return over the distinct morphologies in a method's final sampled population. Reported secondarily so rare and dominant designs contribute equally.
_Avoid_: treating it as the method's expected deployment return

**Benchmark diversity**:
Three cross-method distribution summaries over the 4,096 design-only samples: unique fraction for support breadth; empirical entropy and effective body count for probability concentration; and mean normalized typed-token distance in the canonical grammar encoding for structural spread. Codesign committance remains a method diagnostic.
_Avoid_: using raw unique count alone, comparing method-specific latent embeddings

**Benchmark stability**:
Nominal-task fall rate and episode length for native pairs. These measure whether locomotion remains upright and sustained under the standard evaluation task; they are not robustness evidence.
_Avoid_: calling nominal stability "robustness" without a perturbation or generalisation suite

**Top-K-of-M**:
The mean return of the K highest-returning native morphology–controller pairs from an equal-budget sample of M pairs. **Top-1-of-M** is the corresponding maximum; both are cross-method selection metrics.
_Avoid_: best morph, global mode, greedy body

**Body source** (eval):
Where each evaluated body comes from, held independent of the control policy. Three, all unrolling the generator's grammar-masked MDP (`net.sample` mode): **general** — the trained generator's stochastic draw (in-distribution performance); **best morph** — greedy/argmax decode, the generator's *committed* body; **random** — uniform over the grammar-valid set, i.e. a random policy on the same MDP, doubling as the diversity reference and the baseline for **gen-advantage-over-random**.

**Morphology-search baseline**:
A non-learned or evolutionary comparator to learned **codesign**, coupled to a concurrently trained controller within one run.
_Avoid_: body source (an eval-time population, not a training algorithm), random eval

**Codesign benchmark suite**:
The four required comparators that contextualize learned **codesign**: the **fixed-base-morph baseline**, **uniform-action generation policy**, **faithful NGE**, and **faithful BodyGen**.
_Avoid_: genetic algorithm, GA (the evolutionary comparator is specifically NGE)

**Codesign benchmark design space**:
The shared typed morphology grammar used by every method in the benchmark suite: eight fixed limb slots, each absent or holding one to three typed effectors followed by a typed terminal cap. The robot has at least one effector; effector type determines module length.
_Avoid_: count-only design space, method-specific morphology spaces

**Shared benchmark start**:
The codesign **base morph** `[1,4,6]` is the initial morphology for every benchmark method. Adaptive methods may diverge only after this common starting point. It supplies morphology only: controllers use method-native random initialization under matched training seeds, with no free pretrained policy.
_Avoid_: method-specific initial morphology distributions, giving one method uncounted controller pretraining

**Fixed-base-morph baseline**:
A controller-only baseline whose morphology is always the codesign **base morph**. It uses the codesign control stack without learned morphology generation.
_Avoid_: fixed canonical body

**Uniform-action generation policy**:
A no-learning baseline that chooses uniformly among the valid actions at every step of the morphology-generation MDP and uses the codesign control stack. It is not a uniform distribution over completed morphologies.
_Avoid_: uniform random bodies, uniform morphology sampling, random eval (an eval-time body source)

**Faithful NGE**:
The full Neural Graph Evolution method, including its native graph controller, parent-to-child policy sharing, population selection, and Graph Mutation with Uncertainty, adapted only to the shared task and morphology design space.
_Avoid_: NGE-style, generic genetic algorithm, NGE search with the codesign controller

The readable training entry point is:

```bash
python scripts/train_ant_nge.py train --seed 42 --name s42
```

Before committing GPU time, exercise the same end-to-end path with the explicit
non-paper smoke preset:

```bash
python scripts/train_ant_nge.py train --seed 42 --name smoke_s42 --smoke
```

Resume only from a saved between-generation boundary:

```bash
python scripts/train_ant_nge.py train \
  --resume runs/benchmarks/nge/nge_nervenetpp/s42/checkpoints/generation_0005.pth
```

Evaluate the frozen surviving population through the same comparison script:

```bash
python scripts/benchmark_eval.py \
  nge=runs/benchmarks/nge/nge_nervenetpp/s42 \
  --epochs final
```

**NGE graph mutations**:
The four NGE mutation classes—**Add-Node**, **Add-Graph**, **Del-Graph**, and **Pert-Graph**—interpreted as grammar-preserving operations on the benchmark's typed radial limb chains.

**Faithful BodyGen**:
The full BodyGen method, including MoSAT, TopoPE, topology and attribute design policies, enhanced temporal credit assignment, and its method-native value networks, adapted only to the shared task and morphology design space. Its topology policy retains Add/Delete/NoChange; its attribute policy selects the benchmark's categorical effector and cap types.
_Avoid_: BodyGen-inspired, a BodyGen generator paired with the codesign controller

**Native-pair evaluation**:
The benchmark evaluation in which every morphology is controlled by the controller produced with it by the same method. A method is compared as an end-to-end morphology–controller system.
_Avoid_: substituting the codesign controller into NGE or BodyGen

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
