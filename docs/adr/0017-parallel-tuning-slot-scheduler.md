---
status: accepted
---

# Tuning runs one trial per pinned GPU slot, scheduled by an ask/tell loop

## Context & Decision

Tuning ran serially: one trial, one subprocess, `study.optimize()` blocking until it exited. A sweep of ~200 trials at ~2 h each is ~400 h. The new tuning box has 2× RTX PRO 6000 Blackwell (96 GB each), partitioned by **MIG into 8 × 24 GB instances**.

The design this replaces was premised on "VSim allows one process at a time, so many algorithms must share one gym via environment groups". That premise is **wrong**. VSim's actual limits are **one gym per process** (a process-global singleton, `codesigner/backend/simulation.py:28`) and **one GPU per gym** (`create_gym(cuda_device=…)`). vlearn's own multi-GPU path is `torchrun --nproc-per-node=2`, i.e. concurrent processes, and the licence is node-locked to the machine rather than per-seat. Nothing prevents eight trial processes.

`scripts/tune.py` therefore owns a pool of **slots** and drives Optuna manually: `study.ask()` whenever a slot is free and budget remains, write the trial config, launch a subprocess pinned to that slot, poll, `study.tell()` on exit. A slot is a **device string**, not a MIG instance — MIG is merely how the server supplies eight of them, while a workstation supplies one per whole GPU and `slots=1` reproduces the old serial tuner exactly.

Supporting choices, each load-bearing:

- **`TPESampler(constant_liar=True)`** above one slot. In-flight trials are otherwise invisible to the sampler, which proposes near-duplicates of whatever is already running.
- **Per-trial seed through the config's `params.seed`, never the trainer's `--seed` flag.** `--seed` additionally forces `cudnn.deterministic`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `set_num_threads(1)` (`train_utils.py:835-851`); the serial tuner avoided this only by accident.
- **A failure taxonomy, because "exit != 0" is no longer one thing.** OOM is a real property of the hyperparameters: score it `0.0` (rewards are strictly positive; TPE is rank-based) and never relaunch, so the sampler learns the region is unavailable. The gym-rebuild race (`docs/troubleshooting/resample_rebuild_crash.md`) is transient: **resume from the newest checkpoint**, capped at `max_attempts`. OOM is tested first because an OOM cascades into a generic CUDA error that would otherwise be misfiled as transient and resumed forever.
- **`start_new_session=True` + `killpg`**, and explicit `SIGINT`/`SIGTERM` handlers. On stop, the whole cohort is killed and every in-flight trial is marked `FAIL` and re-enqueued.
- **One cached `EventAccumulator` per trial.** First `Reload()` costs ~0.343 s, subsequent ones ~0 s; the serial tuner built a fresh accumulator per poll, which is what made polling look like a scaling problem.

## Considered Options

- **`torchrun --multi_gpu` — one trial spread across both cards.** Rejected: it optimises the latency of a single trial, but the sweep's unit of work is a *trial*, and its throughput is what matters. The epoch is 86% gradient update, so DDP might roughly halve one trial's wall-clock while halving the number of concurrent trials — no net gain, plus gradient-sync complexity and a shared blast radius.
- **One gym with environment groups, many algorithms in one process.** Rejected: the premise was wrong (see above), and it would buy nothing even if true, since only 10% of the epoch is physics. It also couples the trials — one OOM or one rebuild crash takes down all eight, and a process-global singleton gives no memory isolation.
- **Whole cards, 2 slots.** Rejected: real process memory at 4096 envs is 11.5 GB, so a 96 GB card would sit ~88% idle. Four 24 GB slices per card still leave ~2× headroom for swept depth/width/`minibatch_size`.
- **MPS instead of MIG.** Rejected: MPS shares memory, so one trial's OOM can cascade into its neighbours' — which would be scored as bad hyperparameters. MIG's hard partition makes a slot's memory bound a real bound.
- **Threads or asyncio within one process.** Rejected outright: the VSim gym is a process-global singleton.
- **A second script rather than rewriting `tune.py`.** Rejected: with a device-generic slot pool, `slots=1` *is* the previous behaviour, so a second script would be duplicated logic that drifts.
- **Bounding per-trial CPU (`OMP_NUM_THREADS`).** Deliberately not done: left unbounded so a trial gets whatever the box has spare.

## Consequences

- **MIG geometry does not survive a reboot.** A systemd unit must re-apply `nvidia-smi mig -cgi/-cci`, or the box silently reverts to two whole GPUs and the tuner discovers 2 slots instead of 8.
- **Compute-only mode means that box can never run `play`, `--video` or the follow camera.** Those stay on the workstation.
- A trial sees exactly one device and calls it `cuda:0`, so the four hardcoded `cuda:0` sites (`train_utils.py:658,850,944`, `algorithm.py:137,291`) need **no change**.
- Slots are idle-checked before being claimed, so a neighbour's job cannot cause an OOM that gets scored as a bad hyperparameter. On a MIG box the driver may attribute a compute app to the *parent* GPU rather than the instance; the check matches UUIDs exactly, so it **under-matches** rather than over-matches. That is the safe direction — over-matching would mark all four slices of a card busy the moment one was used — and unattributable apps are reported at startup. `--allow-busy` exists for the workstation, whose desktop session permanently occupies its only GPU.
- **Per-trial CPU is unbounded, so trial duration is neighbour-dependent.** The duration row of the results plot reads as "roughly", not as a measurement.
- Trials are parallel but all bookkeeping happens in the one scheduler process, so **SQLite needs no concurrency story**.
- Checkpointing must be on for resume to be possible (`save_frequency: 100`), so the tuner keeps only the newest couple per trial and deletes them all when the trial ends; at ~19 MB/save, retaining everything would cost ~150 GB over a sweep.
- Launches are staggered (`launch_stagger_seconds`) so the initial `finalize()` scene builds do not contend for pinned host memory. This desynchronises only the *start*; trials drift back into alignment.
- A resumed trial writes a second event file into the same summaries dir, duplicating up to `save_frequency` epochs mid-stream. Harmless for a tail-mean objective; it would matter for positional epoch indexing, which only pruning uses, and pruning is off (ADR-0018).
- If the tuner is `SIGKILL`ed it cannot clean up: trials are left `RUNNING` in the DB and their processes orphaned. The next start reconciles the DB (mark `FAIL`, re-enqueue) and the idle check keeps the orphans' slots out of the pool until they are reaped.
