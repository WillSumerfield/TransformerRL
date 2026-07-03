# The resample rebuild crash (pinned-host-memory use-after-free)

An intermittent crash at morphology resample, when `AntMultiMorphEnv._rebuild()` tears down and
recreates the gym. Documented here because it's a **low-frequency race** that resists a clean repro,
so the response is mitigation + a safety net rather than a verified root-cause fix.

## Symptom

At a resample (`_maybe_resample` → `env.resample()` → `_rebuild()` → `v.delete_gym()` + recreate the
4096 articulations), the process dies with either:

- `.../vsim/extension/VsDefaultAllocators.cpp : 428): FATAL: Error deallocating pinned host memory`
  (vsim calling `cudaFreeHost` on a corrupted/already-freed pointer), or
- `torch.AcceleratorError: CUDA error: invalid argument`, surfacing at the first CUDA op *after* the
  rebuild (e.g. `self._reset_buf[:] = True` in `reset`) — an async error reported late.

**Low frequency:** some full runs reach 3000 epochs (~47 rebuilds) fine; others die within the first
few resamples. Crash resamples observed: #1 (ep 63), #2 (ep 127), #3 (ep 189) across different runs.

## What it is

An **async use-after-free around `delete_gym()`**: in-flight async work still references a pinned host
buffer that `delete_gym` frees. The bad free is inside vsim's C++ allocator — not our Python.

Established by experiment:

- **It's a race.** `CUDA_LAUNCH_BLOCKING=1` survived 6 resamples cleanly; serializing launches hides it.
- **It needs a full training window's worth of in-flight work.** Forcing frequent rebuilds
  (`resample_interval: 0.05`, little compute between them) never crashed in 13 rebuilds. Every
  historical crash was at `resample_interval: 1` (a rebuild every ~63 epochs).
- **Independent of `--seed`/determinism** (a determinism-off run crashed) and **not** a debug-artifact
  (`train_phase1_s42_v2` crashed with no debug instrumentation active).
- **`torch.cuda.synchronize()` alone is insufficient** — vsim drives its own stream/pipeline that
  torch's device sync does not cover (it exposes `end_streaming` / `_check_for_cuda_errors`).

## Why it won't reproduce on demand

0 crashes in ~16 rebuild attempts on the clean tree (compile on/off × interval 0.05/1, sailing past all
three historical crash points). So **compute-sanitizer is not viable** — memcheck's heavy timing
perturbation would suppress the race just like `LAUNCH_BLOCKING`, and there's no minimal repro to run
under it. Root-causing inside vsim is deferred as low-value.

## Mitigations

1. **Drain before teardown — APPLIED** (`ant_multimorph._rebuild()`): `torch.cuda.synchronize()` +
   vsim `end_streaming()` + `_check_for_cuda_errors()` before dropping refs / `delete_gym()`. Flushes
   both torch and vsim in-flight work so nothing dangles into the free. Cost **~0.2 ms vs ~14 s** per
   rebuild (negligible). Targets the leading hypothesis but is **UNVERIFIED** — with no repro we can't
   prove it removes the crash; it's free insurance, not a guarantee.

2. **Auto-resume — RECOMMENDED, NOT YET IMPLEMENTED.** rl_games checkpoints every `save_frequency`
   (50) epochs. A thin wrapper that relaunches training from the latest checkpoint on non-zero exit
   turns a rare rebuild crash into one auto-restart (≤50 epochs lost) instead of a dead run. This is
   the robust safety net given the drain can't be verified; add it before long unattended sweeps.

3. **Guaranteed workaround:** `CUDA_LAUNCH_BLOCKING=1` eliminates the race by serialization, at a
   throughput cost. Use for a single run that absolutely must not die.

## If it recurs

Get a reliable repro first (grind `resample_interval: 1` rebuilds until it fires — the crash needs the
full inter-rebuild work), then `compute-sanitizer --tool memcheck` around a rebuild, or debug vsim's
pinned-buffer lifecycle in `delete_gym`. Until a repro exists, prefer mitigation #2 over chasing it.
