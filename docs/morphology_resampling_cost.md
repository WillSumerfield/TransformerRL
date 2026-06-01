# The cost of morphology resampling

How expensive it is to resample the full ant's morphology set mid-training, why the cost lives where
it does, and how to pick a resample cadence. The architectural decision (resampling *is* a full gym
rebuild, vs. alternatives) is recorded in
[ADR-0005](adr/0005-runtime-morphology-resampling-via-gym-rebuild.md); this doc is the cost model
that informs the cadence knob. Benchmarked 2026-06-01 (single `cuda:0`; absolute seconds are
machine-specific — the *ratios* are the durable part).

## Why resampling means a full rebuild

The full ant runs one `EnvironmentGroup` per body (epm=1, so 4096 groups). A resample draws a fresh
set of variable-length bodies; **vsim bakes link geometry at `gym.finalize()` and offers no in-place
mutation of segment lengths** (only mass/inertia and joint properties are runtime-writable). So new
lengths require tearing the whole sim down and rebuilding it:

```
drop env refs  →  v.delete_gym()  →  v.create_gym()
   →  regenerate + import 4096 .vsim defs  →  finalize()  →  allocate buffers/commands
```

Because lengths are continuous, essentially every body is new each resample — **no body can be
reused**, so the rebuild is all-or-nothing. This is the floor, not an implementation we can shave.

## What a rebuild costs

| Stage | Time @ 4096 groups |
|---|---|
| `delete_gym` | 4.4 s |
| reconstruct (gen + import + finalize + alloc) | 10.0 s |
| **full rebuild** | **14.4 s ± 0.14** |
| (initial construct, for reference) | 10.7 s |

The cost is vsim scene construction — `import_definitions` per group, `finalize` (scene re-lock +
contact-buffer alloc over 4096 envs), and our buffer/command allocation. It is flat and repeatable
(±1%).

### Generating the `.vsim` files is *not* the cost

Producing a full set of 4096 variable-length bodies — sample → build XML → write to disk — takes
**0.53 s** (216 ms building strings + 313 ms disk writes, ~17 KB/file). That is ~5% of the
reconstruct and ~4% of the whole rebuild. **Don't optimize the generator** to speed up resampling;
the time is in vsim, not in our Python.

## The denominator: why an episode is slow (and the overhead small)

The rebuild is a fixed 14.4 s; whether that hurts depends on how much training happens between
resamples. Real full-ant PPO runs at `fps_total ≈ 70k` env-frames/s, so **one 1000-step episode ≈
58.5 s of wall-clock** — not the ~1.5 s the raw env ceiling would suggest. Per epoch (~0.94 s):

| Phase | Share |
|---|---|
| sim stepping | ~13% |
| action inference (rollout) | ~5% |
| **PPO update** | **~82%** |

The update dominates because each collected sample is pushed through the net `mini_epochs=4` times
with forward **+ backward**, against a simulator that costs ~1.8 µs/env/step (a tuned GPU physics
kernel; host overhead already removed — see [group_count_throughput.md](group_count_throughput.md)).
A consequence: **PPO wall-time is independent of morphology count** — the dominant term is batched
net compute, blind to how many distinct bodies are in the batch (the network always pads to 8 legs /
16 DOF and masks). So resampling more bodies never makes a step slower; it only adds rebuild stalls.

## Picking a cadence

Overhead of resampling every `K` episodes:

```
overhead = rebuild / (K × episode + rebuild) = 14.4 / (K × 58.5 + 14.4)
```

| K (episodes/resample) | training between resamples | rebuild overhead |
|---:|---:|---:|
| 1 | 58.5 s | **~20%** |
| 2 | 117 s | ~11% |
| 5 | 293 s | ~5% |
| 10 | 585 s | ~2% |

**Recommendation:** expose a `resample_interval` (episodes) config knob and default it to keep
overhead in the single digits (K≈5 → ~5%). Every-episode resampling (K=1) is tolerable (~20%) if
maximal morphology turnover matters more than throughput, but it is the worst case. Two notes when
re-deciding:

- The overhead scales with `episode_time`, which is **machine- and config-dependent** (`fps_total`,
  `max_episode_length`, net size). Re-measure `fps_total` on the target machine before trusting the
  table — `episode_time = max_episode_length × num_actors / fps_total`.
- Resampling fewer bodies per rebuild does **not** help: `finalize` rebuilds the entire scene
  regardless, so a partial resample costs the same 14.4 s. The only lever is frequency (K).

## Reproducing

- **Generation:** `scripts/bench_vsim_gen.py` — samples 4096 bodies, times build vs write over Z=10
  repeats (+1 warmup). Pure CPU, no gym.
- **Rebuild:** `scripts/bench_gym_rebuild.py` — initial construct, then R=3 steady-state rebuilds
  (`delete_gym` + reconstruct) at 4096 groups in one process (one gym at a time; `delete_gym`
  between). `torch.cuda.synchronize()` brackets each construct.
- **Episode time:** run `scripts/train_ant_full.py train --max_epochs 40 --seed 42` and read steady
  `fps_total`; `episode_time = 1000 × 4096 / fps_total`.
