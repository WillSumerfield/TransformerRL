# Making a vsim env group-count-independent

A guide to why per-`EnvironmentGroup` overhead slows a vsim env, how to measure it, and the
techniques that remove it. Applying these to `AntMultiMorphEnv` took it from inversely-scaling to
**flat ~2.9M env-steps/s regardless of group count**; the ordered change history, per-stage numbers,
and reproduction live in [ADR-0004](./adr/0004-vectorize-antmultimorph-across-groups.md). Benchmark
first run 2026-06-01.

## Background

vlearn's `EnvironmentGroup` is one vsim physics build shared by a batch of envs. This repo uses
**one group per morphology** (see [ADR 0001](./adr/0001-environment-group-per-morphology.md)):
real limb removal needs a distinct vsim body, so the adaptive ant runs 46 groups, the full ant up
to 131. A natural worry follows: does having many groups (each then holding fewer envs) hurt
throughput, independent of the bodies themselves?

To answer that in isolation we hold the **body fixed** — every group runs the *same* classic
4-limb ant — and only vary the group count. So this measures the raw cost of group *granularity*,
not of morphology diversity. (This deliberately breaks the "one group per morphology" convention;
it's a benchmark, not a model of real use.)

## What was tested

Both scripts take a fixed **4096 classic-ant envs** (limbs at 45/135/225/315°) and slice them into
N groups of `4096 / N` envs each, sweeping `N ∈ [1, 4, 16, 64, 256, 1024, 4096]`. For each N they
run 20 warmup + 200 timed steps and report **throughput = total_envs × timed_steps / wall_seconds**
(env-steps/s), with `torch.cuda.synchronize()` bracketing the timed block (vsim is async — without
the sync the timing is meaningless). Each N runs in its **own subprocess** for a fresh CUDA context
and gym, since the gym and the kinematic-handler `_gym` class singleton don't tear down cleanly
in-process.

The two scripts differ in *what layer* they measure:

### 1. `scripts/bench_group_throughput.py` — the vsim floor

Talks to the vsim `gym` directly (modeled on vlearn demo `502_heterogeneous_environment_groups`).
Each timed step does only the simulator primitives:

- per-group random motor-buffer fill (a Python loop, one iteration per group)
- one batched `set_motor_forces` across all groups
- `gym.step()` (a single global solve over all 4096 envs, regardless of grouping)
- one batched `get_articulation_kinematic_states` reading root pose per group

No observations, no reward, no resets. This is the **best case** — the most vsim could deliver.

### 2. `scripts/bench_group_throughput_env.py` — the real training cost

Drives the actual `AntMultiMorphEnv` through its real `env.step()`, built with N **duplicate**
classic-ant morphs (`morphologies=[{2,4,6,8}] * N`) so it produces N groups of the same body. On
top of the vsim primitives above, each step now also pays everything the training loop pays:

- per-group observation scatter into the 139-D padded obs buffer (a Python loop per group)
- full reward / termination / truncation compute
- per-env resets as the random-action ants fall over (adds the per-group kinematic *set* path)
- dof position/velocity and force-sensor reads, not just root pose

This is the **end-to-end** number a real run actually sees.

## Results

Charts: [`data/group_throughput.png`](../data/group_throughput.png) (raw vsim) and
[`data/group_throughput_env.png`](../data/group_throughput_env.png) (full env). Raw tables in
`data/group_throughput.md` and `data/group_throughput_env.md`.

| Groups | Envs/group | Raw vsim (env-steps/s) | Rel. | Full env (env-steps/s) | Rel. |
|-------:|-----------:|-----------------------:|-----:|-----------------------:|-----:|
| 1      | 4096       | 3,604,280              | 1.00 | 2,318,681              | 1.00 |
| 4      | 1024       | 3,616,231              | 1.00 | 1,217,617              | 0.53 |
| 16     | 256        | 3,541,859              | 0.98 | 495,345                | 0.21 |
| 64     | 64         | 3,424,950              | 0.95 | 168,317                | 0.07 |
| 256    | 16         | 2,875,705              | 0.80 | 49,582                 | 0.02 |
| 1024   | 4          | 881,494                | 0.25 | 12,644                 | 0.005 |
| 4096   | 1          | 210,623                | 0.06 | 3,159                  | 0.001 |

## Interpretation

**vsim barely cares about group count; our env code cares a lot.** The two curves tell opposite
stories:

- **Raw vsim is flat through 64 groups** (≤5% loss), then falls off only once envs-per-group gets
  very small (25% of baseline at 1024 groups, 6% at 4096). The physics solve is global over all
  4096 envs no matter how they're grouped; that high-N falloff is **not vsim** but this bench's own
  per-group loop (a `motor_buf.uniform_()` per group plus a per-group handler list). A loop-free env
  stays flat past this bench at high group counts — so read this column as a *bench* floor, not the
  true vsim ceiling.

- **The full env declines from the very first split** — throughput roughly *halves for every 4×
  more groups*, scaling almost inversely with group count the whole way. At 64 groups it's already
  at 7% while raw vsim is still at 95%.

The gap between the curves is **per-group Python/torch overhead in `AntMultiMorphEnv`**: the
`for g in groups:` loops in `pre_physics_step`, `compute_observations` (obs scatter), `reset_idx`,
and reward. That work is essentially un-batched — each extra group adds a fixed slug of host-side
Python the GPU can't hide — so it dominates long before vsim itself struggles.

## The playbook: making the per-step host work O(1) in group count

The cost lives in `for g in groups:` loops, not vsim. Remove them in this order:

1. **Batch every per-group GPU call; never host-sync per group.** Collect per-group commands into
   one array (`gym.create_gpu_array`) and issue a single `get` / `set` / `set_motor_forces`. Delete
   any per-group `.any()` / `.all()` — each is a CPU↔GPU sync that stalls the pipeline N times.
2. **Use a masked set instead of per-group conditional writes.** A vsim set command can carry a
   `masks_buffer`; vsim applies the write only where the mask is true (the VSim *Command mask*
   term). Fill the set buffers for *all* envs and let the mask gate them — no per-group branch or
   gather. **Caveat:** this only wins if the fill is itself O(1) (steps 3–4); an unconditional
   per-group *fill* costs more than the host syncs it replaces, so at high group count you regress.
   Sync removal and fill-vectorization must land together.
3. **Alias uniform-width state into one global tensor.** Quantities that are the same width for
   every morphology (here, root pose/vel) live in a single `(N, …)` tensor; back each group's
   command with a contiguous **row-slice** (`wrap_gpu_buffer` accepts a tensor view). The batched
   get then writes straight into the global tensor, and everything downstream (obs root block,
   reward, reset bookkeeping) becomes a whole-tensor op.
4. **Flatten ragged state and precompute a gather index.** Variable-width / permuted data (DOF and
   force-sensor values) can't share a rectangular tensor. Put every group's data end-to-end in one
   **flat buffer**, alias each group's command to a reshaped contiguous slice of it, and compute
   once an index mapping flat ↔ padded obs slots. The per-step scatter/gather is then a single
   `flat[gather_idx] * mask` (inactive slots gather index 0 and the mask zeroes them).
5. **Hoist constants out of the step loop.** Anything that doesn't change per step (here, the
   reset-to-init root pose) is computed once at allocate, not rebuilt every reset.

## Verifying a change

- **Bit-identical parity.** Run a fixed action sequence with `reset_noise_scale=0` (deterministic
  reset) before and after; assert the obs/reward/term trajectory is unchanged. Catches index and
  aliasing bugs while resets fire. (Vectorizing the reset noise into one flat draw reorders the RNG
  stream, so only the noise-off trajectory is bit-identical — noise-on stays statistically equal.)
- **Ragged cross-check.** With a mixed-limb-count morphology set, reconstruct the obs DOF/sensor
  regions independently from the flat buffers and compare — varying widths exercise the gather
  indices that a single-width set does not.
- **If you try CUDA-graph capture: confirm physics actually advanced.** vsim's `gym.step` does not
  record into a torch CUDA graph — capture *succeeds* but replays stale physics. Always check a
  vsim-produced value (e.g. root pose) changes across replays before trusting any timing.

## The ceiling

Once de-looped, the step is **~95% vsim physics** and group-count-independent — almost no host
overhead remains. CUDA-graph capture and `torch.compile` were evaluated and do **not** help (vsim
won't capture; in-place buffer mutation disables inductor's cudagraphs). The simulator is the floor;
stop there. Full results, per-stage numbers, and reproduction are in
[ADR-0004](./adr/0004-vectorize-antmultimorph-across-groups.md).
