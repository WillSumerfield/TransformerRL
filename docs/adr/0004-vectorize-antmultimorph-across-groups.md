# Vectorizing AntMultiMorphEnv across EnvironmentGroups

`AntMultiMorphEnv` runs one `EnvironmentGroup` per morphology ([ADR 0001](0001-environment-group-per-morphology.md)),
and its per-step `pre_physics_step` / `compute_observations` / `compute_reward` / `reset_idx` were
`for g in groups:` loops — so per-step host work, and throughput, scaled roughly inversely with the
morphology count (7% of the 1-group rate at 64 groups). We rewrote the step to be fully vectorized
across groups; it is now group-count-independent (~2.9M env-steps/s for 1…1024 groups). vsim's
physics solve is already group-agnostic, so the win was purely removing host-side per-group
Python/torch work. The reusable technique is distilled in
[docs/group_count_throughput.md](../group_count_throughput.md).

## Changes, in order (and what each revealed)

1. **Masked batched reset.** Fill the set buffers for all envs and issue one batched
   `set_articulation_kinematic_states` whose per-group `reset_buf` mask gates which envs are
   written; removed the per-group `.any()` host syncs, the N→1 set dispatch, and a redundant
   pre-reset get. *Revealed:* on its own this **regressed** high group counts (−21% @64, −35%
   @4096). Removing the per-group skip made the reset *fill* unconditional, and an O(N) fill costs
   more than the host syncs it replaced. Lesson: sync removal only pays once the fill is also O(1) —
   the "fill all + masked set" idiom is optimal for vlearn's single-group envs but a loss for many
   groups until the loops are vectorized.
2. **Global root buffers.** Replaced `ArticulationKinematicStateHandler` (which allocates its own
   non-aliasable buffers) with our own get/set commands, so root pose/vel — uniform width across
   morphologies — live in single global tensors with each group's command backed by a contiguous
   row-slice (`wrap_gpu_buffer` accepts a tensor view). Reward, torso obs, old-root save, action
   masking, reset bookkeeping, and the now-constant root reset-fill became whole-tensor. *Result:*
   2–3× over baseline; the regression flipped to a win.
3. **Flat buffers + precomputed gather indices.** Ragged DOF/sensor data (variable width + a
   per-morphology slot permutation) went into one flat buffer per quantity, each group's command
   aliasing a reshaped contiguous slice; precomputed indices turned the last four per-group loops
   (motor gather, DOF and sensor obs scatter, DOF reset-fill) into single whole-tensor ops.
   *Result:* throughput went **flat** in group count. *Revealed:* the raw-vsim floor bench
   (`bench_group_throughput.py`) has its own per-group loop, so it was never the true ceiling — the
   loop-free env exceeds it past ~256 groups.
4. **CUDA-graph capture / `torch.compile` — rejected.** Full-step CUDA-graph capture *froze
   physics* (vsim's `gym.step` does not record into a torch graph; capture silently replayed stale
   state — caught only by checking the root pose stopped moving). `torch.compile` was −4% (in-place
   buffer mutation disables inductor's cudagraphs; vsim graph-breaks fragment the rest). The step is
   ~95% vsim physics, so there is essentially no host overhead left to recover.

## Results (env-steps/s; 4096 classic-ant envs sliced into N groups of the same body)

| Groups | Baseline | +masked reset | +global root | +flat buffers | Final vs base |
|-------:|---------:|--------------:|-------------:|--------------:|--------------:|
| 1      | 2,318,681 | 2,807,798 | 2,823,876 | 2,903,872 | 1.25× |
| 4      | 1,217,617 | 1,715,292 | 2,483,537 | 2,936,474 | 2.41× |
| 16     | 495,345   | 546,954   | 1,457,615 | 2,900,980 | 5.9× |
| 64     | 168,317   | 132,449   | 432,769   | 2,925,050 | 17.4× |
| 256    | 49,582    | 35,439    | 109,891   | 2,901,114 | 58× |
| 1024   | 12,644    | 8,740     | 26,954    | 2,852,354 | 226× |
| 4096   | 3,159     | 2,049     | 6,747     | 2,670,037 | 845× |

`+masked reset` is faster ≤16 groups but slower from 64 up — the regression noted above. 4096
single-env groups is pathological; the final env is flat (~2.9M) for 1…1024.

## Reproducing / affirming

- **Benchmark:** `scripts/bench_group_throughput_env.py` (full env) and
  `scripts/bench_group_throughput.py` (raw-vsim floor). 4096 classic-ant envs (legs {2,4,6,8})
  sliced into N ∈ {1,4,16,64,256,1024,4096} groups of the same body, 20 warmup + 200 timed steps,
  `throughput = total_envs × steps / wall`, with `torch.cuda.synchronize()` bracketing the timed
  block (vsim is async). Each N runs in its **own subprocess** — vsim allows one gym per process (a
  second `create_gym` fails license validation), which is also why per-config isolation is needed.
  Single `cuda:0`.
- **Correctness:** a noise-off parity harness (`reset_noise_scale=0`, fixed actions, resets firing)
  is **bit-identical** to the pre-refactor env at every stage; the ragged gathers were additionally
  cross-checked against an independent per-group reconstruction on a mixed-leg-count morphology set.
  The flat reset-noise draw reorders the RNG stream, so only the noise-off trajectory is
  bit-identical; noise-on is statistically equivalent.

## Consequences

One-group-per-morphology (ADR 0001) no longer carries a throughput penalty — morphology count is
effectively free at runtime. The step has no per-group Python loop, so anyone adding an obs field or
a new per-DOF/per-sensor quantity must extend the flat-buffer + gather-index machinery in
`allocate_buffers`, not reintroduce a `for g in groups:` loop in the step path.
