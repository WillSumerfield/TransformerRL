# Runtime morphology resampling via full gym rebuild

The full ant trains a controller to generalize across bodies. With a fixed sampled set, the policy
only ever sees `num_envs` morphologies for the whole run. To keep it meeting unseen bodies we
**resample the morphology set partway through training** — and because vsim bakes link geometry at
`finalize()` with no in-place mutation, that means a **full in-process gym rebuild**, not a cheap
swap.

## Decision

Resampling is a periodic **full teardown + rebuild of the gym on the same env object**, triggered
from the training agent:

- **Trigger.** `LoggingA2CAgent` counts env-steps at each epoch boundary and, every
  `resample_interval` episodes (`resample_interval × max_episode_length` steps), calls
  `env.resample()` then refreshes its cached rollout-start obs (`self.obs = self.env_reset()`).
  Off by default (`resample_interval: 0`); only the full-ant config sets it. Guarded so it no-ops
  unless the env samples morphologies.
- **Rebuild.** `AntMultiMorphEnv.resample()` draws a fresh set from a persistent seeded rng, drops
  all gym-backed references, `delete_gym()`, recreates the gym (`MultiGroupEnvironmentGpu._create_gym`,
  factored out so init and rebuild share it), then `create_envs` / `allocate_buffers` / `finalize`.
  `num_envs` and the obs shape (139) are invariant, so rl_games' experience buffer is untouched.
- **Reproducibility.** A single `random.Random(seed)` on the env feeds the initial draw and every
  resample, so the whole morphology stream is reproducible from the run seed.

Cost and cadence are measured in [docs/morphology_resampling_cost.md](../morphology_resampling_cost.md)
(rebuild ~14.4 s; episode ~58.5 s; default K=5 ≈ 5% overhead).

## Alternatives considered

- **In-place geometry mutation.** vsim only exposes runtime writes for joint properties and link
  mass/inertia — *not* link geometry (segment lengths, collision shapes, joint origins), which is
  locked at `finalize()`. New lengths are impossible without a rebuild.
- **Process-restart resampling.** Checkpoint, kill, relaunch with a new set. Sidesteps the
  teardown, but loses optimizer/normalizer state and adds orchestration; rejected once we confirmed
  `delete_gym` + `create_gym` works in-process.
- **Pre-build a fixed pool of bodies.** Lengths are continuous, so a finite pool can't cover them;
  and simulating extra inactive envs wastes the physics solve every step.
- **Partial resample (swap only some bodies).** `finalize()` rebuilds the entire scene regardless,
  so resampling 1 body costs the same as resampling all 4096. Frequency (K), not granularity, is the
  only lever.

## Consequences

- **A resample stalls training ~14.4 s.** The `resample_interval` knob trades morphology turnover
  against this overhead; see the cost doc for the K-vs-overhead table.
- **Checkpoint resume restarts the stream.** The rng position and step counter are not checkpointed,
  so a resumed run's morphology sequence begins again from the seed rather than continuing. Fresh
  runs are fully reproducible.
- **The resample hard-resets every env**, ending all in-flight episodes mid-stride. The agent zeroes
  `current_rewards`/`current_lengths` so the partial episodes don't pollute reward logging, and
  resets the per-morph logging metadata (the bodies — hence the labels — changed).
- **The input normalizer carries over.** RMS stats stay valid because the sampled length/topology
  distribution is stationary across resamples.
- **The step path is unchanged.** Resampling only rebuilds between epochs; per-step throughput and
  the group-vectorized hot path ([ADR-0004](0004-vectorize-antmultimorph-across-groups.md)) are
  untouched.
