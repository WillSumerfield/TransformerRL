# Test mode owns its rollout loop (reuse the player for loading only)

> **OBSOLETE (2026-07-26):** `test` mode, `--data-type`/`--num-samples`, the morph-value sweep, and
> `transformer_rl/rollout.py` have all been removed. Headless checkpoint evaluation now lives in a
> standalone `eval.py` (not a run mode). The **own-the-loop decision below still stands as the design
> eval.py inherits** — build the player, `restore()` only, drive a custom loop reading `(mu, value)`
> directly — but the `test`-specific specifics (`_TestDone`, `--data-type`, per-Sample resample flags)
> are historical.

Supersedes the loop half of [ADR-0003](0003-test-mode-via-env-wrapper-exception.md).

ADR-0003 made `test` mode reuse rl_games' player *loop* (`runner.run({play:True})`), with a
`_TestTracker` env wrapper intercepting `step()` to accumulate per-env reward and a `_TestDone`
exception to break out. That works when the only thing `test` needs is reward/done — both live on the
env, which the wrapper sees.

A new analysis need breaks that assumption. The morph-value sweep wants, per step, the critic's
**value estimate** — and between population draws it wants to call `env.resample()`. Neither is
reachable through the env wrapper: the value is computed inside the player's model forward and
discarded (`get_action` returns only the action), and `player.run()` owns the loop, so there is no
seam to inject a `resample()` between [Samples](../../experiments/CONTEXT.md). ADR-0003 itself noted
rl_games exposes no stable hook between player creation and `run()`.

The pull is to keep two loops — `player.run()` for the existing per-morph summary, a custom loop only
for full capture — but two rollout implementations of the same eval would drift.

## Decision

`test` mode **always** runs its own rollout loop; rl_games' player is reused only for what it does
well — restoring the checkpoint (actor/critic weights, `running_mean_std`, `value_mean_std`).

- **Build the player, skip `player.run()`.** Construct the player and call `restore(checkpoint)`, then
  drive our own loop instead of `runner.run({play:True})`. Each step calls the model forward directly
  to get `(mu, value)` together; the deterministic action is `mu`.
- **One loop, flag-gated logging.** The loop runs `K` episodes per env across `S` Samples, calling
  `env.resample()` + `env.reset()` between Samples. `--data-type summary` (default) keeps the existing
  per-morph CSV/PNG/MD; `--data-type full` writes one self-contained `.npz` (per-step value/reward
  traces + per-env morph features). `--num-samples` defaults to `1` (no resample) and **hard-errors** if `>1` on a
  non-`sample_morphs` env, so adaptive-ant `test --compare`/`--test-set` are unchanged; the sweep uses
  `--num-samples 5 --num-episodes 1 --data-type full`.
- **`_TestDone` retired.** Owning the loop removes the exception-as-control-flow; the loop simply ends
  when every env has finished its `K` episodes for the current Sample.
- **The rollout engine lives in `transformer_rl/rollout.py`**, imported by `test` mode and reusable by
  future single-use `experiments/` scripts.

## Trade-off

Reimplementing the loop risks regressing the summary/`--compare` outputs that previously came for free
from `player.run()` — mitigated by reusing the player's load path verbatim (the fragile part ADR-0003
rightly avoided) and by re-verifying summary parity. Accepted because the alternative — two divergent
rollout loops — is worse over time, and because value capture + mid-rollout `resample()` are
structurally impossible through the env-wrapper seam.
