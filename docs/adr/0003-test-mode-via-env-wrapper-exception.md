# Test mode drives rl_games player via env wrapper + exception

> **Loop half superseded by [ADR-0007](0007-test-mode-owns-rollout-loop.md):** `test` mode now owns
> its rollout loop (no `player.run()`, no `_TestDone`); the player is reused for checkpoint *loading*
> only. The "reuse the player's load path, not a hand-rolled one" rationale below still holds.

`test` mode reuses rl_games' `PlayerContinuous` loop rather than reimplementing model loading and inference from scratch. A `_TestTracker` wrapper is inserted between the raw env and `NewToOldAPICompatilibity`; it intercepts `step()` to accumulate per-env episode rewards. When every env slot has completed `num_episodes` episodes, the tracker raises `_TestDone` (a custom exception subclass of `Exception`), which propagates up through the player loop and is caught at the `runner.run()` call site in `run_training`.

**Alternatives considered:**
- *Custom model loading loop* — required replicating rl_games' checkpoint loading, obs normalization, and inference plumbing (all version-specific internals). High fragility.
- *Subclass `PlayerContinuous.run()`* — cleaner in principle, but rl_games doesn't expose a stable hook between player creation and `run()` in `Runner.run_play()`; would require monkeypatching the runner.

**Trade-off:** The exception-as-control-flow is unusual but self-contained — `_TestDone` is a named type, the raise site is a single line in `_TestTracker.step()`, and the catch site in `run_training` is adjacent to the existing recorder-stop handler that uses the same pattern.

## Render shutdown (play / random)

The same exception-as-control-flow pattern terminates the **rendering** loop, where no clean stop exists either: rl_games' `player.run()` (play) and our hand-rolled `_run_random` loop both call `env.step()`, which calls `render()` — there is no return value the caller can poll mid-`step()`.

`render()` raises a named sentinel **`RenderFinished`** (subclass of `Exception`, defined in `envs/multigroup_environment.py` beside the raise site) whenever `render_finished and raise_exception`. `render_finished` flips `True` on any of:
- the viewer window being closed (`gym_render.render()` returns `True`),
- the `--num-episodes` step cap (`_PlayLimiter.step()` after `num_episodes × max_episode_length` steps),
- the video recorder hitting its frame budget with `stop_env=True` (play-mode `--video`).

Caught in two places — the `runner.run()` call site (play) and the `_run_random` loop — both treating `RenderFinished` as a clean exit and letting every other `Exception` propagate as a real crash.

**Why named, not bare:** the original raise was a bare `Exception`, caught only when a video recorder had finished (`recorder.done`); plain `play`/`random` shutdown therefore surfaced as an uncaught traceback, and a *genuine* bug couldn't be told apart from an intended stop. The named sentinel makes intended shutdown unambiguous — same rationale as `_TestDone` above.

**Rejected alternative:** poll `render_finished` between steps instead of raising. Rejected because `render()` runs *inside* `env.step()`, so the loop can't observe the flag until the step that set it has already returned — the window-close frame would still need an exception (or a deeper rl_games hook) to unwind.
