# Test mode drives rl_games player via env wrapper + exception

`test` mode reuses rl_games' `PlayerContinuous` loop rather than reimplementing model loading and inference from scratch. A `_TestTracker` wrapper is inserted between the raw env and `NewToOldAPICompatilibity`; it intercepts `step()` to accumulate per-env episode rewards. When every env slot has completed `num_episodes` episodes, the tracker raises `_TestDone` (a custom exception subclass of `Exception`), which propagates up through the player loop and is caught at the `runner.run()` call site in `run_training`.

**Alternatives considered:**
- *Custom model loading loop* — required replicating rl_games' checkpoint loading, obs normalization, and inference plumbing (all version-specific internals). High fragility.
- *Subclass `PlayerContinuous.run()`* — cleaner in principle, but rl_games doesn't expose a stable hook between player creation and `run()` in `Runner.run_play()`; would require monkeypatching the runner.

**Trade-off:** The exception-as-control-flow is unusual but self-contained — `_TestDone` is a named type, the raise site is a single line in `_TestTracker.step()`, and the catch site in `run_training` is adjacent to the existing recorder-stop handler that uses the same pattern.
