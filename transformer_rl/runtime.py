"""What this process is running: the one ModuleLibrary, the seed body, and the Task's obs layout.

rl_games builds the network from a config dict and hands it no env, so the live library cannot
reach it as an argument -- and rebuilding it there from a name in the network config is the
duplication D14 removes: two instances of the same library, silently free to disagree. `run_training`
constructs one, names it once, and parks it here; the network and the agent read it back.

Anything holding an env should read `env.module_library` instead -- the Task carries it. This exists
for the callers that have no env to ask, and it holds the two facts the Task deliberately does not:
which body seeds the generator, and (until the env is up) nothing else.

One process runs one library, for the same reason it runs one gym.
"""

_LIBRARY = None
_BASE_MORPHOLOGY = None
_OBS_LAYOUT = None


def set_run(library=None, base_morphology=None, obs_layout=None) -> None:
    """Record whichever of the three are known yet. The library and the seed body are settled
    before the Runner starts; the layout only exists once a Task has been set up."""
    global _LIBRARY, _BASE_MORPHOLOGY, _OBS_LAYOUT
    if library is not None:
        _LIBRARY = library
    if base_morphology is not None:
        _BASE_MORPHOLOGY = base_morphology
    if obs_layout is not None:
        _OBS_LAYOUT = obs_layout


def library():
    assert _LIBRARY is not None, "no ModuleLibrary set -- run_training sets it before the Runner"
    return _LIBRARY


def base_morphology():
    assert _BASE_MORPHOLOGY is not None, "no seed body set -- run_training sets it from env.base_morphology"
    return _BASE_MORPHOLOGY


def obs_layout() -> dict:
    assert _OBS_LAYOUT is not None, "no obs layout set -- the Task publishes it at setup()"
    return _OBS_LAYOUT
