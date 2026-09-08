"""What experiment 5's no-generator arms ARE: a name, and the algorithm that name resolves to.

Split out of `scripts/optimize_baselines.py` because the training run is no longer the only thing
that has to construct these algorithms. `rerank` reconstitutes a finished run's algorithm to read
its checkpoint, and the specialization pass after it does the same to continue training -- and an
algorithm built from different overrides than the run that wrote a checkpoint loads a
differently-shaped network out of it, silently where the shapes happen to agree. One constructor,
so a `--set` the launcher passes reaches every later pass over the run it produced.

Here rather than in `transformer_rl/` because the arm is an experimental condition, not a component:
`RandomBodyAlgorithm` is the reusable object, and "which of the two baselines is this" is a fact
about experiment 5. `scripts/eval.py` already imports the harness this way round.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from transformer_rl.algorithm import CodesignAlgorithm
from transformer_rl.random_body import RandomBodyAlgorithm

_ROOT = Path(__file__).resolve().parent.parent.parent

# The first two arms are TransformerRL's control-only comparisons.  The native
# arms are intentionally run by SoftwarePackage's maintained runners, not
# adapted to CodesignAlgorithm.
LOCAL_ARMS = ("fixed_body", "random_generator")
NATIVE_ARMS = ("nge", "bodygen", "stackelberg")
ARMS = (*LOCAL_ARMS, *NATIVE_ARMS)

EXPERIMENT = "codesign_baselines"
"""The run-dir leaf both sides must agree on: `runs/<task>_codesign/codesign_baselines`. Defined
once and imported by `launch`, because the launcher derives done-detection and resume from the path
IT composes while the script writes to the path the ALGORITHM composes -- and the two disagreeing
does not fail, it just means every finished run reads as never-started."""


def software_package_root() -> Path:
    """Source checkout providing the native runners and their benchmark YAMLs.

    The wheel supplies `codesigner`, but does not install the repository's
    scripts and configs.  Set SOFTWARE_PACKAGE_ROOT on a remote machine when
    its sibling-checkout layout differs from this workstation's.
    """
    configured = os.environ.get("SOFTWARE_PACKAGE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return _ROOT.parents[2] / "AIRA_packages" / "SoftwarePackage"


def native_runner(arm: str) -> Path:
    """Maintained SoftwarePackage runner for one native baseline method."""
    if arm not in NATIVE_ARMS:
        raise ValueError(f"{arm!r} is not a native SoftwarePackage arm")
    return software_package_root() / "scripts" / f"train_{arm}.py"


def native_config(arm: str) -> Path:
    """Maintained SoftwarePackage config for one native baseline method."""
    if arm not in NATIVE_ARMS:
        raise ValueError(f"{arm!r} is not a native SoftwarePackage arm")
    return software_package_root() / "config" / f"{arm}.yaml"


def native_checkpoint(arm: str, run_dir: Path) -> Path | None:
    """Newest resumable checkpoint in this runner's native on-disk format."""
    checkpoints = run_dir / "checkpoints"
    if arm == "nge":
        matches = sorted(checkpoints.glob("generation_*.pt"))
        return matches[-1] if matches else None
    if arm == "bodygen":
        latest = checkpoints / "latest_resume.pth"
        if latest.is_file():
            return latest
        matches = sorted(checkpoints.glob("epoch_*.p"))
        return matches[-1] if matches else None
    if arm == "stackelberg":
        matches = sorted(checkpoints.glob("update_*.pth"))
        return matches[-1] if matches else None
    raise ValueError(f"{arm!r} is not a native SoftwarePackage arm")


def native_finished(arm: str, run_dir: Path) -> bool:
    """Whether the native runner wrote its completion artifact."""
    if arm in {"nge", "bodygen"}:
        return (run_dir / "summary.json").is_file()
    if arm == "stackelberg":
        return (run_dir / "checkpoints" / "final.pth").is_file()
    raise ValueError(f"{arm!r} is not a native SoftwarePackage arm")


def overrides_from_sets(sets, **ppo) -> dict:
    """`--set KEY=VAL` strings as the nested dict an Algorithm takes, plus any `params.config` keys
    passed by name. Values are YAML-parsed, the same way `train_utils` parses the trainer's."""
    overrides: dict = {}
    ppo = {k: v for k, v in ppo.items() if v is not None}
    if ppo:
        overrides["params"] = {"config": ppo}
    for kv in sets:
        key, _, val = kv.partition("=")
        *path, leaf = key.split(".")
        d = overrides
        for p in path:
            d = d.setdefault(p, {})
        d[leaf] = yaml.safe_load(val)
    return overrides


def algorithm_for(arm: str, config, *, name: str | None = None, seed: int | None = None,
                  overrides: dict | None = None, experiment: str = EXPERIMENT):
    """The arm's algorithm, already carrying this run's config overrides.

    `fixed_body` gets an ordinary `CodesignAlgorithm` under a distinct name: it is not a different
    algorithm, it is this one constrained from outside by `fix_morphologies` (D18). Only
    `random_generator` needs a class of its own, because replacing the body source is a change to
    the window boundary rather than to what may be built.
    """
    if arm not in LOCAL_ARMS:
        raise ValueError(
            f"{arm!r} is a native SoftwarePackage method; launch it through the harness"
        )
    config = Path(config)
    if not config.is_absolute():
        # A path relative to the repo root OR a bare name under configs/, since the launcher passes
        # the first and a human at a shell types the second.
        rooted = _ROOT / config
        config = rooted if rooted.exists() else _ROOT / "configs" / config
    cls = RandomBodyAlgorithm if arm == "random_generator" else CodesignAlgorithm
    kwargs = {} if arm == "random_generator" else {"name": "fixed_body_control"}
    return cls(config, run_name=name, overrides=overrides or {}, seed=seed,
               experiment=experiment, **kwargs)
