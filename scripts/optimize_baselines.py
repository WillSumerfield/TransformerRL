"""Experiment 5's cross-method baselines: the two conditions that use no learned generator.

    python scripts/optimize_baselines.py fixed_body       --config ppo_ant_codesign_tuned.yaml \
        --name baselines_fixed_body_s42 --seed 42
    python scripts/optimize_baselines.py random_generator --config ppo_ant_codesign_tuned.yaml \
        --name baselines_random_generator_s42 --seed 42

Both are **body sources**, not algorithms in their own right: same network, same config, same Task,
same control stack, same budget as the tuned codesign run they are compared against, differing only
in where a window's bodies come from. Which is why they are two flags on one script rather than two
scripts, and why they go through the package's entry points rather than around them --
`optimize_control` IS the fixed-body baseline (D18: a fixed generator is how a control-only
baseline is expressed), so there is no wrapper class to write.

    fixed_body        the **reference morphology**, held still, controlled as well as the budget
                      allows. The normalizer every other method's score is reported against, so it
                      is deliberately a good body rather than the seed body a codesign run starts
                      from.
    random_generator  the **random-design baseline**: the population redrawn every window from the
                      uniform-size draw. See `transformer_rl/random_body.py`.

Checkpoints are the package's (`RunDirectory`: latest/best/config.json/metrics.json), not
rl_games' `nn/*.pth`, and `resume=True` means *run or continue* -- so a crashed run is restarted by
re-issuing the identical command, and the launcher passes no checkpoint path.
"""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from argparse import ArgumentParser
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from codesigner import optimize
from codesigner.control import optimize_control
from codesigner.metrics.record import MetricRecord

from experiments.harness.baselines import LOCAL_ARMS, algorithm_for, overrides_from_sets
from transformer_rl.morphology import reference_body


def build(args):
    """The arm's algorithm, already carrying this run's config overrides."""
    return algorithm_for(args.arm, args.config, name=args.name, seed=args.seed,
                         overrides=overrides_from_sets(args.set_keys, num_actors=args.num_envs,
                                                       max_epochs=args.max_epochs))


def main():
    ap = ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("arm", choices=LOCAL_ARMS)
    ap.add_argument("--config", required=True,
                    help="run config (names the task); path or a name under configs/")
    ap.add_argument("--name", required=True,
                    help="run name; the leaf output dir. Required, unlike the codesign script's: "
                         "the run dir IS the resume key here, and a timestamped default would "
                         "start a new run every time a crashed one was re-issued")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None, dest="num_envs")
    ap.add_argument("--max-epochs", type=int, default=None, dest="max_epochs")
    ap.add_argument("--quiet-iterations", action="store_true",
                    help="drop the per-epoch iteration tick; keep only the per-window run tick")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL", dest="set_keys",
                    help="override any config key by dotted path, repeatable; VAL is YAML-parsed")
    args = ap.parse_args()

    algorithm = build(args)
    library = algorithm.make_library()
    task = algorithm.make_task()

    def on_iteration(p):
        if p.tick % 20 == 0 and p.reward is not None:
            print(f"    epoch {p.tick}: reward {p.reward:.1f}", flush=True)

    run_dir = Path(algorithm._cfg["params"]["config"]["train_dir"],
                   algorithm._cfg["params"]["config"]["full_experiment_name"])
    common = dict(on_iteration=None if args.quiet_iterations else on_iteration,
                  checkpoint_dir=run_dir, resume=True)

    if args.arm == "fixed_body":
        # ONE body, and the package tiles it across the whole population -- a fixed set is a
        # whitelist of what may be built, never a layout, so every morph group runs the reference
        # rather than one group running it while the rest idle.
        body = reference_body(library)
        print(f"[baselines] fixed_body on {body}", flush=True)
        # `population=0` switches off the recorder's per-tick draw, and with it every exploration
        # metric. Not tidiness: this arm's generator head EXISTS and is never trained, and the
        # shared trunk means control updates drift it -- so a draw from it is noise, and left on it
        # reports the fixed-body baseline discovering design modes it has no mechanism to discover.
        # The random-design arm keeps the draw, because there its draw is the body source itself.
        best, policy, record = optimize_control(task, algorithm, library, [body],
                                                metrics=MetricRecord(population=0), **common)
        generator = None
    else:
        # `retain=None`, the same policy `optimize_control` forces on the other arm and for the
        # same reason: retained checkpoints exist so a protocol can read a run a third and two
        # thirds of the way through its DESIGN SEARCH, and neither of these arms has one. The draw
        # is the same distribution in window 48 as in window 1.
        best, policy, generator, record = optimize(algorithm, task, library, retain=None, **common)

    print(f"\nbest reward {best:.2f}")
    print(f"  policy    {policy}")
    print(f"  run dir   {run_dir}")
    if generator is not None:
        shortlist = algorithm.shortlist.entries
        print(f"  committed {generator.generate(1, deterministic=True)[0]}")
        print(f"  shortlist {len(shortlist)} bodies, "
              f"R {shortlist[0]['score']:.1f}..{shortlist[-1]['score']:.1f} -- RAW, one episode "
              f"each; re-evaluate before believing the order")
    for name, value in record.summary().items():
        print(f"    {name:24s} {value}")


if __name__ == "__main__":
    main()
