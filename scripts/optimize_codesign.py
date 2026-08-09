"""Single-network codesign, driven through the package (`codesigner.optimize`).

The same agent, network and config as `train_codesign_single.py` -- what differs is who owns the
loop. Here the package's driver does: it hands the algorithm a Task and a ModuleLibrary, calls
`run()` until the algorithm reports finished, and fires a progress tick and a checkpoint after each
one. One `run()` is one resample window.

Use the other script for day-to-day training (it has play mode, video, the follow camera and the
full CLI). Use this one to exercise the package contract, and as the worked example of what
implementing `Algorithm` gets you.

    python scripts/optimize_codesign.py --name my_run --max-epochs 200
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

from transformer_rl.algorithm import CodesignAlgorithm


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="run config (names the task); path or a name under configs/")
    ap.add_argument("--name", default=None, help="run name; the leaf output dir")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None, dest="num_envs")
    ap.add_argument("--max-epochs", type=int, default=None, dest="max_epochs")
    ap.add_argument("--quiet-iterations", action="store_true",
                    help="drop the per-epoch iteration tick; keep only the per-window run tick")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL", dest="set_keys",
                    help="override any config key by dotted path, repeatable; VAL is YAML-parsed. "
                         "e.g. --set env.max_episode_length=16")
    args = ap.parse_args()

    overrides = {}
    ppo = {}
    if args.num_envs is not None:
        ppo["num_actors"] = args.num_envs
    if args.max_epochs is not None:
        ppo["max_epochs"] = args.max_epochs
    if ppo:
        overrides["params"] = {"config": ppo}
    for kv in args.set_keys:
        import yaml
        key, _, val = kv.partition("=")
        *path, leaf = key.split(".")
        d = overrides
        for p in path:
            d = d.setdefault(p, {})
        d[leaf] = yaml.safe_load(val)

    config = args.config if Path(args.config).is_absolute() else _ROOT / "configs" / args.config
    algorithm = CodesignAlgorithm(config, run_name=args.name, overrides=overrides, seed=args.seed)

    # The library and the Task are the caller's to build and the driver's to hand over -- the Task
    # arrives unsized, because how many envs and which bodies are the algorithm's choice (D7).
    library = algorithm.make_library()
    task = algorithm.make_task()

    def on_iteration(record):
        if record.step % 20 == 0 and record.reward is not None:
            print(f"    epoch {record.step}: reward {record.reward:.1f}", flush=True)

    run_dir = Path(algorithm._cfg["params"]["config"]["train_dir"],
                   algorithm._cfg["params"]["config"]["full_experiment_name"])
    best, policy, generator = optimize(
        algorithm, task, library,
        on_iteration=None if args.quiet_iterations else on_iteration,
        checkpoint_path=run_dir / "codesign.ckpt",
    )

    print(f"\nbest reward {best:.2f}")
    print(f"  policy    {policy}")
    print(f"  generator {generator}")
    best_body = generator.generate(1, deterministic=True)[0]
    print(f"  best body {best_body}")


if __name__ == "__main__":
    main()
