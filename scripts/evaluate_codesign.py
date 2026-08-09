"""Score a codesign checkpoint on a named set of bodies (`codesigner.evaluate`).

Out-of-process and unrelated to training: this reconstructs the module library from the
checkpoint's own provenance, rebuilds the network, and drives the Task through `ControlPolicy.act`.

Which bodies to score is an explicit choice (D27), not something carried over from the run:

    --bodies generator   the checkpoint's own generator, deterministic (its committed design)
    --bodies sampled     a draw from that generator, so a distribution is scored, not a point
    --bodies stable      the fixed suite of stable topologies, for comparability across runs
    --bodies seed        the seed body alone

    python scripts/evaluate_codesign.py runs/.../codesign.ckpt --bodies stable
"""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from argparse import ArgumentParser
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from codesigner import checkpoint as _checkpoint, evaluate

from transformer_rl.algorithm import CodesignAlgorithm
from transformer_rl.morphology import seed_body, stable_slot_sets


def main():
    ap = ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--config", required=True,
                    help="run config (names the task); path or a name under configs/")
    ap.add_argument("--bodies", default="generator",
                    choices=["generator", "sampled", "stable", "seed"])
    ap.add_argument("--num-bodies", type=int, default=8, dest="num_bodies",
                    help="how many to draw, for --bodies sampled")
    ap.add_argument("--envs-per-body", type=int, default=64, dest="envs_per_body")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = args.config if Path(args.config).is_absolute() else _ROOT / "configs" / args.config
    algorithm = CodesignAlgorithm(config)
    task = algorithm.make_task()
    # The library comes from the CHECKPOINT, not from the config -- the config names whichever
    # library its next training run would use, which need not be the one this checkpoint was trained
    # against. Building the config's library here instead just moves the failure to `evaluate`'s
    # provenance check, which is right to refuse it.
    # The task IS the config's, though, and is checked here so a checkpoint from another task is
    # refused before any scene is built rather than after the rollout has been set up.
    library = _checkpoint.load(args.checkpoint, map_location="cpu",
                               task=algorithm.task_key)["library"]
    print(f"module library from checkpoint: {library.name}")

    if args.bodies == "stable":
        bodies = [seed_body(library, slots=sorted(s)) for s in stable_slot_sets(library.n_slots)]
    elif args.bodies == "seed":
        bodies = [seed_body(library)]
    else:
        # The generator has to be loaded before it can be drawn from, and evaluate loads the
        # checkpoint itself, so these body sources read the payload twice. Cheap next to the rollout.
        algorithm.assign_task_and_modlib(task, library)
        algorithm.load_checkpoint(args.checkpoint, map_location="cpu")
        generator = algorithm.morphology_generator()
        bodies = (generator.generate(1, deterministic=True) if args.bodies == "generator"
                  else generator.generate(args.num_bodies, deterministic=False))

    scores = evaluate(args.checkpoint, task, algorithm, bodies,
                      envs_per_body=args.envs_per_body, seed=args.seed)

    print(f"\n{args.bodies}: {len(bodies)} bodies x {args.envs_per_body} envs")
    for body, score in sorted(zip(bodies, scores), key=lambda p: -p[1]):
        print(f"  {score:9.2f}  {body}")
    print(f"  mean {sum(scores) / len(scores):.2f}")


if __name__ == "__main__":
    main()
