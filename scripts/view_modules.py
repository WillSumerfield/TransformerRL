"""Static gallery of the Phase-5 module vocabulary: one grid cell per module type, floating so
nothing clips the ground. Purely for eyeballing geometry -- no policy, no training.

Each cell is a full 8-limb body with every limb identical, so neighbour clearance (do the caps
collide with each other?) is visible. Effector cells differ visually only in link length/cylinder
convention -- twist vs knee is a joint-AXIS difference and looks the same standing still.

  uv run python scripts/view_modules.py                 # floating, joints at zero
  uv run python scripts/view_modules.py --knee 0.6      # constant knee torque -> stance pose
  uv run python scripts/view_modules.py --fall          # gravity on; watch them land
"""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from argparse import ArgumentParser
from math import ceil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch
import vlearn as v
from codesigner.components.modular_libraries import SimpleModuleLibrary
from codesigner.components.tasks import Ant
from codesigner.interfaces import ModuleType, Morphology
# Nominal stance knee bend (rad) -- the angle a cap's pitch is built to cancel, i.e. where caps sit
# flat. Matches SimpleModuleLibrary's internal _KNEE_NOMINAL_BEND (dev-tool default only, not read
# from the library -- ModuleLibrary doesn't expose it).
_NOMINAL_KNEE_BEND = 1.134464045365651486


def _uniform(ml, chain, cap) -> Morphology:
    """Body where every slot carries the same effector chain and the same cap."""
    return Morphology.from_names(ml, {s: (tuple(chain), cap) for s in range(ml.n_slots)})


def gallery(ml) -> list:
    """(label, morphology) per grid cell: 3 effector demos then 4 cap demos. "swing"/"knee"/"bare"
    are OUR choice of canonical chain/cap (matches transformer_rl.morphology.CANONICAL_*), not
    something queried from the library."""
    cells = [(f"eff:{e}", _uniform(ml, [e], "bare")) for e in ml.names(ModuleType.EFFECTOR)]
    cells += [(f"cap:{c}", _uniform(ml, ["swing", "knee"], c))
              for c in ml.names(ModuleType.CAP)]
    return cells


def main():
    p = ArgumentParser()
    p.add_argument("--knee", type=float, default=_NOMINAL_KNEE_BEND,
                   help="knee angle (rad) to hold every limb at; defaults to the nominal bend the "
                        "cap pitch is built around, i.e. the pose where caps sit flat. 0 = splayed")
    p.add_argument("--fall", action="store_true", help="enable gravity (default: float in place)")
    p.add_argument("--spacing", type=float, default=3.0)
    args = p.parse_args()

    library = SimpleModuleLibrary()
    cells = gallery(library)
    cols = max(1, ceil(len(cells) ** 0.5))
    print(f"[gallery] {len(cells)} cells, {cols} per row (+x = column, +z = row):")
    for i, (label, m) in enumerate(cells):
        print(f"  [{i // cols},{i % cols}] {label}")

    device = torch.device("cuda:0")
    env = Ant(
        device=device,
        rendering=True,
        with_window=True,
        raise_exception=False,
        enable_scene_query=False,
        rootOffset=(v.Vec3(0, 0, 0), v.Quat(0, 0, 0, 1)),
        gravity=v.Vec3(0, -9.81, 0) if args.fall else v.Vec3(0, 0, 0),
        reset_noise_scale=0.0,          # identical, jitter-free pose in every cell
        healthy_y_range=(-1e9, 1e9),    # never terminate; the gallery is meant to sit still
        max_episode_length=10 ** 9,
        spacing=args.spacing,
    )
    # 1 env per cell, explicit per-group bodies -- the gallery IS the body list.
    env.setup(library, len(cells), len(cells), [m for _, m in cells], seed=0)
    layout = env.obs_layout()

    # Hold the knees at a fixed angle by writing the reset DOF pose (zero gravity + zero torque, so
    # the bodies just sit there).
    pose = torch.zeros((env.total_num_envs, layout["n_modules"]), device=device)
    # depth-major slot order: slot(s, d) = d * n_slots + s, 0-based. Depth 1 == the knee.
    pose[:, layout["n_slots"]:2 * layout["n_slots"]] = args.knee
    env._flat_dof_init[:] = pose.reshape(-1)[env._motor_src_idx]

    act = torch.zeros_like(pose)
    env.reset()
    while not env.render_finished:
        env.step(act)


if __name__ == "__main__":
    main()
