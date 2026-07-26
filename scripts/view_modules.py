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
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import torch
import vlearn as v

from envs.ant_envs.ant_multimorph import AntMultiMorphEnv, _N_LIMBS, _N_DOFS_FULL, _slot
from envs.ant_envs.build_vsim import Morphology, _CAP_PITCH
from transformer_rl.vocab import (EFF_SWING, EFF_KNEE, EFF_TWIST, EFF_NAMES,
                                  CAP_BARE, CAP_FOOT, CAP_PAD, CAP_BALL, CAP_NAMES)

_LIMBS = range(1, _N_LIMBS + 1)


def _uniform(chain, cap) -> Morphology:
    """Body where every limb carries the same effector chain and the same cap."""
    return Morphology.from_design({n: list(chain) for n in _LIMBS}, {n: cap for n in _LIMBS})


def gallery() -> list:
    """(label, morphology) per grid cell: 3 effector demos then 4 cap demos."""
    cells = [(f"eff:{EFF_NAMES[e]}", _uniform([e], CAP_BARE))
             for e in (EFF_SWING, EFF_KNEE, EFF_TWIST)]
    cells += [(f"cap:{CAP_NAMES[c]}", _uniform([EFF_SWING, EFF_KNEE], c))
              for c in (CAP_BARE, CAP_FOOT, CAP_PAD, CAP_BALL)]
    return cells


def main():
    p = ArgumentParser()
    p.add_argument("--knee", type=float, default=_CAP_PITCH,
                   help="knee angle (rad) to hold every limb at; defaults to the nominal bend the "
                        "cap pitch is built around, i.e. the pose where caps sit flat. 0 = splayed")
    p.add_argument("--fall", action="store_true", help="enable gravity (default: float in place)")
    p.add_argument("--spacing", type=float, default=3.0)
    args = p.parse_args()

    cells = gallery()
    cols = max(1, ceil(len(cells) ** 0.5))
    print(f"[gallery] {len(cells)} cells, {cols} per row (+x = column, +z = row):")
    for i, (label, m) in enumerate(cells):
        print(f"  [{i // cols},{i % cols}] {label}")

    device = torch.device("cuda:0")
    env = AntMultiMorphEnv(
        len(cells), device,
        morphologies=[m for _, m in cells],
        sample_morphs=False,
        rendering=True,
        with_window=True,
        raise_exception=False,
        gravity=v.Vec3(0, -9.81, 0) if args.fall else v.Vec3(0, 0, 0),
        reset_noise_scale=0.0,          # identical, jitter-free pose in every cell
        healthy_y_range=(-1e9, 1e9),    # never terminate; the gallery is meant to sit still
        max_episode_length=10 ** 9,
        spacing=args.spacing,
        seed=0,
    )

    # Hold the knees at a fixed angle by writing the reset DOF pose (zero gravity + zero torque, so
    # the bodies just sit there). Limbs 4 and 6 have a negated ankle axis, hence a negated angle.
    pose = torch.zeros((env.total_num_envs, _N_DOFS_FULL), device=device)
    for n in _LIMBS:
        pose[:, _slot(n, 2)] = -args.knee if n in (4, 6) else args.knee
    env._flat_dof_init[:] = pose.reshape(-1)[env._motor_src_idx]

    act = torch.zeros_like(pose)
    env.reset()
    while not env.render_finished:
        env.step(act)


if __name__ == "__main__":
    main()
