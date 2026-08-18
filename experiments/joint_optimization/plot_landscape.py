"""3D surfaces of the joint-optimization landscapes: raw f, the clamped L1 (dead zone grey), and
the composed L2. Use it to tune c against a new f -- grey is the part of the domain where the
fidelity gate does nothing.

Plotly/WebGL rather than mplot3d, which repaints every quad on the CPU and rotates at ~5 fps.

  python experiments/joint_optimization/plot_landscape.py [--c 0.25] [--n 201] [--save fig.html]
"""

from __future__ import annotations

import argparse

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

try:  # runs both as `python -m experiments.joint_optimization.plot_landscape` and as a plain file
    from .env import BOUNDS, DEFAULT_C, f, g, grid
except ImportError:
    from env import BOUNDS, DEFAULT_C, f, g, grid

DEAD_GREY = [[0.0, "#a6a6a6"], [1.0, "#a6a6a6"]]


def _surface(fig, x, y, z, col, *, dead=None):
    """One surface panel. `dead` splits it into a viridis trace and a flat grey trace, each
    NaN-masked to its own region, so the dead zone reads as inert rather than as 'low'."""
    if dead is None:
        fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale="Viridis", showscale=False), 1, col)
        return
    nan = torch.tensor(float("nan"))
    fig.add_trace(
        go.Surface(x=x, y=y, z=torch.where(dead, nan, z), colorscale="Viridis", showscale=False),
        1,
        col,
    )
    fig.add_trace(
        go.Surface(x=x, y=y, z=torch.where(dead, z, nan), colorscale=DEAD_GREY, showscale=False),
        1,
        col,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c", type=float, default=DEFAULT_C, help="clamp offset: L1 = min(f - c, 0)")
    p.add_argument("--n", type=int, default=201, help="grid resolution per axis")
    p.add_argument("--save", type=str, default=None, help="write .html (or .png, needs kaleido)")
    args = p.parse_args()

    with torch.no_grad():
        xy = grid(args.n)
        f_raw = f(xy)
        l1 = torch.clamp(f_raw - args.c, max=0.0)
        l2 = g(xy, l1)
    axis = torch.linspace(*BOUNDS, args.n)
    dead = f_raw >= args.c

    titles = ("raw f", f"L1 = min(f - {args.c}, 0)", "L2 = g(x, y, L1)")
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "surface"}] * 3], subplot_titles=titles)
    _surface(fig, axis, axis, f_raw, 1)
    _surface(fig, axis, axis, l1, 2, dead=dead)
    _surface(fig, axis, axis, l2, 3)

    scene = dict(xaxis_title="x", yaxis_title="y", aspectmode="cube")
    fig.update_layout(
        title=f"grey = dead zone, {100 * dead.float().mean():.1f}% of domain (L1 = 0, gate inert)",
        scene=scene,
        scene2=scene,
        scene3=scene,
        height=620,
        margin=dict(l=10, r=10, t=90, b=10),
    )

    if args.save:
        if args.save.endswith(".png"):
            fig.write_image(args.save, width=1800, height=620, scale=2)
        else:
            fig.write_html(args.save)
        print(f"wrote {args.save}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
