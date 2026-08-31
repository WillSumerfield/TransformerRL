"""The Simple-ES primitives shared by the designer and the controller.

Both optimizers are the same object: a moving point `mu` that samples around itself, optionally
improves each sample by climbing the hill it landed on, evaluates, and steps toward a rank-weighted
recombination of the better half. They differ only in which axis they move along, what distance
attenuates their climb, and how often they update.

Everything here is shape-agnostic and batched over a leading run dimension, so an entire sweep
(all configs x all seeds) advances as one tensor program. See ADR-0023 for why the three
configuration axes are defined this way.
"""

from __future__ import annotations

import torch

from landscape import to_coord, to_index


def rank_weights(m: int, device="cpu") -> torch.Tensor:
    """CMA-style log weights over `m` elites, descending, summing to 1."""
    w = torch.log(torch.tensor(m + 0.5, device=device)) - torch.log(
        torch.arange(1, m + 1, device=device, dtype=torch.float32)
    )
    return w / w.sum()


def recombine(mu: torch.Tensor, x: torch.Tensor, v: torch.Tensor, alpha: float, top_percent: float) -> torch.Tensor:
    """Step `mu` toward the rank-weighted mean of the better half of `x`.

    `x` and `v` are (..., P); higher `v` is better. The result stays inside the convex hull of the
    samples, so the update cannot diverge however wide the spread. `alpha` is a global constant --
    step size is deliberately left coupled to spread (ADR-0023).
    """
    m = max(int(x.shape[-1] * top_percent), 1)
    # topk, not a full argsort: the controller keeps 40 of 4096, and sorting the rest is both
    # slower and a 4x larger int64 allocation, which costs chunks (see `sweep.chunk_size`).
    elite = torch.topk(v, m, dim=-1).indices
    target = (torch.gather(x, -1, elite) * rank_weights(m, x.device)).sum(-1)
    return mu + alpha * (target - mu)


@torch.compile(dynamic=True)
def climb(
    x: torch.Tensor,
    target: torch.Tensor,
    dist2: torch.Tensor,
    g: torch.Tensor,
    e: torch.Tensor,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """Partially climb each sample up its hill.

    `target` is the local peak, `dist2` the squared distance from the optimizer's centre in the
    joint space, `g` the generalization radius, `e` the probability of leaving a sample raw.

    g -> inf lands exactly on target, g -> 0 never moves; e = 1 disables the climb entirely, which
    is why exploration gates generalization rather than sitting beside it.

    Fused: like `landscape.f` this is elementwise over the sample cloud and bandwidth-bound, and
    `dynamic=True` absorbs the ragged final chunk without a recompile.
    """
    w = torch.exp(-0.5 * dist2 / g.clamp_min(1e-8) ** 2)
    moved = x + w * (target - x)
    raw = torch.rand(x.shape, device=x.device, generator=gen) < e
    return torch.where(raw, x, moved)


def designer_propose(
    mu_d: torch.Tensor,
    mu_a: torch.Tensor,
    sig_d: torch.Tensor,
    e_d: torch.Tensor,
    g_d: torch.Tensor,
    T_d: torch.Tensor,
    n: int,
    P_d: int,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample `P_d` designs and climb them on the slice through the controller's mean action.

    The designer is unconditional, so its samples sit at (d, mu_a) and its centre at (mu_d, mu_a):
    the action term of the joint distance vanishes by construction. It climbs `f(., mu_a)` -- the
    controller's *current* competence, one iteration stale -- not the oracle marginal, so its view
    of design quality stays mediated by the controller (ADR-0023).
    """
    mu_d, mu_a = mu_d.unsqueeze(-1), mu_a.unsqueeze(-1)
    d = torch.normal(mu_d.expand(-1, P_d), sig_d.unsqueeze(-1).expand(-1, P_d), generator=gen)

    tgt = to_coord(T_d[to_index(d, n), to_index(mu_a, n).expand_as(d)], n)
    return climb(d, tgt, (d - mu_d) ** 2, g_d.unsqueeze(-1), e_d.unsqueeze(-1), gen)


def controller_act(
    d: torch.Tensor,
    mu_a: torch.Tensor,
    mu_d_prev: torch.Tensor,
    sig_c: torch.Tensor,
    e_c: torch.Tensor,
    g_c: torch.Tensor,
    T_a: torch.Tensor,
    n: int,
    P_a: int,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample `P_a` actions for each of the given designs and climb them.

    Returns (R, P_d, P_a). The sampled action still decides *which* hill is climbed, so
    controller spread keeps its job; only the strength of the climb is attenuated, by joint
    distance from (mu_d_prev, mu_a). That is what makes designer spread compete with controller
    generalization: designs far from what the controller has recently seen get played badly.
    """
    d = d.unsqueeze(-1)
    mu_a = mu_a.unsqueeze(-1).unsqueeze(-1)
    shape = (d.shape[0], d.shape[1], P_a)

    a = torch.normal(mu_a.expand(shape), sig_c.view(-1, 1, 1).expand(shape), generator=gen)

    tgt = to_coord(T_a[to_index(d, n).expand_as(a), to_index(a, n)], n)
    dist2 = (d - mu_d_prev.view(-1, 1, 1)) ** 2 + (a - mu_a) ** 2
    return climb(a, tgt, dist2, g_c.view(-1, 1, 1), e_c.view(-1, 1, 1), gen)
