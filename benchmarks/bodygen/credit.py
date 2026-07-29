"""BodyGen's enhanced temporal credit assignment (Enhanced-TCA).

The implementation follows ``BodyGenAgent.estimate_advantages`` from the
pinned camera-ready source.  Control transitions use GAE.  Both design stages
instead regress to the undiscounted return remaining in their complete
episode, which carries the eventual execution reward back through all six
simulator-free design decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .design import ATTRIBUTE, CONTROL, TOPOLOGY


class ReturnNormalizer(nn.Module):
    """Serializable running return scale used separately by design/control.

    BodyGen's camera-ready defaults use scaling without demeaning. ``demean``
    stays explicit so the two upstream normalization equations are auditable;
    the faithful training config always constructs both instances with
    ``demean=False``.
    """

    def __init__(
        self,
        *,
        demean: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.demean = bool(demean)
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros((), dtype=dtype))
        self.register_buffer("variance", torch.ones((), dtype=dtype))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1).to(
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        if values.numel() == 0:
            return
        batch_variance, batch_mean = torch.var_mean(
            values, unbiased=False
        )
        batch_count = values.numel()
        old_count = self.count.to(self.mean.dtype)
        total_count = old_count + batch_count
        old_weight = old_count / total_count
        batch_weight = 1.0 - old_weight
        delta = batch_mean - self.mean
        self.variance.copy_(
            old_weight * self.variance
            + batch_weight * batch_variance
            + old_weight * batch_weight * delta.square()
        )
        self.mean.copy_(old_weight * self.mean + batch_weight * batch_mean)
        self.count.add_(batch_count)

    def normalize(
        self,
        values: torch.Tensor,
        *,
        update: bool = False,
    ) -> torch.Tensor:
        if update:
            self.update(values)
        if not bool(self.count):
            return values
        result = values
        if self.demean:
            result = result - self.mean.to(result)
        return result / (torch.sqrt(self.variance).to(result) + 1.0e-8)

    def unscale(self, values: torch.Tensor) -> torch.Tensor:
        result = values * (torch.sqrt(self.variance).to(values) + 1.0e-8)
        if self.demean:
            result = result + self.mean.to(values)
        return result


@dataclass(frozen=True)
class CreditAssignment:
    """PPO advantages and value targets plus their auditable raw returns."""

    advantages: torch.Tensor
    returns: torch.Tensor
    design_returns: torch.Tensor
    control_returns: torch.Tensor


def _vector(name: str, value: torch.Tensor, length: int | None = None) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch tensor")
    result = value.reshape(-1)
    if length is not None and result.numel() != length:
        raise ValueError(f"{name} must contain {length} transitions")
    return result


def _normalise_group(
    advantages: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    selected = advantages[mask]
    if selected.numel() == 0:
        return
    # Upstream uses torch.std's sample standard deviation.  A one-transition
    # defensive case cannot estimate that scale, so it is only demeaned.
    centred = selected - selected.mean()
    if selected.numel() > 1:
        # Match ``BodyGenAgent.estimate_advantages`` exactly. Unlike the
        # running normalizers, upstream adds no epsilon to this standard
        # deviation.
        centred = centred / selected.std(unbiased=True)
    advantages[mask] = centred


def enhanced_temporal_credit_assignment(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    stages: torch.Tensor,
    *,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
    design_normalizer: ReturnNormalizer | None = None,
    control_normalizer: ReturnNormalizer | None = None,
    update_normalizers: bool = True,
) -> CreditAssignment:
    """Compute camera-ready Enhanced-TCA over complete, contiguous episodes.

    ``values`` and ``next_values`` must be in raw reward units.  Termination
    prevents value bootstrapping; truncation keeps the supplied bootstrap value
    but ends the GAE recursion.  Design returns are deliberately undiscounted
    and stop at either kind of episode boundary.

    The returned value targets are normalized separately when the two
    normalizers are supplied.  Advantages always begin in raw reward units and
    are standardized independently for design and control transitions.
    """

    rewards = _vector("rewards", rewards)
    count = rewards.numel()
    values = _vector("values", values, count)
    next_values = _vector("next_values", next_values, count)
    terminated = _vector("terminated", terminated, count).bool()
    truncated = _vector("truncated", truncated, count).bool()
    stages = _vector("stages", stages, count).long()
    if count == 0:
        raise ValueError("Enhanced-TCA needs at least one transition")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    if torch.any((stages < 0) | (stages > CONTROL)):
        raise ValueError("stage ids must be topology, attribute or control")

    done = terminated | truncated
    advantages = torch.zeros_like(rewards)
    design_returns = torch.zeros_like(rewards)
    next_advantage = rewards.new_zeros(())
    next_design_return = rewards.new_zeros(())

    for index in range(count - 1, -1, -1):
        bootstrap = (~terminated[index]).to(rewards.dtype)
        continuation = (~done[index]).to(rewards.dtype)
        delta = (
            rewards[index]
            + gamma * next_values[index] * bootstrap
            - values[index]
        )
        next_advantage = (
            delta
            + gamma * gae_lambda * next_advantage * continuation
        )
        advantages[index] = next_advantage
        next_design_return = (
            rewards[index] + next_design_return * continuation
        )
        design_returns[index] = next_design_return

    control_returns = values + advantages
    returns = control_returns.clone()
    design_mask = stages != CONTROL
    control_mask = ~design_mask
    raw_design_advantages = design_returns - values
    advantages[design_mask] = raw_design_advantages[design_mask]
    returns[design_mask] = design_returns[design_mask]

    if design_normalizer is not None:
        returns[design_mask] = design_normalizer.normalize(
            design_returns[design_mask],
            update=update_normalizers,
        )
    if control_normalizer is not None:
        returns[control_mask] = control_normalizer.normalize(
            control_returns[control_mask],
            update=update_normalizers,
        )

    if normalize_advantages:
        _normalise_group(advantages, design_mask)
        _normalise_group(advantages, control_mask)

    return CreditAssignment(
        advantages=advantages,
        returns=returns,
        design_returns=design_returns,
        control_returns=control_returns,
    )
