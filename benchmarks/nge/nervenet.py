"""PyTorch NerveNet++ controller used by the VSim NGE port.

The policy follows the upstream data flow: per-node observation and structural
embeddings, mean-aggregated graph messages, a recurrent GRU node state carried
between environment steps, and one shared Gaussian action head for controlling
nodes.  PPO unrolls this recurrence in short sequences (truncated BPTT).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .graph import NGEGraph, NODE_ATTRIBUTE_SIZE


# VSim observation offsets.  They are repeated here deliberately so importing
# and unit-testing NerveNet does not load VSim's native shared libraries.
N_ACTIONS = 32
OBSERVATION_SIZE = 893
_O_SIN = 13
_O_COS = 45
_O_VEL = 77
_O_ACT = 109
_O_RELPOS = 141
_O_RELROT = 237
_O_RELVEL = 429
_O_SENSOR = 621
_O_LENGTH = 669
_O_MASK = 701
PHYSICAL_OBSERVATION_SIZE = 669
NODE_OBSERVATION_SIZE = 28


def _orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class RunningMeanStd:
    """Small serialisable observation normaliser, inherited with a policy."""

    def __init__(self, size: int, device: torch.device) -> None:
        self.mean = torch.zeros(size, dtype=torch.float32, device=device)
        self.var = torch.ones(size, dtype=torch.float32, device=device)
        self.count = 1.0e-4

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.reshape(-1, values.shape[-1]).float()
        if values.numel() == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = float(values.shape[0])
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        first = self.var * self.count
        second = batch_var * batch_count
        correction = delta.square() * self.count * batch_count / total
        self.mean = new_mean
        self.var = (first + second + correction) / total
        self.count = total

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        normalized = (values - self.mean) / torch.sqrt(self.var + 1.0e-8)
        return normalized.clamp(-10.0, 10.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean.copy_(state["mean"].to(self.mean.device))
        self.var.copy_(state["var"].to(self.var.device))
        self.count = float(state["count"])

    def clone(self) -> "RunningMeanStd":
        result = RunningMeanStd(self.mean.numel(), self.mean.device)
        result.load_state_dict(self.state_dict())
        return result


def normalize_observation(
    observation: torch.Tensor,
    normalizer: RunningMeanStd,
) -> torch.Tensor:
    """Normalize physical channels while retaining exact typed-grammar bits."""
    physical = normalizer.normalize(observation[..., :PHYSICAL_OBSERVATION_SIZE])
    return torch.cat(
        (physical, observation[..., PHYSICAL_OBSERVATION_SIZE:]),
        dim=-1,
    )


def node_observations(
    observation: torch.Tensor,
    graph: NGEGraph,
) -> torch.Tensor:
    """Split the padded VSim vector into root/module observations.

    Returns ``[batch, graph_nodes, 28]`` in the same node order as
    :meth:`NGEGraph.node_attributes`.
    """
    if observation.ndim != 2 or observation.shape[-1] != OBSERVATION_SIZE:
        raise ValueError(
            f"observation must have shape (batch, {OBSERVATION_SIZE})"
        )
    batch = observation.shape[0]
    result = observation.new_zeros((batch, graph.num_nodes, NODE_OBSERVATION_SIZE))
    result[:, 0, :13] = observation[:, :13]

    row = 1
    for limb, chain in enumerate(graph.effectors):
        for depth in range(len(chain)):
            slot = depth * 8 + limb
            result[:, row, 0] = observation[:, _O_SIN + slot]
            result[:, row, 1] = observation[:, _O_COS + slot]
            result[:, row, 2] = observation[:, _O_VEL + slot]
            result[:, row, 3] = observation[:, _O_ACT + slot]
            result[:, row, 4:7] = observation[
                :, _O_RELPOS + 3 * slot : _O_RELPOS + 3 * (slot + 1)
            ]
            result[:, row, 7:13] = observation[
                :, _O_RELROT + 6 * slot : _O_RELROT + 6 * (slot + 1)
            ]
            result[:, row, 13:19] = observation[
                :, _O_RELVEL + 6 * slot : _O_RELVEL + 6 * (slot + 1)
            ]
            if depth == len(chain) - 1:
                result[:, row, 19:25] = observation[
                    :, _O_SENSOR + 6 * limb : _O_SENSOR + 6 * (limb + 1)
                ]
                result[:, row, 27] = 1.0
            result[:, row, 25] = observation[:, _O_LENGTH + slot]
            result[:, row, 26] = observation[:, _O_MASK + slot]
            row += 1
    return result


class NerveNetPlusPlus(nn.Module):
    """Graph-independent recurrent Gaussian controller."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        if hidden_size % 2:
            raise ValueError("hidden_size must be even")
        half = hidden_size // 2
        self.hidden_size = hidden_size
        self.root_observation = nn.Linear(NODE_OBSERVATION_SIZE, half)
        self.body_observation = nn.Linear(NODE_OBSERVATION_SIZE, half)
        self.attribute_embedding = nn.Linear(NODE_ATTRIBUTE_SIZE, half)
        self.message = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.root_update = nn.GRUCell(2 * hidden_size, hidden_size)
        self.body_update = nn.GRUCell(2 * hidden_size, hidden_size)
        self.action_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        # The upstream implementation learns a state-independent scale per
        # controller output.  VSim's fixed padded action layout makes all 32
        # scales graph-independent, so parent weights copy without remapping.
        self.log_std = nn.Parameter(torch.zeros(N_ACTIONS))
        self.apply(_orthogonal_init)
        _orthogonal_init(self.action_mean[-1], gain=0.01)

    def initial_hidden(
        self,
        batch_size: int,
        graph: NGEGraph,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            graph.num_nodes,
            self.hidden_size,
            device=device,
        )

    def _input_features(
        self,
        observation: torch.Tensor,
        graph: NGEGraph,
    ) -> torch.Tensor:
        node_obs = node_observations(observation, graph)
        root = torch.tanh(self.root_observation(node_obs[:, :1]))
        if graph.num_nodes > 1:
            body = torch.tanh(self.body_observation(node_obs[:, 1:]))
            obs_embedding = torch.cat((root, body), dim=1)
        else:  # Defensive; valid graphs always contain an actuator.
            obs_embedding = root
        attributes = torch.as_tensor(
            graph.node_attributes(),
            dtype=observation.dtype,
            device=observation.device,
        )
        attr_embedding = torch.tanh(self.attribute_embedding(attributes))
        attr_embedding = attr_embedding.unsqueeze(0).expand(observation.shape[0], -1, -1)
        return torch.cat((obs_embedding, attr_embedding), dim=-1)

    def forward_step(
        self,
        observation: torch.Tensor,
        graph: NGEGraph,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one environment timestep.

        Returns padded action means/log-standard-deviations and the new node
        state.  Only graph actuator slots contain policy outputs.
        """
        features = self._input_features(observation, graph)
        if hidden.shape != (
            observation.shape[0],
            graph.num_nodes,
            self.hidden_size,
        ):
            raise ValueError("hidden state shape does not match batch and graph")

        edges = graph.edges()
        source = torch.as_tensor(
            [edge[0] for edge in edges],
            dtype=torch.long,
            device=observation.device,
        )
        target = torch.as_tensor(
            [edge[1] for edge in edges],
            dtype=torch.long,
            device=observation.device,
        )
        sent = self.message(features[:, source])
        messages = torch.zeros_like(features)
        messages.index_add_(1, target, sent)
        degrees = torch.zeros(
            graph.num_nodes,
            dtype=observation.dtype,
            device=observation.device,
        )
        degrees.index_add_(0, target, torch.ones_like(target, dtype=observation.dtype))
        messages = messages / degrees.clamp_min(1.0).view(1, -1, 1)
        update_input = torch.cat((messages, features), dim=-1)

        root_state = self.root_update(
            update_input[:, 0].reshape(-1, 2 * self.hidden_size),
            hidden[:, 0].reshape(-1, self.hidden_size),
        ).view(observation.shape[0], 1, self.hidden_size)
        body_state = self.body_update(
            update_input[:, 1:].reshape(-1, 2 * self.hidden_size),
            hidden[:, 1:].reshape(-1, self.hidden_size),
        ).view(observation.shape[0], graph.num_nodes - 1, self.hidden_size)
        new_hidden = torch.cat((root_state, body_state), dim=1)

        node_means = self.action_mean(body_state).squeeze(-1)
        mean = observation.new_zeros((observation.shape[0], N_ACTIONS))
        slots = torch.as_tensor(
            graph.action_slots(), dtype=torch.long, device=observation.device
        )
        mean[:, slots] = node_means
        log_std = self.log_std.clamp(-5.0, 2.0).unsqueeze(0).expand_as(mean)
        return mean, log_std, new_hidden

    def forward_sequence(
        self,
        observations: torch.Tensor,
        graph: NGEGraph,
        hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unroll a ``[time, batch, obs]`` sequence for truncated BPTT."""
        means, log_stds = [], []
        for time in range(observations.shape[0]):
            keep = (~episode_starts[time]).to(hidden.dtype).view(-1, 1, 1)
            hidden = hidden * keep
            mean, log_std, hidden = self.forward_step(
                observations[time], graph, hidden
            )
            means.append(mean)
            log_stds.append(log_std)
        return torch.stack(means), torch.stack(log_stds), hidden


class ValueNetwork(nn.Module):
    """The separate two-layer fully connected baseline used by upstream NGE."""

    def __init__(self, observation_size: int = OBSERVATION_SIZE) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.apply(_orthogonal_init)
        _orthogonal_init(self.network[-1], gain=1.0)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


def action_mask(
    graph: NGEGraph,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    mask = torch.zeros(N_ACTIONS, dtype=dtype, device=device)
    mask[list(graph.action_slots())] = 1.0
    return mask


def gaussian_log_prob(
    action: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Log probability summed only over the graph's active controllers."""
    variance = torch.exp(2.0 * log_std)
    per_action = -0.5 * (
        (action - mean).square() / variance
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )
    return (per_action * mask).sum(dim=-1)


def gaussian_entropy(log_std: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_action = log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))
    return (per_action * mask).sum(dim=-1)


@dataclass
class ControllerState:
    """Networks and normalisation state owned by one species."""

    policy: NerveNetPlusPlus
    value: ValueNetwork
    normalizer: RunningMeanStd

    @classmethod
    def create(
        cls,
        device: torch.device,
        *,
        hidden_size: int = 64,
    ) -> "ControllerState":
        return cls(
            NerveNetPlusPlus(hidden_size).to(device),
            ValueNetwork().to(device),
            RunningMeanStd(PHYSICAL_OBSERVATION_SIZE, device),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        self.normalizer.load_state_dict(state["normalizer"])

    def inherited_copy(self) -> "ControllerState":
        result = self.create(
            self.normalizer.mean.device,
            hidden_size=self.policy.hidden_size,
        )
        result.load_state_dict(self.state_dict())
        return result
