"""MoSAT policy/value networks for the faithful BodyGen baseline.

The camera-ready method uses separate self-attention trunks for the topology,
attribute and control actors and another three for their critics.  Variable
trees are padded within each batch and padding is excluded from attention and
all body-level probabilities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from transformer_rl.vocab import N_CAP, N_EFF

from .design import (
    ADD,
    ATTRIBUTE,
    CONTROL,
    DESIGN_FEATURE_SIZE,
    N_DESIGN_STEPS,
    N_TOPOLOGY_ACTIONS,
    N_TOPOLOGY_WAVES,
    TOPOLOGY,
    TOPOLOGY_EMBEDDINGS,
    BodyGenDesign,
    DesignBatchTrace,
    DesignTrace,
    DesignTransition,
    apply_attribute_actions,
    apply_topology_actions,
    design_node_features,
)


OBSERVATION_SIZE = 893
ACTION_SIZE = 32
PHYSICAL_NODE_FEATURE_SIZE = 28
NODE_FEATURE_SIZE = PHYSICAL_NODE_FEATURE_SIZE + DESIGN_FEATURE_SIZE

# VSim's padded ant observation offsets.  Keeping this torch-only avoids
# importing VSim's native library in core/unit tests.
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


@dataclass(frozen=True)
class NodeBatch:
    """Padded node features and masks for a variable-design batch."""

    features: torch.Tensor
    mask: torch.Tensor
    topology_ids: torch.Tensor
    effector_mask: torch.Tensor
    terminal_mask: torch.Tensor
    action_slots: torch.Tensor


@dataclass(frozen=True)
class ControlOutput:
    """A padded VSim control distribution and its native critic value."""

    mean: torch.Tensor
    log_std: torch.Tensor
    value: torch.Tensor
    action: torch.Tensor
    action_mask: torch.Tensor


class RunningObservationNormalizer(nn.Module):
    """The one observation normalizer shared by all six BodyGen trunks."""

    def __init__(
        self,
        size: int = NODE_FEATURE_SIZE,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros(size, dtype=dtype))
        self.register_buffer("variance", torch.ones(size, dtype=dtype))

    @torch.no_grad()
    def update(self, features: torch.Tensor, mask: torch.Tensor) -> None:
        values = features[mask].detach().to(self.mean)
        if values.numel() == 0:
            return
        batch_variance, batch_mean = torch.var_mean(
            values, dim=0, unbiased=False
        )
        batch_count = values.shape[0]
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

    def forward(
        self,
        features: torch.Tensor,
        *,
        update: bool = False,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if update:
            if mask is None:
                raise ValueError("a padding mask is required when updating")
            self.update(features, mask)
        if not bool(self.count):
            return features
        normalized = (
            (features - self.mean.to(features))
            / (torch.sqrt(self.variance).to(features) + 1.0e-8)
        )
        return normalized.clamp(-10.0, 10.0)


class MaskedSelfAttention(nn.Module):
    """BodyGen's single-head scaled dot-product attention."""

    def __init__(
        self,
        hidden_size: int,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.scale = hidden_size**-0.5

    def forward(
        self,
        hidden: torch.Tensor,
        valid_nodes: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(
            ~valid_nodes[:, None, :],
            -torch.inf,
        )
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)


class MoSATBlock(nn.Module):
    """One camera-ready Pre-LN/Post-LN MoSAT block."""

    def __init__(
        self,
        hidden_size: int,
        *,
        layer_norm: str = "pre",
        feed_forward_ratio: int = 4,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if layer_norm not in {"none", "pre", "post"}:
            raise ValueError("layer_norm must be none, pre or post")
        if feed_forward_ratio < 1:
            raise ValueError("feed_forward_ratio must be positive")
        self.layer_norm = layer_norm
        self.attention = MaskedSelfAttention(hidden_size, dtype=dtype)
        self.norm1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.feed_forward = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size * feed_forward_ratio,
                dtype=dtype,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size * feed_forward_ratio,
                hidden_size,
                dtype=dtype,
            ),
        )
        self.norm2 = nn.LayerNorm(hidden_size, dtype=dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        valid_nodes: torch.Tensor,
    ) -> torch.Tensor:
        if self.layer_norm == "pre":
            hidden = hidden + self.attention(self.norm1(hidden), valid_nodes)
            return hidden + self.feed_forward(self.norm2(hidden))
        if self.layer_norm == "post":
            hidden = self.norm1(
                hidden + self.attention(hidden, valid_nodes)
            )
            return self.norm2(hidden + self.feed_forward(hidden))
        hidden = hidden + self.attention(hidden, valid_nodes)
        return hidden + self.feed_forward(hidden)


class MoSAT(nn.Module):
    """Padded morphology self-attention transformer with TopoPE."""

    def __init__(
        self,
        input_size: int = NODE_FEATURE_SIZE,
        *,
        hidden_size: int = 64,
        num_blocks: int = 3,
        layer_norm: str = "pre",
        topology_embeddings: int = TOPOLOGY_EMBEDDINGS,
        feed_forward_ratio: int = 4,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_blocks < 1:
            raise ValueError("hidden_size and num_blocks must be positive")
        if topology_embeddings < TOPOLOGY_EMBEDDINGS:
            raise ValueError("TopoPE needs at least 256 embeddings")
        self.input = nn.Linear(input_size, hidden_size, dtype=dtype)
        self.topology_embedding = nn.Embedding(
            topology_embeddings,
            hidden_size,
            dtype=dtype,
        )
        self.blocks = nn.ModuleList(
            MoSATBlock(
                hidden_size,
                layer_norm=layer_norm,
                feed_forward_ratio=feed_forward_ratio,
                dtype=dtype,
            )
            for _ in range(num_blocks)
        )
        self.to(dtype=dtype)

    def forward(
        self,
        features: torch.Tensor,
        valid_nodes: torch.Tensor,
        topology_ids: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, nodes, channels]")
        if valid_nodes.shape != features.shape[:2]:
            raise ValueError("valid_nodes does not match padded features")
        if topology_ids.shape != features.shape[:2]:
            raise ValueError("topology_ids does not match padded features")
        if not torch.all(valid_nodes.any(dim=1)):
            raise ValueError("every body needs at least its root node")
        hidden = self.input(features)
        hidden = hidden + self.topology_embedding(topology_ids)
        for block in self.blocks:
            hidden = block(hidden, valid_nodes)
        return hidden * valid_nodes.unsqueeze(-1)


class TopologyActor(nn.Module):
    def __init__(self, **trunk_options: object) -> None:
        super().__init__()
        self.trunk = MoSAT(**trunk_options)
        self.logits = nn.Linear(
            self.trunk.input.out_features,
            N_TOPOLOGY_ACTIONS,
            dtype=self.trunk.input.weight.dtype,
        )

    def forward(self, batch: NodeBatch) -> torch.Tensor:
        hidden = self.trunk(
            batch.features, batch.mask, batch.topology_ids
        )
        return self.logits(hidden)


class AttributeActor(nn.Module):
    def __init__(self, **trunk_options: object) -> None:
        super().__init__()
        self.trunk = MoSAT(**trunk_options)
        hidden_size = self.trunk.input.out_features
        dtype = self.trunk.input.weight.dtype
        self.effector_logits = nn.Linear(hidden_size, N_EFF, dtype=dtype)
        self.cap_logits = nn.Linear(hidden_size, N_CAP, dtype=dtype)
        for head in (self.effector_logits, self.cap_logits):
            head.weight.data.mul_(0.1)
            head.bias.data.zero_()

    def forward(
        self,
        batch: NodeBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(
            batch.features, batch.mask, batch.topology_ids
        )
        return self.effector_logits(hidden), self.cap_logits(hidden)


class ControlActor(nn.Module):
    def __init__(
        self,
        *,
        action_size: int = ACTION_SIZE,
        initial_log_std: float = -0.5,
        **trunk_options: object,
    ) -> None:
        super().__init__()
        self.action_size = int(action_size)
        self.trunk = MoSAT(**trunk_options)
        self.action_mean = nn.Linear(
            self.trunk.input.out_features,
            1,
            dtype=self.trunk.input.weight.dtype,
        )
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.zero_()
        dtype = self.action_mean.weight.dtype
        self.log_std = nn.Parameter(
            torch.full((1,), initial_log_std, dtype=dtype)
        )

    def forward(
        self,
        batch: NodeBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(
            batch.features, batch.mask, batch.topology_ids
        )
        node_means = self.action_mean(hidden).squeeze(-1)
        mean = hidden.new_zeros((hidden.shape[0], self.action_size))
        action_mask = torch.zeros(
            hidden.shape[0],
            self.action_size,
            dtype=torch.bool,
            device=hidden.device,
        )
        for row in range(hidden.shape[0]):
            active_nodes = batch.effector_mask[row]
            slots = batch.action_slots[row, active_nodes]
            if torch.any(slots < 0) or torch.any(slots >= self.action_size):
                raise ValueError("design contains a slot outside the action vector")
            mean[row, slots] = node_means[row, active_nodes]
            action_mask[row, slots] = True
        # The pinned BodyGen policy learns this scalar without clipping it.
        # Clamping here would silently change both the distribution and its
        # gradient once long training runs crossed either bound.
        log_std = self.log_std.expand_as(mean)
        return mean, log_std, action_mask


class MoSATCritic(nn.Module):
    """A stage-native MoSAT followed by upstream's root-value MLP."""

    def __init__(
        self,
        *,
        critic_hidden: Sequence[int] = (512, 256),
        **trunk_options: object,
    ) -> None:
        super().__init__()
        self.trunk = MoSAT(**trunk_options)
        widths = (
            self.trunk.input.out_features,
            *(int(width) for width in critic_hidden),
            1,
        )
        if any(width < 1 for width in widths):
            raise ValueError("critic widths must be positive")
        dtype = self.trunk.input.weight.dtype
        layers: list[nn.Module] = []
        for input_width, output_width in zip(widths[:-2], widths[1:-1]):
            layers.extend(
                (
                    nn.Linear(input_width, output_width, dtype=dtype),
                    nn.Tanh(),
                )
            )
        layers.append(nn.Linear(widths[-2], widths[-1], dtype=dtype))
        self.value_head = nn.Sequential(*layers)
        final = self.value_head[-1]
        assert isinstance(final, nn.Linear)
        final.weight.data.mul_(0.1)
        final.bias.data.zero_()

    def forward(self, batch: NodeBatch) -> torch.Tensor:
        hidden = self.trunk(
            batch.features, batch.mask, batch.topology_ids
        )
        # BodyGen's value is the transformed root-node value.
        return self.value_head(hidden[:, 0]).squeeze(-1)


def _physical_node_features(
    observation: torch.Tensor,
    design: BodyGenDesign,
) -> torch.Tensor:
    if observation.ndim != 1 or observation.numel() != OBSERVATION_SIZE:
        raise ValueError(
            f"each observation must contain {OBSERVATION_SIZE} values"
        )
    result = observation.new_zeros(
        (design.num_nodes, PHYSICAL_NODE_FEATURE_SIZE)
    )
    result[0, :13] = observation[:13]
    row = 1
    for limb, chain in enumerate(design.effectors):
        for depth in range(len(chain)):
            slot = depth * 8 + limb
            result[row, 0] = observation[_O_SIN + slot]
            result[row, 1] = observation[_O_COS + slot]
            result[row, 2] = observation[_O_VEL + slot]
            result[row, 3] = observation[_O_ACT + slot]
            result[row, 4:7] = observation[
                _O_RELPOS + 3 * slot : _O_RELPOS + 3 * (slot + 1)
            ]
            result[row, 7:13] = observation[
                _O_RELROT + 6 * slot : _O_RELROT + 6 * (slot + 1)
            ]
            result[row, 13:19] = observation[
                _O_RELVEL + 6 * slot : _O_RELVEL + 6 * (slot + 1)
            ]
            if depth == len(chain) - 1:
                result[row, 19:25] = observation[
                    _O_SENSOR + 6 * limb : _O_SENSOR + 6 * (limb + 1)
                ]
                result[row, 27] = 1.0
            result[row, 25] = observation[_O_LENGTH + slot]
            result[row, 26] = observation[_O_MASK + slot]
            row += 1
    return result


def _node_batch(
    designs: Sequence[BodyGenDesign],
    *,
    observations: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> NodeBatch:
    if not designs:
        raise ValueError("at least one design is required")
    if observations is not None:
        if observations.shape != (len(designs), OBSERVATION_SIZE):
            raise ValueError(
                f"observations must have shape ({len(designs)}, {OBSERVATION_SIZE})"
            )
        observations = observations.to(device=device, dtype=dtype)

    max_nodes = max(design.num_nodes for design in designs)
    features = torch.zeros(
        len(designs),
        max_nodes,
        NODE_FEATURE_SIZE,
        device=device,
        dtype=dtype,
    )
    mask = torch.zeros(
        len(designs), max_nodes, device=device, dtype=torch.bool
    )
    topology_ids = torch.zeros(
        len(designs), max_nodes, device=device, dtype=torch.long
    )
    effector_mask = torch.zeros_like(mask)
    terminal_mask = torch.zeros_like(mask)
    action_slots = torch.full(
        (len(designs), max_nodes),
        -1,
        device=device,
        dtype=torch.long,
    )

    for row, design in enumerate(designs):
        nodes = design.num_nodes
        structural = design_node_features(
            design, device=device, dtype=dtype
        )
        if observations is None:
            physical = torch.zeros(
                nodes,
                PHYSICAL_NODE_FEATURE_SIZE,
                device=device,
                dtype=dtype,
            )
        else:
            physical = _physical_node_features(observations[row], design)
        features[row, :nodes] = torch.cat((physical, structural), dim=-1)
        mask[row, :nodes] = True
        topology_ids[row, :nodes] = torch.as_tensor(
            design.topology_ids(), device=device, dtype=torch.long
        )
        effector_mask[row, 1:nodes] = True
        slots = design.action_slots()
        action_slots[row, 1:nodes] = torch.as_tensor(
            slots, device=device, dtype=torch.long
        )
        for node_index, node in enumerate(design.nodes):
            if (
                not node.is_root
                and node in design.terminal_nodes
            ):
                terminal_mask[row, node_index] = True
    return NodeBatch(
        features=features,
        mask=mask,
        topology_ids=topology_ids,
        effector_mask=effector_mask,
        terminal_mask=terminal_mask,
        action_slots=action_slots,
    )


def _sample_categorical(
    logits: torch.Tensor,
    active: torch.Tensor,
    *,
    deterministic: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    result = torch.zeros(
        logits.shape[:2], dtype=torch.long, device=logits.device
    )
    valid_logits = logits[active]
    if deterministic:
        sampled = valid_logits.argmax(dim=-1)
    else:
        probabilities = torch.softmax(valid_logits, dim=-1)
        sampled = torch.multinomial(
            probabilities, 1, generator=generator
        ).squeeze(-1)
    result[active] = sampled
    return result


def _categorical_statistics(
    logits: torch.Tensor,
    actions: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    node_log_prob = log_probabilities.gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)
    node_entropy = -(probabilities * log_probabilities).sum(dim=-1)
    node_log_prob = node_log_prob * active
    node_entropy = node_entropy * active
    return node_log_prob.sum(dim=-1), node_entropy.sum(dim=-1)


class BodyGenNetworks(nn.Module):
    """All six native BodyGen trunks plus one shared observation normalizer."""

    def __init__(
        self,
        observation_size: int = OBSERVATION_SIZE,
        action_size: int = ACTION_SIZE,
        *,
        hidden_size: int = 64,
        num_blocks: int = 3,
        layer_norm: str = "pre",
        topology_embeddings: int = TOPOLOGY_EMBEDDINGS,
        feed_forward_ratio: int = 4,
        critic_hidden: Sequence[int] = (512, 256),
        initial_control_log_std: float = -0.5,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if observation_size != OBSERVATION_SIZE:
            raise ValueError(
                f"the shared VSim observation has size {OBSERVATION_SIZE}"
            )
        trunk_options = {
            "input_size": NODE_FEATURE_SIZE,
            "hidden_size": hidden_size,
            "num_blocks": num_blocks,
            "layer_norm": layer_norm,
            "topology_embeddings": topology_embeddings,
            "feed_forward_ratio": feed_forward_ratio,
            "dtype": dtype,
        }
        self.observation_size = observation_size
        self.action_size = action_size
        self.normalizer = RunningObservationNormalizer(dtype=dtype)

        # These are intentionally six separate module instances.  Sharing a
        # trunk would no longer be the camera-ready BodyGen architecture.
        self.topology_actor = TopologyActor(**trunk_options)
        self.attribute_actor = AttributeActor(**trunk_options)
        self.control_actor = ControlActor(
            action_size=action_size,
            initial_log_std=initial_control_log_std,
            **trunk_options,
        )
        self.topology_critic = MoSATCritic(
            critic_hidden=critic_hidden, **trunk_options
        )
        self.attribute_critic = MoSATCritic(
            critic_hidden=critic_hidden, **trunk_options
        )
        self.control_critic = MoSATCritic(
            critic_hidden=critic_hidden, **trunk_options
        )
        self.to(dtype=dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _batch(
        self,
        designs: Sequence[BodyGenDesign],
        observations: torch.Tensor | None = None,
    ) -> NodeBatch:
        batch = _node_batch(
            designs,
            observations=observations,
            device=self.device,
            dtype=self.dtype,
        )
        return NodeBatch(
            features=self.normalizer(batch.features),
            mask=batch.mask,
            topology_ids=batch.topology_ids,
            effector_mask=batch.effector_mask,
            terminal_mask=batch.terminal_mask,
            action_slots=batch.action_slots,
        )

    @torch.no_grad()
    def update_observation_normalizer(
        self,
        designs: Sequence[BodyGenDesign],
        observations: torch.Tensor | None = None,
    ) -> None:
        raw = _node_batch(
            designs,
            observations=observations,
            device=self.device,
            dtype=self.dtype,
        )
        self.normalizer.update(raw.features, raw.mask)

    def _topology_statistics(
        self,
        designs: Sequence[BodyGenDesign],
        actions: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self._batch(designs)
        logits = self.topology_actor(batch)
        action_tensor = torch.zeros(
            batch.mask.shape, dtype=torch.long, device=self.device
        )
        for row, choices in enumerate(actions):
            if len(choices) != designs[row].num_nodes:
                raise ValueError("saved topology actions do not match their design")
            action_tensor[row, : len(choices)] = torch.as_tensor(
                choices, device=self.device, dtype=torch.long
            )
        log_prob, entropy = _categorical_statistics(
            logits, action_tensor, batch.mask
        )
        value = self.topology_critic(batch)
        return log_prob, entropy, value

    def _attribute_statistics(
        self,
        designs: Sequence[BodyGenDesign],
        effector_actions: Sequence[Sequence[int]],
        cap_actions: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self._batch(designs)
        effector_logits, cap_logits = self.attribute_actor(batch)
        effector_tensor = torch.zeros(
            batch.mask.shape, dtype=torch.long, device=self.device
        )
        cap_tensor = torch.zeros_like(effector_tensor)
        for row, (effectors, caps) in enumerate(
            zip(effector_actions, cap_actions)
        ):
            effector_tensor[row, batch.effector_mask[row]] = torch.as_tensor(
                effectors, device=self.device, dtype=torch.long
            )
            cap_tensor[row, batch.terminal_mask[row]] = torch.as_tensor(
                caps, device=self.device, dtype=torch.long
            )
        effector_log_prob, effector_entropy = _categorical_statistics(
            effector_logits, effector_tensor, batch.effector_mask
        )
        cap_log_prob, cap_entropy = _categorical_statistics(
            cap_logits, cap_tensor, batch.terminal_mask
        )
        value = self.attribute_critic(batch)
        return (
            effector_log_prob + cap_log_prob,
            effector_entropy + cap_entropy,
            value,
        )

    @torch.no_grad()
    def sample_designs(
        self,
        count: int,
        generator: torch.Generator | None,
        deterministic: bool = False,
    ) -> tuple[list[BodyGenDesign], DesignBatchTrace]:
        """Sample five topology waves and one categorical attribute wave."""

        if count < 1:
            raise ValueError("count must be positive")
        designs = [BodyGenDesign.canonical() for _ in range(count)]
        histories: list[list[DesignTransition]] = [
            [] for _ in range(count)
        ]

        for _ in range(N_TOPOLOGY_WAVES):
            batch = self._batch(designs)
            logits = self.topology_actor(batch)
            actions = _sample_categorical(
                logits,
                batch.mask,
                deterministic=deterministic,
                generator=generator,
            )
            next_designs: list[BodyGenDesign] = []
            for row, design in enumerate(designs):
                choices = tuple(
                    int(choice)
                    for choice in actions[row, : design.num_nodes].tolist()
                )
                histories[row].append(
                    DesignTransition(
                        stage=TOPOLOGY,
                        design=design,
                        topology_actions=choices,
                    )
                )
                next_designs.append(
                    apply_topology_actions(design, choices)
                )
            designs = next_designs

        batch = self._batch(designs)
        effector_logits, cap_logits = self.attribute_actor(batch)
        effector_actions = _sample_categorical(
            effector_logits,
            batch.effector_mask,
            deterministic=deterministic,
            generator=generator,
        )
        cap_actions = _sample_categorical(
            cap_logits,
            batch.terminal_mask,
            deterministic=deterministic,
            generator=generator,
        )
        final_designs: list[BodyGenDesign] = []
        traces: list[DesignTrace] = []
        for row, design in enumerate(designs):
            effectors = tuple(
                int(choice)
                for choice in effector_actions[
                    row, batch.effector_mask[row]
                ].tolist()
            )
            caps = tuple(
                int(choice)
                for choice in cap_actions[
                    row, batch.terminal_mask[row]
                ].tolist()
            )
            histories[row].append(
                DesignTransition(
                    stage=ATTRIBUTE,
                    design=design,
                    effector_actions=effectors,
                    cap_actions=caps,
                )
            )
            final = apply_attribute_actions(design, effectors, caps)
            final_designs.append(final)
            traces.append(
                DesignTrace(tuple(histories[row]), final)
            )
        return final_designs, DesignBatchTrace(tuple(traces))

    @torch.no_grad()
    def sample_design(
        self,
        generator: torch.Generator | None,
        deterministic: bool = False,
    ) -> tuple[BodyGenDesign, DesignTrace]:
        designs, batch = self.sample_designs(
            1, generator, deterministic=deterministic
        )
        return designs[0], batch.episodes[0]

    def evaluate_design(
        self,
        trace: DesignBatchTrace,
    ) -> dict[str, torch.Tensor]:
        """Re-evaluate stored design actions under the current policy."""

        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for step in range(N_DESIGN_STEPS):
            transitions = [
                episode.transitions[step]
                for episode in trace.episodes
            ]
            designs = [transition.design for transition in transitions]
            if step < N_TOPOLOGY_WAVES:
                statistics = self._topology_statistics(
                    designs,
                    [
                        transition.topology_actions
                        for transition in transitions
                    ],
                )
            else:
                statistics = self._attribute_statistics(
                    designs,
                    [
                        transition.effector_actions
                        for transition in transitions
                    ],
                    [
                        transition.cap_actions
                        for transition in transitions
                    ],
                )
            log_prob, entropy, value = statistics
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
        return {
            "log_prob": torch.stack(log_probs, dim=1),
            "entropy": torch.stack(entropies, dim=1),
            "values": torch.stack(values, dim=1),
        }

    def evaluate_design_transitions(
        self,
        transitions: Sequence[DesignTransition],
    ) -> dict[str, torch.Tensor]:
        """Evaluate only the design transitions in one PPO minibatch.

        Upstream's ``batch_design`` path groups stages before slicing 2,048
        transition minibatches. Re-evaluating the entire 50k+ design trace for
        every such slice would change neither gradients nor outputs, but would
        multiply memory and compute enough to make short-episode batches
        impractical.
        """

        if not transitions:
            raise ValueError("at least one design transition is required")
        log_prob: list[torch.Tensor | None] = [None] * len(transitions)
        entropy: list[torch.Tensor | None] = [None] * len(transitions)
        values: list[torch.Tensor | None] = [None] * len(transitions)

        for stage in (TOPOLOGY, ATTRIBUTE):
            positions = [
                index
                for index, transition in enumerate(transitions)
                if transition.stage == stage
            ]
            if not positions:
                continue
            selected = [transitions[index] for index in positions]
            designs = [transition.design for transition in selected]
            if stage == TOPOLOGY:
                statistics = self._topology_statistics(
                    designs,
                    [
                        transition.topology_actions
                        for transition in selected
                    ],
                )
            else:
                statistics = self._attribute_statistics(
                    designs,
                    [
                        transition.effector_actions
                        for transition in selected
                    ],
                    [transition.cap_actions for transition in selected],
                )
            for local, position in enumerate(positions):
                log_prob[position] = statistics[0][local]
                entropy[position] = statistics[1][local]
                values[position] = statistics[2][local]

        if any(item is None for item in log_prob + entropy + values):
            raise RuntimeError("unknown stage in a saved design transition")
        return {
            "log_prob": torch.stack(
                [item for item in log_prob if item is not None]
            ),
            "entropy": torch.stack(
                [item for item in entropy if item is not None]
            ),
            "values": torch.stack(
                [item for item in values if item is not None]
            ),
        }

    def control(
        self,
        observations: torch.Tensor,
        designs: Sequence[BodyGenDesign],
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> ControlOutput:
        batch = self._batch(designs, observations)
        mean, log_std, action_mask = self.control_actor(batch)
        value = self.control_critic(batch)
        if deterministic:
            action = mean
        else:
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            action = mean + log_std.exp() * noise
        action = action * action_mask
        return ControlOutput(
            mean=mean,
            log_std=log_std,
            value=value,
            action=action,
            action_mask=action_mask,
        )

    def evaluate_control(
        self,
        observations: torch.Tensor,
        designs: Sequence[BodyGenDesign],
        raw_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self.control(
            observations, designs, deterministic=True
        )
        actions = raw_actions.to(
            device=self.device, dtype=self.dtype
        )
        if actions.shape != output.mean.shape:
            raise ValueError(
                f"actions must have shape {tuple(output.mean.shape)}"
            )
        variance = torch.exp(2.0 * output.log_std)
        per_action_log_prob = -0.5 * (
            (actions - output.mean).square() / variance
            + 2.0 * output.log_std
            + math.log(2.0 * math.pi)
        )
        per_action_entropy = (
            output.log_std
            + 0.5 * (1.0 + math.log(2.0 * math.pi))
        )
        mask = output.action_mask
        return {
            "log_prob": (per_action_log_prob * mask).sum(dim=-1),
            "entropy": (per_action_entropy * mask).sum(dim=-1),
            "value": output.value,
        }

    def values(
        self,
        stage: int | str,
        observations: torch.Tensor | None,
        designs: Sequence[BodyGenDesign],
    ) -> torch.Tensor:
        """Small evaluator-facing stage-value helper."""

        names = {
            TOPOLOGY: "topology",
            ATTRIBUTE: "attribute",
            CONTROL: "control",
        }
        if isinstance(stage, str):
            name = stage
        elif stage in names:
            name = names[stage]
        else:
            raise ValueError("unknown BodyGen stage")
        if name == "control":
            if observations is None:
                raise ValueError("control values need VSim observations")
            return self.control(
                observations, designs, deterministic=True
            ).value
        if observations is not None:
            raise ValueError("design-stage values do not use VSim observations")
        batch = self._batch(designs)
        if name == "topology":
            return self.topology_critic(batch)
        if name == "attribute":
            return self.attribute_critic(batch)
        raise ValueError("unknown BodyGen stage")
