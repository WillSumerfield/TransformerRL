"""Graph Mutation with Uncertainty (GM-UC).

The surrogate is trained on prior species fitness and ranks an over-generated
candidate set with one shared inference-dropout sample per generation.  Reusing
the same mask for every candidate is important: it is one Thompson-sampled
model, not unrelated noise added to each graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .graph import NGEGraph, NODE_ATTRIBUTE_SIZE


class GraphFitnessSurrogate(nn.Module):
    """Attribute-only GNN that predicts one amortised-fitness score."""

    def __init__(self, hidden_size: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.attribute = nn.Linear(NODE_ATTRIBUTE_SIZE, hidden_size)
        self.message = nn.Linear(hidden_size, hidden_size)
        self.update = nn.GRUCell(hidden_size, hidden_size)
        self.output_hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        for name, parameter in self.update.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)

    def sample_masks(
        self,
        rng: np.random.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One dropout realization reused for a complete candidate ranking."""
        keep = 1.0 - self.dropout

        def draw(size: int) -> torch.Tensor:
            values = (rng.random(size) < keep).astype(np.float32) / keep
            return torch.as_tensor(values, device=device)

        return (
            draw(NODE_ATTRIBUTE_SIZE),
            draw(self.hidden_size),
            draw(self.hidden_size),
        )

    def _drop(
        self,
        value: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is not None:
            return value * mask
        return F.dropout(value, p=self.dropout, training=self.training)

    def forward_graph(
        self,
        graph: NGEGraph,
        masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        device = self.attribute.weight.device
        attrs = torch.as_tensor(
            graph.node_attributes(),
            dtype=self.attribute.weight.dtype,
            device=device,
        )
        first, second, third = masks if masks is not None else (None, None, None)
        hidden = torch.tanh(self.attribute(self._drop(attrs, first)))

        edges = graph.edges()
        source = torch.as_tensor(
            [edge[0] for edge in edges], dtype=torch.long, device=device
        )
        target = torch.as_tensor(
            [edge[1] for edge in edges], dtype=torch.long, device=device
        )
        # Three graph propagation rounds match the small upstream pruning GNN.
        for _ in range(3):
            sent = torch.tanh(self.message(self._drop(hidden[source], second)))
            aggregated = torch.zeros_like(hidden)
            aggregated.index_add_(0, target, sent)
            degree = torch.zeros(
                graph.num_nodes, dtype=hidden.dtype, device=device
            )
            degree.index_add_(
                0, target, torch.ones_like(target, dtype=hidden.dtype)
            )
            aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)
            hidden = self.update(aggregated, hidden)

        graph_hidden = hidden.mean(dim=0)
        graph_hidden = torch.tanh(
            self.output_hidden(self._drop(graph_hidden, third))
        )
        return self.output(graph_hidden).squeeze(-1)


@dataclass(frozen=True)
class FitnessExample:
    species_id: int
    graph: NGEGraph
    fitness: float


class GraphMutationWithUncertainty:
    """Owns the surrogate, its historical labels, optimizer and Thompson RNG."""

    def __init__(
        self,
        device: torch.device,
        *,
        hidden_size: int = 64,
        dropout: float = 0.5,
        learning_rate: float = 3.0e-4,
        batch_size: int = 64,
        gradient_steps: int = 1,
        temperature: float = 0.1,
    ) -> None:
        if temperature <= 0:
            raise ValueError("GM-UC temperature must be positive")
        self.device = device
        self.model = GraphFitnessSurrogate(hidden_size, dropout).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = int(batch_size)
        self.gradient_steps = int(gradient_steps)
        self.temperature = float(temperature)
        self.history: dict[int, FitnessExample] = {}
        self.updates = 0

    def observe(
        self,
        examples: Iterable[tuple[int, NGEGraph, float]],
    ) -> None:
        # Upstream stores one latest record per persistent species ID.
        for species_id, graph, fitness in examples:
            if not np.isfinite(fitness):
                raise ValueError("GM-UC fitness labels must be finite")
            self.history[int(species_id)] = FitnessExample(
                int(species_id), graph, float(fitness)
            )

    def train_generation(self, rng: np.random.Generator) -> float:
        """Take the upstream-style small regression update after a generation."""
        if len(self.history) < 2 or self.gradient_steps <= 0:
            return float("nan")
        examples = list(self.history.values())
        fitness = np.asarray([example.fitness for example in examples])
        mean = float(fitness.mean())
        std = max(float(fitness.std()), 1.0e-8)
        losses: list[float] = []
        self.model.train()
        for _ in range(self.gradient_steps):
            sample_size = min(self.batch_size, len(examples))
            indices = rng.integers(0, len(examples), size=sample_size)
            predictions = torch.stack(
                [self.model.forward_graph(examples[int(index)].graph) for index in indices]
            )
            targets = torch.as_tensor(
                [(examples[int(index)].fitness - mean) / std for index in indices],
                dtype=predictions.dtype,
                device=self.device,
            )
            loss = F.mse_loss(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach()))
            self.updates += 1
        return float(np.mean(losses))

    @torch.no_grad()
    def rank(
        self,
        candidates: list[Any],
        keep: int,
        rng: np.random.Generator,
    ) -> list[Any]:
        """Return the top candidates under one Thompson-sampled surrogate."""
        if keep < 0 or keep > len(candidates):
            raise ValueError("invalid number of GM-UC candidates to retain")
        if keep == 0:
            return []
        self.model.eval()
        masks = self.model.sample_masks(rng, self.device)
        scores = []
        for candidate in candidates:
            prediction = float(self.model.forward_graph(candidate.graph, masks))
            # The pinned code adds standard Gumbel noise and divides the
            # prediction by this temperature before ranking.
            uniform = float(rng.uniform(np.finfo(float).eps, 1.0))
            gumbel = -np.log(-np.log(uniform))
            scores.append(prediction / self.temperature + gumbel)
        order = np.argsort(np.asarray(scores))[::-1][:keep]
        return [candidates[int(index)] for index in order]

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": [
                {
                    "species_id": example.species_id,
                    "graph": example.graph.to_dict(),
                    "fitness": example.fitness,
                }
                for example in self.history.values()
            ],
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.history = {
            int(item["species_id"]): FitnessExample(
                int(item["species_id"]),
                NGEGraph.from_dict(item["graph"]),
                float(item["fitness"]),
            )
            for item in state["history"]
        }
        self.updates = int(state["updates"])
