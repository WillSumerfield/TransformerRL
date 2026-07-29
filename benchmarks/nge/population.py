"""Readable population-selection loop for Neural Graph Evolution."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .gm_uc import GraphMutationWithUncertainty
from .graph import Mutation, NGEGraph, mutate


@dataclass(frozen=True)
class Species:
    """Graph and genealogy metadata; controller state lives in the trainer."""

    species_id: int
    graph: NGEGraph
    parent_id: int | None
    birth_generation: int
    mutation: str
    fitness: float | None = None
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "species_id": self.species_id,
            "graph": self.graph.to_dict(),
            "parent_id": self.parent_id,
            "birth_generation": self.birth_generation,
            "mutation": self.mutation,
            "fitness": self.fitness,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Species":
        return cls(
            species_id=int(data["species_id"]),
            graph=NGEGraph.from_dict(data["graph"]),
            parent_id=(
                None if data.get("parent_id") is None else int(data["parent_id"])
            ),
            birth_generation=int(data["birth_generation"]),
            mutation=str(data["mutation"]),
            fitness=(
                None if data.get("fitness") is None else float(data["fitness"])
            ),
            rank=None if data.get("rank") is None else int(data["rank"]),
        )


@dataclass(frozen=True)
class Candidate:
    graph: NGEGraph
    parent_id: int
    mutation: Mutation


@dataclass(frozen=True)
class EvolutionResult:
    eliminated_ids: tuple[int, ...]
    child_parent_ids: dict[int, int]
    gm_uc_loss: float


class Population:
    """Persistent species bank with elimination, reproduction, and GM-UC."""

    def __init__(
        self,
        species: list[Species],
        *,
        generation: int = 0,
        next_species_id: int | None = None,
    ) -> None:
        if not species:
            raise ValueError("NGE population cannot be empty")
        ids = [item.species_id for item in species]
        if len(ids) != len(set(ids)):
            raise ValueError("species IDs must be unique")
        self.species = list(species)
        self.generation = int(generation)
        self.next_species_id = (
            max(ids) + 1 if next_species_id is None else int(next_species_id)
        )

    @classmethod
    def initial(
        cls,
        size: int,
        graph: NGEGraph,
    ) -> "Population":
        if size < 2:
            raise ValueError("NGE requires at least two species")
        return cls(
            [
                Species(
                    species_id=index + 1,
                    graph=graph,
                    parent_id=None,
                    birth_generation=0,
                    mutation="initial",
                )
                for index in range(size)
            ]
        )

    @property
    def size(self) -> int:
        return len(self.species)

    def by_id(self) -> dict[int, Species]:
        return {item.species_id: item for item in self.species}

    def assign_fitness(self, fitness: dict[int, float]) -> None:
        expected = {item.species_id for item in self.species}
        if set(fitness) != expected:
            missing = sorted(expected - set(fitness))
            extra = sorted(set(fitness) - expected)
            raise ValueError(f"fitness IDs mismatch; missing={missing}, extra={extra}")
        if not all(np.isfinite(value) for value in fitness.values()):
            raise ValueError("species fitness must be finite")
        self.species = [
            replace(item, fitness=float(fitness[item.species_id]))
            for item in self.species
        ]

    def evolve(
        self,
        gm_uc: GraphMutationWithUncertainty,
        rng: np.random.Generator,
        *,
        elimination_rate: float,
        candidate_pool_size: int,
        mutation_probabilities: dict[str, float],
        node_perturb_probability: float,
    ) -> EvolutionResult:
        """Perform Algorithm 1's remove, mutate, prune, and inherit bookkeeping."""
        if any(item.fitness is None for item in self.species):
            raise ValueError("all species need fitness before selection")
        if not 0.0 < elimination_rate < 1.0:
            raise ValueError("elimination_rate must lie strictly between zero and one")
        eliminate = int(np.floor(self.size * elimination_rate))
        eliminate = max(1, min(eliminate, self.size - 1))

        ranked = sorted(
            self.species,
            key=lambda item: (float(item.fitness), item.species_id),
            reverse=True,
        )
        ranked = [replace(item, rank=index + 1) for index, item in enumerate(ranked)]
        survivors = ranked[: self.size - eliminate]
        eliminated = ranked[self.size - eliminate :]

        gm_uc.observe(
            (
                item.species_id,
                item.graph,
                float(item.fitness),
            )
            for item in ranked
        )
        gm_uc_loss = gm_uc.train_generation(rng)

        if candidate_pool_size < self.size:
            raise ValueError("candidate_pool_size cannot be smaller than population")
        new_candidate_count = candidate_pool_size - len(survivors)
        candidates: list[Candidate] = []
        for _ in range(new_candidate_count):
            parent = survivors[int(rng.integers(len(survivors)))]
            mutation = mutate(
                parent.graph,
                rng,
                mutation_probabilities,
                node_perturb_probability=node_perturb_probability,
            )
            candidates.append(
                Candidate(mutation.graph, parent.species_id, mutation)
            )

        selected = gm_uc.rank(candidates, eliminate, rng)
        child_parent_ids: dict[int, int] = {}
        children: list[Species] = []
        for candidate in selected:
            child_id = self.next_species_id
            self.next_species_id += 1
            child_parent_ids[child_id] = candidate.parent_id
            children.append(
                Species(
                    species_id=child_id,
                    graph=candidate.graph,
                    parent_id=candidate.parent_id,
                    birth_generation=self.generation + 1,
                    mutation=candidate.mutation.operation,
                )
            )

        # Fitness is generation-local.  Survivors retain learned controllers but
        # must be evaluated again beside the new children.
        self.species = [replace(item, fitness=None, rank=None) for item in survivors]
        self.species.extend(children)
        self.generation += 1
        if self.size != len(ranked):
            raise RuntimeError("selection changed the configured population size")
        return EvolutionResult(
            tuple(item.species_id for item in eliminated),
            child_parent_ids,
            gm_uc_loss,
        )

    def sample(
        self,
        count: int,
        rng: np.random.Generator,
    ) -> list[Species]:
        """Uniform samples from the final surviving species distribution."""
        if count <= 0:
            raise ValueError("sample count must be positive")
        indices = rng.integers(0, self.size, size=count)
        return [self.species[int(index)] for index in indices]

    def state_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "next_species_id": self.next_species_id,
            "species": [item.to_dict() for item in self.species],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Population":
        return cls(
            [Species.from_dict(item) for item in state["species"]],
            generation=int(state["generation"]),
            next_species_id=int(state["next_species_id"]),
        )
