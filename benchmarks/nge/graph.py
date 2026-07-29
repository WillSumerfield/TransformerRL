"""NGE's robot graph and its four paper mutation primitives.

The original NGE code mutates arbitrary MuJoCo trees.  This benchmark instead
uses the shared ant grammar: eight fixed radial attachment slots, each holding
zero to three typed actuators and one typed terminal cap.  The mutations below
are the direct tree operations that remain inside that grammar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from transformer_rl.vocab import CANON_CAP, N_CAP, N_EFF, canonical_eff


N_LIMBS = 8
MAX_EFFECTORS = 3
PADDED_DEPTH = MAX_EFFECTORS + 1
NODE_ATTRIBUTE_SIZE = 16
MUTATION_NAMES = ("add_node", "add_graph", "del_graph", "pert_graph")


def _slot(limb: int, depth: int) -> int:
    """Return the environment's depth-major actuator slot (both inputs 0-based)."""
    return depth * N_LIMBS + limb


@dataclass(frozen=True)
class NGEGraph:
    """One valid body in the benchmark's typed tree grammar."""

    effectors: tuple[tuple[int, ...], ...]
    caps: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.effectors) != N_LIMBS or len(self.caps) != N_LIMBS:
            raise ValueError("an NGE graph must contain exactly eight limb slots")
        if not any(self.effectors):
            raise ValueError("an NGE graph needs at least one actuator")
        for chain in self.effectors:
            if len(chain) > MAX_EFFECTORS:
                raise ValueError("a limb cannot contain more than three effectors")
            if any(kind < 0 or kind >= N_EFF for kind in chain):
                raise ValueError("invalid effector type")
        if any(kind < 0 or kind >= N_CAP for kind in self.caps):
            raise ValueError("invalid cap type")

    @classmethod
    def canonical(cls, base_legs: Sequence[int] = (1, 4, 6)) -> "NGEGraph":
        """The exact shared starting morphology, using 1-based limb numbers."""
        active = {int(limb) - 1 for limb in base_legs}
        if not active or min(active) < 0 or max(active) >= N_LIMBS:
            raise ValueError("base_legs must be non-empty values in [1, 8]")
        chains = tuple(
            tuple(canonical_eff(depth) for depth in range(2))
            if limb in active
            else ()
            for limb in range(N_LIMBS)
        )
        return cls(chains, (CANON_CAP,) * N_LIMBS)

    @classmethod
    def from_arrays(
        cls,
        counts: Sequence[int],
        eff_sub: np.ndarray | Sequence[Sequence[int]],
        cap_sub: Sequence[int],
    ) -> "NGEGraph":
        count_array = np.asarray(counts, dtype=np.int64)
        effectors = np.asarray(eff_sub, dtype=np.int64)
        caps = np.asarray(cap_sub, dtype=np.int64)
        if count_array.shape != (N_LIMBS,):
            raise ValueError("counts must have shape (8,)")
        if effectors.ndim != 2 or effectors.shape[0] != N_LIMBS:
            raise ValueError("eff_sub must have shape (8, depth)")
        if caps.shape != (N_LIMBS,):
            raise ValueError("cap_sub must have shape (8,)")
        return cls(
            tuple(
                tuple(int(value) for value in effectors[limb, : count_array[limb]])
                for limb in range(N_LIMBS)
            ),
            tuple(int(value) for value in caps),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NGEGraph":
        return cls(
            tuple(tuple(int(value) for value in chain) for chain in data["effectors"]),
            tuple(int(value) for value in data["caps"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "effectors": [list(chain) for chain in self.effectors],
            "caps": list(self.caps),
        }

    @property
    def num_actuators(self) -> int:
        return sum(len(chain) for chain in self.effectors)

    @property
    def num_nodes(self) -> int:
        # NerveNet nodes are the root and actuated physical modules.  A cap's
        # type is an attribute on its terminal module because caps have no DOF.
        return 1 + self.num_actuators

    @property
    def key(self) -> tuple[Any, ...]:
        return (*self.effectors, self.caps)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts = np.asarray([len(chain) for chain in self.effectors], dtype=np.int64)
        effectors = np.full((N_LIMBS, PADDED_DEPTH), -1, dtype=np.int64)
        for limb, chain in enumerate(self.effectors):
            effectors[limb, : len(chain)] = chain
        return counts, effectors, np.asarray(self.caps, dtype=np.int64)

    def action_slots(self) -> tuple[int, ...]:
        """Padded VSim action slot for each non-root graph node."""
        return tuple(
            _slot(limb, depth)
            for limb, chain in enumerate(self.effectors)
            for depth in range(len(chain))
        )

    def edges(self) -> tuple[tuple[int, int], ...]:
        """Undirected NerveNet edges, represented as directed pairs.

        The tree edge links each module to its physical parent.  As in the
        upstream ``Rb`` root-connection option, the root also talks directly to
        every controlling module.  Duplicate edges are removed.
        """
        edges: set[tuple[int, int]] = set()
        node = 1
        for chain in self.effectors:
            previous = 0
            for _ in chain:
                for source, target in ((previous, node), (node, previous)):
                    edges.add((source, target))
                for source, target in ((0, node), (node, 0)):
                    edges.add((source, target))
                previous = node
                node += 1
        return tuple(sorted(edges))

    def node_attributes(self) -> np.ndarray:
        """Fixed-width structural attributes consumed by policy and GM-UC."""
        attributes = np.zeros((self.num_nodes, NODE_ATTRIBUTE_SIZE), dtype=np.float32)
        attributes[0, 0] = 1.0  # root
        row = 1
        for limb, chain in enumerate(self.effectors):
            angle = limb * np.pi / 4.0
            for depth, effector in enumerate(chain):
                attr = attributes[row]
                attr[1] = 1.0
                attr[2 + effector] = 1.0
                if depth == len(chain) - 1:
                    attr[5 + self.caps[limb]] = 1.0
                    attr[14] = 1.0
                attr[9] = np.sin(angle)
                attr[10] = np.cos(angle)
                attr[11 + depth] = 1.0
                attr[15] = float(len(chain)) / MAX_EFFECTORS
                row += 1
        return attributes


@dataclass(frozen=True)
class Mutation:
    """A mutated graph plus enough genealogy metadata to audit it."""

    graph: NGEGraph
    operation: str
    source_limb: int | None = None
    target_limb: int | None = None
    subtree_depth: int | None = None
    mirrored: bool = False


def _different_integer(current: int, size: int, rng: np.random.Generator) -> int:
    draw = int(rng.integers(size - 1))
    return draw + (draw >= current)


def add_node(graph: NGEGraph, rng: np.random.Generator) -> Mutation:
    """Append one uniformly typed actuator at a valid tree tip."""
    targets = [limb for limb, chain in enumerate(graph.effectors) if len(chain) < MAX_EFFECTORS]
    if not targets:
        raise ValueError("Add-Node has no valid attachment point")
    target = int(rng.choice(targets))
    chains = [list(chain) for chain in graph.effectors]
    chains[target].append(int(rng.integers(N_EFF)))
    caps = list(graph.caps)
    if len(chains[target]) == 1:
        caps[target] = int(rng.integers(N_CAP))
    return Mutation(
        NGEGraph(tuple(tuple(chain) for chain in chains), tuple(caps)),
        "add_node",
        target_limb=target,
        subtree_depth=len(chains[target]) - 1,
    )


def _mirrored_limb(limb: int) -> int:
    # Reflection across the forward/back axis: 2<->8, 3<->7, 4<->6.
    return (-limb) % N_LIMBS


def add_graph(graph: NGEGraph, rng: np.random.Generator) -> Mutation:
    """Copy a sampled limb subtree and append it at another valid tip."""
    sources = [
        (limb, depth)
        for limb, chain in enumerate(graph.effectors)
        for depth in range(len(chain))
    ]
    targets = [limb for limb, chain in enumerate(graph.effectors) if len(chain) < MAX_EFFECTORS]
    if not sources or not targets:
        raise ValueError("Add-Graph has no valid source or attachment point")

    source, depth = sources[int(rng.integers(len(sources)))]
    mirrored = bool(rng.integers(2))
    mirror_target = _mirrored_limb(source)
    if mirrored and mirror_target in targets:
        target = mirror_target
    else:
        target = int(rng.choice(targets))

    chains = [list(chain) for chain in graph.effectors]
    room = MAX_EFFECTORS - len(chains[target])
    subtree = list(graph.effectors[source][depth : depth + room])
    if not subtree:
        raise ValueError("Add-Graph sampled an empty subtree")
    chains[target].extend(subtree)
    caps = list(graph.caps)
    caps[target] = graph.caps[source]
    return Mutation(
        NGEGraph(tuple(tuple(chain) for chain in chains), tuple(caps)),
        "add_graph",
        source_limb=source,
        target_limb=target,
        subtree_depth=depth,
        mirrored=mirrored and target == mirror_target,
    )


def del_graph(graph: NGEGraph, rng: np.random.Generator) -> Mutation:
    """Remove a sampled actuator and every descendant below it."""
    choices = [
        (limb, depth)
        for limb, chain in enumerate(graph.effectors)
        for depth in range(len(chain))
        if graph.num_actuators - (len(chain) - depth) >= 1
    ]
    if not choices:
        raise ValueError("Del-Graph cannot remove the final actuator")
    source, depth = choices[int(rng.integers(len(choices)))]
    chains = [list(chain) for chain in graph.effectors]
    del chains[source][depth:]
    caps = list(graph.caps)
    if not chains[source]:
        caps[source] = CANON_CAP
    return Mutation(
        NGEGraph(tuple(tuple(chain) for chain in chains), tuple(caps)),
        "del_graph",
        source_limb=source,
        subtree_depth=depth,
    )


def pert_graph(
    graph: NGEGraph,
    rng: np.random.Generator,
    node_probability: float = 0.1,
) -> Mutation:
    """Perturb discrete attributes throughout one sampled subtree.

    NGE adds Gaussian noise to continuous XML attributes.  The shared benchmark
    grammar has categorical module attributes, so crossing a category boundary
    is its discrete counterpart.  At least one attribute is always changed.
    """
    if not 0.0 <= node_probability <= 1.0:
        raise ValueError("node_probability must lie in [0, 1]")
    choices = [
        (limb, depth)
        for limb, chain in enumerate(graph.effectors)
        for depth in range(len(chain))
    ]
    source, depth = choices[int(rng.integers(len(choices)))]
    chains = [list(chain) for chain in graph.effectors]
    caps = list(graph.caps)
    changed = False
    for index in range(depth, len(chains[source])):
        if rng.random() < node_probability:
            chains[source][index] = _different_integer(
                chains[source][index], N_EFF, rng
            )
            changed = True
    if rng.random() < node_probability:
        caps[source] = _different_integer(caps[source], N_CAP, rng)
        changed = True
    if not changed:
        choices_to_force = list(range(depth, len(chains[source]))) + ["cap"]
        chosen = choices_to_force[int(rng.integers(len(choices_to_force)))]
        if chosen == "cap":
            caps[source] = _different_integer(caps[source], N_CAP, rng)
        else:
            index = int(chosen)
            chains[source][index] = _different_integer(
                chains[source][index], N_EFF, rng
            )
    return Mutation(
        NGEGraph(tuple(tuple(chain) for chain in chains), tuple(caps)),
        "pert_graph",
        source_limb=source,
        subtree_depth=depth,
    )


def mutate(
    graph: NGEGraph,
    rng: np.random.Generator,
    probabilities: dict[str, float] | None = None,
    *,
    node_perturb_probability: float = 0.1,
) -> Mutation:
    """Sample and apply one of the four mutation primitives."""
    probabilities = probabilities or {name: 0.25 for name in MUTATION_NAMES}
    if set(probabilities) != set(MUTATION_NAMES):
        raise ValueError(f"mutation probabilities must name {MUTATION_NAMES}")
    weights = np.asarray([probabilities[name] for name in MUTATION_NAMES], dtype=float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("mutation probabilities must be non-negative and sum to one")
    operations = {
        "add_node": add_node,
        "add_graph": add_graph,
        "del_graph": del_graph,
        "pert_graph": lambda value, generator: pert_graph(
            value, generator, node_perturb_probability
        ),
    }

    # A saturated/minimal graph can make one operation invalid.  Resampling the
    # primitive matches the upstream rejection-style mutation loop.
    for _ in range(64):
        name = str(rng.choice(MUTATION_NAMES, p=weights))
        try:
            result = operations[name](graph, rng)
        except ValueError:
            continue
        if result.graph != graph:
            return result
    raise RuntimeError("could not produce a valid graph mutation after 64 attempts")
