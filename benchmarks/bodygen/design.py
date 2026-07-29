"""BodyGen's two design stages in the shared typed ant grammar.

BodyGen performs five topology decisions followed by one attribute decision
before the resulting body is simulated.  The upstream method edits a general
MuJoCo tree.  Here the same decisions are restricted to the benchmark's eight
radial limb slots, with at most three actuated effectors per limb.

The code in this module is deliberately simulator-free.  A topology action is
applied to a snapshot of the nodes that existed at the start of that wave, so
newly added nodes cannot act until the next wave.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from transformer_rl.vocab import CANON_CAP, N_CAP, N_EFF, canonical_eff


N_LIMBS = 8
MAX_EFFECTORS = 3
# The environment/evaluator reserves the fourth depth position for the cap.
PADDED_DEPTH = MAX_EFFECTORS + 1
N_TOPOLOGY_WAVES = 5
N_DESIGN_STEPS = N_TOPOLOGY_WAVES + 1
TOPOLOGY_EMBEDDINGS = 256

TOPOLOGY = 0
ATTRIBUTE = 1
CONTROL = 2

NO_CHANGE = 0
ADD = 1
DELETE = 2
N_TOPOLOGY_ACTIONS = 3

# Root/effector kind, four depth buckets, eight radial slots, and the typed
# effector/cap categories.  The camera-ready crawler observes depth but does
# not receive explicit Add/Delete-validity flags.
DESIGN_FEATURE_SIZE = 2 + 4 + N_LIMBS + N_EFF + N_CAP


@dataclass(frozen=True, order=True)
class DesignNode:
    """A node in the root-first, limb-major BodyGen tree."""

    limb: int | None
    depth: int

    @property
    def is_root(self) -> bool:
        return self.limb is None


@dataclass(frozen=True)
class BodyGenDesign:
    """One valid morphology in the benchmark's typed radial grammar."""

    effectors: tuple[tuple[int, ...], ...]
    caps: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.effectors) != N_LIMBS or len(self.caps) != N_LIMBS:
            raise ValueError("a BodyGen design must contain exactly eight limb slots")
        if not any(self.effectors):
            raise ValueError("a BodyGen design needs at least one effector")
        for chain in self.effectors:
            if len(chain) > MAX_EFFECTORS:
                raise ValueError("a limb cannot contain more than three effectors")
            if any(kind < 0 or kind >= N_EFF for kind in chain):
                raise ValueError("invalid effector type")
        if any(kind < 0 or kind >= N_CAP for kind in self.caps):
            raise ValueError("invalid cap type")

    @classmethod
    def canonical(
        cls,
        base_legs: Sequence[int] = (1, 4, 6),
    ) -> "BodyGenDesign":
        """Return the shared starting body.

        Limb numbers are one-based at this public boundary.  Each initial limb
        has the canonical two-effector swing/knee chain and a bare cap.
        """

        active = {int(limb) - 1 for limb in base_legs}
        if not active or min(active) < 0 or max(active) >= N_LIMBS:
            raise ValueError("base_legs must be non-empty values in [1, 8]")
        effectors = tuple(
            tuple(canonical_eff(depth) for depth in range(2))
            if limb in active
            else ()
            for limb in range(N_LIMBS)
        )
        return cls(effectors, (CANON_CAP,) * N_LIMBS)

    @classmethod
    def from_arrays(
        cls,
        counts: Sequence[int] | np.ndarray,
        eff_sub: Sequence[Sequence[int]] | np.ndarray,
        cap_sub: Sequence[int] | np.ndarray,
    ) -> "BodyGenDesign":
        count_array = np.asarray(counts, dtype=np.int64)
        effector_array = np.asarray(eff_sub, dtype=np.int64)
        cap_array = np.asarray(cap_sub, dtype=np.int64)
        if count_array.shape != (N_LIMBS,):
            raise ValueError("counts must have shape (8,)")
        if effector_array.ndim != 2 or effector_array.shape[0] != N_LIMBS:
            raise ValueError("eff_sub must have shape (8, depth)")
        if cap_array.shape != (N_LIMBS,):
            raise ValueError("cap_sub must have shape (8,)")
        if np.any(count_array < 0) or np.any(count_array > MAX_EFFECTORS):
            raise ValueError("counts must lie in [0, 3]")
        if np.any(count_array > effector_array.shape[1]):
            raise ValueError("eff_sub does not contain every requested effector")
        return cls(
            tuple(
                tuple(
                    int(kind)
                    for kind in effector_array[limb, : count_array[limb]]
                )
                for limb in range(N_LIMBS)
            ),
            tuple(int(kind) for kind in cap_array),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BodyGenDesign":
        return cls(
            tuple(
                tuple(int(kind) for kind in chain)
                for chain in data["effectors"]
            ),
            tuple(int(kind) for kind in data["caps"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "effectors": [list(chain) for chain in self.effectors],
            "caps": list(self.caps),
        }

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays accepted directly by :class:`AntCodesignEnv`."""

        counts = np.asarray(
            [len(chain) for chain in self.effectors], dtype=np.int64
        )
        effectors = np.full(
            (N_LIMBS, PADDED_DEPTH), -1, dtype=np.int64
        )
        for limb, chain in enumerate(self.effectors):
            effectors[limb, : len(chain)] = chain
        caps = np.asarray(self.caps, dtype=np.int64)
        return counts, effectors, caps

    @property
    def num_effectors(self) -> int:
        return sum(len(chain) for chain in self.effectors)

    @property
    def num_nodes(self) -> int:
        return 1 + self.num_effectors

    @property
    def nodes(self) -> tuple[DesignNode, ...]:
        return (DesignNode(None, -1),) + tuple(
            DesignNode(limb, depth)
            for limb, chain in enumerate(self.effectors)
            for depth in range(len(chain))
        )

    @property
    def terminal_nodes(self) -> tuple[DesignNode, ...]:
        return tuple(
            DesignNode(limb, len(chain) - 1)
            for limb, chain in enumerate(self.effectors)
            if chain
        )

    def action_slots(self) -> tuple[int, ...]:
        """VSim's depth-major action slot for every non-root node."""

        return tuple(
            depth * N_LIMBS + limb
            for limb, chain in enumerate(self.effectors)
            for depth in range(len(chain))
        )

    def topology_ids(self) -> tuple[int, ...]:
        return tuple(topology_id(node) for node in self.nodes)


def topology_id(node: DesignNode) -> int:
    """Stable TopoPE index from the node's root path.

    Root is zero.  A root child is identified by its one-based radial slot and
    every subsequent (only) child adds digit one at the next base-nine place.
    With little-endian path digits, the deepest possible IDs are 91..98 and
    therefore fit the upstream 256-entry embedding without truncation.
    """

    if node.is_root:
        return 0
    assert node.limb is not None
    value = node.limb + 1
    for place in range(1, node.depth + 1):
        value += 9**place
    return value


def _integers(values: Sequence[int] | np.ndarray | torch.Tensor) -> list[int]:
    if torch.is_tensor(values):
        values = values.detach().cpu().reshape(-1).tolist()
    elif isinstance(values, np.ndarray):
        values = values.reshape(-1).tolist()
    return [int(value) for value in values]


def can_add(design: BodyGenDesign, node: DesignNode) -> bool:
    if node.is_root:
        return any(not chain for chain in design.effectors)
    assert node.limb is not None
    chain = design.effectors[node.limb]
    return node.depth == len(chain) - 1 and len(chain) < MAX_EFFECTORS


def can_delete(design: BodyGenDesign, node: DesignNode) -> bool:
    if node.is_root or design.num_effectors <= 1:
        return False
    assert node.limb is not None
    chain = design.effectors[node.limb]
    return node.depth == len(chain) - 1


def design_node_features(
    design: BodyGenDesign,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Structural/type observations in ``design.nodes`` order."""

    result = torch.zeros(
        design.num_nodes,
        DESIGN_FEATURE_SIZE,
        dtype=dtype,
        device=device,
    )
    depth_start = 2
    limb_start = depth_start + 4
    effector_start = limb_start + N_LIMBS
    cap_start = effector_start + N_EFF

    for row, node in enumerate(design.nodes):
        if node.is_root:
            result[row, 0] = 1.0
            result[row, depth_start] = 1.0
        else:
            assert node.limb is not None
            result[row, 1] = 1.0
            result[row, depth_start + node.depth + 1] = 1.0
            result[row, limb_start + node.limb] = 1.0
            kind = design.effectors[node.limb][node.depth]
            result[row, effector_start + kind] = 1.0
            if node.depth == len(design.effectors[node.limb]) - 1:
                result[row, cap_start + design.caps[node.limb]] = 1.0
    return result


def apply_topology_actions(
    design: BodyGenDesign,
    actions: Sequence[int] | np.ndarray | torch.Tensor,
) -> BodyGenDesign:
    """Apply one BodyGen topology wave.

    Actions correspond to ``design.nodes`` at the start of the wave.  They are
    processed in that stable root-first order.  Invalid actions are no-ops,
    matching upstream BodyGen.  This ordering also guarantees that concurrent
    deletions cannot remove the final remaining effector.
    """

    snapshot_nodes = design.nodes
    choices = _integers(actions)
    if len(choices) != len(snapshot_nodes):
        raise ValueError(
            f"expected {len(snapshot_nodes)} topology actions, got {len(choices)}"
        )
    if any(choice < 0 or choice >= N_TOPOLOGY_ACTIONS for choice in choices):
        raise ValueError("topology actions must be NoChange, Add or Delete")

    chains = [list(chain) for chain in design.effectors]
    caps = list(design.caps)

    # Nodes are deliberately still interpreted against the snapshot.  The
    # mutable chains only hold the accumulated result.
    for node, action in zip(snapshot_nodes, choices):
        if action == NO_CHANGE:
            continue
        if node.is_root:
            if action == ADD:
                empty = next(
                    (limb for limb, chain in enumerate(chains) if not chain),
                    None,
                )
                if empty is not None:
                    chains[empty].append(canonical_eff(0))
                    caps[empty] = CANON_CAP
            continue

        assert node.limb is not None
        snapshot_chain = design.effectors[node.limb]
        was_terminal = node.depth == len(snapshot_chain) - 1
        if not was_terminal:
            continue

        if action == ADD and len(snapshot_chain) < MAX_EFFECTORS:
            # The current chain can only differ if the root filled this slot.
            # A snapshot node cannot belong to an originally empty slot.
            if len(chains[node.limb]) == len(snapshot_chain):
                chains[node.limb].append(canonical_eff(len(snapshot_chain)))
        elif action == DELETE:
            current_total = sum(len(chain) for chain in chains)
            if (
                current_total > 1
                and len(chains[node.limb]) == len(snapshot_chain)
            ):
                chains[node.limb].pop()
                if not chains[node.limb]:
                    caps[node.limb] = CANON_CAP

    return BodyGenDesign(
        tuple(tuple(chain) for chain in chains),
        tuple(caps),
    )


def apply_attribute_actions(
    design: BodyGenDesign,
    effector_actions: Sequence[int] | np.ndarray | torch.Tensor,
    cap_actions: Sequence[int] | np.ndarray | torch.Tensor,
) -> BodyGenDesign:
    """Apply BodyGen's single simultaneous categorical attribute step.

    Effector actions are in ``design.nodes[1:]`` order.  Cap actions are in
    ``design.terminal_nodes`` order.  They are absolute category selections,
    not deltas from the canonical body.
    """

    effector_kinds = _integers(effector_actions)
    cap_kinds = _integers(cap_actions)
    if len(effector_kinds) != design.num_effectors:
        raise ValueError(
            f"expected {design.num_effectors} effector attributes, "
            f"got {len(effector_kinds)}"
        )
    if len(cap_kinds) != len(design.terminal_nodes):
        raise ValueError(
            f"expected {len(design.terminal_nodes)} cap attributes, "
            f"got {len(cap_kinds)}"
        )
    if any(kind < 0 or kind >= N_EFF for kind in effector_kinds):
        raise ValueError(f"effector attributes must lie in [0, {N_EFF})")
    if any(kind < 0 or kind >= N_CAP for kind in cap_kinds):
        raise ValueError(f"cap attributes must lie in [0, {N_CAP})")

    iterator = iter(effector_kinds)
    chains = tuple(
        tuple(next(iterator) for _ in chain)
        for chain in design.effectors
    )
    caps = list(design.caps)
    for node, kind in zip(design.terminal_nodes, cap_kinds):
        assert node.limb is not None
        caps[node.limb] = kind
    return BodyGenDesign(chains, tuple(caps))


@dataclass(frozen=True)
class DesignTransition:
    """One immutable design-stage state/action pair."""

    stage: int
    design: BodyGenDesign
    topology_actions: tuple[int, ...] = ()
    effector_actions: tuple[int, ...] = ()
    cap_actions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in (TOPOLOGY, ATTRIBUTE):
            raise ValueError("a design transition must be topology or attribute")
        if self.stage == TOPOLOGY:
            if len(self.topology_actions) != self.design.num_nodes:
                raise ValueError("topology action count does not match design nodes")
            if self.effector_actions or self.cap_actions:
                raise ValueError("topology transitions cannot contain attributes")
        else:
            if self.topology_actions:
                raise ValueError("attribute transitions cannot contain topology actions")
            if len(self.effector_actions) != self.design.num_effectors:
                raise ValueError("effector action count does not match design")
            if len(self.cap_actions) != len(self.design.terminal_nodes):
                raise ValueError("cap action count does not match design")


@dataclass(frozen=True)
class DesignTrace:
    """The six design transitions belonging to one complete episode."""

    transitions: tuple[DesignTransition, ...]
    final_design: BodyGenDesign

    def __post_init__(self) -> None:
        if len(self.transitions) != N_DESIGN_STEPS:
            raise ValueError("a BodyGen trace must contain five topology and one attribute step")
        stages = tuple(transition.stage for transition in self.transitions)
        if stages != (TOPOLOGY,) * N_TOPOLOGY_WAVES + (ATTRIBUTE,):
            raise ValueError("BodyGen design stages are not in native order")


@dataclass(frozen=True)
class DesignBatchTrace:
    """A batch of independent immutable design traces."""

    episodes: tuple[DesignTrace, ...]

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("a design batch needs at least one episode")

    def __len__(self) -> int:
        return len(self.episodes)

    def select(self, index: int) -> DesignTrace:
        return self.episodes[index]

    @classmethod
    def concatenate(
        cls,
        batches: Iterable["DesignBatchTrace"],
    ) -> "DesignBatchTrace":
        return cls(
            tuple(
                episode
                for batch in batches
                for episode in batch.episodes
            )
        )
