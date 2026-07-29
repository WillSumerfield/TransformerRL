"""Population-grouped VSim environment for Neural Graph Evolution training."""
from __future__ import annotations

from benchmarks.nge.graph import NGEGraph

from .ant_multimorph import AntMultiMorphEnv
from .build_vsim import Morphology


def morphology_from_graph(graph: NGEGraph) -> Morphology:
    effectors = {
        limb + 1: list(chain)
        for limb, chain in enumerate(graph.effectors)
        if chain
    }
    caps = {
        limb + 1: graph.caps[limb]
        for limb, chain in enumerate(graph.effectors)
        if chain
    }
    return Morphology.from_design(effectors, caps)


class AntNGEEnv(AntMultiMorphEnv):
    """One morphology group per species and equal environments per group."""

    def __init__(self, num_envs, device, *, graphs: list[NGEGraph], **kwargs):
        if not graphs:
            raise ValueError("NGE needs a non-empty graph population")
        if num_envs % len(graphs):
            raise ValueError(
                "parallel environments must divide evenly across NGE species"
            )
        kwargs["sample_morphs"] = False
        kwargs["morphologies"] = [morphology_from_graph(graph) for graph in graphs]
        super().__init__(num_envs, device, **kwargs)
