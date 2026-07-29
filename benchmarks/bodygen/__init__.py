"""Faithful BodyGen baseline core.

Evaluator-facing imports are added beside these core exports by
``benchmarks.bodygen.method``; keeping the algorithmic pieces available here
makes training scripts and CPU tests independent of VSim's native libraries.
"""

from .credit import (
    CreditAssignment,
    ReturnNormalizer,
    enhanced_temporal_credit_assignment,
)
from .design import (
    ADD,
    ATTRIBUTE,
    CONTROL,
    DELETE,
    NO_CHANGE,
    TOPOLOGY,
    BodyGenDesign,
    DesignBatchTrace,
    DesignNode,
    DesignTrace,
    DesignTransition,
    apply_attribute_actions,
    apply_topology_actions,
    design_node_features,
    topology_id,
)
from .mosat import BodyGenNetworks, ControlOutput, MoSAT
from .method import BodyGenMethod, checkpoints_for_bodygen_run, load_bodygen

__all__ = [
    "ADD",
    "ATTRIBUTE",
    "CONTROL",
    "DELETE",
    "NO_CHANGE",
    "TOPOLOGY",
    "BodyGenDesign",
    "BodyGenMethod",
    "BodyGenNetworks",
    "ControlOutput",
    "CreditAssignment",
    "DesignBatchTrace",
    "DesignNode",
    "DesignTrace",
    "DesignTransition",
    "MoSAT",
    "ReturnNormalizer",
    "apply_attribute_actions",
    "apply_topology_actions",
    "checkpoints_for_bodygen_run",
    "design_node_features",
    "enhanced_temporal_credit_assignment",
    "load_bodygen",
    "topology_id",
]
