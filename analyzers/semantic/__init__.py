"""
analyzers/semantic — Semantic & Entity Analysis Layer (v2.0 Phase B)
====================================================================
Day 54: Semantic Entity Set Overlap Engine (EntityTracker).
Day 55: Local Cross-Encoder NLI Engine (NLIEngine).
"""

from analyzers.semantic.entity_tracker import (
    EntityDeltaReport,
    EntityTracker,
    compare_entity_sets,
    extract_entities,
)

__all__ = [
    "EntityTracker",
    "EntityDeltaReport",
    "extract_entities",
    "compare_entity_sets",
]
