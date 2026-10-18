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
from analyzers.semantic.nli_engine import (
    NLIClaimScore,
    NLIEngine,
    NLIEvaluationReport,
    NLILabel,
    evaluate_pair_lexical_fallback,
    get_nli_engine,
    split_claims,
)

__all__ = [
    "EntityTracker",
    "EntityDeltaReport",
    "extract_entities",
    "compare_entity_sets",
    "NLIEngine",
    "NLILabel",
    "NLIClaimScore",
    "NLIEvaluationReport",
    "split_claims",
    "evaluate_pair_lexical_fallback",
    "get_nli_engine",
]
