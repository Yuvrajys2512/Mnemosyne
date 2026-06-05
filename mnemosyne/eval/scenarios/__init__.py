"""All evaluation scenarios, grouped by dimension."""
from .consolidation import SCENARIOS as CONSOLIDATION_SCENARIOS
from .edge_cases import SCENARIOS as EDGE_CASE_SCENARIOS
from .recall_accuracy import SCENARIOS as RECALL_SCENARIOS
from .retention import SCENARIOS as RETENTION_SCENARIOS

ALL_SCENARIOS = (
    RECALL_SCENARIOS
    + CONSOLIDATION_SCENARIOS
    + RETENTION_SCENARIOS
    + EDGE_CASE_SCENARIOS
)

__all__ = [
    "ALL_SCENARIOS",
    "RECALL_SCENARIOS",
    "CONSOLIDATION_SCENARIOS",
    "RETENTION_SCENARIOS",
    "EDGE_CASE_SCENARIOS",
]
