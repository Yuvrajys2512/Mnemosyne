from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConsolidationResult:
    episodes_processed: int = 0
    facts_created: int = 0
    facts_updated: int = 0
    facts_superseded: int = 0
    llm_calls_made: int = 0
    tokens_used: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class Consolidator:
    def __init__(self, session_id: str, model: str) -> None:
        self.session_id = session_id
        self.model = model

    def consolidate(self) -> ConsolidationResult:
        raise NotImplementedError

    async def aconsolidate(self) -> ConsolidationResult:
        raise NotImplementedError
