from __future__ import annotations

import os

from mnemosyne.consolidator import ConsolidationResult, Consolidator
from mnemosyne.core.episodic import EpisodicStore
from mnemosyne.core.semantic import SemanticStore
from mnemosyne.core.working import WorkingMemory
from mnemosyne.scoring import ScoringEngine, ScoringWeights
from mnemosyne.types import EventType, Memory


class Mnemosyne:
    def __init__(
        self,
        session_id: str,
        storage_path: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        working_memory_tokens: int = 1500,
        consolidation_model: str = "claude-haiku-4-5-20251001",
        auto_consolidate_threshold: int = 10,
        scoring_weights: ScoringWeights | None = None,
        anthropic_api_key: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.storage_path = storage_path or os.path.expanduser(
            f"~/.mnemosyne/sessions/{session_id}"
        )

        self.episodic = EpisodicStore(session_id, self.storage_path)
        self.semantic = SemanticStore(session_id, self.storage_path)
        self.working = WorkingMemory(token_budget=working_memory_tokens)
        self.scorer = ScoringEngine(weights=scoring_weights)
        self._consolidator = Consolidator(
            session_id=session_id,
            model=consolidation_model,
        )
        self._auto_consolidate_threshold = auto_consolidate_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        event_type: EventType = EventType.USER_MESSAGE,
        importance: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        return self.episodic.add(
            content,
            event_type=event_type,
            importance=importance,
            metadata=metadata,
        )

    def recall(
        self,
        query: str,
        token_budget: int | None = None,
        return_raw: bool = False,
    ) -> str | list[Memory]:
        raise NotImplementedError

    def consolidate(self, force: bool = False) -> ConsolidationResult:
        return self._consolidator.consolidate()

    async def aconsolidate(self) -> ConsolidationResult:
        return await self._consolidator.aconsolidate()
