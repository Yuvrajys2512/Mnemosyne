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
        working_memory_tokens: int = 1_500,
        consolidation_model: str = "claude-haiku-4-5-20251001",
        auto_consolidate_threshold: int = 10,
        scoring_weights: ScoringWeights | None = None,
        anthropic_api_key: str | None = None,
        # Private params used in tests — prefixed with _ to signal non-public use
        _ephemeral: bool = False,
        _embedding_function=None,
    ) -> None:
        self.session_id = session_id
        self.storage_path = storage_path or os.path.expanduser(
            f"~/.mnemosyne/sessions/{session_id}"
        )

        self.episodic = EpisodicStore(
            session_id=session_id,
            storage_path=self.storage_path,
            embedding_model=embedding_model,
            _ephemeral=_ephemeral,
            _embedding_function=_embedding_function,
        )
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
        # 1. Retrieve candidates with their cosine similarities from ChromaDB
        candidates_with_sim = self.episodic.query(query, n_results=30)

        # 2. Re-rank using the composite score (recency + importance + frequency)
        ranked = self.scorer.rank(candidates_with_sim)

        # 3. Greedy selection within token budget (highest-scored first)
        memories = [m for m, _ in ranked]
        selected = self.working.select(
            memories,
            token_budget=token_budget or self.working.token_budget,
        )

        # 4. Increment access counts for everything we surface
        if selected:
            ids = [m.id for m in selected]
            counts = [m.access_count + 1 for m in selected]
            self.episodic.increment_access(ids, counts)

        if return_raw:
            return selected
        return self.working.format_for_prompt(selected)

    def consolidate(self, force: bool = False) -> ConsolidationResult:
        return self._consolidator.consolidate()

    async def aconsolidate(self) -> ConsolidationResult:
        return await self._consolidator.aconsolidate()
