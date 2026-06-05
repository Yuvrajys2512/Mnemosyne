from __future__ import annotations

import asyncio
import logging
import os

from mnemosyne.consolidator import ConsolidationResult, Consolidator
from mnemosyne.core.episodic import EpisodicStore
from mnemosyne.core.semantic import SemanticStore
from mnemosyne.core.working import WorkingMemory
from mnemosyne.providers import build_providers
from mnemosyne.scoring import ScoringEngine, ScoringWeights
from mnemosyne.types import EventType, Memory

logger = logging.getLogger(__name__)


class Mnemosyne:
    def __init__(
        self,
        session_id: str,
        storage_path: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        working_memory_tokens: int = 1_500,
        # LLM providers for consolidation
        groq_api_key: str | None = None,
        groq_model: str = "llama-3.3-70b-versatile",
        ollama_model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434/v1",
        # Scoring
        scoring_weights: ScoringWeights | None = None,
        # Auto-consolidation
        auto_consolidate_threshold: int = 20,
        # Private — for tests only
        _ephemeral: bool = False,
        _embedding_function=None,
        _provider=None,      # inject a mock LLM provider in tests
    ) -> None:
        self._auto_consolidate_threshold = auto_consolidate_threshold
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
        self.semantic = SemanticStore(
            session_id=session_id,
            storage_path=self.storage_path,
            embedding_model=embedding_model,
            _ephemeral=_ephemeral,
            _embedding_function=_embedding_function,
        )
        self.working = WorkingMemory(token_budget=working_memory_tokens)
        self.scorer = ScoringEngine(weights=scoring_weights)

        if _provider is not None:
            primary, fallback = _provider, None
        else:
            primary, fallback = build_providers(
                groq_api_key=groq_api_key,
                groq_model=groq_model,
                ollama_model=ollama_model,
                ollama_base_url=ollama_base_url,
            )

        self._consolidator = Consolidator(
            session_id=session_id,
            episodic=self.episodic,
            semantic=self.semantic,
            provider=primary,
            fallback=fallback,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        event_type: EventType = EventType.USER_MESSAGE,
        importance: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        id_ = self.episodic.add(
            content,
            event_type=event_type,
            importance=importance,
            metadata=metadata,
        )
        if id_:
            self._maybe_auto_consolidate()
        return id_

    def _maybe_auto_consolidate(self) -> None:
        """Fire a background consolidation task when the threshold is hit.

        Only works inside a running asyncio event loop (e.g. a LangGraph node).
        In a purely synchronous context the user must call consolidate() manually.
        """
        if self.episodic.count_unconsolidated() < self._auto_consolidate_threshold:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.aconsolidate())
        except RuntimeError:
            pass  # no running loop — silent skip, user calls consolidate() manually

    def recall(
        self,
        query: str,
        token_budget: int | None = None,
        return_raw: bool = False,
    ) -> str | list[Memory]:
        try:
            return self._recall(query, token_budget=token_budget, return_raw=return_raw)
        except Exception:
            logger.exception("recall() failed — returning empty context to avoid crashing agent")
            return [] if return_raw else ""

    def _recall(
        self,
        query: str,
        token_budget: int | None = None,
        return_raw: bool = False,
    ) -> str | list[Memory]:
        # 1. Retrieve candidates from both memory layers
        episodic_candidates = self.episodic.query(query, n_results=20)
        semantic_candidates = self.semantic.query(query, n_results=20)
        all_candidates: list[tuple[Memory, float]] = episodic_candidates + semantic_candidates

        # 2. Re-rank with the composite score
        ranked = self.scorer.rank(all_candidates)

        # 3. Greedy token-budget selection
        memories = [m for m, _ in ranked]
        selected = self.working.select(
            memories,
            token_budget=token_budget or self.working.token_budget,
        )

        # 4. Increment access counts
        if selected:
            ep_ids = [m.id for m in selected if m.id.startswith("ep_")]
            ep_counts = [m.access_count + 1 for m in selected if m.id.startswith("ep_")]
            if ep_ids:
                self.episodic.increment_access(ep_ids, ep_counts)

            sem_ids = [m.id for m in selected if m.id.startswith("sem_")]
            sem_counts = [m.access_count + 1 for m in selected if m.id.startswith("sem_")]
            if sem_ids:
                self.semantic.increment_access(sem_ids, sem_counts)

        if return_raw:
            return selected
        return self.working.format_for_prompt(selected)

    def consolidate(self) -> ConsolidationResult:
        return self._consolidator.consolidate()

    async def aconsolidate(self) -> ConsolidationResult:
        return await self._consolidator.aconsolidate()
