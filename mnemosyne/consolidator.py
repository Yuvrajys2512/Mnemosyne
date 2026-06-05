from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from mnemosyne.types import EpisodicEvent, FactType, SemanticFact

logger = logging.getLogger(__name__)

# ── Extraction prompt ─────────────────────────────────────────────────────────

_SYSTEM = """You extract durable facts from conversation events for long-term agent memory.

Extract only facts that would still be useful in a future session. Focus on:
- User preferences and working style
- Project context (what is being built, key constraints)
- Decisions made and why
- Key facts established as ground truth

Rules:
- One fact per discrete piece of information
- Write as declarative statements: "User prefers X", "Project uses Y"
- importance: a single decimal number between 0.0 and 1.0 (e.g. 0.9). Use 0.9 for core preferences, 0.5 for general context, 0.3 for passing mentions. If the same fact repeats in the batch, use a higher value such as 0.95.
- confidence: a single decimal number between 0.0 and 1.0. Use 0.9 if explicitly stated, 0.6 if inferred, 0.3 if speculative.
- SKIP: filler ("ok", "thanks"), one-off task instructions, time-bound information
- IMPORTANT: importance and confidence must be plain numbers like 0.9 — never arithmetic expressions like 0.9 + 0.1.

fact_type must be one of:
  user_preference, user_background, project_context, task_constraint,
  established_fact, decision, open_question

Respond with ONLY valid JSON — no markdown fences, no commentary:
{"facts": [{"content": "...", "fact_type": "...", "importance": 0.0, "confidence": 0.0}]}
If nothing is worth remembering, return: {"facts": []}"""

_VALID_FACT_TYPES = {ft.value for ft in FactType}

# ── Result dataclass ──────────────────────────────────────────────────────────

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


# ── Consolidator ──────────────────────────────────────────────────────────────

class Consolidator:
    def __init__(
        self,
        session_id: str,
        episodic=None,    # EpisodicStore — injected by Mnemosyne
        semantic=None,    # SemanticStore — injected by Mnemosyne
        provider=None,    # LLMProvider (Groq / Ollama)
        fallback=None,    # LLMProvider used if primary fails
    ) -> None:
        self.session_id = session_id
        self._episodic = episodic
        self._semantic = semantic
        self._provider = provider
        self._fallback = fallback

    def consolidate(self) -> ConsolidationResult:
        if not self._provider:
            logger.warning("No LLM provider configured — skipping consolidation.")
            return ConsolidationResult()

        events = self._episodic.get_unconsolidated(limit=20)
        if not events:
            return ConsolidationResult()

        t_start = time.time()
        result = ConsolidationResult()

        for batch in _batch_by_time(events, window_seconds=600):
            try:
                facts, calls = self._extract(batch)
                result.llm_calls_made += calls
            except Exception as e:
                result.errors.append(f"extraction failed: {e}")
                logger.error("Consolidation extraction failed: %s", e)
                continue

            for fact in facts:
                existing_count = self._semantic._collection.count()
                self._semantic.upsert(fact)
                new_count = self._semantic._collection.count()
                if new_count > existing_count:
                    result.facts_created += 1
                else:
                    result.facts_updated += 1

            self._episodic.mark_consolidated([e.id for e in batch])
            result.episodes_processed += len(batch)

        result.duration_seconds = round(time.time() - t_start, 3)
        return result

    async def aconsolidate(self) -> ConsolidationResult:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.consolidate)

    # ── private ───────────────────────────────────────────────────────────────

    def _extract(self, batch: list[EpisodicEvent]) -> tuple[list[SemanticFact], int]:
        """Call the LLM (with fallback) and parse the JSON response."""
        user_prompt = "Events:\n" + _format_events(batch)

        raw = self._call_with_fallback(_SYSTEM, user_prompt)
        facts = _parse_response(raw, session_id=self.session_id)
        return facts, 1

    def _call_with_fallback(self, system: str, user: str) -> str:
        try:
            return self._provider.complete(system, user)
        except Exception as e:
            if self._fallback:
                logger.warning(
                    "Primary provider (%s) failed: %s — trying fallback (%s).",
                    self._provider.name, e, self._fallback.name,
                )
                return self._fallback.complete(system, user)
            raise


# ── Helpers ───────────────────────────────────────────────────────────────────

def _batch_by_time(
    events: list[EpisodicEvent],
    window_seconds: float,
) -> list[list[EpisodicEvent]]:
    """Group events that occurred within `window_seconds` of each other."""
    if not events:
        return []

    batches: list[list[EpisodicEvent]] = []
    current: list[EpisodicEvent] = [events[0]]

    for event in events[1:]:
        if event.timestamp - current[0].timestamp <= window_seconds:
            current.append(event)
        else:
            batches.append(current)
            current = [event]

    batches.append(current)
    return batches


def _format_events(events: list[EpisodicEvent]) -> str:
    lines = []
    for i, e in enumerate(events, 1):
        prefix = e.event_type.value.replace("_", " ").title()
        lines.append(f"{i}. [{prefix}] {e.content}")
    return "\n".join(lines)


def _parse_response(raw: str, session_id: str) -> list[SemanticFact]:
    """Parse the LLM JSON response into SemanticFact objects."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from inside the string (some models add commentary)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("Could not parse JSON from LLM response: %s", raw[:200])
            return []
        try:
            data = json.loads(raw[start:end])
        except json.JSONDecodeError:
            logger.warning("JSON extraction failed from: %s", raw[:200])
            return []

    facts = []
    for item in data.get("facts", []):
        content = item.get("content", "").strip()
        if not content:
            continue

        raw_type = item.get("fact_type", "established_fact")
        fact_type = FactType(raw_type) if raw_type in _VALID_FACT_TYPES else FactType.ESTABLISHED_FACT

        importance = float(item.get("importance", 0.5))
        confidence = float(item.get("confidence", 0.8))

        facts.append(SemanticFact(
            content=content,
            session_id=session_id,
            fact_type=fact_type,
            importance=round(max(0.0, min(1.0, importance)), 4),
            confidence=round(max(0.0, min(1.0, confidence)), 4),
        ))

    return facts
