from __future__ import annotations

import time

from mnemosyne.types import EpisodicEvent, EventType, FactType, Memory, SemanticFact


def _estimate_tokens(text: str) -> int:
    """~4 chars per token for English prose, with a 20% buffer."""
    return int(len(text) / 4 * 1.2)


def _format_age(timestamp: float) -> str:
    age = time.time() - timestamp
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age / 60)}m ago"
    if age < 86_400:
        return f"{int(age / 3600)}h ago"
    return f"{int(age / 86_400)}d ago"


class WorkingMemory:
    def __init__(self, token_budget: int = 1_500) -> None:
        self.token_budget = token_budget

    def select(
        self,
        candidates: list[Memory],
        token_budget: int | None = None,
    ) -> list[Memory]:
        """Greedy selection: take highest-ranked memories that fit the budget."""
        if not candidates:
            return []

        budget = token_budget if token_budget is not None else self.token_budget
        selected: list[Memory] = []
        used = 0

        for memory in candidates:
            cost = _estimate_tokens(memory.content)
            if used + cost > budget:
                continue
            selected.append(memory)
            used += cost

        return selected

    def format_for_prompt(self, memories: list[Memory]) -> str:
        """Format selected memories into a context block for the system prompt."""
        if not memories:
            return ""

        sections: dict[str, list[Memory]] = {
            "preferences": [],
            "project_context": [],
            "facts": [],
            "recent": [],
        }

        for m in memories:
            if isinstance(m, SemanticFact):
                if m.fact_type in (FactType.USER_PREFERENCE, FactType.USER_BACKGROUND):
                    sections["preferences"].append(m)
                elif m.fact_type in (FactType.PROJECT_CONTEXT, FactType.TASK_CONSTRAINT):
                    sections["project_context"].append(m)
                else:
                    sections["facts"].append(m)
            else:
                sections["recent"].append(m)

        lines = ["=== Memory Context ==="]

        if sections["preferences"]:
            lines.append("\n[User Preferences]")
            for m in sections["preferences"]:
                lines.append(f"• {m.content}")

        if sections["project_context"]:
            lines.append("\n[Project Context]")
            for m in sections["project_context"]:
                lines.append(f"• {m.content}")

        if sections["facts"]:
            lines.append("\n[Established Facts]")
            for m in sections["facts"]:
                assert isinstance(m, SemanticFact)
                lines.append(f"• {m.content}")

        if sections["recent"]:
            lines.append("\n[Recent Context]")
            sorted_recent = sorted(sections["recent"], key=lambda e: e.timestamp)
            for m in sorted_recent:
                assert isinstance(m, EpisodicEvent)
                age = _format_age(m.timestamp)
                prefix = _event_prefix(m.event_type)
                # Truncate long events to keep the context block readable
                snippet = m.content[:300] + ("…" if len(m.content) > 300 else "")
                lines.append(f"• [{age}] {prefix}{snippet}")

        lines.append("\n======================")
        return "\n".join(lines)


def _event_prefix(event_type: EventType) -> str:
    return {
        EventType.USER_MESSAGE:   "User: ",
        EventType.AGENT_RESPONSE: "Agent: ",
        EventType.TOOL_CALL:      "Tool call: ",
        EventType.TOOL_RESULT:    "Tool result: ",
        EventType.OBSERVATION:    "Observation: ",
        EventType.SYSTEM:         "",
    }.get(event_type, "")
