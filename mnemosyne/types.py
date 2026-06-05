from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    USER_MESSAGE   = "user_message"
    AGENT_RESPONSE = "agent_response"
    TOOL_CALL      = "tool_call"
    TOOL_RESULT    = "tool_result"
    OBSERVATION    = "observation"
    SYSTEM         = "system"


class FactType(str, Enum):
    USER_PREFERENCE  = "user_preference"
    USER_BACKGROUND  = "user_background"
    PROJECT_CONTEXT  = "project_context"
    TASK_CONSTRAINT  = "task_constraint"
    ESTABLISHED_FACT = "established_fact"
    DECISION         = "decision"
    OPEN_QUESTION    = "open_question"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class EpisodicEvent:
    content: str
    session_id: str
    event_type: EventType
    id: str = field(default_factory=lambda: _new_id("ep"))
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    consolidated: bool = False
    access_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class SemanticFact:
    content: str
    session_id: str
    fact_type: FactType
    id: str = field(default_factory=lambda: _new_id("sem"))
    source_episode_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    importance: float = 0.5
    confidence: float = 0.8
    access_count: int = 0
    contradicted_by: str | None = None
    supersedes: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def timestamp(self) -> float:
        """Uniform interface with EpisodicEvent — returns last_updated."""
        return self.last_updated


# Union type used throughout the scoring and working memory layers
Memory = EpisodicEvent | SemanticFact
