from __future__ import annotations

from mnemosyne.types import EpisodicEvent, EventType


class EpisodicStore:
    def __init__(self, session_id: str, storage_path: str) -> None:
        self.session_id = session_id
        self.storage_path = storage_path

    def add(
        self,
        content: str,
        event_type: EventType = EventType.USER_MESSAGE,
        importance: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        raise NotImplementedError

    def query(
        self,
        query_text: str,
        n_results: int = 20,
        where: dict | None = None,
    ) -> list[EpisodicEvent]:
        raise NotImplementedError

    def get_recent(self, n: int = 5) -> list[EpisodicEvent]:
        raise NotImplementedError

    def get_unconsolidated(self, limit: int = 20) -> list[EpisodicEvent]:
        raise NotImplementedError

    def mark_consolidated(self, ids: list[str]) -> None:
        raise NotImplementedError

    def count_unconsolidated(self) -> int:
        raise NotImplementedError
