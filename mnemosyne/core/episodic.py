from __future__ import annotations

import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from mnemosyne.types import EpisodicEvent, EventType

# Short responses that carry no information worth storing
_TRIVIAL = {
    "ok", "okay", "yes", "no", "sure", "thanks", "thank you",
    "got it", "sounds good", "great", "nice", "cool", "alright",
}

_PREFERENCE_KW = ["prefer", "always", "never", "must", "hate", "love",
                  "like", "dislike", "requirement", "constraint"]
_CORRECTION_KW = ["wrong", "incorrect", "actually", "correction",
                  "fix", "mistake", "not right"]

# Known ChromaDB metadata keys — stripped from the event.metadata dict
_RESERVED_META_KEYS = {
    "session_id", "event_type", "timestamp",
    "importance", "consolidated", "access_count",
}


class EpisodicStore:
    def __init__(
        self,
        session_id: str,
        storage_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        _ephemeral: bool = False,
        _embedding_function=None,
    ) -> None:
        self.session_id = session_id

        if _ephemeral:
            client = chromadb.EphemeralClient()
        else:
            chroma_path = os.path.join(storage_path, "chroma")
            os.makedirs(chroma_path, exist_ok=True)
            client = chromadb.PersistentClient(path=chroma_path)

        ef = _embedding_function or SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self._collection = client.get_or_create_collection(
            name="episodic",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        event_type: EventType = EventType.USER_MESSAGE,
        importance: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        stripped = content.strip()
        if len(stripped) < 3 or stripped.lower() in _TRIVIAL:
            return ""

        extra: dict = {}
        if len(content) > 2_000:
            extra["truncated"] = True
            content = content[:2_000]

        event = EpisodicEvent(
            content=content,
            session_id=self.session_id,
            event_type=event_type,
            importance=importance if importance is not None
                       else _infer_importance(content, event_type),
        )

        self._collection.add(
            ids=[event.id],
            documents=[event.content],
            metadatas=[{
                "session_id": event.session_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp,
                "importance": event.importance,
                "consolidated": False,
                "access_count": 0,
                **extra,
                **(metadata or {}),
            }],
        )
        return event.id

    def mark_consolidated(self, ids: list[str]) -> None:
        if not ids:
            return
        self._collection.update(
            ids=ids,
            metadatas=[{"consolidated": True} for _ in ids],
        )

    def increment_access(self, ids: list[str], counts: list[int]) -> None:
        if not ids:
            return
        self._collection.update(
            ids=ids,
            metadatas=[{"access_count": c} for c in counts],
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 20,
        where: dict | None = None,
    ) -> list[tuple[EpisodicEvent, float]]:
        total = self._collection.count()
        if total == 0:
            return []

        base = {"session_id": self.session_id}
        effective_where = {"$and": [base, where]} if where else base

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, total),
            where=effective_where,
        )
        return _parse_query(results)  # returns list[tuple[EpisodicEvent, float]]

    def get_recent(self, n: int = 5) -> list[EpisodicEvent]:
        results = self._collection.get(
            where={"session_id": self.session_id},
            include=["documents", "metadatas"],
        )
        events = _parse_get(results)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:n]

    def get_unconsolidated(self, limit: int = 20) -> list[EpisodicEvent]:
        results = self._collection.get(
            where={"$and": [
                {"session_id": self.session_id},
                {"consolidated": False},
            ]},
            include=["documents", "metadatas"],
            limit=limit,
        )
        events = _parse_get(results)
        events.sort(key=lambda e: e.timestamp)
        return events

    def count_unconsolidated(self) -> int:
        results = self._collection.get(
            where={"$and": [
                {"session_id": self.session_id},
                {"consolidated": False},
            ]},
            include=[],
        )
        return len(results["ids"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_importance(content: str, event_type: EventType) -> float:
    base = 0.5
    lower = content.lower()

    if any(kw in lower for kw in _PREFERENCE_KW):
        base += 0.2
    if any(kw in lower for kw in _CORRECTION_KW):
        base += 0.15
    if event_type == EventType.AGENT_RESPONSE:
        base -= 0.1
    if event_type == EventType.TOOL_RESULT:
        base -= 0.2

    return round(max(0.0, min(1.0, base)), 4)


def _build_event(id_: str, doc: str, meta: dict) -> EpisodicEvent:
    extra = {k: v for k, v in meta.items() if k not in _RESERVED_META_KEYS}
    return EpisodicEvent(
        id=id_,
        content=doc,
        session_id=meta["session_id"],
        event_type=EventType(meta["event_type"]),
        timestamp=float(meta["timestamp"]),
        importance=float(meta.get("importance", 0.5)),
        consolidated=bool(meta.get("consolidated", False)),
        access_count=int(meta.get("access_count", 0)),
        metadata=extra,
    )


def _parse_query(results: dict) -> list[tuple[EpisodicEvent, float]]:
    # ChromaDB cosine distance is in [0, 2]: 0=identical, 2=opposite.
    # Convert to similarity in [0, 1]: sim = 1 - distance / 2
    # (i.e. distance 0 → sim 1.0, distance 2 → sim 0.0)
    return [
        (_build_event(id_, doc, meta), 1.0 - distance / 2.0)
        for id_, doc, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def _parse_get(results: dict) -> list[EpisodicEvent]:
    return [
        _build_event(id_, doc, meta)
        for id_, doc, meta in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        )
    ]
