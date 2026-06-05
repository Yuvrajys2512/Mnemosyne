from __future__ import annotations

import os
import time

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from mnemosyne.types import FactType, SemanticFact

_RESERVED_META_KEYS = {
    "session_id", "fact_type", "created_at", "last_updated",
    "importance", "confidence", "access_count",
    "source_episode_ids", "contradicted_by", "supersedes",
}

# Similarity thresholds for upsert decisions
_REINFORCE_THRESHOLD = 0.92   # nearly identical → reinforce
_MERGE_THRESHOLD     = 0.85   # similar but different → merge/update


class SemanticStore:
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
            chroma_path = os.path.join(storage_path, "chroma_semantic")
            os.makedirs(chroma_path, exist_ok=True)
            client = chromadb.PersistentClient(path=chroma_path)

        ef = _embedding_function or SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self._collection = client.get_or_create_collection(
            name="semantic",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(self, fact: SemanticFact) -> str:
        self._collection.add(
            ids=[fact.id],
            documents=[fact.content],
            metadatas=[_to_meta(fact)],
        )
        return fact.id

    def update(self, id_: str, **fields) -> None:
        """Update specific metadata fields (and optionally the document text)."""
        documents = None
        meta_updates: dict = {}

        for k, v in fields.items():
            if k == "content":
                documents = [v]
            elif k == "source_episode_ids":
                meta_updates["source_episode_ids"] = ",".join(v) if v else ""
            elif k in ("contradicted_by", "supersedes"):
                meta_updates[k] = v or ""
            else:
                meta_updates[k] = v

        if meta_updates and documents:
            self._collection.update(ids=[id_], documents=documents, metadatas=[meta_updates])
        elif meta_updates:
            self._collection.update(ids=[id_], metadatas=[meta_updates])
        elif documents:
            self._collection.update(ids=[id_], documents=documents)

    def supersede(self, old_id: str, new_fact: SemanticFact) -> str:
        """Mark old_id as superseded and insert the new fact."""
        new_id = self.add(new_fact)
        self.update(old_id, contradicted_by=new_id)
        return new_id

    def upsert(self, fact: SemanticFact) -> str:
        """
        Smart insert: create, reinforce, or merge depending on similarity
        to existing facts.

        sim > 0.92  → reinforce (same fact confirmed again, boost importance)
        sim 0.85–0.92 → merge (similar fact, update to newer/more specific version)
        sim < 0.85  → insert as new fact
        """
        similar = self.find_similar(fact.content, threshold=_MERGE_THRESHOLD)

        if not similar:
            return self.add(fact)

        best, sim = similar[0]

        if sim >= _REINFORCE_THRESHOLD:
            self._reinforce(best)
            return best.id
        else:
            self._merge(best, fact)
            return best.id

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
    ) -> list[tuple[SemanticFact, float]]:
        total = self._collection.count()
        if total == 0:
            return []

        # Exclude superseded facts from retrieval
        base = {
            "$and": [
                {"session_id": self.session_id},
                {"contradicted_by": ""},  # "" is our sentinel for None
            ]
        }
        effective_where = {"$and": [base, where]} if where else base

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, total),
            where=effective_where,
        )
        return _parse_query(results)

    def find_similar(
        self,
        content: str,
        threshold: float = _MERGE_THRESHOLD,
    ) -> list[tuple[SemanticFact, float]]:
        """Return facts with cosine similarity ≥ threshold, sorted descending."""
        total = self._collection.count()
        if total == 0:
            return []

        results = self._collection.query(
            query_texts=[content],
            n_results=min(5, total),
            where={
                "$and": [
                    {"session_id": self.session_id},
                    {"contradicted_by": ""},
                ]
            },
        )
        pairs = _parse_query(results)
        return [(f, s) for f, s in pairs if s >= threshold]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reinforce(self, fact: SemanticFact) -> None:
        """Bump importance slightly when the same fact is confirmed again."""
        self.update(
            fact.id,
            importance=round(min(1.0, fact.importance + 0.05), 4),
            last_updated=time.time(),
            access_count=fact.access_count + 1,
        )

    def _merge(self, existing: SemanticFact, new_fact: SemanticFact) -> None:
        """Update to the newer/more specific version, keeping the better importance."""
        merged_importance = round(
            min(1.0, max(existing.importance, new_fact.importance) + 0.02), 4
        )
        self.update(
            existing.id,
            content=new_fact.content,
            last_updated=time.time(),
            importance=merged_importance,
        )


# ── Serialization helpers ─────────────────────────────────────────────────────

def _to_meta(fact: SemanticFact) -> dict:
    return {
        "session_id":         fact.session_id,
        "fact_type":          fact.fact_type.value,
        "created_at":         fact.created_at,
        "last_updated":       fact.last_updated,
        "importance":         fact.importance,
        "confidence":         fact.confidence,
        "access_count":       fact.access_count,
        "source_episode_ids": ",".join(fact.source_episode_ids),
        "contradicted_by":    fact.contradicted_by or "",
        "supersedes":         fact.supersedes or "",
        **{k: v for k, v in fact.metadata.items() if k not in _RESERVED_META_KEYS},
    }


def _build_fact(id_: str, doc: str, meta: dict) -> SemanticFact:
    raw_ids = meta.get("source_episode_ids", "")
    extra = {k: v for k, v in meta.items() if k not in _RESERVED_META_KEYS}
    return SemanticFact(
        id=id_,
        content=doc,
        session_id=meta["session_id"],
        fact_type=FactType(meta["fact_type"]),
        created_at=float(meta.get("created_at", 0.0)),
        last_updated=float(meta.get("last_updated", 0.0)),
        importance=float(meta.get("importance", 0.5)),
        confidence=float(meta.get("confidence", 0.8)),
        access_count=int(meta.get("access_count", 0)),
        source_episode_ids=[x for x in raw_ids.split(",") if x],
        contradicted_by=meta.get("contradicted_by") or None,
        supersedes=meta.get("supersedes") or None,
        metadata=extra,
    )


def _parse_query(results: dict) -> list[tuple[SemanticFact, float]]:
    return [
        (_build_fact(id_, doc, meta), 1.0 - distance / 2.0)
        for id_, doc, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
