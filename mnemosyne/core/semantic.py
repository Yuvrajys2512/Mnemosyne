from __future__ import annotations

from mnemosyne.types import SemanticFact


class SemanticStore:
    def __init__(self, session_id: str, storage_path: str) -> None:
        self.session_id = session_id
        self.storage_path = storage_path

    def add(self, fact: SemanticFact) -> str:
        raise NotImplementedError

    def query(
        self,
        query_text: str,
        n_results: int = 20,
        where: dict | None = None,
    ) -> list[SemanticFact]:
        raise NotImplementedError

    def find_similar(self, content: str, threshold: float = 0.85) -> list[SemanticFact]:
        raise NotImplementedError

    def update(self, id: str, **fields) -> None:
        raise NotImplementedError

    def supersede(self, old_id: str, new_fact: SemanticFact) -> str:
        raise NotImplementedError
