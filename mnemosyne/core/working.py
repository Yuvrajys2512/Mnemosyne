from __future__ import annotations

from mnemosyne.types import Memory


class WorkingMemory:
    def __init__(self, token_budget: int = 1500) -> None:
        self.token_budget = token_budget

    def select(self, candidates: list[tuple[Memory, float]]) -> list[Memory]:
        raise NotImplementedError

    def format_for_prompt(self, memories: list[Memory]) -> str:
        raise NotImplementedError
