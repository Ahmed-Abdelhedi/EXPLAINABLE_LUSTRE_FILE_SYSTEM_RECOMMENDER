from __future__ import annotations

from typing import List, Optional, Protocol

from .models import FieldObservation, PendingQuestion


class ExtractorPort(Protocol):
    domain: str

    def extract(
        self,
        text: str,
        *,
        message_id: str,
        pending_question: Optional[PendingQuestion] = None,
    ) -> List[FieldObservation]:
        ...


class WeightingPort(Protocol):
    def is_ready(self, active_preferences: list[str]) -> bool:
        ...

    def next_question(self) -> Optional[str]:
        ...

    def consume_answer(self, text: str) -> None:
        ...

    def final_weights(self) -> Optional[dict[str, float]]:
        ...
