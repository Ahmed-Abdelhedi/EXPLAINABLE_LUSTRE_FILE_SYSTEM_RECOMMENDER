from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .models import FieldObservation, PendingQuestion
from .ports import ExtractorPort


@dataclass
class ExtractionRouter:
    extractors: Iterable[ExtractorPort]

    def route(
        self,
        text: str,
        *,
        message_id: str,
        pending_question: Optional[PendingQuestion],
    ) -> List[FieldObservation]:
        observations: List[FieldObservation] = []

        for extractor in self.extractors:
            observations.extend(
                extractor.extract(
                    text,
                    message_id=message_id,
                    pending_question=pending_question,
                )
            )

        return observations
