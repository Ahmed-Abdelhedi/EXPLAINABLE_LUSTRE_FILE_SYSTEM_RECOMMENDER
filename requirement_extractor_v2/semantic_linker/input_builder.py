from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..models import Quantity, QuantityDimension

from .compatibility import allowed_fields_for_dimension
from .labels import SemanticField


Q_OPEN = "[Q]"
Q_CLOSE = "[/Q]"


@dataclass(frozen=True)
class SemanticLinkerInput:
    """
    Immutable input contract prepared for one Semantic Linker prediction.

    One unresolved Quantity produces one SemanticLinkerInput.

    The full original user message is preserved. Only the target quantity
    is surrounded by the internal [Q]...[/Q] markers.
    """

    quantity_id: str
    original_text: str
    marked_text: str

    raw_quantity: str
    normalized_quantity: str
    value: object
    unit: Optional[str]
    dimension: QuantityDimension
    corrected: bool

    previous_question: Optional[str] = None
    candidate_fields: Tuple[SemanticField, ...] = ()

    def to_dict(self) -> dict:
        return {
            "quantity_id": self.quantity_id,
            "original_text": self.original_text,
            "marked_text": self.marked_text,
            "raw_quantity": self.raw_quantity,
            "normalized_quantity": self.normalized_quantity,
            "value": self.value,
            "unit": self.unit,
            "dimension": self.dimension.value,
            "corrected": self.corrected,
            "previous_question": self.previous_question,
            "candidate_fields": [
                field.value for field in self.candidate_fields
            ],
        }


class SemanticLinkerInputBuilder:
    """
    Build the exact input consumed later by the Transformer.

    Responsibilities
    ----------------
    - preserve the complete original user message;
    - mark exactly ONE target quantity with [Q]...[/Q];
    - preserve the raw evidence written by the user;
    - expose deterministic Quantity metadata;
    - expose the C2 candidate FIELD mask;
    - optionally carry the previous assistant question.

    Non-responsibilities
    --------------------
    - no sentence splitting;
    - no clause extraction;
    - no field prediction;
    - no role prediction;
    - no normalization of the whole user message;
    - no LLM call.
    """

    def build(
        self,
        text: str,
        quantity: Quantity,
        previous_question: Optional[str] = None,
    ) -> SemanticLinkerInput:
        self._validate_target(text, quantity)

        marked_text = self.mark_target_quantity(
            text=text,
            quantity=quantity,
        )

        candidate_fields = allowed_fields_for_dimension(
            quantity.dimension
        )

        return SemanticLinkerInput(
            quantity_id=quantity.id,
            original_text=text,
            marked_text=marked_text,
            raw_quantity=quantity.raw,
            normalized_quantity=quantity.normalized,
            value=quantity.value,
            unit=quantity.unit,
            dimension=quantity.dimension,
            corrected=quantity.corrected,
            previous_question=previous_question,
            candidate_fields=candidate_fields,
        )

    def mark_target_quantity(
        self,
        text: str,
        quantity: Quantity,
    ) -> str:
        """
        Mark exactly the target Quantity using its ORIGINAL offsets.

        Example
        -------
        text:
            "Need 500 TiB for two hunderd hosts."

        target:
            raw="two hunderd"

        output:
            "Need 500 TiB for [Q]two hunderd[/Q] hosts."

        The normalized form ("two hundred") is metadata only. It does not
        replace the user's original evidence inside marked_text.
        """
        self._validate_target(text, quantity)

        return (
            text[: quantity.start]
            + Q_OPEN
            + text[quantity.start : quantity.end]
            + Q_CLOSE
            + text[quantity.end :]
        )

    @staticmethod
    def _validate_target(
        text: str,
        quantity: Quantity,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text:
            raise ValueError("text must not be empty")

        if quantity.start < 0:
            raise ValueError(
                f"{quantity.id}: start offset must be >= 0"
            )

        if quantity.end <= quantity.start:
            raise ValueError(
                f"{quantity.id}: end offset must be greater than start"
            )

        if quantity.end > len(text):
            raise ValueError(
                f"{quantity.id}: end offset {quantity.end} exceeds "
                f"text length {len(text)}"
            )

        actual_raw = text[quantity.start : quantity.end]

        if actual_raw != quantity.raw:
            raise ValueError(
                f"{quantity.id}: Quantity.raw does not match the "
                f"original text span. expected={quantity.raw!r}, "
                f"actual={actual_raw!r}"
            )

        # [Q] and [/Q] are reserved internal tokens that will later be
        # added to the Transformer tokenizer. Rejecting pre-existing
        # markers prevents an input from containing multiple apparent
        # targets.
        if Q_OPEN in text or Q_CLOSE in text:
            raise ValueError(
                "The original text already contains reserved Semantic "
                "Linker markers [Q] or [/Q]"
            )