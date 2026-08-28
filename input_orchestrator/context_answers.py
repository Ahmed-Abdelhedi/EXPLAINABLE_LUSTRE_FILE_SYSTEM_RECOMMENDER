from __future__ import annotations

import re
from typing import List, Optional

from .models import (
    Evidence,
    FieldObservation,
    FieldState,
    PendingQuestion,
)
from .ratio_parser import parse_read_write_ratio


YES = {
    "yes", "y", "oui", "yeah", "yep",
    "required", "mandatory", "obligatoire",
}
NO = {
    "no", "n", "non", "nope",
    "not required", "pas obligatoire", "pas necessaire",
}
SKIP = {
    "unknown", "i don't know", "dont know", "skip",
    "no constraint", "not specified", "aucune contrainte",
    "je ne sais pas", "passer",
}

PREFERENCE_LEVELS = {
    "very low": "VERY_LOW",
    "very_low": "VERY_LOW",
    "tres faible": "VERY_LOW",
    "très faible": "VERY_LOW",
    "not important at all": "VERY_LOW",
    "pas important du tout": "VERY_LOW",

    "low": "LOW",
    "faible": "LOW",
    "not important": "LOW",
    "unimportant": "LOW",
    "peu important": "LOW",
    "pas important": "LOW",

    "medium": "MEDIUM",
    "moderate": "MEDIUM",
    "moyen": "MEDIUM",
    "moyenne": "MEDIUM",

    "high": "HIGH",
    "important": "HIGH",
    "eleve": "HIGH",
    "élevé": "HIGH",

    "very high": "VERY_HIGH",
    "very_high": "VERY_HIGH",
    "critical": "VERY_HIGH",
    "essential": "VERY_HIGH",
    "tres eleve": "VERY_HIGH",
    "très élevé": "VERY_HIGH",

    "no signal": "NO_SIGNAL",
    "no preference": "NO_SIGNAL",
    "aucune preference": "NO_SIGNAL",
    "aucune préférence": "NO_SIGNAL",
}


def _compact(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.strip().lower().strip(" .!?"),
    )


def _observation(
    *,
    target: str,
    value,
    state: FieldState,
    text: str,
    message_id: str,
    explicit_correction: bool = False,
) -> FieldObservation:
    return FieldObservation(
        field=target,
        value=value,
        state=state,
        source="ORCHESTRATOR_CONTEXT",
        evidence=Evidence(
            text=text,
            source="ORCHESTRATOR_CONTEXT",
        ),
        message_id=message_id,
        explicit_correction=explicit_correction,
    )


def contextual_observation(
    text: str,
    *,
    pending_question: Optional[PendingQuestion],
    message_id: str,
) -> List[FieldObservation]:
    if pending_question is None:
        return []

    compact = _compact(text)
    target = pending_question.target_field
    expected = pending_question.expected_answer_type
    repair_correction = bool(
        pending_question.context.get(
            "validation_repair",
            False,
        )
    )

    if compact in SKIP:
        return [
            _observation(
                target=target,
                value=None,
                state=FieldState.DECLINED,
                text=text,
                message_id=message_id,
                explicit_correction=repair_correction,
            )
        ]

    if expected == "yes_no":
        if compact in YES:
            return [
                _observation(
                    target=target,
                    value=True,
                    state=FieldState.VERIFIED,
                    text=text,
                    message_id=message_id,
                    explicit_correction=repair_correction,
                )
            ]
        if compact in NO:
            return [
                _observation(
                    target=target,
                    value=False,
                    state=FieldState.VERIFIED,
                    text=text,
                    message_id=message_id,
                    explicit_correction=repair_correction,
                )
            ]

    if expected in {
        "positive_integer",
        "positive_integer_years",
    }:
        match = re.fullmatch(
            r"\s*(\d+)(?:\s*(?:years?|ans?))?\s*",
            text,
            flags=re.I,
        )
        if match:
            value = int(match.group(1))
            if value > 0:
                return [
                    _observation(
                        target=target,
                        value=value,
                        state=FieldState.VERIFIED,
                        text=text,
                        message_id=message_id,
                        explicit_correction=repair_correction,
                    )
                ]

    if expected == "ratio":
        ratio = parse_read_write_ratio(
            text,
            pending_ratio_question=True,
        )
        if ratio is not None:
            return [
                _observation(
                    target=target,
                    value=ratio,
                    state=FieldState.VERIFIED,
                    text=text,
                    message_id=message_id,
                    explicit_correction=repair_correction,
                )
            ]

    if expected == "access_type":
        mapping = {
            "sequential": "sequential",
            "sequentiel": "sequential",
            "séquentiel": "sequential",
            "random": "random",
            "aleatoire": "random",
            "aléatoire": "random",
            "mixed": "mixed",
            "mixte": "mixed",
        }
        if compact in mapping:
            return [
                _observation(
                    target=target,
                    value=mapping[compact],
                    state=FieldState.VERIFIED,
                    text=text,
                    message_id=message_id,
                    explicit_correction=repair_correction,
                )
            ]

    if expected == "preference":
        level = PREFERENCE_LEVELS.get(compact)
        if level is not None:
            return [
                _observation(
                    target=target,
                    value=level,
                    state=FieldState.VERIFIED,
                    text=text,
                    message_id=message_id,
                    explicit_correction=repair_correction,
                )
            ]

    if expected == "conflict_choice":
        previous = pending_question.context.get("previous_value")
        new = pending_question.context.get("new_value")

        previous_tokens = {
            _compact(str(previous)),
            "previous", "old", "first",
            "precedent", "précédent",
        }
        new_tokens = {
            _compact(str(new)),
            "new", "latest", "last", "second",
            "nouveau", "nouvelle", "dernier",
            "derniere", "dernière",
        }

        if compact in previous_tokens:
            return [
                _observation(
                    target=target,
                    value=previous,
                    state=FieldState.VERIFIED,
                    text=text,
                    message_id=message_id,
                    explicit_correction=True,
                )
            ]

        if compact in new_tokens:
            return [
                _observation(
                    target=target,
                    value=new,
                    state=FieldState.VERIFIED,
                    text=text,
                    message_id=message_id,
                    explicit_correction=True,
                )
            ]

    return []
