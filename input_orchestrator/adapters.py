from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from .models import (
    Evidence,
    FieldObservation,
    FieldState,
    PendingQuestion,
)


def _state(value: Any) -> FieldState:
    if isinstance(value, FieldState):
        return value
    if value is None:
        return FieldState.UNRESOLVED
    raw = str(value).upper()
    aliases = {
        "VERIFIED": FieldState.VERIFIED,
        "RESOLVED": FieldState.VERIFIED,
        "NO_EVIDENCE": FieldState.MISSING,
        "MISSING": FieldState.MISSING,
        "PARTIAL": FieldState.PARTIAL,
        "UNRESOLVED": FieldState.UNRESOLVED,
        "ABSTAIN": FieldState.UNRESOLVED,
        "CONFLICT": FieldState.CONFLICT,
        "DECLINED": FieldState.DECLINED,
    }
    return aliases.get(raw, FieldState.UNRESOLVED)


@dataclass
class CallableExtractorAdapter:
    """
    Adapter for an existing extractor callable.

    `call` may return:
      - a dict with field names directly,
      - an object exposing to_dict(),
      - a dict containing `requirement_fields`,
      - detailed per-field dicts with value/status/source/evidence/confidence.

    This keeps the orchestrator independent from internal extractor classes.
    """
    domain: str
    call: Callable[..., Any]
    allowed_fields: Iterable[str]

    def extract(
        self,
        text: str,
        *,
        message_id: str,
        pending_question: Optional[PendingQuestion] = None,
    ) -> List[FieldObservation]:
        kwargs: Dict[str, Any] = {}

        if pending_question is not None:
            kwargs["pending_question"] = pending_question

        try:
            raw = self.call(text, **kwargs)
        except TypeError:
            # Existing production extractors may not accept orchestrator context.
            raw = self.call(text)

        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()

        if not isinstance(raw, dict):
            return []

        observations: List[FieldObservation] = []
        requirement_fields = raw.get("requirement_fields", {})
        if not isinstance(requirement_fields, dict):
            requirement_fields = {}

        for field_name in self.allowed_fields:
            detail = raw.get(field_name)

            if isinstance(detail, dict):
                value = detail.get("value")
                state = _state(detail.get("status"))
                source = str(detail.get("source") or self.domain)
                evidence = Evidence(
                    text=detail.get("evidence"),
                    source=source,
                    confidence=detail.get("semantic_confidence")
                    or detail.get("confidence"),
                )
                observations.append(
                    FieldObservation(
                        field=field_name,
                        value=value,
                        state=state,
                        source=source,
                        evidence=evidence,
                        message_id=message_id,
                    )
                )
                continue

            if field_name in requirement_fields:
                value = requirement_fields[field_name]
                if value is not None:
                    observations.append(
                        FieldObservation(
                            field=field_name,
                            value=value,
                            state=FieldState.VERIFIED,
                            source=self.domain,
                            evidence=Evidence(source=self.domain),
                            message_id=message_id,
                        )
                    )
                continue

            if field_name in raw and raw[field_name] is not None:
                observations.append(
                    FieldObservation(
                        field=field_name,
                        value=raw[field_name],
                        state=FieldState.VERIFIED,
                        source=self.domain,
                        evidence=Evidence(source=self.domain),
                        message_id=message_id,
                    )
                )

        return observations
