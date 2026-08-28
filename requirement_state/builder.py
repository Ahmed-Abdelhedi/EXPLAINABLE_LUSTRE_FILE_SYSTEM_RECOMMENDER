from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .contract import REQUIREMENT_FIELDS
from .models import (
    FinalRequirementState,
    PreferenceWeights,
    RequirementFieldStatus,
    RequirementFieldTrace,
)


class RequirementStateBuilder:
    """
    Step 6.2A schema builder.

    This component only translates an already-collected orchestrator session
    into the typed FinalRequirementState schema.

    It intentionally does NOT:
      - resolve new conflicts,
      - invent missing values,
      - run BWM,
      - perform sizing,
      - decide the final validation policy.

    Those behaviors belong to Step 6.2B.
    """

    @staticmethod
    def _status(value: Any) -> RequirementFieldStatus:
        raw = getattr(value, "value", value)
        return RequirementFieldStatus(str(raw))

    def from_session(
        self,
        session,
        *,
        bwm_metadata: Optional[Mapping[str, Any]] = None,
    ) -> FinalRequirementState:
        state = FinalRequirementState()

        missing = []
        unresolved = []

        for field_name in REQUIREMENT_FIELDS:
            record = session.get(field_name)
            status = self._status(record.state)

            trace = RequirementFieldTrace(
                field=field_name,
                status=status,
                value=record.value,
                source=record.source,
                evidence=record.evidence,
                confidence=record.confidence,
                message_id=record.message_id,
                revision=record.revision,
            )
            state.field_traces[field_name] = trace

            if status == RequirementFieldStatus.VERIFIED:
                setattr(
                    state,
                    field_name,
                    record.value,
                )
            elif status == RequirementFieldStatus.MISSING:
                missing.append(field_name)
            elif status in {
                RequirementFieldStatus.PARTIAL,
                RequirementFieldStatus.UNRESOLVED,
                RequirementFieldStatus.CONFLICT,
            }:
                unresolved.append(field_name)

        state.missing_fields = missing
        state.unresolved_fields = unresolved

        weights_record = session.get(
            "preference_weights"
        )

        weights_status = self._status(
            weights_record.state
        )

        state.field_traces[
            "preference_weights"
        ] = RequirementFieldTrace(
            field="preference_weights",
            status=weights_status,
            value=weights_record.value,
            source=weights_record.source,
            evidence=weights_record.evidence,
            confidence=weights_record.confidence,
            message_id=weights_record.message_id,
            revision=weights_record.revision,
        )

        if (
            weights_status
            == RequirementFieldStatus.VERIFIED
            and isinstance(
                weights_record.value,
                dict,
            )
        ):
            metadata: Dict[str, Any] = dict(
                bwm_metadata or {}
            )

            state.preference_weights = (
                PreferenceWeights.from_mapping(
                    weights_record.value,
                    method=metadata.get(
                        "method",
                        weights_record.source,
                    ),
                    xi_star=metadata.get(
                        "xi_star"
                    ),
                    consistency_status=metadata.get(
                        "consistency_status"
                    ),
                    source=metadata.get(
                        "source",
                        weights_record.source,
                    ),
                )
            )

        return state
