from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .field_registry import PREFERENCE_FIELDS
from .models import (
    Conflict,
    FieldObservation,
    FieldState,
    ObservationKind,
)
from .session_state import WorkingSessionState


ACCEPTABLE_UPDATE_STATES = {
    FieldState.VERIFIED,
    FieldState.PARTIAL,
    FieldState.UNRESOLVED,
    FieldState.DECLINED,
}


@dataclass
class MergeResult:
    updated_fields: List[str]
    conflicts: List[Conflict]


class ResultMerger:
    def merge(
        self,
        session: WorkingSessionState,
        observations: Iterable[FieldObservation],
    ) -> MergeResult:
        updated: List[str] = []
        conflicts: List[Conflict] = []
        value_observations = []
        preference_relation_changed = False

        for obs in observations:
            if obs.kind == ObservationKind.RELATION:
                relation = (
                    dict(obs.value)
                    if isinstance(obs.value, dict)
                    else {}
                )
                relation["message_id"] = obs.message_id
                relation["source"] = obs.source
                before = len(session.preference_relations)
                session.add_preference_relation(relation)
                preference_relation_changed = (
                    preference_relation_changed
                    or len(session.preference_relations) > before
                )
                continue

            if obs.kind == ObservationKind.ABSTENTION:
                continue

            value_observations.append(obs)

        ranked = sorted(
            value_observations,
            key=lambda obs: {
                FieldState.VERIFIED: 0,
                FieldState.DECLINED: 1,
                FieldState.PARTIAL: 2,
                FieldState.CONFLICT: 3,
                FieldState.UNRESOLVED: 4,
                FieldState.MISSING: 5,
            }[obs.state],
        )

        seen_verified_in_message = set()

        for obs in ranked:
            record = session.get(obs.field)

            if (
                obs.field in seen_verified_in_message
                and obs.state != FieldState.VERIFIED
            ):
                continue

            if obs.state == FieldState.MISSING:
                continue

            if (
                record.state == FieldState.VERIFIED
                and obs.state == FieldState.VERIFIED
                and record.value != obs.value
            ):
                if obs.explicit_correction:
                    self._apply(record, obs)
                    updated.append(obs.field)
                    seen_verified_in_message.add(obs.field)
                    continue

                conflict = Conflict(
                    field=obs.field,
                    previous_value=record.value,
                    new_value=obs.value,
                    previous_message_id=record.message_id,
                    new_message_id=obs.message_id,
                    reason=(
                        "NEW_VERIFIED_VALUE_CONTRADICTS_"
                        "EXISTING_VERIFIED_VALUE"
                    ),
                )
                conflicts.append(conflict)
                record.history.append(record.snapshot())
                record.state = FieldState.CONFLICT
                continue

            if (
                record.state == FieldState.VERIFIED
                and obs.state in {
                    FieldState.UNRESOLVED,
                    FieldState.PARTIAL,
                }
            ):
                continue

            if obs.state in ACCEPTABLE_UPDATE_STATES:
                self._apply(record, obs)
                updated.append(obs.field)
                if obs.state == FieldState.VERIFIED:
                    seen_verified_in_message.add(obs.field)

        preference_value_changed = any(
            field_name in PREFERENCE_FIELDS
            for field_name in updated
        )

        if (
            preference_value_changed
            or preference_relation_changed
        ):
            session.reset_bwm_dialogue()

        return MergeResult(
            updated_fields=sorted(set(updated)),
            conflicts=conflicts,
        )

    @staticmethod
    def _apply(record, obs: FieldObservation) -> None:
        if record.revision > 0 or record.state != FieldState.MISSING:
            record.history.append(record.snapshot())

        record.value = obs.value
        record.state = obs.state
        record.source = obs.source
        record.evidence = obs.evidence.text
        record.confidence = obs.evidence.confidence
        record.message_id = obs.message_id
        record.revision += 1
