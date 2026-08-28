from __future__ import annotations

from typing import List, Optional

from .ratio_parser import parse_read_write_ratio

from .models import (
    Evidence,
    FieldObservation,
    FieldState,
    ObservationKind,
    PendingQuestion,
)


QUANTITY_FIELDS = {
    "requested_usable_capacity_tib",
    "client_count",
    "average_file_size_gb",
    "max_file_size_gb",
    "total_file_count",
    "read_write_ratio",
    "target_read_gbps",
    "target_write_gbps",
    "max_budget_usd",
    "max_power_w",
    "annual_growth_percent",
    "planning_horizon_years",
}

PREFERENCE_FIELD_BY_DIMENSION = {
    "cost": "cost_priority",
    "power": "power_priority",
    "performance": "performance_priority",
    "reliability": "reliability_priority",
}


class QuantityProductionAdapter:
    domain = "quantity"

    def __init__(self, pipeline=None) -> None:
        if pipeline is None:
            from requirement_extractor_v2.verified_pipeline import (
                VerifiedRequirementPipeline,
            )
            pipeline = VerifiedRequirementPipeline()

        self.pipeline = pipeline
        self.last_result = None

    def extract(
        self,
        text: str,
        *,
        message_id: str,
        pending_question: Optional[PendingQuestion] = None,
    ) -> List[FieldObservation]:
        previous_question = None
        previous_question_field = None
        requested_unit = None

        if (
            pending_question is not None
            and pending_question.target_field in QUANTITY_FIELDS
        ):
            from requirement_extractor_v2.models import ParamName

            previous_question = pending_question.question
            try:
                previous_question_field = ParamName(
                    pending_question.target_field
                )
            except ValueError:
                previous_question_field = None

            requested_unit = pending_question.context.get(
                "requested_unit"
            )

        pending_ratio = (
            pending_question is not None
            and pending_question.target_field == "read_write_ratio"
        )
        explicit_ratio = parse_read_write_ratio(
            text,
            pending_ratio_question=pending_ratio,
        )

        result = self.pipeline.process(
            text,
            previous_question=previous_question,
            previous_question_field=previous_question_field,
            requested_unit=requested_unit,
        )
        self.last_result = result

        correction = bool(
            pending_question is not None
            and pending_question.context.get(
                "validation_repair",
                False,
            )
        )
        scope = getattr(result, "scope", None)
        intent = getattr(scope, "intent", None)

        if intent is not None:
            intent_value = getattr(intent, "value", str(intent))
            correction = intent_value == "CORRECTION"

        observations: List[FieldObservation] = []

        for decision in result.decisions:
            field = getattr(decision, "field", None)
            if field is None:
                continue

            field_name = getattr(field, "value", str(field))
            if field_name not in QUANTITY_FIELDS:
                continue

            if (
                field_name == "read_write_ratio"
                and explicit_ratio is not None
            ):
                continue

            status_obj = getattr(decision, "status", None)
            status = getattr(status_obj, "value", str(status_obj))

            if status == "VERIFIED":
                observations.append(
                    FieldObservation(
                        field=field_name,
                        value=decision.value,
                        state=FieldState.VERIFIED,
                        source="QUANTITY_VERIFIER",
                        evidence=Evidence(
                            text=getattr(decision, "evidence", None),
                            source="QUANTITY_VERIFIER",
                        ),
                        message_id=message_id,
                        explicit_correction=correction,
                        metadata={
                            "unit": getattr(decision, "unit", None),
                            "reasons": list(
                                getattr(decision, "reasons", [])
                            ),
                        },
                    )
                )
                continue

            if (
                pending_question is not None
                and pending_question.target_field == field_name
                and status in {
                    "AMBIGUOUS",
                    "INVALID",
                    "UNRESOLVED",
                }
            ):
                observations.append(
                    FieldObservation(
                        field=field_name,
                        value=None,
                        state=FieldState.UNRESOLVED,
                        source="QUANTITY_VERIFIER",
                        evidence=Evidence(
                            text=getattr(decision, "evidence", None),
                            source="QUANTITY_VERIFIER",
                        ),
                        message_id=message_id,
                    )
                )

        if explicit_ratio is not None:
            observations.append(
                FieldObservation(
                    field="read_write_ratio",
                    value=explicit_ratio,
                    state=FieldState.VERIFIED,
                    source="DETERMINISTIC_RATIO_PARSER",
                    evidence=Evidence(
                        text=text,
                        source="DETERMINISTIC_RATIO_PARSER",
                    ),
                    message_id=message_id,
                    explicit_correction=correction,
                    metadata={
                        "canonical_structure":
                            "read_percent/write_percent",
                    },
                )
            )

        return observations


class PreferenceProductionAdapter:
    domain = "preference"

    def __init__(
        self,
        *,
        signal_detector,
        layer2_provider,
    ) -> None:
        self.signal_detector = signal_detector
        self.layer2_provider = layer2_provider
        self.last_signal_result = None
        self.last_layer2_result = None

    def extract(
        self,
        text: str,
        *,
        message_id: str,
        pending_question: Optional[PendingQuestion] = None,
    ) -> List[FieldObservation]:
        signal = self.signal_detector.predict(text)
        self.last_signal_result = signal

        if not bool(
            getattr(signal, "has_preference_signal", False)
        ):
            return []

        layer2 = self.layer2_provider()
        result = layer2.extract(text)
        self.last_layer2_result = result

        observations: List[FieldObservation] = []

        for dimension, dimension_result in result.dimensions.items():
            dimension_name = getattr(
                dimension,
                "value",
                str(dimension),
            )
            field_name = PREFERENCE_FIELD_BY_DIMENSION.get(
                dimension_name
            )

            if field_name is None:
                continue

            status_obj = getattr(
                dimension_result,
                "status",
                None,
            )
            status = getattr(
                status_obj,
                "value",
                str(status_obj),
            )
            level = getattr(
                dimension_result,
                "level",
                None,
            )

            if status == "RESOLVED" and level is not None:
                level_value = getattr(
                    level,
                    "value",
                    str(level),
                )

                confidence = getattr(
                    dimension_result,
                    "intensity_confidence",
                    None,
                )
                if confidence is None:
                    confidence = getattr(
                        dimension_result,
                        "presence_probability",
                        None,
                    )

                source_obj = getattr(
                    dimension_result,
                    "source",
                    None,
                )
                source = getattr(
                    source_obj,
                    "value",
                    "PREFERENCE_LAYER2",
                )

                observations.append(
                    FieldObservation(
                        field=field_name,
                        value=level_value,
                        state=FieldState.VERIFIED,
                        source=str(source),
                        evidence=Evidence(
                            text=getattr(
                                dimension_result,
                                "evidence",
                                None,
                            ),
                            source=str(source),
                            confidence=confidence,
                        ),
                        message_id=message_id,
                        explicit_correction=bool(
                            pending_question is not None
                            and pending_question.target_field == field_name
                            and pending_question.context.get(
                                "validation_repair",
                                False,
                            )
                        ),
                    )
                )

            elif (
                pending_question is not None
                and pending_question.target_field == field_name
                and status in {
                    "NEEDS_FALLBACK",
                    "UNRESOLVED",
                }
            ):
                observations.append(
                    FieldObservation(
                        field=field_name,
                        value=None,
                        state=FieldState.UNRESOLVED,
                        source="PREFERENCE_LAYER2",
                        evidence=Evidence(
                            text=getattr(
                                dimension_result,
                                "evidence",
                                None,
                            ),
                            source="PREFERENCE_LAYER2",
                        ),
                        message_id=message_id,
                    )
                )

            # NO_SIGNAL is intentionally not merged across turns.
            # It means "no preference signal in this turn for this dimension"
            # and must not erase an older confirmed preference.

        for relation in result.relations:
            payload = (
                relation.to_dict()
                if hasattr(relation, "to_dict")
                else dict(relation)
            )

            observations.append(
                FieldObservation(
                    field="__preference_relation__",
                    value=payload,
                    state=FieldState.VERIFIED,
                    source="PREFERENCE_LAYER2_RELATION",
                    evidence=Evidence(
                        text=payload.get("evidence"),
                        source="PREFERENCE_LAYER2_RELATION",
                    ),
                    message_id=message_id,
                    kind=ObservationKind.RELATION,
                )
            )

        return observations


class CategoricalProductionAdapter:
    domain = "categorical"

    def __init__(self, extractor) -> None:
        self.extractor = extractor
        self.last_result = None

    def extract(
        self,
        text: str,
        *,
        message_id: str,
        pending_question: Optional[PendingQuestion] = None,
    ) -> List[FieldObservation]:
        ha_context = None
        access_context = None

        if pending_question is not None:
            if pending_question.target_field == "ha_required":
                ha_context = pending_question.question
            elif pending_question.target_field == "access_type":
                access_context = pending_question.question

        result = self.extractor.extract(
            text,
            ha_question_context=ha_context,
            access_question_context=access_context,
        )
        self.last_result = result

        payload = (
            result.to_dict()
            if hasattr(result, "to_dict")
            else dict(result)
        )

        observations: List[FieldObservation] = []

        for field_name in ("ha_required", "access_type"):
            detail = payload.get(field_name)
            if not isinstance(detail, dict):
                continue

            status = str(detail.get("status", "")).upper()

            if status == "VERIFIED":
                source = str(
                    detail.get("source")
                    or "CATEGORICAL"
                )
                observations.append(
                    FieldObservation(
                        field=field_name,
                        value=detail.get("value"),
                        state=FieldState.VERIFIED,
                        source=source,
                        evidence=Evidence(
                            text=detail.get("evidence"),
                            source=source,
                            confidence=detail.get(
                                "semantic_confidence"
                            ),
                        ),
                        message_id=message_id,
                        explicit_correction=bool(
                            pending_question is not None
                            and pending_question.target_field == field_name
                            and pending_question.context.get(
                                "validation_repair",
                                False,
                            )
                        ),
                    )
                )

            elif (
                pending_question is not None
                and pending_question.target_field == field_name
                and status == "UNRESOLVED"
            ):
                source = str(
                    detail.get("source")
                    or "CATEGORICAL"
                )
                observations.append(
                    FieldObservation(
                        field=field_name,
                        value=None,
                        state=FieldState.UNRESOLVED,
                        source=source,
                        evidence=Evidence(
                            text=detail.get("evidence"),
                            source=source,
                            confidence=detail.get(
                                "semantic_confidence"
                            ),
                        ),
                        message_id=message_id,
                    )
                )

            # NO_EVIDENCE is not converted into False and does not overwrite
            # an older confirmed value.

        return observations
