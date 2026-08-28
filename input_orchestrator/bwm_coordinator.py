from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from .field_registry import WEIGHT_FIELD
from .models import (
    FieldState,
    PendingQuestion,
)
from .session_state import WorkingSessionState


@dataclass(frozen=True)
class BWMAction:
    complete: bool
    status: str
    question: Optional[str] = None
    expected_answer_type: Optional[str] = None
    context: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "complete": self.complete,
            "status": self.status,
            "question": self.question,
            "expected_answer_type": self.expected_answer_type,
            "context": dict(self.context or {}),
        }


class BWMCoordinator:
    """
    Conversation adapter around the already-frozen
    FormalPreferenceWeightingLayer.

    It never computes weights itself. It only:
      - rebuilds the aggregate qualitative preference state;
      - stores explicit Best/Worst choices;
      - stores explicit user BWM judgments on the 1..9 scale;
      - calls FormalPreferenceWeightingLayer.run();
      - persists a WEIGHTS_READY result in preference_weights.

    Numerical weights always come from the frozen Linear BWM solver.
    """

    def __init__(
        self,
        weighting_layer=None,
        *,
        enabled: bool = False,
    ) -> None:
        self.weighting_layer = weighting_layer
        self.enabled = bool(enabled)

    @classmethod
    def from_frozen_layer(cls) -> "BWMCoordinator":
        from preference_extractor.weighting import (
            FormalPreferenceWeightingLayer,
        )

        return cls(
            weighting_layer=FormalPreferenceWeightingLayer(),
            enabled=True,
        )

    @staticmethod
    def _compact(text: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            text.strip().lower().strip(" .!?"),
        )

    @staticmethod
    def _dimension_alias(text: str) -> Optional[str]:
        value = BWMCoordinator._compact(text)

        aliases = {
            "cost": "cost",
            "cout": "cost",
            "coût": "cost",
            "budget": "cost",

            "power": "power",
            "power efficiency": "power",
            "energy": "power",
            "energie": "power",
            "énergie": "power",
            "consommation": "power",

            "performance": "performance",
            "performances": "performance",

            "reliability": "reliability",
            "reliabilite": "reliability",
            "fiability": "reliability",
            "fiabilite": "reliability",
            "fiabilité": "reliability",
        }

        return aliases.get(value)

    @staticmethod
    def _yes_no(text: str) -> Optional[bool]:
        value = BWMCoordinator._compact(text)

        if value in {
            "yes", "y", "oui", "confirm", "confirmed",
            "correct", "true",
        }:
            return True

        if value in {
            "no", "n", "non", "false",
        }:
            return False

        return None

    def consume_pending_answer(
        self,
        *,
        text: str,
        session: WorkingSessionState,
        pending_question: Optional[PendingQuestion],
    ) -> bool:
        """
        Consume only a pending BWM question.

        The complete user message is still routed through the extractors by
        the orchestrator afterwards, so rich answers can carry additional
        Requirement information.
        """
        if (
            not self.enabled
            or pending_question is None
            or pending_question.target_field != WEIGHT_FIELD
        ):
            return False

        expected = pending_question.expected_answer_type
        dialogue = session.bwm_dialogue
        dialogue["last_input_error"] = None

        if expected == "bwm_judgment":
            match = re.fullmatch(
                r"\s*([1-9])\s*",
                text,
            )

            if match is None:
                dialogue["last_input_error"] = (
                    "BWM judgment must be one integer from 1 to 9."
                )
                return False

            comparison_id = pending_question.context.get(
                "comparison_id"
            )

            if not comparison_id:
                dialogue["last_input_error"] = (
                    "Missing BWM comparison identifier."
                )
                return False

            answers = dialogue.setdefault("answers", {})
            answers[str(comparison_id)] = int(
                match.group(1)
            )
            return True

        if expected == "bwm_best":
            dimension = self._dimension_alias(text)

            if dimension is None:
                dialogue["last_input_error"] = (
                    "Answer with one active preference dimension."
                )
                return False

            active = set(
                pending_question.context.get(
                    "active_dimensions",
                    [],
                )
            )

            if dimension not in active:
                dialogue["last_input_error"] = (
                    f"{dimension} is not an active preference."
                )
                return False

            dialogue["explicit_best"] = dimension
            return True

        if expected == "bwm_worst":
            dimension = self._dimension_alias(text)

            if dimension is None:
                dialogue["last_input_error"] = (
                    "Answer with one active preference dimension."
                )
                return False

            active = set(
                pending_question.context.get(
                    "active_dimensions",
                    [],
                )
            )

            if dimension not in active:
                dialogue["last_input_error"] = (
                    f"{dimension} is not an active preference."
                )
                return False

            dialogue["explicit_worst"] = dimension
            return True

        if expected == "bwm_single_confirmation":
            answer = self._yes_no(text)

            if answer is None:
                dialogue["last_input_error"] = (
                    "Please answer yes or no."
                )
                return False

            dialogue["single_active_confirmed"] = answer
            dialogue["single_active_rejected"] = not answer
            return True

        return False

    def evaluate(
        self,
        session: WorkingSessionState,
        *,
        message_id: str,
    ) -> BWMAction:
        if not self.enabled:
            return BWMAction(
                complete=True,
                status="BWM_DISABLED",
            )

        extraction = self._build_extraction(session)

        from preference_extractor.layer2.labels import (
            PreferenceDimension,
        )
        from preference_extractor.weighting.models import (
            WeightingStatus,
        )

        dialogue = session.bwm_dialogue

        explicit_best = (
            PreferenceDimension(
                dialogue["explicit_best"]
            )
            if dialogue.get("explicit_best")
            else None
        )

        explicit_worst = (
            PreferenceDimension(
                dialogue["explicit_worst"]
            )
            if dialogue.get("explicit_worst")
            else None
        )

        if self.weighting_layer is None:
            raise RuntimeError(
                "BWM weighting layer is not configured."
            )

        result = self.weighting_layer.run(
            extraction,
            explicit_best=explicit_best,
            explicit_worst=explicit_worst,
            bwm_answers=dict(
                dialogue.get("answers", {})
            ),
            single_active_confirmed=bool(
                dialogue.get(
                    "single_active_confirmed",
                    False,
                )
            ),
        )

        payload = result.to_dict()
        status = result.status.value

        dialogue["last_status"] = status
        dialogue["last_result"] = payload

        # ---------------------------------------------------------
        # Terminal states
        # ---------------------------------------------------------
        if result.status == WeightingStatus.WEIGHTS_READY:
            weights = result.all_four_weights()

            if abs(sum(weights.values()) - 1.0) > 1e-8:
                raise RuntimeError(
                    "Formal BWM returned non-normalized weights."
                )

            record = session.get(WEIGHT_FIELD)

            if record.state != FieldState.MISSING:
                record.history.append(record.snapshot())

            record.value = weights
            record.state = FieldState.VERIFIED
            record.source = (
                result.method
                or "FORMAL_PREFERENCE_WEIGHTING"
            )
            record.evidence = (
                f"status={status};"
                f"xi_star={result.xi_star};"
                f"consistency={result.consistency.status.value}"
            )
            record.confidence = None
            record.message_id = message_id
            record.revision += 1

            return BWMAction(
                complete=True,
                status=status,
            )

        if result.status == WeightingStatus.NO_ACTIVE_PREFERENCE:
            record = session.get(WEIGHT_FIELD)

            if record.state != FieldState.MISSING:
                record.history.append(record.snapshot())

            record.value = None
            record.state = FieldState.DECLINED
            record.source = "FORMAL_BWM:NO_ACTIVE_PREFERENCE"
            record.evidence = (
                "No active preference; no numerical weight vector invented."
            )
            record.confidence = None
            record.message_id = message_id
            record.revision += 1

            return BWMAction(
                complete=True,
                status=status,
            )

        # ---------------------------------------------------------
        # Dialogue states
        # ---------------------------------------------------------
        if (
            result.status
            == WeightingStatus.NEEDS_SINGLE_CRITERION_CONFIRMATION
        ):
            active = [
                dimension.value
                for dimension in result.active_dimensions
            ]
            dimension = (
                active[0]
                if active
                else "the active criterion"
            )

            if dialogue.get("single_active_rejected"):
                return BWMAction(
                    complete=False,
                    status=status,
                    question=(
                        f"{dimension} is currently the only active "
                        "preference, but you did not confirm assigning it "
                        "100% of the preference weight. Please restate at "
                        "least one additional preference in your next "
                        "message, or answer yes to confirm this single "
                        "criterion."
                    ),
                    expected_answer_type="bwm_single_confirmation",
                    context={
                        "active_dimensions": active,
                        "single_active_dimension": dimension,
                    },
                )

            return BWMAction(
                complete=False,
                status=status,
                question=(
                    f"{dimension} is the only active preference. "
                    "Should it be the only ranking objective and receive "
                    "100% of the preference weight? Answer yes or no."
                ),
                expected_answer_type="bwm_single_confirmation",
                context={
                    "active_dimensions": active,
                    "single_active_dimension": dimension,
                },
            )

        if result.status == WeightingStatus.NEEDS_BEST_WORST:
            active = [
                dimension.value
                for dimension in result.active_dimensions
            ]
            active_text = ", ".join(active)

            if result.best is None:
                return BWMAction(
                    complete=False,
                    status=status,
                    question=(
                        "Which active preference is the MOST important? "
                        f"Choose one of: {active_text}."
                    ),
                    expected_answer_type="bwm_best",
                    context={
                        "active_dimensions": active,
                    },
                )

            if result.worst is None:
                return BWMAction(
                    complete=False,
                    status=status,
                    question=(
                        "Which active preference is the LEAST important? "
                        f"Choose one of: {active_text}."
                    ),
                    expected_answer_type="bwm_worst",
                    context={
                        "active_dimensions": active,
                    },
                )

            raise RuntimeError(
                "NEEDS_BEST_WORST returned with both endpoints resolved."
            )

        if (
            result.status
            == WeightingStatus.NEEDS_BWM_COMPARISONS
        ):
            if not result.missing_questions:
                raise RuntimeError(
                    "NEEDS_BWM_COMPARISONS without a missing question."
                )

            question = result.missing_questions[0]

            prefix = ""
            if dialogue.get("last_input_error"):
                prefix = (
                    dialogue["last_input_error"]
                    + " "
                )

            return BWMAction(
                complete=False,
                status=status,
                question=prefix + question.prompt,
                expected_answer_type="bwm_judgment",
                context={
                    "comparison_id":
                        question.comparison_id,
                    "kind": question.kind,
                    "left": question.left.value,
                    "right": question.right.value,
                    "scale_min": question.scale_min,
                    "scale_max": question.scale_max,
                },
            )

        if result.status == WeightingStatus.BLOCKED_UNRESOLVED:
            return BWMAction(
                complete=False,
                status=status,
                question=(
                    "Preference weighting is blocked because at least one "
                    "preference dimension is unresolved. Please clarify the "
                    "preference before weighting can continue."
                ),
                expected_answer_type="bwm_blocked_preference",
                context={
                    "violations": list(result.violations),
                },
            )

        if result.status in {
            WeightingStatus.INVALID_BWM_JUDGMENTS,
            WeightingStatus.INCONSISTENT_PREFERENCES,
        }:
            return BWMAction(
                complete=False,
                status=status,
                question=(
                    "The preference judgments are inconsistent with the "
                    "verified preference order. Please review and restate "
                    "the conflicting preference information before final "
                    "validation."
                ),
                expected_answer_type="bwm_inconsistency",
                context={
                    "violations": list(result.violations),
                    "result": payload,
                },
            )

        raise RuntimeError(
            f"Unsupported formal weighting status: {status}"
        )

    def _build_extraction(
        self,
        session: WorkingSessionState,
    ):
        from preference_extractor.layer2.labels import (
            PreferenceDimension,
            PreferenceLevel,
            ResolutionSource,
            ResolutionStatus,
        )
        from preference_extractor.layer2.schemas import (
            DimensionPreferenceResult,
            PreferenceExtractionResult,
            PreferenceRelation,
        )

        field_by_dimension = {
            PreferenceDimension.COST: "cost_priority",
            PreferenceDimension.POWER: "power_priority",
            PreferenceDimension.PERFORMANCE:
                "performance_priority",
            PreferenceDimension.RELIABILITY:
                "reliability_priority",
        }

        dimensions = {}

        for dimension, field_name in field_by_dimension.items():
            record = session.get(field_name)

            if record.state == FieldState.VERIFIED:
                if record.value == "NO_SIGNAL":
                    dimensions[dimension] = (
                        DimensionPreferenceResult(
                            dimension=dimension,
                            status=ResolutionStatus.NO_SIGNAL,
                            source=ResolutionSource.NONE,
                            level=None,
                            evidence=record.evidence,
                        )
                    )
                else:
                    dimensions[dimension] = (
                        DimensionPreferenceResult(
                            dimension=dimension,
                            status=ResolutionStatus.RESOLVED,
                            source=ResolutionSource.NONE,
                            level=PreferenceLevel(
                                str(record.value)
                            ),
                            evidence=record.evidence,
                        )
                    )
                continue

            if record.state == FieldState.DECLINED:
                dimensions[dimension] = (
                    DimensionPreferenceResult(
                        dimension=dimension,
                        status=ResolutionStatus.NO_SIGNAL,
                        source=ResolutionSource.NONE,
                        level=None,
                        evidence=record.evidence,
                    )
                )
                continue

            dimensions[dimension] = (
                DimensionPreferenceResult(
                    dimension=dimension,
                    status=ResolutionStatus.UNRESOLVED,
                    source=ResolutionSource.NONE,
                    level=None,
                    evidence=record.evidence,
                )
            )

        relations = []

        for item in session.preference_relations:
            try:
                higher = PreferenceDimension(
                    str(item.get("higher"))
                )
                lower = PreferenceDimension(
                    str(item.get("lower"))
                )
            except ValueError:
                continue

            relations.append(
                PreferenceRelation(
                    higher=higher,
                    lower=lower,
                    evidence=str(
                        item.get("evidence")
                        or ""
                    ),
                    relation_type=str(
                        item.get(
                            "relation_type",
                            "MORE_IMPORTANT_THAN",
                        )
                    ),
                )
            )

        return PreferenceExtractionResult(
            text="SESSION_AGGREGATE",
            dimensions=dimensions,
            relations=relations,
        )
