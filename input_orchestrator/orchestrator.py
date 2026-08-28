from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4

from .bwm_coordinator import BWMCoordinator
from .collection_gate import collection_gap_report
from .conflict_resolver import ConflictResolver
from .context_answers import contextual_observation
from .field_registry import WEIGHT_FIELD
from .models import (
    ConversationState,
    FieldObservation,
    OrchestratorResponse,
    PendingQuestion,
)
from .policies import DEFAULT_POLICY, OrchestrationPolicy
from .question_planner import QuestionPlanner
from .readiness import is_ready_for_final_validation
from .result_merger import ResultMerger
from .router import ExtractionRouter
from .session_state import WorkingSessionState


@dataclass
class InputOrchestrator:
    router: ExtractionRouter
    policy: OrchestrationPolicy = DEFAULT_POLICY
    bwm: Optional[BWMCoordinator] = None
    production_components: Dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.merger = ResultMerger()
        self.conflict_resolver = ConflictResolver()
        self.question_planner = QuestionPlanner(self.policy)

        if self.bwm is None:
            # Generic/unit-test orchestrators remain usable without importing
            # the heavy Preference package. Production wiring injects the
            # real frozen BWM coordinator explicitly.
            self.bwm = BWMCoordinator(
                enabled=False
            )

    def new_session(
        self,
        session_id: Optional[str] = None,
    ) -> WorkingSessionState:
        return WorkingSessionState(
            session_id=session_id or str(uuid4())
        )

    def handle_message(
        self,
        text: str,
        session: WorkingSessionState,
    ) -> OrchestratorResponse:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "User message must be a non-empty string."
            )

        message_id = session.next_message_id()
        previous_pending = session.pending_question

        # ---------------------------------------------------------
        # 0) Consume a pending BWM answer first.
        # ---------------------------------------------------------
        bwm_answer_consumed = False

        if self.bwm is not None:
            bwm_answer_consumed = (
                self.bwm.consume_pending_answer(
                    text=text,
                    session=session,
                    pending_question=previous_pending,
                )
            )

        # 1) Pending Requirement question gives meaning to short answers.
        # BWM questions have their own parser and must not be converted into
        # normal Requirement fields.
        context_obs = []

        if (
            previous_pending is None
            or previous_pending.target_field != WEIGHT_FIELD
        ):
            context_obs = contextual_observation(
                text,
                pending_question=previous_pending,
                message_id=message_id,
            )

        # 2) Analyze the complete message with all extractors EXCEPT for a
        # pure BWM control turn.
        #
        # Examples intentionally NOT routed to Quantity/Preference/Categorical:
        #   - "continue" when raw collection is already complete and BWM starts
        #   - "3" as an exact BWM 1..9 answer
        #   - "reliability" as an exact Best/Worst answer
        #   - "yes" as an exact single-active confirmation
        #
        # A rich answer is still routed normally. For example:
        #   "3, and the write target is 8 GB/s"
        # is not consumed as a pure BWM answer and therefore reaches all
        # extractors.
        pre_route_gaps = collection_gap_report(
            session,
            self.policy,
        )

        # If a short targeted clarification has already been parsed
        # deterministically by contextual_observation(), do not send that
        # same short answer through unrelated domains. This prevents cases
        # such as "not important" (answering power_priority) from being
        # misread by the categorical HA extractor.
        #
        # Rich answers are unaffected because contextual_observation() uses
        # strict/full-match parsing; a sentence carrying additional
        # Requirement information falls through to the normal extractors.
        skip_extractor_routing = bool(context_obs) or (
            self._is_pure_bwm_control_turn(
                text=text,
                previous_pending=previous_pending,
                bwm_answer_consumed=bwm_answer_consumed,
                raw_collection_complete=pre_route_gaps.complete,
            )
        )

        extracted = (
            []
            if skip_extractor_routing
            else self.router.route(
                text,
                message_id=message_id,
                pending_question=previous_pending,
            )
        )

        observations = self._deduplicate_context_priority(
            context_obs + extracted,
            previous_pending,
        )

        merge_result = self.merger.merge(
            session,
            observations,
        )

        if (
            previous_pending is not None
            and (
                previous_pending.target_field
                in merge_result.updated_fields
                or (
                    previous_pending.target_field
                    == WEIGHT_FIELD
                    and bwm_answer_consumed
                )
            )
        ):
            session.clear_pending_question()

        # ---------------------------------------------------------
        # 3) Conflicts always have first priority.
        # ---------------------------------------------------------
        if merge_result.conflicts:
            conflict = merge_result.conflicts[0]

            pending = self.conflict_resolver.question_for(
                conflict,
                question_id=self._question_id(session),
                message_id=message_id,
            )

            session.pending_question = pending
            session.conversation_state = (
                ConversationState.RESOLVING_CONFLICT
            )

            return OrchestratorResponse(
                conversation_state=session.conversation_state,
                assistant_message=pending.question,
                updated_fields=merge_result.updated_fields,
                conflicts=merge_result.conflicts,
                pending_question=pending,
                ready_for_final_validation=False,
                diagnostics={
                    "message_id": message_id,
                    "observations": len(observations),
                    "collection_gate": collection_gap_report(
                        session,
                        self.policy,
                    ).to_dict(),
                },
            )

        # ---------------------------------------------------------
        # 4) Raw Requirement clarification gate.
        # ---------------------------------------------------------
        gaps = collection_gap_report(
            session,
            self.policy,
        )

        conditional_growth_gap = (
            "planning_horizon_years"
            in gaps.blocking_fields
            and "planning_horizon_years"
            in gaps.conditional_required_fields
        )

        if (
            (
                not self.policy.ready_when_core_complete_and_no_conflict
                and not gaps.complete
            )
            or conditional_growth_gap
        ):
            pending = self.question_planner.next_question(
                session,
                question_id=self._question_id(session),
                message_id=message_id,
            )

            if pending is None:
                raise RuntimeError(
                    "Collection gate found blocking fields but "
                    "QuestionPlanner returned no clarification."
                )

            session.pending_question = pending
            session.conversation_state = (
                ConversationState.WAITING_FOR_ANSWER
            )

            return OrchestratorResponse(
                conversation_state=session.conversation_state,
                assistant_message=pending.question,
                updated_fields=merge_result.updated_fields,
                conflicts=[],
                pending_question=pending,
                ready_for_final_validation=False,
                diagnostics={
                    "message_id": message_id,
                    "observations": len(observations),
                    "collection_gate": gaps.to_dict(),
                    "decision": (
                        "CLARIFICATION_REQUIRED_BEFORE_VALIDATION"
                    ),
                },
            )

        # ---------------------------------------------------------
        # 5) Formal preference weighting phase.
        # ---------------------------------------------------------
        if self.bwm is not None and self.bwm.enabled:
            action = self.bwm.evaluate(
                session,
                message_id=message_id,
            )

            if not action.complete:
                pending = PendingQuestion(
                    question_id=self._question_id(session),
                    target_field=WEIGHT_FIELD,
                    question=action.question or (
                        "Preference weighting requires clarification."
                    ),
                    expected_answer_type=(
                        action.expected_answer_type
                        or "bwm_judgment"
                    ),
                    created_after_message_id=message_id,
                    context=dict(action.context or {}),
                )

                session.pending_question = pending
                session.conversation_state = (
                    ConversationState.BWM_ELICITATION
                )

                return OrchestratorResponse(
                    conversation_state=session.conversation_state,
                    assistant_message=pending.question,
                    updated_fields=merge_result.updated_fields,
                    conflicts=[],
                    pending_question=pending,
                    ready_for_final_validation=False,
                    diagnostics={
                        "message_id": message_id,
                        "collection_gate": gaps.to_dict(),
                        "bwm": action.to_dict(),
                    },
                )

        # ---------------------------------------------------------
        # 6) READY only after raw fields + BWM terminal state.
        # ---------------------------------------------------------
        ready = is_ready_for_final_validation(
            session,
            self.policy,
        )

        if (
            self.bwm is not None
            and self.bwm.enabled
            and session.get(WEIGHT_FIELD).state.value
            not in {"VERIFIED", "DECLINED"}
        ):
            ready = False

        if ready:
            session.pending_question = None
            session.conversation_state = (
                ConversationState.READY_FOR_FINAL_VALIDATION
            )

            return OrchestratorResponse(
                conversation_state=session.conversation_state,
                assistant_message=None,
                updated_fields=merge_result.updated_fields,
                conflicts=[],
                pending_question=None,
                ready_for_final_validation=True,
                diagnostics={
                    "message_id": message_id,
                    "collection_gate": gaps.to_dict(),
                    "bwm_status": session.bwm_dialogue.get(
                        "last_status"
                    ),
                },
            )

        # ---------------------------------------------------------
        # 7) Generic fallback.
        # ---------------------------------------------------------
        pending = self.question_planner.next_question(
            session,
            question_id=self._question_id(session),
            message_id=message_id,
        )

        session.pending_question = pending
        session.conversation_state = (
            ConversationState.WAITING_FOR_ANSWER
            if pending is not None
            else ConversationState.COLLECTING
        )

        return OrchestratorResponse(
            conversation_state=session.conversation_state,
            assistant_message=(
                None if pending is None else pending.question
            ),
            updated_fields=merge_result.updated_fields,
            conflicts=[],
            pending_question=pending,
            ready_for_final_validation=False,
            diagnostics={
                "message_id": message_id,
                "observations": len(observations),
                "collection_gate": gaps.to_dict(),
            },
        )

    @staticmethod
    def _is_pure_bwm_control_turn(
        *,
        text: str,
        previous_pending: Optional[PendingQuestion],
        bwm_answer_consumed: bool,
        raw_collection_complete: bool,
    ) -> bool:
        # Exact BWM answers are parsed by BWMCoordinator using deliberately
        # strict parsers. If consumed, there is no additional Requirement
        # content to extract.
        if (
            previous_pending is not None
            and previous_pending.target_field == WEIGHT_FIELD
            and bwm_answer_consumed
        ):
            return True

        # When collection is already complete, these are orchestration control
        # tokens rather than Requirement content. Avoid sending them to the
        # quantity fallback/LLM.
        if (
            previous_pending is None
            and raw_collection_complete
        ):
            compact = " ".join(
                text.strip().lower().strip(" .!?").split()
            )

            if compact in {
                "continue",
                "continue please",
                "next",
                "go on",
                "proceed",
                "continuer",
                "continuez",
                "suivant",
                "on continue",
            }:
                return True

        return False

    @staticmethod
    def _question_id(
        session: WorkingSessionState,
    ) -> str:
        return f"Q{session.message_counter:04d}"

    @staticmethod
    def _deduplicate_context_priority(
        observations: list[FieldObservation],
        pending: Optional[PendingQuestion],
    ) -> list[FieldObservation]:
        if pending is None:
            return observations

        contextual = [
            obs
            for obs in observations
            if (
                obs.field == pending.target_field
                and obs.source == "ORCHESTRATOR_CONTEXT"
            )
        ]

        if not contextual:
            return observations

        return contextual + [
            obs
            for obs in observations
            if obs.field != pending.target_field
        ]
