from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from .conversation_scope_resolver import (
    ConversationScopeResolver,
)
from .deterministic_verifier import DeterministicVerifier
from .models import (
    ParamName,
    Quantity,
    QuantityDimension,
    ScopeIntent,
    ScopeResolution,
    SemanticLink,
    SemanticRole,
    VerificationDecision,
    VerificationStatus,
)
from .selective_cascade import (
    QuantityRouteTrace,
    SelectiveCascade,
    SelectiveCascadeResult,
)


# =====================================================================
# CONTEXTUAL BINDING CONTRACT
# =====================================================================

# A short answer to an active clarification question does not need
# Semantic Linker / LLM inference: the conversation state already provides
# the FIELD. The role below is the deterministic business role associated
# with that clarification field.
_CONTEXTUAL_ROLE_BY_FIELD = {
    ParamName.requested_usable_capacity_tib:
        SemanticRole.TARGET,

    ParamName.client_count:
        SemanticRole.TOTAL_COUNT,

    ParamName.average_file_size_gb:
        SemanticRole.AVERAGE_VALUE,

    ParamName.max_file_size_gb:
        SemanticRole.MAXIMUM_LIMIT,

    ParamName.total_file_count:
        SemanticRole.TOTAL_COUNT,

    ParamName.read_write_ratio:
        SemanticRole.RATIO_COMPONENT,

    ParamName.target_read_gbps:
        SemanticRole.TARGET,

    ParamName.target_write_gbps:
        SemanticRole.TARGET,

    ParamName.max_budget_usd:
        SemanticRole.MAXIMUM_LIMIT,

    ParamName.max_power_w:
        SemanticRole.MAXIMUM_LIMIT,

    ParamName.annual_growth_percent:
        SemanticRole.GROWTH_RATE,
}


_CONTEXTUAL_DIMENSION_BY_FIELD = {
    ParamName.requested_usable_capacity_tib:
        QuantityDimension.CAPACITY,

    ParamName.client_count:
        QuantityDimension.UNKNOWN,

    ParamName.average_file_size_gb:
        QuantityDimension.FILE_SIZE,

    ParamName.max_file_size_gb:
        QuantityDimension.FILE_SIZE,

    ParamName.total_file_count:
        QuantityDimension.UNKNOWN,

    ParamName.read_write_ratio:
        QuantityDimension.PERCENT,

    ParamName.target_read_gbps:
        QuantityDimension.THROUGHPUT,

    ParamName.target_write_gbps:
        QuantityDimension.THROUGHPUT,

    ParamName.max_budget_usd:
        QuantityDimension.MONEY,

    ParamName.max_power_w:
        QuantityDimension.POWER,

    ParamName.annual_growth_percent:
        QuantityDimension.PERCENT,
}


@dataclass
class VerifiedPipelineResult:
    """
    Result of the complete quantitative requirement-extraction pipeline.

    Integrated routing:

        ConversationScopeResolver
                ↓
        ┌──── OUT_OF_SCOPE ────> stop quantitative extraction
        │
        ├──── short contextual answer
        │           ↓
        │     QuantityScanner
        │           ↓
        │     deterministic FIELD/ROLE binding
        │           ↓
        │     DeterministicVerifier
        │
        └──── new/correction requirement
                    ↓
              SelectiveCascade
                    ↓
              DeterministicVerifier

    Only VERIFIED decisions are exposed as accepted requirement values.

    Note:
        access_type and ha_required are intentionally outside this
        quantitative pipeline. Their scope can still be identified by
        ConversationScopeResolver, but they are not converted into
        VerificationDecision objects here.
    """

    text: str
    cascade: SelectiveCascadeResult
    decisions: List[VerificationDecision]
    scope: Optional[ScopeResolution] = None

    # ================================================================
    # DERIVED VIEWS
    # ================================================================

    @property
    def verified(self) -> List[VerificationDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.status
            == VerificationStatus.VERIFIED
        ]

    @property
    def ambiguous(self) -> List[VerificationDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.status
            == VerificationStatus.AMBIGUOUS
        ]

    @property
    def invalid(self) -> List[VerificationDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.status
            == VerificationStatus.INVALID
        ]

    @property
    def unresolved(self) -> List[VerificationDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.status
            == VerificationStatus.UNRESOLVED
        ]

    # ================================================================
    # VERIFIED REQUIREMENT VALUES
    # ================================================================

    def verified_values(self) -> Dict[str, object]:
        """
        Return only fields that passed deterministic verification.
        """

        values: Dict[str, object] = {}

        for decision in self.verified:
            if decision.field is None:
                continue

            values[
                decision.field.value
            ] = decision.value

        return values

    def to_dict(self) -> dict:
        return {
            "text": self.text,

            "scope": (
                None
                if self.scope is None
                else self.scope.to_dict()
            ),

            "cascade":
                self.cascade.to_dict(),

            "decisions": [
                decision.to_dict()
                for decision
                in self.decisions
            ],

            "verified_values":
                self.verified_values(),

            "summary": {
                "verified":
                    len(self.verified),

                "ambiguous":
                    len(self.ambiguous),

                "invalid":
                    len(self.invalid),

                "unresolved":
                    len(self.unresolved),
            },
        }


class VerifiedRequirementPipeline:
    """
    Quantitative Requirement Extractor V2 pipeline with conversation scope.

    ConversationScopeResolver is now the first routing stage.

    Important safety rule:
        A short answer is bound directly to the active clarification FIELD
        only when ConversationScopeResolver explicitly classifies the turn as
        ANSWER_TO_PREVIOUS_QUESTION.

    Rich messages are never force-bound to the previous question.
    """

    def __init__(
        self,
        cascade: Optional[
            SelectiveCascade
        ] = None,
        verifier: Optional[
            DeterministicVerifier
        ] = None,
        scope_resolver: Optional[
            ConversationScopeResolver
        ] = None,
    ) -> None:

        self.cascade = (
            cascade
            or SelectiveCascade()
        )

        self.verifier = (
            verifier
            or DeterministicVerifier()
        )

        self.scope_resolver = (
            scope_resolver
            or ConversationScopeResolver()
        )

    # =================================================================
    # EMPTY / STOP RESULT
    # =================================================================

    @staticmethod
    def _empty_cascade(
        text: str,
    ) -> SelectiveCascadeResult:
        return SelectiveCascadeResult(
            text=text,
            quantities=[],
            links=[],
            unresolved_quantity_ids=[],
            traces={},
        )

    # =================================================================
    # CONTEXTUAL SHORT-ANSWER HELPERS
    # =================================================================

    @staticmethod
    def _apply_inherited_unit(
        quantities: List[Quantity],
        scope: ScopeResolution,
    ) -> List[Quantity]:
        """
        Apply the unit inherited from the active clarification question.

        This is done only for quantities that do not already contain an
        explicit unit.

        The dimension is also restored from the active field because a bare
        number such as "200" has UNKNOWN dimension before conversation context
        is applied.
        """

        if (
            scope.target_field is None
            or scope.inherited_unit is None
        ):
            return quantities

        expected_dimension = (
            _CONTEXTUAL_DIMENSION_BY_FIELD.get(
                scope.target_field
            )
        )

        if expected_dimension is None:
            return quantities

        updated: List[Quantity] = []

        for quantity in quantities:

            if quantity.unit is not None:
                updated.append(
                    quantity
                )
                continue

            updated.append(
                replace(
                    quantity,
                    unit=scope.inherited_unit,
                    dimension=expected_dimension,
                )
            )

        return updated

    @staticmethod
    def _contextual_links(
        text: str,
        quantities: List[Quantity],
        target_field: ParamName,
    ) -> List[SemanticLink]:

        role = (
            _CONTEXTUAL_ROLE_BY_FIELD.get(
                target_field
            )
        )

        if role is None:
            return []

        return [
            SemanticLink(
                quantity_id=quantity.id,
                field=target_field,
                role=role,
                evidence=quantity.raw,
                resolver="conversation_scope",
            )
            for quantity in quantities
        ]

    @staticmethod
    def _contextual_traces(
        quantities: List[Quantity],
    ) -> Dict[str, QuantityRouteTrace]:

        traces: Dict[
            str,
            QuantityRouteTrace,
        ] = {}

        for quantity in quantities:
            traces[
                quantity.id
            ] = QuantityRouteTrace(
                quantity_id=quantity.id,
                explicit_attempted=False,
                explicit_resolved=False,
                semantic_attempted=False,
                semantic_accepted=False,
                llm_attempted=False,
                llm_resolved=False,
                final_resolver="conversation_scope",
                final_status="resolved",
            )

        return traces

    def _process_contextual_answer(
        self,
        text: str,
        scope: ScopeResolution,
    ) -> VerifiedPipelineResult:
        """
        Resolve a short answer from conversation state without Semantic Linker
        or LLM inference.
        """

        target_field = (
            scope.target_field
        )

        # access_type / ha_required are non-quantitative and intentionally
        # remain outside this quantitative pipeline.
        if (
            target_field is None
            or target_field
            not in _CONTEXTUAL_ROLE_BY_FIELD
        ):
            return VerifiedPipelineResult(
                text=text,
                scope=scope,
                cascade=self._empty_cascade(
                    text
                ),
                decisions=[],
            )

        quantities = (
            self.cascade.scanner.scan(
                text
            )
        )

        quantities = (
            self._apply_inherited_unit(
                quantities=quantities,
                scope=scope,
            )
        )

        if not quantities:
            return VerifiedPipelineResult(
                text=text,
                scope=scope,
                cascade=self._empty_cascade(
                    text
                ),
                decisions=[],
            )

        links = (
            self._contextual_links(
                text=text,
                quantities=quantities,
                target_field=target_field,
            )
        )

        link_ids = {
            link.quantity_id
            for link in links
        }

        cascade_result = (
            SelectiveCascadeResult(
                text=text,
                quantities=quantities,
                links=links,
                unresolved_quantity_ids=[
                    quantity.id
                    for quantity in quantities
                    if quantity.id
                    not in link_ids
                ],
                traces=self._contextual_traces(
                    quantities
                ),
            )
        )

        decisions = (
            self.verifier.verify_many(
                quantities=quantities,
                links=links,
                source_text=text,
            )
        )

        return VerifiedPipelineResult(
            text=text,
            scope=scope,
            cascade=cascade_result,
            decisions=decisions,
        )

    # =================================================================
    # PUBLIC API
    # =================================================================

    def process(
        self,
        text: str,
        previous_question: Optional[str] = None,
        previous_question_field: Optional[
            ParamName
        ] = None,
        requested_unit: Optional[str] = None,
    ) -> VerifiedPipelineResult:

        # -------------------------------------------------------------
        # 1. Conversation / scope analysis
        # -------------------------------------------------------------

        scope = (
            self.scope_resolver.resolve(
                user_text=text,
                previous_question_field=
                    previous_question_field,
                requested_unit=
                    requested_unit,
                previous_question=
                    previous_question,
            )
        )

        # -------------------------------------------------------------
        # 2. Explicitly out-of-scope messages stop here
        # -------------------------------------------------------------

        if (
            scope.intent
            == ScopeIntent.OUT_OF_SCOPE
        ):
            return VerifiedPipelineResult(
                text=text,
                scope=scope,
                cascade=self._empty_cascade(
                    text
                ),
                decisions=[],
            )

        # -------------------------------------------------------------
        # 3. Short answer to active clarification
        # -------------------------------------------------------------

        if (
            scope.intent
            == ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
        ):
            return (
                self._process_contextual_answer(
                    text=text,
                    scope=scope,
                )
            )

        # -------------------------------------------------------------
        # 4. NEW_REQUIREMENT / CORRECTION
        #
        # Scope has explicitly decided not to bind this turn to the active
        # question. Therefore the old previous question is NOT injected
        # into Semantic Linker / LLM, preventing stale-context bias.
        # -------------------------------------------------------------

        cascade_result = (
            self.cascade.resolve(
                text=text,
                previous_question=None,
            )
        )

        # -------------------------------------------------------------
        # 5. Deterministic verification
        # -------------------------------------------------------------

        decisions = (
            self.verifier.verify_many(
                quantities=
                    cascade_result.quantities,

                links=
                    cascade_result.links,

                source_text=text,
            )
        )

        # -------------------------------------------------------------
        # 6. Safe result
        # -------------------------------------------------------------

        return VerifiedPipelineResult(
            text=text,
            scope=scope,
            cascade=cascade_result,
            decisions=decisions,
        )

    def info(self) -> dict:
        return {
            "cascade":
                self.cascade.info(),

            "verifier":
                "DeterministicVerifier",

            "conversation_scope":
                "ConversationScopeResolver",

            "conversation_scope_integrated":
                True,
        }