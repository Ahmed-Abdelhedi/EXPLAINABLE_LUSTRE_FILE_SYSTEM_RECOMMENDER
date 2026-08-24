from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from .candidate_relation_resolver import (
    CandidateRelationResolution,
    CandidateRelationResolver,
    CandidateRelationType,
)
from .conversation_scope_resolver import ConversationScopeResolver
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
from .relation_aware_verifier import RelationAwareDeterministicVerifier
from .robust_explicit_pattern_resolver import RobustExplicitPatternResolver
from .robust_quantity_scanner import RobustQuantityScanner
from .selective_cascade import (
    QuantityRouteTrace,
    SelectiveCascade,
    SelectiveCascadeResult,
)


# =====================================================================
# CONTEXTUAL BINDING CONTRACT
# =====================================================================

_CONTEXTUAL_ROLE_BY_FIELD = {
    ParamName.requested_usable_capacity_tib: SemanticRole.TARGET,
    ParamName.client_count: SemanticRole.TOTAL_COUNT,
    ParamName.average_file_size_gb: SemanticRole.AVERAGE_VALUE,
    ParamName.max_file_size_gb: SemanticRole.MAXIMUM_LIMIT,
    ParamName.total_file_count: SemanticRole.TOTAL_COUNT,
    ParamName.read_write_ratio: SemanticRole.RATIO_COMPONENT,
    ParamName.target_read_gbps: SemanticRole.TARGET,
    ParamName.target_write_gbps: SemanticRole.TARGET,
    ParamName.max_budget_usd: SemanticRole.MAXIMUM_LIMIT,
    ParamName.max_power_w: SemanticRole.MAXIMUM_LIMIT,
    ParamName.annual_growth_percent: SemanticRole.GROWTH_RATE,

    # Step 2.1:
    # The field name already encodes the canonical time unit (years), so a
    # short answer such as "3" can be safely bound only when the active
    # clarification explicitly targets planning_horizon_years.
    ParamName.planning_horizon_years: SemanticRole.TARGET,
}

_CONTEXTUAL_DIMENSION_BY_FIELD = {
    ParamName.requested_usable_capacity_tib: QuantityDimension.CAPACITY,
    ParamName.client_count: QuantityDimension.UNKNOWN,
    ParamName.average_file_size_gb: QuantityDimension.FILE_SIZE,
    ParamName.max_file_size_gb: QuantityDimension.FILE_SIZE,
    ParamName.total_file_count: QuantityDimension.UNKNOWN,
    ParamName.read_write_ratio: QuantityDimension.PERCENT,
    ParamName.target_read_gbps: QuantityDimension.THROUGHPUT,
    ParamName.target_write_gbps: QuantityDimension.THROUGHPUT,
    ParamName.max_budget_usd: QuantityDimension.MONEY,
    ParamName.max_power_w: QuantityDimension.POWER,
    ParamName.annual_growth_percent: QuantityDimension.PERCENT,

    # Do not add a new QuantityDimension in Step 2.1 because the existing
    # Semantic Linker compatibility tables and trained Transformer do not know
    # a duration class. The relation-aware verifier has a dedicated guarded
    # contract for this field.
    ParamName.planning_horizon_years: QuantityDimension.UNKNOWN,
}


@dataclass
class VerifiedPipelineResult:
    """
    Result of the complete quantitative Requirement Extractor V2 pipeline.

    Final routing:

        ConversationScopeResolver
                ↓
        QuantityScanner / contextual binding
                ↓
        Explicit Resolver
                ↓ unresolved only
        Semantic Linker
                ↓ abstention only
        LLM fallback
                ↓
        CandidateRelationResolver
                ↓
        RelationAwareDeterministicVerifier

    Only VERIFIED decisions are exposed as accepted requirement values.
    """

    text: str
    cascade: SelectiveCascadeResult
    decisions: List[VerificationDecision]
    scope: Optional[ScopeResolution] = None
    relation_resolution: Optional[CandidateRelationResolution] = None

    @property
    def verified(self) -> List[VerificationDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.status == VerificationStatus.VERIFIED
        ]

    @property
    def ambiguous(self) -> List[VerificationDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.status == VerificationStatus.AMBIGUOUS
        ]

    @property
    def invalid(self) -> List[VerificationDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.status == VerificationStatus.INVALID
        ]

    @property
    def unresolved(self) -> List[VerificationDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.status == VerificationStatus.UNRESOLVED
        ]

    @property
    def needs_clarification(self) -> bool:
        if self.ambiguous:
            return True

        if self.relation_resolution is None:
            return False

        return self.relation_resolution.has_blocking_relation

    def verified_values(self) -> Dict[str, object]:
        """Return only fields that passed all deterministic safety gates."""
        values: Dict[str, object] = {}

        for decision in self.verified:
            if decision.field is None:
                continue

            values[
                decision.field.value
            ] = decision.value

        return values

    def relation_clarifications(self) -> List[str]:
        """Generate deterministic clarification prompts for blocking relations."""
        if self.relation_resolution is None:
            return []

        questions: List[str] = []

        for relation in self.relation_resolution.relations:
            if not relation.blocks_automatic_acceptance:
                continue

            fields = (
                ", ".join(
                    field.value
                    for field in relation.fields
                )
                or "the requirement"
            )

            if relation.relation == CandidateRelationType.ALTERNATIVE:
                question = (
                    f"Several alternative values were provided for {fields}. "
                    "Which value should be used?"
                )

            elif relation.relation == CandidateRelationType.RANGE:
                question = (
                    f"A range was provided for {fields}, but the current "
                    "requirement contract expects one scalar value. "
                    "Which value should be used?"
                )

            elif relation.relation == CandidateRelationType.COMPARISON:
                question = (
                    f"The message compares several values for {fields}. "
                    "Which value is the actual requirement?"
                )

            elif relation.relation == CandidateRelationType.CONFLICT:
                question = (
                    f"Conflicting values were detected for {fields}. "
                    "Please provide the authoritative value."
                )

            else:
                question = (
                    f"The relationship between candidate values for {fields} "
                    "could not be resolved safely. "
                    "Please clarify the intended value."
                )

            if question not in questions:
                questions.append(question)

        return questions

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "scope": (
                None
                if self.scope is None
                else self.scope.to_dict()
            ),
            "cascade": self.cascade.to_dict(),
            "relation_resolution": (
                None
                if self.relation_resolution is None
                else self.relation_resolution.to_dict()
            ),
            "decisions": [
                decision.to_dict()
                for decision in self.decisions
            ],
            "verified_values": self.verified_values(),
            "needs_clarification": self.needs_clarification,
            "relation_clarifications": self.relation_clarifications(),
            "summary": {
                "verified": len(self.verified),
                "ambiguous": len(self.ambiguous),
                "invalid": len(self.invalid),
                "unresolved": len(self.unresolved),
            },
        }


class VerifiedRequirementPipeline:
    """
    Quantitative Requirement Extractor V2 with relation-aware safety.

    ``planning_horizon_years`` is now part of this quantitative pipeline.
    ``access_type`` and ``ha_required`` remain intentionally outside it and
    will be implemented in the separate categorical layer.
    """

    def __init__(
        self,
        cascade: Optional[SelectiveCascade] = None,
        verifier: Optional[DeterministicVerifier] = None,
        scope_resolver: Optional[ConversationScopeResolver] = None,
        relation_resolver: Optional[CandidateRelationResolver] = None,
    ) -> None:
        self.cascade = cascade or SelectiveCascade()

        # Step 1 robustness adapters. They preserve the public SelectiveCascade
        # contract while strengthening only the deterministic scanner/resolver
        # stages. If a caller injects custom components, those components are
        # wrapped rather than discarded.
        scanner = getattr(
            self.cascade,
            "scanner",
            None,
        )

        if (
            scanner is not None
            and not isinstance(
                scanner,
                RobustQuantityScanner,
            )
        ):
            self.cascade.scanner = RobustQuantityScanner(
                base_scanner=scanner,
            )

        explicit_resolver = getattr(
            self.cascade,
            "explicit_resolver",
            None,
        )

        if (
            explicit_resolver is not None
            and not isinstance(
                explicit_resolver,
                RobustExplicitPatternResolver,
            )
        ):
            self.cascade.explicit_resolver = RobustExplicitPatternResolver(
                base_resolver=explicit_resolver,
            )

        self.scope_resolver = (
            scope_resolver
            or ConversationScopeResolver()
        )

        self.relation_resolver = (
            relation_resolver
            or CandidateRelationResolver()
        )

        self.base_verifier = (
            verifier
            or DeterministicVerifier()
        )

        self.verifier = RelationAwareDeterministicVerifier(
            base_verifier=self.base_verifier,
            relation_resolver=self.relation_resolver,
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

    def _empty_relation_resolution(
        self,
        text: str,
    ) -> CandidateRelationResolution:
        return self.relation_resolver.resolve(
            text=text,
            quantities=[],
            links=[],
        )

    # =================================================================
    # CONTEXTUAL SHORT-ANSWER HELPERS
    # =================================================================

    @staticmethod
    def _apply_inherited_unit(
        quantities: List[Quantity],
        scope: ScopeResolution,
    ) -> List[Quantity]:
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
                updated.append(quantity)
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
        role = _CONTEXTUAL_ROLE_BY_FIELD.get(
            target_field
        )

        if role is None:
            return []

        links: List[SemanticLink] = []

        for quantity in quantities:
            # For a planning-horizon clarification, preserve the whole short
            # reply as evidence. The base QuantityScanner may represent
            # "3 years" as raw="3" because duration is intentionally not a
            # storage/power/throughput QuantityDimension. The conversation
            # scope already proves which field is being answered, and the
            # full reply preserves the explicit "years/ans" wording when it
            # is present.
            evidence = (
                text.strip()
                if target_field == ParamName.planning_horizon_years
                else quantity.raw
            )

            links.append(
                SemanticLink(
                    quantity_id=quantity.id,
                    field=target_field,
                    role=role,
                    evidence=evidence,
                    resolver="conversation_scope",
                )
            )

        return links

    @staticmethod
    def _contextual_traces(
        quantities: List[Quantity],
    ) -> Dict[str, QuantityRouteTrace]:
        traces: Dict[str, QuantityRouteTrace] = {}

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
        target_field = scope.target_field

        # access_type / ha_required are non-quantitative and intentionally
        # remain outside this quantitative pipeline.
        if (
            target_field is None
            or target_field not in _CONTEXTUAL_ROLE_BY_FIELD
        ):
            return VerifiedPipelineResult(
                text=text,
                scope=scope,
                cascade=self._empty_cascade(text),
                decisions=[],
                relation_resolution=(
                    self._empty_relation_resolution(
                        text
                    )
                ),
            )

        quantities = self.cascade.scanner.scan(
            text
        )

        quantities = self._apply_inherited_unit(
            quantities=quantities,
            scope=scope,
        )

        if not quantities:
            return VerifiedPipelineResult(
                text=text,
                scope=scope,
                cascade=self._empty_cascade(text),
                decisions=[],
                relation_resolution=(
                    self._empty_relation_resolution(
                        text
                    )
                ),
            )

        links = self._contextual_links(
            text=text,
            quantities=quantities,
            target_field=target_field,
        )

        link_ids = {
            link.quantity_id
            for link in links
        }

        cascade_result = SelectiveCascadeResult(
            text=text,
            quantities=quantities,
            links=links,
            unresolved_quantity_ids=[
                quantity.id
                for quantity in quantities
                if quantity.id not in link_ids
            ],
            traces=self._contextual_traces(
                quantities
            ),
        )

        relation_resolution = (
            self.relation_resolver.resolve(
                text=text,
                quantities=quantities,
                links=links,
            )
        )

        decisions = self.verifier.verify_many(
            quantities=quantities,
            links=links,
            source_text=text,
            relation_resolution=relation_resolution,
        )

        return VerifiedPipelineResult(
            text=text,
            scope=scope,
            cascade=cascade_result,
            decisions=decisions,
            relation_resolution=relation_resolution,
        )

    # =================================================================
    # PUBLIC API
    # =================================================================

    def process(
        self,
        text: str,
        previous_question: Optional[str] = None,
        previous_question_field: Optional[ParamName] = None,
        requested_unit: Optional[str] = None,
    ) -> VerifiedPipelineResult:
        # 1. Conversation / scope analysis.
        scope = self.scope_resolver.resolve(
            user_text=text,
            previous_question_field=previous_question_field,
            requested_unit=requested_unit,
            previous_question=previous_question,
        )

        # 2. Out-of-scope messages stop before quantitative extraction.
        if scope.intent == ScopeIntent.OUT_OF_SCOPE:
            return VerifiedPipelineResult(
                text=text,
                scope=scope,
                cascade=self._empty_cascade(text),
                decisions=[],
                relation_resolution=(
                    self._empty_relation_resolution(
                        text
                    )
                ),
            )

        # 3. Short answer to an active clarification question.
        if (
            scope.intent
            == ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
        ):
            return self._process_contextual_answer(
                text=text,
                scope=scope,
            )

        # 4. NEW_REQUIREMENT / CORRECTION.
        # Stale previous-question context is intentionally not injected into
        # the semantic/LLM cascade once scope classified this as a new turn.
        cascade_result = self.cascade.resolve(
            text=text,
            previous_question=None,
        )

        # 5. Resolve cross-candidate relations before final acceptance.
        relation_resolution = (
            self.relation_resolver.resolve(
                text=text,
                quantities=cascade_result.quantities,
                links=cascade_result.links,
            )
        )

        # 6. Whole-interpretation deterministic verification.
        decisions = self.verifier.verify_many(
            quantities=cascade_result.quantities,
            links=cascade_result.links,
            source_text=text,
            relation_resolution=relation_resolution,
        )

        return VerifiedPipelineResult(
            text=text,
            scope=scope,
            cascade=cascade_result,
            decisions=decisions,
            relation_resolution=relation_resolution,
        )

    def info(self) -> dict:
        return {
            "cascade": self.cascade.info(),
            "verifier": self.verifier.info(),
            "conversation_scope": "ConversationScopeResolver",
            "conversation_scope_integrated": True,
            "candidate_relation_resolver": self.relation_resolver.info(),
            "candidate_relation_resolver_integrated": True,
            "robust_quantity_scanner": self.cascade.scanner.info(),
            "robust_explicit_resolver": self.cascade.explicit_resolver.info(),
            "planning_horizon_years_integrated": True,
            "planning_horizon_transformer_label_added": False,
        }