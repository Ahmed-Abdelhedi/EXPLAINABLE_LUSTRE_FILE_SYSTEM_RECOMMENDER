from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Optional

from .candidate_relation_resolver import (
    CandidateRelation,
    CandidateRelationResolution,
    CandidateRelationResolver,
    CandidateRelationType,
)
from .deterministic_verifier import DeterministicVerifier
from .models import (
    ParamName,
    Quantity,
    SemanticLink,
    SemanticRole,
    VerificationDecision,
    VerificationStatus,
)


class RelationAwareDeterministicVerifier:
    """
    Whole-interpretation verification wrapper.

    Order of work:
        semantic links
            -> CandidateRelationResolver
            -> ordinary deterministic candidate validation
            -> field-level relation safety gate
            -> structured-field completeness checks

    The wrapped DeterministicVerifier remains the authority for units,
    field/dimension compatibility, roles, evidence and numeric validity.
    This class adds the missing cross-candidate safety constraints.

    Important structured-field rule
    -------------------------------
    The current V2 contract represents ``read_write_ratio`` as two percentage
    components (read and write). A single component such as ``70% read`` is
    therefore not complete enough to update the requirement state. It remains
    AMBIGUOUS until the complementary component is supplied explicitly.
    """

    _UNSAFE_RELATIONS = {
        CandidateRelationType.ALTERNATIVE,
        CandidateRelationType.RANGE,
        CandidateRelationType.COMPARISON,
        CandidateRelationType.CONFLICT,
        CandidateRelationType.UNRESOLVED,
    }

    def __init__(
        self,
        base_verifier: Optional[DeterministicVerifier] = None,
        relation_resolver: Optional[CandidateRelationResolver] = None,
    ) -> None:
        self.base_verifier = base_verifier or DeterministicVerifier()
        self.relation_resolver = relation_resolver or CandidateRelationResolver()
        self.last_relation_resolution: Optional[CandidateRelationResolution] = None

    def verify(self, *args, **kwargs) -> VerificationDecision:
        """Backward-compatible single-candidate verification."""
        return self.base_verifier.verify(*args, **kwargs)

    def verify_many(
        self,
        quantities: Iterable[Quantity],
        links: Iterable[SemanticLink],
        source_text: Optional[str] = None,
        relation_resolution: Optional[CandidateRelationResolution] = None,
    ) -> List[VerificationDecision]:
        quantity_list = list(quantities)
        link_list = list(links)
        text = source_text or ""

        if relation_resolution is None:
            relation_resolution = self.relation_resolver.resolve(
                text=text,
                quantities=quantity_list,
                links=link_list,
            )

        self.last_relation_resolution = relation_resolution

        # Individual technical checks happen only after the relation analysis
        # has been established for the whole message interpretation.
        decisions = self.base_verifier.verify_many(
            quantities=quantity_list,
            links=link_list,
            source_text=source_text,
        )

        unsafe_by_quantity = {}
        superseded_by_quantity = {}

        for relation in relation_resolution.relations:
            if relation.relation in self._UNSAFE_RELATIONS:
                for quantity_id in relation.blocked_quantity_ids:
                    unsafe_by_quantity.setdefault(quantity_id, []).append(relation)

            if relation.relation == CandidateRelationType.CORRECTION:
                for quantity_id in relation.blocked_quantity_ids:
                    superseded_by_quantity.setdefault(quantity_id, []).append(relation)

        guarded: List[VerificationDecision] = []
        for decision in decisions:
            quantity_id = decision.quantity_id

            # Unsafe relation always wins over an otherwise valid scalar
            # candidate. Invalid candidates remain INVALID because they already
            # failed a stronger deterministic check.
            unsafe_relations = unsafe_by_quantity.get(quantity_id, [])
            if unsafe_relations and decision.status == VerificationStatus.VERIFIED:
                guarded.append(
                    self._override(
                        decision=decision,
                        status=VerificationStatus.AMBIGUOUS,
                        relations=unsafe_relations,
                        extra_reason="automatic_acceptance_blocked_by_candidate_relation",
                    )
                )
                continue

            # In a correction such as "change 20 to 30 GB/s", the old value is
            # not ambiguous: it is superseded. It must simply not be accepted.
            superseded_relations = superseded_by_quantity.get(quantity_id, [])
            if superseded_relations and decision.status == VerificationStatus.VERIFIED:
                guarded.append(
                    self._override(
                        decision=decision,
                        status=VerificationStatus.UNRESOLVED,
                        relations=superseded_relations,
                        extra_reason="superseded_by_explicit_correction",
                    )
                )
                continue

            guarded.append(decision)

        # Structured-field safety is intentionally performed after the generic
        # relation gate. This catches incomplete structures that contain only
        # one scalar candidate and therefore do not create a multi-candidate
        # relation by themselves.
        return self._apply_structured_field_safety(guarded)

    def info(self) -> dict:
        return {
            "name": "RelationAwareDeterministicVerifier",
            "base_verifier": type(self.base_verifier).__name__,
            "relation_resolver": self.relation_resolver.info(),
            "structured_field_rules": {
                "read_write_ratio": "requires_two_verified_ratio_components_summing_to_100",
            },
        }

    def _apply_structured_field_safety(
        self,
        decisions: List[VerificationDecision],
    ) -> List[VerificationDecision]:
        """
        Validate structured fields after scalar verification.

        Current V2 representation of read_write_ratio:
            read component  -> one percentage
            write component -> one percentage

        Safety policy:
        - exactly one verified ratio component -> AMBIGUOUS;
        - exactly two verified components whose sum is 100 -> keep VERIFIED;
        - two or more verified components whose sum is not 100 -> AMBIGUOUS.

        The verifier does not invent the missing complement. For example,
        ``70% read`` is not silently converted to ``70/30``.
        """

        ratio_indices = [
            index
            for index, decision in enumerate(decisions)
            if (
                decision.status == VerificationStatus.VERIFIED
                and decision.field == ParamName.read_write_ratio
                and decision.role == SemanticRole.RATIO_COMPONENT
            )
        ]

        if not ratio_indices:
            return decisions

        output = list(decisions)

        if len(ratio_indices) == 1:
            index = ratio_indices[0]
            output[index] = self._override_without_relation(
                decision=output[index],
                status=VerificationStatus.AMBIGUOUS,
                reason="incomplete_read_write_ratio_requires_both_components",
            )
            return output

        values: List[float] = []
        numeric = True
        for index in ratio_indices:
            try:
                values.append(float(output[index].value))
            except (TypeError, ValueError):
                numeric = False
                break

        # If values cannot be checked numerically, automatic acceptance would
        # not be justified for this structured field.
        if not numeric:
            for index in ratio_indices:
                output[index] = self._override_without_relation(
                    decision=output[index],
                    status=VerificationStatus.AMBIGUOUS,
                    reason="read_write_ratio_components_not_numeric",
                )
            return output

        if len(ratio_indices) != 2 or abs(sum(values) - 100.0) > 1e-9:
            for index in ratio_indices:
                output[index] = self._override_without_relation(
                    decision=output[index],
                    status=VerificationStatus.AMBIGUOUS,
                    reason="read_write_ratio_components_must_form_complete_100_percent_pair",
                )

        return output

    @staticmethod
    def _override(
        decision: VerificationDecision,
        status: VerificationStatus,
        relations: List[CandidateRelation],
        extra_reason: str,
    ) -> VerificationDecision:
        reasons = list(decision.reasons)
        if extra_reason not in reasons:
            reasons.append(extra_reason)

        for relation in relations:
            relation_reason = f"candidate_relation:{relation.relation.value}"
            if relation_reason not in reasons:
                reasons.append(relation_reason)
            if relation.reason and relation.reason not in reasons:
                reasons.append(relation.reason)

        return replace(
            decision,
            status=status,
            reasons=reasons,
        )

    @staticmethod
    def _override_without_relation(
        decision: VerificationDecision,
        status: VerificationStatus,
        reason: str,
    ) -> VerificationDecision:
        reasons = list(decision.reasons)
        if reason not in reasons:
            reasons.append(reason)

        return replace(
            decision,
            status=status,
            reasons=reasons,
        )