from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Dict, Iterable, List, Optional

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
            -> planning-horizon deterministic validation
            -> field-level relation safety gate
            -> structured-field completeness checks

    The wrapped DeterministicVerifier remains the authority for the original
    quantitative Requirement Contract fields.

    Step 2.1 adds ``planning_horizon_years`` deliberately outside the XLM-R
    field-label space. That field is therefore verified here through a narrow
    deterministic contract:
        - role must be TARGET;
        - value must be finite;
        - value must be a strictly positive integer;
        - fresh requirements require explicit year evidence;
        - contextual short answers are accepted only when conversation scope
          has already bound the quantity to planning_horizon_years;
        - canonical output unit is ``years``.

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

    _YEAR_EVIDENCE_RE = re.compile(
        r"\b(?:years?|yrs?|ans?|ann[ée]es?)\b",
        re.IGNORECASE,
    )

    _ALLOWED_HORIZON_SOURCE_UNITS = {
        None,
        "year",
        "years",
        "yr",
        "yrs",
        "an",
        "ans",
        "année",
        "années",
        "annee",
        "annees",
    }

    def __init__(
        self,
        base_verifier: Optional[DeterministicVerifier] = None,
        relation_resolver: Optional[CandidateRelationResolver] = None,
    ) -> None:
        self.base_verifier = base_verifier or DeterministicVerifier()
        self.relation_resolver = relation_resolver or CandidateRelationResolver()
        self.last_relation_resolution: Optional[CandidateRelationResolution] = None

    # =================================================================
    # SINGLE-CANDIDATE API
    # =================================================================

    def verify(self, *args, **kwargs) -> VerificationDecision:
        """Backward-compatible single-candidate verification."""

        quantity = kwargs.get("quantity")
        link = kwargs.get("link")
        source_text = kwargs.get("source_text")

        if quantity is None and args:
            quantity = args[0]

        if link is None and len(args) >= 2:
            link = args[1]

        if source_text is None and len(args) >= 3:
            source_text = args[2]

        if (
            isinstance(link, SemanticLink)
            and link.field == ParamName.planning_horizon_years
            and isinstance(quantity, Quantity)
        ):
            return self._verify_planning_horizon(
                quantity=quantity,
                link=link,
                source_text=source_text or quantity.source_text or "",
            )

        return self.base_verifier.verify(
            *args,
            **kwargs,
        )

    # =================================================================
    # BATCH API
    # =================================================================

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

        # -------------------------------------------------------------
        # Individual technical checks.
        # We preserve the base verifier for all existing fields and handle
        # only planning_horizon_years locally.
        # -------------------------------------------------------------
        links_by_quantity: Dict[str, List[SemanticLink]] = {}

        for link in link_list:
            links_by_quantity.setdefault(
                link.quantity_id,
                [],
            ).append(link)

        decisions: List[VerificationDecision] = []
        known_quantity_ids = {
            quantity.id
            for quantity in quantity_list
        }

        for quantity in quantity_list:
            quantity_links = links_by_quantity.get(
                quantity.id,
                [],
            )

            if not quantity_links:
                decisions.append(
                    self.base_verifier.verify(
                        quantity=quantity,
                        link=None,
                        source_text=source_text,
                    )
                )
                continue

            if len(quantity_links) > 1:
                # Preserve exactly the original base-verifier safety policy.
                decisions.extend(
                    self.base_verifier.verify_many(
                        quantities=[quantity],
                        links=quantity_links,
                        source_text=source_text,
                    )
                )
                continue

            link = quantity_links[0]

            if link.field == ParamName.planning_horizon_years:
                decisions.append(
                    self._verify_planning_horizon(
                        quantity=quantity,
                        link=link,
                        source_text=text or quantity.source_text or "",
                    )
                )
                continue

            decisions.append(
                self.base_verifier.verify(
                    quantity=quantity,
                    link=link,
                    source_text=source_text,
                )
            )

        # Defensive check retained from DeterministicVerifier.verify_many:
        # a link may never point to a quantity that does not exist.
        for link in link_list:
            if link.quantity_id in known_quantity_ids:
                continue

            decisions.append(
                VerificationDecision(
                    status=VerificationStatus.INVALID,
                    quantity_id=link.quantity_id,
                    field=link.field,
                    role=link.role,
                    value=None,
                    unit=None,
                    evidence=link.evidence,
                    reasons=[
                        "semantic_link_references_unknown_quantity"
                    ],
                )
            )

        # -------------------------------------------------------------
        # Relation safety.
        # -------------------------------------------------------------
        unsafe_by_quantity = {}
        superseded_by_quantity = {}

        for relation in relation_resolution.relations:
            if relation.relation in self._UNSAFE_RELATIONS:
                for quantity_id in relation.blocked_quantity_ids:
                    unsafe_by_quantity.setdefault(
                        quantity_id,
                        [],
                    ).append(relation)

            if relation.relation == CandidateRelationType.CORRECTION:
                for quantity_id in relation.blocked_quantity_ids:
                    superseded_by_quantity.setdefault(
                        quantity_id,
                        [],
                    ).append(relation)

        guarded: List[VerificationDecision] = []

        for decision in decisions:
            quantity_id = decision.quantity_id

            # Unsafe relation always wins over an otherwise valid scalar
            # candidate. Invalid candidates remain INVALID because they already
            # failed a stronger deterministic check.
            unsafe_relations = unsafe_by_quantity.get(
                quantity_id,
                [],
            )

            if (
                unsafe_relations
                and decision.status == VerificationStatus.VERIFIED
            ):
                guarded.append(
                    self._override(
                        decision=decision,
                        status=VerificationStatus.AMBIGUOUS,
                        relations=unsafe_relations,
                        extra_reason=(
                            "automatic_acceptance_blocked_by_candidate_relation"
                        ),
                    )
                )
                continue

            # In a correction such as "change 20 to 30 GB/s", the old value is
            # not ambiguous: it is superseded. It must simply not be accepted.
            superseded_relations = superseded_by_quantity.get(
                quantity_id,
                [],
            )

            if (
                superseded_relations
                and decision.status == VerificationStatus.VERIFIED
            ):
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
        return self._apply_structured_field_safety(
            guarded
        )

    # =================================================================
    # PLANNING HORIZON
    # =================================================================

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return " ".join(
            (text or "").strip().split()
        )

    def _verify_planning_horizon(
        self,
        quantity: Quantity,
        link: SemanticLink,
        source_text: str,
    ) -> VerificationDecision:
        """
        Deterministic contract for planning_horizon_years.

        No inference of a missing horizon is allowed.
        """
        if link.quantity_id != quantity.id:
            return VerificationDecision(
                status=VerificationStatus.INVALID,
                quantity_id=quantity.id,
                field=link.field,
                role=link.role,
                value=quantity.value,
                unit=quantity.unit,
                evidence=link.evidence,
                reasons=[
                    "quantity_id_mismatch:"
                    f"quantity={quantity.id},"
                    f"link={link.quantity_id}"
                ],
            )

        if link.field != ParamName.planning_horizon_years:
            return VerificationDecision(
                status=VerificationStatus.INVALID,
                quantity_id=quantity.id,
                field=link.field,
                role=link.role,
                value=quantity.value,
                unit=quantity.unit,
                evidence=link.evidence,
                reasons=["planning_horizon_verifier_wrong_field"],
            )

        if link.role != SemanticRole.TARGET:
            return VerificationDecision(
                status=VerificationStatus.INVALID,
                quantity_id=quantity.id,
                field=link.field,
                role=link.role,
                value=quantity.value,
                unit=quantity.unit,
                evidence=link.evidence,
                reasons=[
                    "invalid_field_role_pair:"
                    "planning_horizon_years+"
                    f"{link.role.value}"
                ],
            )

        evidence = (link.evidence or "").strip()
        source_norm = self._normalize_spaces(
            source_text
        ).casefold()
        evidence_norm = self._normalize_spaces(
            evidence
        ).casefold()
        raw_norm = self._normalize_spaces(
            quantity.raw
        ).casefold()

        evidence_errors: List[str] = []

        if not evidence:
            evidence_errors.append(
                "missing_evidence"
            )
        else:
            if (
                source_norm
                and evidence_norm not in source_norm
            ):
                evidence_errors.append(
                    "evidence_not_supported_by_source_text"
                )

            if (
                raw_norm
                and raw_norm not in evidence_norm
            ):
                evidence_errors.append(
                    "evidence_does_not_contain_target_quantity"
                )

        if evidence_errors:
            return VerificationDecision(
                status=VerificationStatus.AMBIGUOUS,
                quantity_id=quantity.id,
                field=link.field,
                role=link.role,
                value=quantity.value,
                unit=quantity.unit,
                evidence=evidence,
                reasons=evidence_errors,
            )

        try:
            numeric = float(
                quantity.value
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return VerificationDecision(
                status=VerificationStatus.INVALID,
                quantity_id=quantity.id,
                field=link.field,
                role=link.role,
                value=quantity.value,
                unit=quantity.unit,
                evidence=evidence,
                reasons=[
                    "planning_horizon_must_be_finite_numeric"
                ],
            )

        if not math.isfinite(numeric):
            return VerificationDecision(
                status=VerificationStatus.INVALID,
                quantity_id=quantity.id,
                field=link.field,
                role=link.role,
                value=quantity.value,
                unit=quantity.unit,
                evidence=evidence,
                reasons=[
                    "planning_horizon_must_be_finite_numeric"
                ],
            )

        if numeric <= 0:
            return VerificationDecision(
                status=VerificationStatus.INVALID,
                quantity_id=quantity.id,
                field=link.field,
                role=link.role,
                value=quantity.value,
                unit="years",
                evidence=evidence,
                reasons=[
                    "planning_horizon_must_be_strictly_positive"
                ],
            )

        if not numeric.is_integer():
            return VerificationDecision(
                status=VerificationStatus.INVALID,
                quantity_id=quantity.id,
                field=link.field,
                role=link.role,
                value=quantity.value,
                unit="years",
                evidence=evidence,
                reasons=[
                    "planning_horizon_years_must_be_integer"
                ],
            )

        source_unit = (
            None
            if quantity.unit is None
            else str(quantity.unit).strip().casefold()
        )

        if source_unit not in self._ALLOWED_HORIZON_SOURCE_UNITS:
            return VerificationDecision(
                status=VerificationStatus.INVALID,
                quantity_id=quantity.id,
                field=link.field,
                role=link.role,
                value=int(numeric),
                unit=quantity.unit,
                evidence=evidence,
                reasons=[
                    "planning_horizon_requires_year_unit:"
                    f"actual={quantity.unit}"
                ],
            )

        is_contextual = (
            link.resolver == "conversation_scope"
        )

        has_explicit_year_evidence = (
            self._YEAR_EVIDENCE_RE.search(
                evidence
            )
            is not None
        )

        if (
            not is_contextual
            and not has_explicit_year_evidence
        ):
            return VerificationDecision(
                status=VerificationStatus.AMBIGUOUS,
                quantity_id=quantity.id,
                field=link.field,
                role=link.role,
                value=int(numeric),
                unit="years",
                evidence=evidence,
                reasons=[
                    "planning_horizon_requires_explicit_year_evidence"
                ],
            )

        return VerificationDecision(
            status=VerificationStatus.VERIFIED,
            quantity_id=quantity.id,
            field=link.field,
            role=link.role,
            value=int(numeric),
            unit="years",
            evidence=evidence,
            reasons=[
                "planning_horizon_field_valid",
                "planning_horizon_role_valid",
                "evidence_supported",
                "planning_horizon_positive_integer",
                "planning_horizon_normalized_to_years",
            ],
        )

    # =================================================================
    # INFO
    # =================================================================

    def info(self) -> dict:
        return {
            "name": "RelationAwareDeterministicVerifier",
            "base_verifier": type(self.base_verifier).__name__,
            "relation_resolver": self.relation_resolver.info(),
            "structured_field_rules": {
                "read_write_ratio": (
                    "requires_two_verified_ratio_components_summing_to_100"
                ),
            },
            "planning_horizon_years": {
                "validation": (
                    "strictly_positive_integer_years"
                ),
                "transformer_label_added": False,
            },
        }

    # =================================================================
    # STRUCTURED FIELD SAFETY
    # =================================================================

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
                reason=(
                    "incomplete_read_write_ratio_requires_both_components"
                ),
            )
            return output

        values: List[float] = []
        numeric = True

        for index in ratio_indices:
            try:
                values.append(
                    float(output[index].value)
                )
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

        if (
            len(ratio_indices) != 2
            or abs(sum(values) - 100.0) > 1e-9
        ):
            for index in ratio_indices:
                output[index] = self._override_without_relation(
                    decision=output[index],
                    status=VerificationStatus.AMBIGUOUS,
                    reason=(
                        "read_write_ratio_components_must_form_"
                        "complete_100_percent_pair"
                    ),
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
            relation_reason = (
                f"candidate_relation:{relation.relation.value}"
            )

            if relation_reason not in reasons:
                reasons.append(relation_reason)

            if (
                relation.reason
                and relation.reason not in reasons
            ):
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