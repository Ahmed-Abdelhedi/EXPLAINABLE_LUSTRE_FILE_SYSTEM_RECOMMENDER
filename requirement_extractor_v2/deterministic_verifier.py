from __future__ import annotations

import math
from decimal import Decimal
from typing import Dict, Iterable, List, Optional

from .models import (
    ParamName,
    Quantity,
    QuantityDimension,
    SemanticLink,
    VerificationDecision,
    VerificationStatus,
)

from .unit_normalizer import (
    normalize_unit_value,
    validate_source_unit_for_field,
)

from .semantic_linker.compatibility import (
    is_field_allowed_for_dimension,
    is_valid_field_role_pair,
)

from .semantic_linker.labels import SemanticField


class DeterministicVerifier:
    """
    Final deterministic verification layer for quantitative extraction.

    Input:
        Quantity + SemanticLink

    Output:
        VerificationDecision

    Possible statuses:
        VERIFIED
        AMBIGUOUS
        INVALID
        UNRESOLVED

    This component never calls:
        - the Transformer;
        - the LLM;
        - the QuantityScanner.

    It never guesses or repairs a semantic mapping.
    """

    # =================================================================
    # FIELD POLICIES
    # =================================================================

    QUANTITATIVE_FIELDS = frozenset(
        {
            ParamName.requested_usable_capacity_tib,
            ParamName.client_count,
            ParamName.average_file_size_gb,
            ParamName.max_file_size_gb,
            ParamName.total_file_count,
            ParamName.read_write_ratio,
            ParamName.target_read_gbps,
            ParamName.target_write_gbps,
            ParamName.max_budget_usd,
            ParamName.max_power_w,
            ParamName.annual_growth_percent,
        }
    )

    STRICTLY_POSITIVE_FIELDS = frozenset(
        {
            ParamName.requested_usable_capacity_tib,
            ParamName.client_count,
            ParamName.average_file_size_gb,
            ParamName.max_file_size_gb,
            ParamName.total_file_count,
            ParamName.target_read_gbps,
            ParamName.target_write_gbps,
            ParamName.max_budget_usd,
            ParamName.max_power_w,
        }
    )

    INTEGER_FIELDS = frozenset(
        {
            ParamName.client_count,
            ParamName.total_file_count,
        }
    )

    # UNKNOWN unitless quantities may safely reach final verification
    # directly only for count-like concepts.
    UNKNOWN_DIRECT_FIELDS = frozenset(
        {
            ParamName.client_count,
            ParamName.total_file_count,
        }
    )

    CANONICAL_UNITS = {
        ParamName.requested_usable_capacity_tib: "TiB",
        ParamName.client_count: None,
        ParamName.average_file_size_gb: "GB",
        ParamName.max_file_size_gb: "GB",
        ParamName.total_file_count: None,
        ParamName.read_write_ratio: "%",
        ParamName.target_read_gbps: "GB/s",
        ParamName.target_write_gbps: "GB/s",
        ParamName.max_budget_usd: "USD",
        ParamName.max_power_w: "W",
        ParamName.annual_growth_percent: "%",
    }

    # =================================================================
    # BASIC HELPERS
    # =================================================================

    @staticmethod
    def _is_number(value) -> bool:
        return (
            isinstance(
                value,
                (int, float, Decimal),
            )
            and not isinstance(value, bool)
        )

    @classmethod
    def _is_finite_number(
        cls,
        value,
    ) -> bool:
        if not cls._is_number(value):
            return False

        try:
            return math.isfinite(
                float(value)
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return False

    @staticmethod
    def _normalize_spaces(
        text: str,
    ) -> str:
        return " ".join(
            (text or "")
            .strip()
            .split()
        )

    # =================================================================
    # DECISION HELPERS
    # =================================================================

    @staticmethod
    def _decision(
        *,
        status: VerificationStatus,
        quantity: Optional[Quantity],
        link: Optional[SemanticLink],
        value=None,
        unit=None,
        reasons: Optional[List[str]] = None,
    ) -> VerificationDecision:

        return VerificationDecision(
            status=status,
            quantity_id=(
                None
                if quantity is None
                else quantity.id
            ),
            field=(
                None
                if link is None
                else link.field
            ),
            role=(
                None
                if link is None
                else link.role
            ),
            value=value,
            unit=unit,
            evidence=(
                ""
                if link is None
                else link.evidence
            ),
            reasons=list(
                reasons or []
            ),
        )

    # =================================================================
    # EVIDENCE VERIFICATION
    # =================================================================

    def _verify_evidence(
        self,
        quantity: Quantity,
        link: SemanticLink,
        source_text: str,
    ) -> List[str]:
        """
        Evidence must:
        - exist;
        - originate from the user text;
        - contain the target quantity.

        Returns a list of ambiguity reasons.
        """

        reasons: List[str] = []

        evidence = (
            link.evidence or ""
        ).strip()

        if not evidence:
            reasons.append(
                "missing_evidence"
            )
            return reasons

        evidence_norm = (
            self._normalize_spaces(
                evidence
            ).casefold()
        )

        source_norm = (
            self._normalize_spaces(
                source_text
            ).casefold()
        )

        if (
            source_norm
            and evidence_norm
            not in source_norm
        ):
            reasons.append(
                "evidence_not_supported_by_source_text"
            )

        raw_norm = (
            self._normalize_spaces(
                quantity.raw
            ).casefold()
        )

        if (
            raw_norm
            and raw_norm
            not in evidence_norm
        ):
            reasons.append(
                "evidence_does_not_contain_target_quantity"
            )

        return reasons

    # =================================================================
    # FIELD / DIMENSION VERIFICATION
    # =================================================================

    def _verify_field_dimension(
        self,
        quantity: Quantity,
        link: SemanticLink,
    ) -> Optional[str]:

        if link.field is None:
            return "semantic_field_is_unresolved"

        if (
            link.field
            not in self.QUANTITATIVE_FIELDS
        ):
            return (
                "non_quantitative_field_in_"
                "quantitative_verifier"
            )

        try:
            semantic_field = (
                SemanticField(
                    link.field.value
                )
            )

        except ValueError:
            return "unknown_semantic_field"

        if not is_field_allowed_for_dimension(
            quantity.dimension,
            semantic_field,
        ):
            return (
                "field_incompatible_with_"
                f"dimension:{quantity.dimension.value}"
            )

        return None

    # =================================================================
    # ROLE VERIFICATION
    # =================================================================

    @staticmethod
    def _verify_role(
        link: SemanticLink,
    ) -> Optional[str]:

        if link.field is None:
            return "field_missing"

        try:
            semantic_field = (
                SemanticField(
                    link.field.value
                )
            )

        except ValueError:
            return "unknown_semantic_field"

        if not is_valid_field_role_pair(
            semantic_field,
            link.role,
        ):
            return (
                "invalid_field_role_pair:"
                f"{link.field.value}+"
                f"{link.role.value}"
            )

        return None

    # =================================================================
    # UNKNOWN DIMENSION POLICY
    # =================================================================

    def _verify_unknown_dimension(
        self,
        quantity: Quantity,
        link: SemanticLink,
    ) -> Optional[str]:
        """
        UNKNOWN is intentionally broad during ML inference.

        Final verification is more conservative.

        A unitless UNKNOWN quantity can be directly verified only for
        count-like fields.

        Other mappings require explicit unit/context resolution first.
        """

        if (
            quantity.dimension
            is not QuantityDimension.UNKNOWN
        ):
            return None

        if (
            link.field
            in self.UNKNOWN_DIRECT_FIELDS
        ):
            return None

        return (
            "unknown_dimension_requires_"
            "explicit_unit_or_context"
        )

    # =================================================================
    # NORMALIZATION
    # =================================================================

    def _normalize_value(
        self,
        quantity: Quantity,
        field: ParamName,
    ):
        """
        Normalize Quantity value into the canonical Requirement Contract
        unit.

        Ratio components are handled separately because they are scalar
        components of the future read_write_ratio structure.
        """

        if field == ParamName.read_write_ratio:
            return (
                quantity.value,
                "%",
            )

        return normalize_unit_value(
            field=field,
            value=quantity.value,
            unit=quantity.unit,
        )

    # =================================================================
    # VALUE VALIDATION
    # =================================================================

    def _validate_value(
        self,
        field: ParamName,
        value,
    ) -> Optional[str]:

        if value is None or value == "":
            return "empty_value"

        # -------------------------------------------------------------
        # All fields handled by this verifier are quantitative
        # -------------------------------------------------------------

        if not self._is_finite_number(
            value
        ):
            return "value_must_be_finite_numeric"

        numeric = float(value)

        # -------------------------------------------------------------
        # Strictly positive fields
        # -------------------------------------------------------------

        if (
            field
            in self.STRICTLY_POSITIVE_FIELDS
            and numeric <= 0
        ):
            return (
                "value_must_be_strictly_positive"
            )

        # -------------------------------------------------------------
        # Integer counts
        # -------------------------------------------------------------

        if (
            field
            in self.INTEGER_FIELDS
            and not numeric.is_integer()
        ):
            return (
                "count_value_must_be_integer"
            )

        # -------------------------------------------------------------
        # Annual growth
        # Existing project policy allows zero growth.
        # -------------------------------------------------------------

        if (
            field
            == ParamName.annual_growth_percent
            and numeric < 0
        ):
            return (
                "annual_growth_cannot_be_negative"
            )

        # -------------------------------------------------------------
        # Ratio component
        # -------------------------------------------------------------

        if (
            field
            == ParamName.read_write_ratio
            and not 0 <= numeric <= 100
        ):
            return (
                "ratio_component_must_be_between_0_and_100"
            )

        return None

    # =================================================================
    # UNIT VALIDATION
    # =================================================================

    def _validate_canonical_unit(
        self,
        field: ParamName,
        unit: Optional[str],
    ) -> Optional[str]:

        expected = (
            self.CANONICAL_UNITS[field]
        )

        if expected is None:

            if unit is not None:
                return (
                    "count_field_must_not_have_unit"
                )

            return None

        if unit != expected:
            return (
                "unexpected_canonical_unit:"
                f"expected={expected},"
                f"actual={unit}"
            )

        return None

    # =================================================================
    # PUBLIC API
    # =================================================================

    def verify(
        self,
        quantity: Quantity,
        link: Optional[SemanticLink],
        source_text: Optional[str] = None,
    ) -> VerificationDecision:
        """
        Verify one Quantity → SemanticLink mapping.
        """

        source_text = (
            source_text
            if source_text is not None
            else quantity.source_text
        )

        # -------------------------------------------------------------
        # 1. No semantic resolution
        # -------------------------------------------------------------

        if link is None:
            return self._decision(
                status=(
                    VerificationStatus.UNRESOLVED
                ),
                quantity=quantity,
                link=None,
                value=quantity.value,
                unit=quantity.unit,
                reasons=[
                    "no_semantic_link"
                ],
            )

        # -------------------------------------------------------------
        # 2. Quantity identity
        # -------------------------------------------------------------

        if (
            link.quantity_id
            != quantity.id
        ):
            return self._decision(
                status=(
                    VerificationStatus.INVALID
                ),
                quantity=quantity,
                link=link,
                value=quantity.value,
                unit=quantity.unit,
                reasons=[
                    "quantity_id_mismatch:"
                    f"quantity={quantity.id},"
                    f"link={link.quantity_id}"
                ],
            )

        # -------------------------------------------------------------
        # 3. FIELD
        # -------------------------------------------------------------

        field_error = (
            self._verify_field_dimension(
                quantity=quantity,
                link=link,
            )
        )

        if field_error is not None:

            status = (
                VerificationStatus.UNRESOLVED
                if link.field is None
                else VerificationStatus.INVALID
            )

            return self._decision(
                status=status,
                quantity=quantity,
                link=link,
                value=quantity.value,
                unit=quantity.unit,
                reasons=[
                    field_error
                ],
            )

        assert link.field is not None

        # -------------------------------------------------------------
        # 4. ROLE
        # -------------------------------------------------------------

        role_error = (
            self._verify_role(link)
        )

        if role_error is not None:
            return self._decision(
                status=(
                    VerificationStatus.INVALID
                ),
                quantity=quantity,
                link=link,
                value=quantity.value,
                unit=quantity.unit,
                reasons=[
                    role_error
                ],
            )

        # -------------------------------------------------------------
        # 5. Explicit source-unit compatibility
        # -------------------------------------------------------------

        try:
            validate_source_unit_for_field(
                field=link.field,
                unit=quantity.unit,
            )

        except ValueError as exc:
            return self._decision(
                status=VerificationStatus.INVALID,
                quantity=quantity,
                link=link,
                value=quantity.value,
                unit=quantity.unit,
                reasons=[str(exc)],
            )

        # -------------------------------------------------------------
        # 6. UNKNOWN-dimension final guard
        # -------------------------------------------------------------

        unknown_reason = (
            self._verify_unknown_dimension(
                quantity=quantity,
                link=link,
            )
        )

        if unknown_reason is not None:
            return self._decision(
                status=(
                    VerificationStatus.AMBIGUOUS
                ),
                quantity=quantity,
                link=link,
                value=quantity.value,
                unit=quantity.unit,
                reasons=[
                    unknown_reason
                ],
            )

        # -------------------------------------------------------------
        # 7. Evidence
        # -------------------------------------------------------------

        evidence_reasons = (
            self._verify_evidence(
                quantity=quantity,
                link=link,
                source_text=source_text or "",
            )
        )

        if evidence_reasons:
            return self._decision(
                status=(
                    VerificationStatus.AMBIGUOUS
                ),
                quantity=quantity,
                link=link,
                value=quantity.value,
                unit=quantity.unit,
                reasons=evidence_reasons,
            )

        # -------------------------------------------------------------
        # 8. Normalize value + unit
        # -------------------------------------------------------------

        try:
            (
                normalized_value,
                normalized_unit,
            ) = self._normalize_value(
                quantity=quantity,
                field=link.field,
            )

        except Exception as exc:
            return self._decision(
                status=(
                    VerificationStatus.INVALID
                ),
                quantity=quantity,
                link=link,
                value=quantity.value,
                unit=quantity.unit,
                reasons=[
                    "normalization_failed:"
                    f"{type(exc).__name__}:"
                    f"{exc}"
                ],
            )

        # -------------------------------------------------------------
        # 9. Value business validation
        # -------------------------------------------------------------

        value_error = (
            self._validate_value(
                field=link.field,
                value=normalized_value,
            )
        )

        if value_error is not None:
            return self._decision(
                status=(
                    VerificationStatus.INVALID
                ),
                quantity=quantity,
                link=link,
                value=normalized_value,
                unit=normalized_unit,
                reasons=[
                    value_error
                ],
            )

        # -------------------------------------------------------------
        # 10. Canonical unit contract
        # -------------------------------------------------------------

        unit_error = (
            self._validate_canonical_unit(
                field=link.field,
                unit=normalized_unit,
            )
        )

        if unit_error is not None:
            return self._decision(
                status=(
                    VerificationStatus.INVALID
                ),
                quantity=quantity,
                link=link,
                value=normalized_value,
                unit=normalized_unit,
                reasons=[
                    unit_error
                ],
            )

        # -------------------------------------------------------------
        # 11. VERIFIED
        # -------------------------------------------------------------

        return self._decision(
            status=(
                VerificationStatus.VERIFIED
            ),
            quantity=quantity,
            link=link,
            value=normalized_value,
            unit=normalized_unit,
            reasons=[
                "field_dimension_valid",
                "field_role_valid",
                "evidence_supported",
                "value_valid",
                "unit_normalized",
            ],
        )

    # =================================================================
    # BATCH API
    # =================================================================

    def verify_many(
        self,
        quantities: Iterable[Quantity],
        links: Iterable[SemanticLink],
        source_text: Optional[str] = None,
    ) -> List[VerificationDecision]:
        """
        Verify an entire SelectiveCascade output.

        Exactly one SemanticLink is expected at most for each Quantity.
        """

        quantity_list = list(
            quantities
        )

        link_list = list(
            links
        )

        links_by_quantity: Dict[
            str,
            List[SemanticLink],
        ] = {}

        for link in link_list:
            links_by_quantity.setdefault(
                link.quantity_id,
                [],
            ).append(link)

        decisions: List[
            VerificationDecision
        ] = []

        known_quantity_ids = {
            quantity.id
            for quantity in quantity_list
        }

        # -------------------------------------------------------------
        # Quantities
        # -------------------------------------------------------------

        for quantity in quantity_list:

            quantity_links = (
                links_by_quantity.get(
                    quantity.id,
                    [],
                )
            )

            if not quantity_links:

                decisions.append(
                    self.verify(
                        quantity=quantity,
                        link=None,
                        source_text=source_text,
                    )
                )

                continue

            if len(quantity_links) > 1:

                decisions.append(
                    self._decision(
                        status=(
                            VerificationStatus.AMBIGUOUS
                        ),
                        quantity=quantity,
                        link=quantity_links[0],
                        value=quantity.value,
                        unit=quantity.unit,
                        reasons=[
                            "multiple_semantic_links_for_same_quantity"
                        ],
                    )
                )

                continue

            decisions.append(
                self.verify(
                    quantity=quantity,
                    link=quantity_links[0],
                    source_text=source_text,
                )
            )

        # -------------------------------------------------------------
        # Defensive check:
        # links pointing to quantities that do not exist
        # -------------------------------------------------------------

        for link in link_list:

            if (
                link.quantity_id
                in known_quantity_ids
            ):
                continue

            decisions.append(
                VerificationDecision(
                    status=(
                        VerificationStatus.INVALID
                    ),
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

        return decisions