from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .deterministic_verifier import DeterministicVerifier
from .models import (
    ParamName,
    VerificationDecision,
    VerificationStatus,
)

from .selective_cascade import (
    SelectiveCascade,
    SelectiveCascadeResult,
)


@dataclass
class VerifiedPipelineResult:
    """
    Result of the complete quantitative semantic pipeline.

    This stage combines:

        QuantityScanner
            ↓
        Explicit Resolver
            ↓
        Semantic Linker
            ↓
        LLM fallback
            ↓
        Deterministic Verifier

    Only VERIFIED decisions are exposed as accepted requirement values.
    """

    text: str
    cascade: SelectiveCascadeResult
    decisions: List[VerificationDecision]

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

        Nothing AMBIGUOUS, INVALID or UNRESOLVED is allowed into this
        dictionary.
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
    Quantitative Requirement Extractor V2 pipeline.

    Important:
        ConversationScopeResolver is NOT integrated here yet.

    Current responsibility:

        one user message
             ↓
        selective semantic cascade
             ↓
        deterministic verification
             ↓
        safe verified values
    """

    def __init__(
        self,
        cascade: Optional[
            SelectiveCascade
        ] = None,
        verifier: Optional[
            DeterministicVerifier
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

    def process(
        self,
        text: str,
        previous_question: Optional[str] = None,
    ) -> VerifiedPipelineResult:

        # -------------------------------------------------------------
        # 1. Semantic cascade
        # -------------------------------------------------------------

        cascade_result = (
            self.cascade.resolve(
                text=text,
                previous_question=
                    previous_question,
            )
        )

        # -------------------------------------------------------------
        # 2. Deterministic verification
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
        # 3. Safe result
        # -------------------------------------------------------------

        return VerifiedPipelineResult(
            text=text,
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
                "not_integrated_yet",
        }