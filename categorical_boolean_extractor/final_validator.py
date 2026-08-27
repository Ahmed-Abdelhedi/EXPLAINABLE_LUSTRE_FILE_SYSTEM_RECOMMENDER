from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .models import (
    FieldResult,
    FieldStatus,
    ResolutionSource,
)
from .semantic.labels import (
    AccessSemanticLabel,
    HASemanticLabel,
)
from .text_utils import fold


class FinalSemanticValidator:
    """
    Final safety gate for residual LLM proposals.

    The LLM never creates a trusted production value directly.
    A VERIFIED proposal must:
      1. use an allowed semantic label,
      2. provide an exact non-empty substring from the user text,
      3. pass the deterministic access safety rule when applicable,
      4. be independently re-accepted by the frozen semantic model.

    Only then is a canonical ha_required/access_type value exposed.
    """

    @staticmethod
    def _exact_evidence(
        text: str,
        evidence: object,
    ) -> Optional[str]:
        if not isinstance(evidence, str):
            return None

        if not evidence:
            return None

        if evidence not in text:
            return None

        return evidence

    def validate_ha(
        self,
        *,
        text: str,
        proposal: Dict[str, Any],
        semantic_verifier: Any,
    ) -> FieldResult:
        status = str(
            proposal.get("status", "UNRESOLVED")
        ).upper()

        raw_label = proposal.get("label")
        label: Optional[str] = (
            raw_label
            if isinstance(raw_label, str)
            else None
        )

        if status == "NO_EVIDENCE":
            return FieldResult(
                field="ha_required",
                status=FieldStatus.NO_EVIDENCE,
                value=None,
                source=ResolutionSource.LLM_FALLBACK,
                reason=(
                    "LLM_NO_EVIDENCE_NO_VALUE_EXPOSED"
                ),
            )

        if status != "VERIFIED":
            return FieldResult(
                field="ha_required",
                status=FieldStatus.UNRESOLVED,
                value=None,
                source=ResolutionSource.NONE,
                reason="LLM_HA_UNRESOLVED",
            )

        allowed_labels = {
            HASemanticLabel.HA_REQUIRED.value,
            HASemanticLabel.HA_NOT_REQUIRED.value,
        }

        if label not in allowed_labels:
            return FieldResult(
                field="ha_required",
                status=FieldStatus.UNRESOLVED,
                value=None,
                source=ResolutionSource.NONE,
                reason=(
                    "LLM_HA_NON_COMMITMENT_"
                    "LABEL_NOT_ACTIONABLE"
                ),
            )

        evidence = self._exact_evidence(
            text,
            proposal.get("evidence"),
        )

        if evidence is None:
            return FieldResult(
                field="ha_required",
                status=FieldStatus.UNRESOLVED,
                value=None,
                source=ResolutionSource.NONE,
                reason=(
                    "LLM_HA_EVIDENCE_NOT_EXACT_SUBSTRING"
                ),
            )

        semantic = semantic_verifier.verify_ha(
            evidence
        )

        if (
            not semantic.accepted
            or semantic.label != label
        ):
            return FieldResult(
                field="ha_required",
                status=FieldStatus.UNRESOLVED,
                value=None,
                source=ResolutionSource.NONE,
                evidence=evidence,
                reason=(
                    "LLM_HA_REJECTED_BY_"
                    "EVIDENCE_SEMANTIC_RECHECK"
                ),
                semantic_label=semantic.label,
                semantic_confidence=(
                    semantic
                    .head_output
                    .top_probability
                ),
            )

        ha_value: bool = (
            label
            == HASemanticLabel.HA_REQUIRED.value
        )

        return FieldResult(
            field="ha_required",
            status=FieldStatus.VERIFIED,
            value=ha_value,
            source=ResolutionSource.LLM_FALLBACK,
            evidence=evidence,
            reason=(
                "LLM_HA_ACCEPTED_AFTER_"
                "SEMANTIC_EVIDENCE_RECHECK"
            ),
            semantic_label=label,
            semantic_confidence=(
                semantic
                .head_output
                .top_probability
            ),
        )

    def validate_access(
        self,
        *,
        text: str,
        proposal: Dict[str, Any],
        semantic_verifier: Any,
    ) -> FieldResult:
        status = str(
            proposal.get("status", "UNRESOLVED")
        ).upper()

        raw_label = proposal.get("label")
        label: Optional[str] = (
            raw_label
            if isinstance(raw_label, str)
            else None
        )

        if status == "NO_EVIDENCE":
            return FieldResult(
                field="access_type",
                status=FieldStatus.NO_EVIDENCE,
                value=None,
                source=ResolutionSource.LLM_FALLBACK,
                reason=(
                    "LLM_NO_EVIDENCE_NO_VALUE_EXPOSED"
                ),
            )

        if status != "VERIFIED":
            return FieldResult(
                field="access_type",
                status=FieldStatus.UNRESOLVED,
                value=None,
                source=ResolutionSource.NONE,
                reason="LLM_ACCESS_UNRESOLVED",
            )

        allowed_labels = {
            AccessSemanticLabel.SEQUENTIAL.value,
            AccessSemanticLabel.RANDOM.value,
            AccessSemanticLabel.MIXED.value,
        }

        if label not in allowed_labels:
            return FieldResult(
                field="access_type",
                status=FieldStatus.UNRESOLVED,
                value=None,
                source=ResolutionSource.NONE,
                reason=(
                    "LLM_ACCESS_NON_ACTIONABLE_LABEL"
                ),
            )

        evidence = self._exact_evidence(
            text,
            proposal.get("evidence"),
        )

        if evidence is None:
            return FieldResult(
                field="access_type",
                status=FieldStatus.UNRESOLVED,
                value=None,
                source=ResolutionSource.NONE,
                reason=(
                    "LLM_ACCESS_EVIDENCE_"
                    "NOT_EXACT_SUBSTRING"
                ),
            )

        folded_evidence: str = fold(evidence)

        has_parallel = bool(
            re.search(
                r"\b("
                r"parallel|concurrent|"
                r"parallele|simultane"
                r")\b",
                folded_evidence,
            )
        )

        has_order = bool(
            re.search(
                r"\b("
                r"sequential|sequentiel|streaming|"
                r"random|aleatoire|mixed|mixte|"
                r"contiguous|scattered|offset"
                r")\b",
                folded_evidence,
            )
        )

        if has_parallel and not has_order:
            return FieldResult(
                field="access_type",
                status=FieldStatus.UNRESOLVED,
                value=None,
                source=ResolutionSource.NONE,
                evidence=evidence,
                reason=(
                    "LLM_PARALLEL_ONLY_EVIDENCE_REJECTED"
                ),
            )

        semantic = (
            semantic_verifier.verify_access(
                evidence
            )
        )

        if (
            not semantic.accepted
            or semantic.label != label
        ):
            return FieldResult(
                field="access_type",
                status=FieldStatus.UNRESOLVED,
                value=None,
                source=ResolutionSource.NONE,
                evidence=evidence,
                reason=(
                    "LLM_ACCESS_REJECTED_BY_"
                    "EVIDENCE_SEMANTIC_RECHECK"
                ),
                semantic_label=semantic.label,
                semantic_confidence=(
                    semantic
                    .head_output
                    .top_probability
                ),
            )

        access_value_by_label = {
            AccessSemanticLabel.SEQUENTIAL.value:
                "sequential",
            AccessSemanticLabel.RANDOM.value:
                "random",
            AccessSemanticLabel.MIXED.value:
                "mixed",
        }

        access_value: str = (
            access_value_by_label[label]
        )

        return FieldResult(
            field="access_type",
            status=FieldStatus.VERIFIED,
            value=access_value,
            source=ResolutionSource.LLM_FALLBACK,
            evidence=evidence,
            reason=(
                "LLM_ACCESS_ACCEPTED_AFTER_"
                "SEMANTIC_EVIDENCE_RECHECK"
            ),
            semantic_label=label,
            semantic_confidence=(
                semantic
                .head_output
                .top_probability
            ),
        )
