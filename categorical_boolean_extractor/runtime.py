from __future__ import annotations
from pathlib import Path
from typing import Optional
from .artifact_contract import FROZEN_ARTIFACT_CONTRACT
from .explicit import AccessTypeExplicitResolver, HAExplicitResolver
from .final_validator import FinalSemanticValidator
from .llm_fallback import CategoricalBooleanLLMFallback
from .models import CategoricalBooleanExtractionResult, FieldResult, FieldStatus, ResolutionSource
from .semantic.labels import AccessSemanticLabel, HASemanticLabel
from .semantic.runtime import SemanticVerifier

class CategoricalBooleanExtractor:
    """
    explicit -> shared XLM-R semantic -> calibrated gate
    -> Qwen only on semantic abstention
    -> exact evidence + semantic evidence re-check
    """
    def __init__(self, *, semantic_verifier: SemanticVerifier, llm_fallback: Optional[CategoricalBooleanLLMFallback]=None) -> None:
        self.ha_explicit = HAExplicitResolver()
        self.access_explicit = AccessTypeExplicitResolver()
        self.semantic_verifier = semantic_verifier
        self.llm_fallback = llm_fallback if llm_fallback is not None else CategoricalBooleanLLMFallback()
        self.final_validator = FinalSemanticValidator()

    @classmethod
    def from_artifact(
        cls,
        artifact_zip,
        *,
        expected_sha256=None,
        device=None,
        llm_fallback=None,
    ):
        semantic = SemanticVerifier.from_artifact_zip(
            artifact_zip,
            expected_sha256=expected_sha256,
            device=device,
        )
        return cls(
            semantic_verifier=semantic,
            llm_fallback=llm_fallback,
        )

    @classmethod
    def from_default_artifact(
        cls,
        *,
        device=None,
        llm_fallback=None,
    ):
        artifact_path = (
            FROZEN_ARTIFACT_CONTRACT.default_artifact_path()
        )
        return cls.from_artifact(
            artifact_path,
            expected_sha256=FROZEN_ARTIFACT_CONTRACT.sha256,
            device=device,
            llm_fallback=llm_fallback,
        )

    def _semantic_ha(self, text: str):
        decision = self.semantic_verifier.verify_ha(text)
        if not decision.accepted:
            return FieldResult(
                "ha_required", FieldStatus.UNRESOLVED, None, ResolutionSource.SEMANTIC_MODEL,
                reason="SEMANTIC_HA_ABSTAIN",
                semantic_label=decision.head_output.top_label,
                semantic_confidence=decision.head_output.top_probability,
            ), True

        label = decision.label
        conf = decision.head_output.top_probability
        if label == HASemanticLabel.HA_REQUIRED.value:
            return FieldResult("ha_required", FieldStatus.VERIFIED, True, ResolutionSource.SEMANTIC_MODEL, text, "SEMANTIC_HA_REQUIRED_ACCEPTED", label, conf), False
        if label == HASemanticLabel.HA_NOT_REQUIRED.value:
            return FieldResult("ha_required", FieldStatus.VERIFIED, False, ResolutionSource.SEMANTIC_MODEL, text, "SEMANTIC_HA_NOT_REQUIRED_ACCEPTED", label, conf), False

        return FieldResult(
            "ha_required", FieldStatus.NO_EVIDENCE, None, ResolutionSource.SEMANTIC_MODEL,
            text, f"SEMANTIC_NON_ACTIONABLE_HA_CLASS:{label}", label, conf
        ), False

    def _semantic_access(self, text: str):
        decision = self.semantic_verifier.verify_access(text)
        if not decision.accepted:
            return FieldResult(
                "access_type", FieldStatus.UNRESOLVED, None, ResolutionSource.SEMANTIC_MODEL,
                reason="SEMANTIC_ACCESS_ABSTAIN",
                semantic_label=decision.head_output.top_label,
                semantic_confidence=decision.head_output.top_probability,
            ), True

        label = decision.label
        conf = decision.head_output.top_probability
        mapping = {
            AccessSemanticLabel.SEQUENTIAL.value: "sequential",
            AccessSemanticLabel.RANDOM.value: "random",
            AccessSemanticLabel.MIXED.value: "mixed",
        }
        if label in mapping:
            return FieldResult("access_type", FieldStatus.VERIFIED, mapping[label], ResolutionSource.SEMANTIC_MODEL, text, f"SEMANTIC_ACCESS_ACCEPTED:{label}", label, conf), False

        return FieldResult(
            "access_type", FieldStatus.UNRESOLVED, None, ResolutionSource.SEMANTIC_MODEL,
            text, "SEMANTIC_NO_SUPPORTED_ACCESS_CLASS", label, conf
        ), False

    def extract(self, text: str, *, ha_question_context=None, access_question_context=None):
        if not isinstance(text, str):
            raise TypeError("CategoricalBooleanExtractor input must be a string.")

        ha = self.ha_explicit.resolve(text, question_context=ha_question_context)
        access = self.access_explicit.resolve(text, question_context=access_question_context)
        semantic_used = False
        semantic_abstained = []
        llm_requested = []

        if ha.status != FieldStatus.VERIFIED:
            semantic_used = True
            ha, abstained = self._semantic_ha(text)
            if abstained:
                semantic_abstained.append("ha_required")
                llm_requested.append("ha_required")

        if access.status != FieldStatus.VERIFIED:
            semantic_used = True
            access, abstained = self._semantic_access(text)
            if abstained:
                semantic_abstained.append("access_type")
                llm_requested.append("access_type")

        before = self.llm_fallback.call_count
        proposals = self.llm_fallback.propose(text=text, requested_fields=llm_requested)
        llm_used = self.llm_fallback.call_count > before

        if "ha_required" in proposals:
            ha = self.final_validator.validate_ha(
                text=text,
                proposal=proposals["ha_required"],
                semantic_verifier=self.semantic_verifier,
            )
        if "access_type" in proposals:
            access = self.final_validator.validate_access(
                text=text,
                proposal=proposals["access_type"],
                semantic_verifier=self.semantic_verifier,
            )

        return CategoricalBooleanExtractionResult(
            text=text,
            ha_required=ha,
            access_type=access,
            semantic_model_used=semantic_used,
            semantic_abstained_fields=semantic_abstained,
            llm_fallback_used=llm_used,
            llm_fallback_fields=list(self.llm_fallback.last_requested_fields) if llm_used else [],
        )
