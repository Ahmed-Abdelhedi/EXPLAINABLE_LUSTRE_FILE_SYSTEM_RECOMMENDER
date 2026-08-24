from __future__ import annotations

from dataclasses import dataclass, field as dc_field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# FINAL REQUIREMENT FIELDS
# ============================================================


class ParamName(str, Enum):
    """Champs officiels du Requirement Contract final."""

    requested_usable_capacity_tib = "requested_usable_capacity_tib"
    client_count = "client_count"
    average_file_size_gb = "average_file_size_gb"
    max_file_size_gb = "max_file_size_gb"
    total_file_count = "total_file_count"
    read_write_ratio = "read_write_ratio"
    access_type = "access_type"
    target_read_gbps = "target_read_gbps"
    target_write_gbps = "target_write_gbps"
    ha_required = "ha_required"
    max_budget_usd = "max_budget_usd"
    max_power_w = "max_power_w"
    annual_growth_percent = "annual_growth_percent"
    planning_horizon_years = "planning_horizon_years"


# ============================================================
# NEW SELECTIVE-CASCADE MODELS
# ============================================================


class ScopeIntent(str, Enum):
    """Classification du message avant extraction."""

    NEW_REQUIREMENT = "NEW_REQUIREMENT"
    ANSWER_TO_PREVIOUS_QUESTION = "ANSWER_TO_PREVIOUS_QUESTION"
    CORRECTION = "CORRECTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNKNOWN = "UNKNOWN"


class QuantityDimension(str, Enum):
    """Dimension physique / métier d'une quantité détectée."""

    CAPACITY = "capacity"
    FILE_SIZE = "file_size"
    THROUGHPUT = "throughput"
    POWER = "power"
    MONEY = "money"
    PERCENT = "percent"
    COUNT = "count"
    UNKNOWN = "unknown"


class QuantityDetection(str, Enum):
    """Méthode utilisée pour détecter / reconstruire une quantité.

    Les valeurs sérialisées restent volontairement stables et explicites
    afin de conserver la provenance de la détection dans les rapports,
    les tests et la future vérification déterministe.
    """

    UNKNOWN = "unknown"
    DIGIT = "digit"
    NUMBER_WORDS = "number_words"
    DIGIT_WITH_UNIT = "digit_with_unit"
    NUMBER_WORDS_WITH_UNIT = "number_words_with_unit"
    FUZZY_NUMBER_WORDS = "fuzzy_number_words"
    FUZZY_NUMBER_WORDS_WITH_UNIT = "fuzzy_number_words_with_unit"


class SemanticRole(str, Enum):
    """Rôle sémantique interne joué par une quantité."""

    MAXIMUM_LIMIT = "maximum_limit"
    MINIMUM_LIMIT = "minimum_limit"
    TARGET = "target"
    CURRENT_VALUE = "current_value"
    EXPECTED_VALUE = "expected_value"
    AVERAGE_VALUE = "average_value"
    TOTAL_COUNT = "total_count"
    RATIO_COMPONENT = "ratio_component"
    GROWTH_RATE = "growth_rate"
    UNSPECIFIED = "unspecified"


class VerificationStatus(str, Enum):
    """Résultat de la future vérification déterministe."""

    VERIFIED = "VERIFIED"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class Quantity:
    """Quantité détectée par le QuantityScanner.

    ``raw`` conserve exactement le texte fourni par l'utilisateur pour le
    span détecté. ``normalized`` conserve la forme utilisée pour le parsing
    lorsque le scanner a normalisé ou corrigé une expression numérique.

    Important : cette structure décrit uniquement *la quantité détectée*.
    Elle ne choisit ni le champ final du Requirement Contract ni son rôle
    sémantique.
    """

    id: str
    raw: str
    value: Any
    unit: Optional[str]
    dimension: QuantityDimension
    start: int
    end: int
    source_text: str = ""

    # Champs ajoutés pour le QuantityScanner enrichi.
    # Ils ont des valeurs par défaut afin de préserver la compatibilité
    # avec les composants déjà validés pendant la migration.
    normalized: str = ""
    detection: QuantityDetection = QuantityDetection.UNKNOWN
    corrected: bool = False

    def __post_init__(self) -> None:
        # Les anciennes constructions de Quantity ne renseignent pas encore
        # ``normalized``. Dans ce cas, le texte brut est déjà la meilleure
        # représentation disponible.
        if not self.normalized:
            self.normalized = self.raw

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "raw": self.raw,
            "normalized": self.normalized,
            "value": self.value,
            "unit": self.unit,
            "dimension": self.dimension.value,
            "start": self.start,
            "end": self.end,
            "source_text": self.source_text,
            "detection": self.detection.value,
            "corrected": self.corrected,
        }


@dataclass
class ScopeResolution:
    """Résultat du futur ConversationScopeResolver."""

    intent: ScopeIntent
    target_field: Optional[ParamName] = None
    inherited_unit: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "target_field": (
                None if self.target_field is None else self.target_field.value
            ),
            "inherited_unit": self.inherited_unit,
            "reason": self.reason,
        }


@dataclass
class SemanticLink:
    """Lien intermédiaire entre une quantité et un champ métier."""

    quantity_id: str
    field: Optional[ParamName]
    role: SemanticRole
    evidence: str = ""
    resolver: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantity_id": self.quantity_id,
            "field": None if self.field is None else self.field.value,
            "role": self.role.value,
            "evidence": self.evidence,
            "resolver": self.resolver,
        }


@dataclass
class VerificationDecision:
    """Décision produite par le futur DeterministicVerifier."""

    status: VerificationStatus
    quantity_id: Optional[str] = None
    field: Optional[ParamName] = None
    role: Optional[SemanticRole] = None
    value: Any = None
    unit: Optional[str] = None
    evidence: str = ""
    reasons: List[str] = dc_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "quantity_id": self.quantity_id,
            "field": None if self.field is None else self.field.value,
            "role": None if self.role is None else self.role.value,
            "value": self.value,
            "unit": self.unit,
            "evidence": self.evidence,
            "reasons": self.reasons,
        }


# ============================================================
# EXISTING PIPELINE MODELS
# Conservés temporairement pour compatibilité pendant la migration.
# ============================================================


class CandidateSource(str, Enum):
    RULE = "RULE"
    NORMALIZER = "NORMALIZER"
    LLM_FALLBACK = "LLM_FALLBACK"
    USER_CLARIFICATION = "USER_CLARIFICATION"
    CONVERSATION_CONTEXT = "CONVERSATION_CONTEXT"
    EXPLICIT_PATTERN = "EXPLICIT_PATTERN"
    SEMANTIC_LINKER = "SEMANTIC_LINKER"


class IssueType(str, Enum):
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_VALUE = "INVALID_VALUE"
    CONFLICTING_VALUES = "CONFLICTING_VALUES"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNSUPPORTED_BY_EVIDENCE = "UNSUPPORTED_BY_EVIDENCE"


class ChatbotStatus(str, Enum):
    VALID = "VALID"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    ERROR = "ERROR"


class PipelineStage(str, Enum):
    REQUIREMENT_COLLECTION = "REQUIREMENT_COLLECTION"
    EXTRACTION = "EXTRACTION"
    VALIDATION = "VALIDATION"
    CLARIFICATION = "CLARIFICATION"
    READY_FOR_CALCULATION = "READY_FOR_CALCULATION"
    CALCULATION = "CALCULATION"
    RECOMMENDATION = "RECOMMENDATION"
    ERROR = "ERROR"


@dataclass
class ExtractedCandidate:
    field: ParamName
    value: Any
    unit: Optional[str]
    evidence: str
    confidence: float
    source: CandidateSource = CandidateSource.RULE
    source_text: str = ""
    turn_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["field"] = self.field.value
        data["source"] = self.source.value
        return data


@dataclass
class ExtractionResult:
    candidates: List[ExtractedCandidate] = dc_field(default_factory=list)
    warnings: List[str] = dc_field(default_factory=list)
    unresolved_fields: List[str] = dc_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": self.warnings,
            "unresolved_fields": self.unresolved_fields,
        }


@dataclass
class FinalFieldValue:
    value: Any
    unit: Optional[str]
    confidence: float
    evidence: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationIssue:
    type: IssueType
    field: ParamName
    message: str
    question: str
    candidates: List[ExtractedCandidate] = dc_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "field": self.field.value,
            "message": self.message,
            "question": self.question,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass
class ValidationReport:
    status: ChatbotStatus
    stage: PipelineStage
    final_json: Dict[str, Optional[FinalFieldValue]]
    issues: List[ValidationIssue]
    missing_fields: List[str]
    conflicting_fields: List[str]
    invalid_fields: List[str]
    unsupported_fields: List[str]
    questions: List[str]

    def to_plain_final_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, item in self.final_json.items():
            out[key] = None if item is None else item.value
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "stage": self.stage.value,
            "final_json": {
                key: None if item is None else item.to_dict()
                for key, item in self.final_json.items()
            },
            "plain_final_json": self.to_plain_final_json(),
            "issues": [issue.to_dict() for issue in self.issues],
            "missing_fields": self.missing_fields,
            "conflicting_fields": self.conflicting_fields,
            "invalid_fields": self.invalid_fields,
            "unsupported_fields": self.unsupported_fields,
            "questions": self.questions,
        }


@dataclass
class RequirementState:
    raw_user_inputs: List[str] = dc_field(default_factory=list)
    extracted_candidates: List[ExtractedCandidate] = dc_field(default_factory=list)
    final_json: Dict[str, Optional[FinalFieldValue]] = dc_field(default_factory=dict)
    issues: List[ValidationIssue] = dc_field(default_factory=list)
    missing_fields: List[str] = dc_field(default_factory=list)
    conflicting_fields: List[str] = dc_field(default_factory=list)
    invalid_fields: List[str] = dc_field(default_factory=list)
    unsupported_fields: List[str] = dc_field(default_factory=list)
    questions: List[str] = dc_field(default_factory=list)

    status: ChatbotStatus = ChatbotStatus.VALID
    stage: PipelineStage = PipelineStage.REQUIREMENT_COLLECTION
    calculation_result: Dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_user_inputs": self.raw_user_inputs,
            "extracted_candidates": [
                candidate.to_dict() for candidate in self.extracted_candidates
            ],
            "final_json": {
                key: None if item is None else item.to_dict()
                for key, item in self.final_json.items()
            },
            "plain_final_json": {
                key: None if item is None else item.value
                for key, item in self.final_json.items()
            },
            "issues": [issue.to_dict() for issue in self.issues],
            "missing_fields": self.missing_fields,
            "conflicting_fields": self.conflicting_fields,
            "invalid_fields": self.invalid_fields,
            "unsupported_fields": self.unsupported_fields,
            "questions": self.questions,
            "status": self.status.value,
            "stage": self.stage.value,
            "calculation_result": self.calculation_result,
        }