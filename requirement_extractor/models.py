from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ParamName(str, Enum):
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


class CandidateSource(str, Enum):
    RULE = "RULE"
    NORMALIZER = "NORMALIZER"
    LLM_FALLBACK = "LLM_FALLBACK"
    USER_CLARIFICATION = "USER_CLARIFICATION"


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
    candidates: List[ExtractedCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unresolved_fields: List[str] = field(default_factory=list)

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
    candidates: List[ExtractedCandidate] = field(default_factory=list)

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
    raw_user_inputs: List[str] = field(default_factory=list)
    extracted_candidates: List[ExtractedCandidate] = field(default_factory=list)
    final_json: Dict[str, Optional[FinalFieldValue]] = field(default_factory=dict)
    issues: List[ValidationIssue] = field(default_factory=list)

    missing_fields: List[str] = field(default_factory=list)
    conflicting_fields: List[str] = field(default_factory=list)
    invalid_fields: List[str] = field(default_factory=list)
    unsupported_fields: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)

    status: ChatbotStatus = ChatbotStatus.VALID
    stage: PipelineStage = PipelineStage.REQUIREMENT_COLLECTION
    calculation_result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_user_inputs": self.raw_user_inputs,
            "extracted_candidates": [
                candidate.to_dict()
                for candidate in self.extracted_candidates
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