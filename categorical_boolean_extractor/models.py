from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

class FieldStatus(str, Enum):
    VERIFIED = "VERIFIED"
    NO_EVIDENCE = "NO_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"
    CONFLICT = "CONFLICT"

class ResolutionSource(str, Enum):
    EXPLICIT_RESOLVER = "EXPLICIT_RESOLVER"
    SEMANTIC_MODEL = "SEMANTIC_MODEL"
    LLM_FALLBACK = "LLM_FALLBACK"
    CONTEXT_RESOLVER = "CONTEXT_RESOLVER"
    NONE = "NONE"

class AccessType(str, Enum):
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    MIXED = "mixed"

FieldValue = Union[bool, str, None]

@dataclass
class FieldResult:
    field: str
    status: FieldStatus
    value: FieldValue
    source: ResolutionSource
    evidence: Optional[str] = None
    reason: Optional[str] = None
    semantic_label: Optional[str] = None
    semantic_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "status": self.status.value,
            "value": self.value,
            "source": self.source.value,
            "evidence": self.evidence,
            "reason": self.reason,
            "semantic_label": self.semantic_label,
            "semantic_confidence": self.semantic_confidence,
        }

@dataclass
class CategoricalBooleanExtractionResult:
    text: str
    ha_required: FieldResult
    access_type: FieldResult
    semantic_model_used: bool = False
    semantic_abstained_fields: List[str] = field(default_factory=list)
    llm_fallback_used: bool = False
    llm_fallback_fields: List[str] = field(default_factory=list)

    def to_requirement_fields(self) -> Dict[str, Any]:
        return {
            "ha_required": self.ha_required.value if self.ha_required.status == FieldStatus.VERIFIED else None,
            "access_type": self.access_type.value if self.access_type.status == FieldStatus.VERIFIED else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "ha_required": self.ha_required.to_dict(),
            "access_type": self.access_type.to_dict(),
            "semantic_model_used": self.semantic_model_used,
            "semantic_abstained_fields": list(self.semantic_abstained_fields),
            "llm_fallback_used": self.llm_fallback_used,
            "llm_fallback_fields": list(self.llm_fallback_fields),
            "requirement_fields": self.to_requirement_fields(),
        }
