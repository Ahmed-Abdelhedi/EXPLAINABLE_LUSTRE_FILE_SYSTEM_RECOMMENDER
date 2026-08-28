from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Dict, List, Optional


class FieldState(str, Enum):
    MISSING = "MISSING"
    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    UNRESOLVED = "UNRESOLVED"
    CONFLICT = "CONFLICT"
    DECLINED = "DECLINED"


class ConversationState(str, Enum):
    COLLECTING = "COLLECTING"
    WAITING_FOR_ANSWER = "WAITING_FOR_ANSWER"
    RESOLVING_CONFLICT = "RESOLVING_CONFLICT"
    BWM_ELICITATION = "BWM_ELICITATION"
    READY_FOR_FINAL_VALIDATION = "READY_FOR_FINAL_VALIDATION"


class ObservationKind(str, Enum):
    VALUE = "VALUE"
    RELATION = "RELATION"
    ABSTENTION = "ABSTENTION"


@dataclass(frozen=True)
class Evidence:
    text: Optional[str] = None
    source: Optional[str] = None
    confidence: Optional[float] = None


@dataclass(frozen=True)
class FieldObservation:
    field: str
    value: Any
    state: FieldState
    source: str
    evidence: Evidence = dataclass_field(default_factory=Evidence)
    message_id: Optional[str] = None
    explicit_correction: bool = False
    kind: ObservationKind = ObservationKind.VALUE
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class FieldRecord:
    field: str
    value: Any = None
    state: FieldState = FieldState.MISSING
    source: Optional[str] = None
    evidence: Optional[str] = None
    confidence: Optional[float] = None
    message_id: Optional[str] = None
    revision: int = 0
    history: List[Dict[str, Any]] = dataclass_field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "state": self.state.value,
            "source": self.source,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "message_id": self.message_id,
            "revision": self.revision,
        }


@dataclass
class PendingQuestion:
    question_id: str
    target_field: str
    question: str
    expected_answer_type: str
    created_after_message_id: str
    context: Dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class Conflict:
    field: str
    previous_value: Any
    new_value: Any
    previous_message_id: Optional[str]
    new_message_id: Optional[str]
    reason: str


@dataclass
class OrchestratorResponse:
    conversation_state: ConversationState
    assistant_message: Optional[str]
    updated_fields: List[str]
    conflicts: List[Conflict]
    pending_question: Optional[PendingQuestion]
    ready_for_final_validation: bool
    diagnostics: Dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_state": self.conversation_state.value,
            "assistant_message": self.assistant_message,
            "updated_fields": list(self.updated_fields),
            "conflicts": [
                {
                    "field": c.field,
                    "previous_value": c.previous_value,
                    "new_value": c.new_value,
                    "previous_message_id": c.previous_message_id,
                    "new_message_id": c.new_message_id,
                    "reason": c.reason,
                }
                for c in self.conflicts
            ],
            "pending_question": (
                None
                if self.pending_question is None
                else {
                    "question_id": self.pending_question.question_id,
                    "target_field": self.pending_question.target_field,
                    "question": self.pending_question.question,
                    "expected_answer_type": self.pending_question.expected_answer_type,
                    "created_after_message_id": self.pending_question.created_after_message_id,
                    "context": dict(self.pending_question.context),
                }
            ),
            "ready_for_final_validation": self.ready_for_final_validation,
            "diagnostics": dict(self.diagnostics),
        }
