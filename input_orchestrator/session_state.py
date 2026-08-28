from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .field_registry import FIELD_SPECS, WEIGHT_FIELD
from .models import (
    ConversationState,
    FieldRecord,
    PendingQuestion,
)


def _default_bwm_dialogue() -> Dict[str, Any]:
    return {
        "explicit_best": None,
        "explicit_worst": None,
        "answers": {},
        "single_active_confirmed": False,
        "single_active_rejected": False,
        "last_status": None,
        "last_result": None,
        "last_input_error": None,
    }


@dataclass
class WorkingSessionState:
    session_id: str
    fields: Dict[str, FieldRecord] = field(default_factory=dict)
    conversation_state: ConversationState = ConversationState.COLLECTING
    pending_question: Optional[PendingQuestion] = None
    message_counter: int = 0

    # Relative preference statements are retained as relations and never
    # converted into invented absolute intensity values.
    preference_relations: List[Dict[str, Any]] = field(default_factory=list)

    # Persistent state for the conversational Linear-BWM elicitation phase.
    bwm_dialogue: Dict[str, Any] = field(default_factory=_default_bwm_dialogue)

    def __post_init__(self) -> None:
        if not self.fields:
            for name in FIELD_SPECS:
                self.fields[name] = FieldRecord(field=name)
            self.fields[WEIGHT_FIELD] = FieldRecord(field=WEIGHT_FIELD)

        defaults = _default_bwm_dialogue()
        for key, value in defaults.items():
            if key not in self.bwm_dialogue:
                self.bwm_dialogue[key] = value

    def next_message_id(self) -> str:
        self.message_counter += 1
        return f"M{self.message_counter:04d}"

    def get(self, field_name: str) -> FieldRecord:
        if field_name not in self.fields:
            self.fields[field_name] = FieldRecord(field=field_name)
        return self.fields[field_name]

    def clear_pending_question(self) -> None:
        self.pending_question = None

    def add_preference_relation(self, relation: Dict[str, Any]) -> None:
        canonical = {
            "higher": relation.get("higher"),
            "lower": relation.get("lower"),
            "relation_type": relation.get(
                "relation_type",
                "MORE_IMPORTANT_THAN",
            ),
            "evidence": relation.get("evidence"),
            "message_id": relation.get("message_id"),
            "source": relation.get("source"),
        }

        signature = (
            canonical["higher"],
            canonical["lower"],
            canonical["relation_type"],
            canonical["evidence"],
        )

        existing = {
            (
                item.get("higher"),
                item.get("lower"),
                item.get("relation_type"),
                item.get("evidence"),
            )
            for item in self.preference_relations
        }

        if signature not in existing:
            self.preference_relations.append(canonical)

    def reset_bwm_dialogue(self) -> None:
        self.bwm_dialogue = _default_bwm_dialogue()

        record = self.get(WEIGHT_FIELD)
        record.value = None
        record.state = type(record.state).MISSING
        record.source = None
        record.evidence = None
        record.confidence = None
        record.message_id = None
        record.revision = 0
        record.history.clear()

    def snapshot(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "conversation_state": self.conversation_state.value,
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
            "preference_relations": [
                dict(item)
                for item in self.preference_relations
            ],
            "bwm_dialogue": {
                "explicit_best": self.bwm_dialogue.get("explicit_best"),
                "explicit_worst": self.bwm_dialogue.get("explicit_worst"),
                "answers": dict(
                    self.bwm_dialogue.get("answers", {})
                ),
                "single_active_confirmed": bool(
                    self.bwm_dialogue.get(
                        "single_active_confirmed",
                        False,
                    )
                ),
                "single_active_rejected": bool(
                    self.bwm_dialogue.get(
                        "single_active_rejected",
                        False,
                    )
                ),
                "last_status": self.bwm_dialogue.get("last_status"),
                "last_result": self.bwm_dialogue.get("last_result"),
                "last_input_error": self.bwm_dialogue.get(
                    "last_input_error"
                ),
            },
            "fields": {
                name: record.snapshot()
                for name, record in self.fields.items()
            },
        }
