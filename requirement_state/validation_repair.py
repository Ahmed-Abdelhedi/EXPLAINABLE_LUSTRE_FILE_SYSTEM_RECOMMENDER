from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from input_orchestrator.field_registry import (
    FIELD_SPECS,
    WEIGHT_FIELD,
)
from input_orchestrator.models import (
    ConversationState,
    PendingQuestion,
)


WEIGHT_ISSUE_CODES = {
    "ACTIVE_PREFERENCES_REQUIRE_WEIGHTS",
    "NON_FINITE_PREFERENCE_WEIGHT",
    "NEGATIVE_PREFERENCE_WEIGHT",
    "PREFERENCE_WEIGHTS_NOT_NORMALIZED",
    "INACTIVE_PREFERENCE_HAS_NONZERO_WEIGHT",
    "BWM_CONSISTENCY_NOT_PASS",
}

ISSUE_PRIORITY = (
    "AVERAGE_FILE_SIZE_EXCEEDS_MAX",
    "GROWTH_REQUIRES_HORIZON",
    "INVALID_READ_WRITE_RATIO_STRUCTURE",
    "INVALID_READ_WRITE_RATIO_VALUE",
    "READ_WRITE_RATIO_NOT_NORMALIZED",
    "NON_POSITIVE_VALUE",
    "NEGATIVE_VALUE",
    "INVALID_POSITIVE_INTEGER",
    "INVALID_PLANNING_HORIZON",
    "INVALID_ACCESS_TYPE",
    "INVALID_HA_VALUE",
    "INVALID_PREFERENCE_LEVEL",
    "FIELD_NOT_RESOLVED",
    "REQUIRED_FIELD_NOT_VERIFIED",
    *tuple(WEIGHT_ISSUE_CODES),
)


@dataclass(frozen=True)
class ValidationRepairAction:
    mode: str
    question: Optional[str] = None
    target_field: Optional[str] = None
    issue_code: Optional[str] = None


def _pick_issue(state) -> Optional[dict]:
    issues = list(state.validation_issues or [])
    if not issues:
        return None

    by_code = {}
    for issue in issues:
        by_code.setdefault(
            issue.get("code"),
            issue,
        )

    for code in ISSUE_PRIORITY:
        if code in by_code:
            return by_code[code]

    return issues[0]


def prepare_validation_repair(
    *,
    state,
    session,
) -> ValidationRepairAction:
    """
    Convert deterministic final-validation failures back into the online
    clarification loop instead of terminating the production process.
    """
    issue = _pick_issue(state)

    if issue is None:
        return ValidationRepairAction(
            mode="NONE",
        )

    code = str(
        issue.get("code")
        or "VALIDATION_ERROR"
    )

    if code in WEIGHT_ISSUE_CODES:
        session.reset_bwm_dialogue()
        session.pending_question = None
        session.conversation_state = (
            ConversationState.COLLECTING
        )
        return ValidationRepairAction(
            mode="RESTART_BWM",
            question=(
                "The preference weighting did not pass final validation. "
                "The BWM elicitation will be restarted."
            ),
            target_field=WEIGHT_FIELD,
            issue_code=code,
        )

    field_name = issue.get("field")

    if code == "AVERAGE_FILE_SIZE_EXCEEDS_MAX":
        field_name = "max_file_size_gb"
        details = issue.get("details") or {}
        avg = details.get(
            "average_file_size_gb"
        )
        maximum = details.get(
            "max_file_size_gb"
        )
        question = (
            "The maximum file size cannot be smaller than the average "
            f"file size. Current average={avg} GB and maximum={maximum} GB. "
            f"Please enter a corrected maximum file size >= {avg} GB."
        )

    elif code == "GROWTH_REQUIRES_HORIZON":
        field_name = "planning_horizon_years"
        question = (
            "Annual growth is greater than zero, so a planning horizon is "
            "required. Over how many years should growth be planned?"
        )

    elif code in {
        "INVALID_READ_WRITE_RATIO_STRUCTURE",
        "INVALID_READ_WRITE_RATIO_VALUE",
        "READ_WRITE_RATIO_NOT_NORMALIZED",
    }:
        field_name = "read_write_ratio"
        question = (
            "Please provide the read/write ratio explicitly, for example "
            "'20/80' for 20% read and 80% write."
        )

    else:
        if field_name not in FIELD_SPECS:
            raise RuntimeError(
                "Final validation produced an issue that cannot be "
                f"repaired interactively: {issue}"
            )

        spec = FIELD_SPECS[field_name]
        question = (
            f"{issue.get('message')} "
            f"{spec.question}"
        )

    if field_name not in FIELD_SPECS:
        raise RuntimeError(
            f"No field specification exists for repair target {field_name!r}."
        )

    spec = FIELD_SPECS[field_name]

    pending = PendingQuestion(
        question_id=(
            f"VQ{session.message_counter:04d}"
        ),
        target_field=field_name,
        question=question,
        expected_answer_type=spec.expected_answer_type,
        created_after_message_id=(
            f"M{session.message_counter:04d}"
        ),
        context={
            "validation_repair": True,
            "validation_issue_code": code,
            "validation_issue": dict(issue),
        },
    )

    session.pending_question = pending
    session.conversation_state = (
        ConversationState.WAITING_FOR_ANSWER
    )

    return ValidationRepairAction(
        mode="ASK_FIELD",
        question=question,
        target_field=field_name,
        issue_code=code,
    )
