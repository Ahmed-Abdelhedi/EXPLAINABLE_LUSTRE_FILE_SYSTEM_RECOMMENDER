from input_orchestrator.collection_gate import collection_gap_report
from input_orchestrator.models import FieldState
from input_orchestrator.policies import DEFAULT_POLICY
from input_orchestrator.question_planner import QuestionPlanner
from input_orchestrator.session_state import WorkingSessionState


def _verify_core(session):
    for field, value in {
        "requested_usable_capacity_tib": 100,
        "client_count": 64,
        "access_type": "sequential",
        "ha_required": True,
    }.items():
        record = session.get(field)
        record.value = value
        record.state = FieldState.VERIFIED


def test_missing_optional_fields_block_validation():
    session = WorkingSessionState("S1")
    _verify_core(session)

    report = collection_gap_report(
        session,
        DEFAULT_POLICY,
    )

    assert report.complete is False
    assert "target_read_gbps" in report.blocking_fields
    assert "max_budget_usd" in report.blocking_fields


def test_declined_optional_field_is_answered():
    session = WorkingSessionState("S2")
    _verify_core(session)

    record = session.get("max_budget_usd")
    record.state = FieldState.DECLINED

    report = collection_gap_report(
        session,
        DEFAULT_POLICY,
    )

    assert "max_budget_usd" not in report.blocking_fields


def test_declined_core_field_remains_blocking_and_is_reasked():
    session = WorkingSessionState("S3")

    record = session.get("requested_usable_capacity_tib")
    record.state = FieldState.DECLINED

    report = collection_gap_report(
        session,
        DEFAULT_POLICY,
    )

    assert (
        "requested_usable_capacity_tib"
        in report.declined_but_required_fields
    )

    question = QuestionPlanner(
        DEFAULT_POLICY
    ).next_question(
        session,
        question_id="Q1",
        message_id="M1",
    )

    assert question is not None
    assert (
        question.target_field
        == "requested_usable_capacity_tib"
    )


def test_growth_makes_horizon_required_even_if_declined():
    session = WorkingSessionState("S4")
    _verify_core(session)

    growth = session.get("annual_growth_percent")
    growth.value = 20
    growth.state = FieldState.VERIFIED

    horizon = session.get("planning_horizon_years")
    horizon.state = FieldState.DECLINED

    report = collection_gap_report(
        session,
        DEFAULT_POLICY,
    )

    assert "planning_horizon_years" in report.blocking_fields
    assert (
        "planning_horizon_years"
        in report.declined_but_required_fields
    )

    question = QuestionPlanner(
        DEFAULT_POLICY
    ).next_question(
        session,
        question_id="Q1",
        message_id="M1",
    )

    assert question is not None
    assert question.target_field == "planning_horizon_years"


def test_all_raw_fields_verified_or_optional_declined_pass_gate():
    session = WorkingSessionState("S5")
    _verify_core(session)

    for name, record in session.fields.items():
        if name == "preference_weights":
            continue

        if record.state == FieldState.MISSING:
            record.state = FieldState.DECLINED

    # Core remains verified.
    _verify_core(session)

    report = collection_gap_report(
        session,
        DEFAULT_POLICY,
    )

    assert report.complete is True
    assert report.blocking_fields == []
