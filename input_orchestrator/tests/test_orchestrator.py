from __future__ import annotations

from input_orchestrator.models import (
    Evidence,
    FieldObservation,
    FieldState,
)
from input_orchestrator.orchestrator import InputOrchestrator
from input_orchestrator.policies import OrchestrationPolicy
from input_orchestrator.router import ExtractionRouter


class ScriptedExtractor:
    domain = "scripted"

    def extract(self, text, *, message_id, pending_question=None):
        lower = text.lower()
        out = []

        mapping = [
            ("100 tib", "requested_usable_capacity_tib", 100.0),
            ("64 clients", "client_count", 64),
            ("sequential", "access_type", "sequential"),
            ("random", "access_type", "random"),
            ("ha mandatory", "ha_required", True),
            ("ha optional", "ha_required", False),
            ("20%", "annual_growth_percent", 20.0),
        ]

        for token, field, value in mapping:
            if token in lower:
                out.append(FieldObservation(
                    field=field,
                    value=value,
                    state=FieldState.VERIFIED,
                    source="SCRIPTED",
                    evidence=Evidence(token, "SCRIPTED"),
                    message_id=message_id,
                ))

        return out


def make_orchestrator(**policy_kwargs):
    return InputOrchestrator(
        router=ExtractionRouter([ScriptedExtractor()]),
        policy=OrchestrationPolicy(**policy_kwargs),
    )


def test_rich_first_message_updates_multiple_fields():
    o = make_orchestrator(
        ask_optional_fields=False,
        ready_when_core_complete_and_no_conflict=True,
    )
    s = o.new_session("T1")

    r = o.handle_message(
        "100 TiB, 64 clients, HA mandatory, sequential",
        s,
    )

    assert s.get("requested_usable_capacity_tib").value == 100.0
    assert s.get("client_count").value == 64
    assert s.get("ha_required").value is True
    assert s.get("access_type").value == "sequential"
    assert r.ready_for_final_validation is True


def test_growth_creates_conditional_horizon_question():
    o = make_orchestrator(
        ask_optional_fields=False,
        ready_when_core_complete_and_no_conflict=True,
    )
    s = o.new_session("T2")

    r = o.handle_message(
        "100 TiB, 64 clients, HA mandatory, sequential, growth 20%",
        s,
    )

    assert r.pending_question is not None
    assert r.pending_question.target_field == "planning_horizon_years"


def test_short_number_uses_pending_context():
    o = make_orchestrator(
        ask_optional_fields=False,
        ready_when_core_complete_and_no_conflict=True,
    )
    s = o.new_session("T3")

    o.handle_message(
        "100 TiB, 64 clients, HA mandatory, sequential, growth 20%",
        s,
    )
    r = o.handle_message("3", s)

    assert s.get("planning_horizon_years").value == 3
    assert r.ready_for_final_validation is True


def test_short_yes_uses_pending_ha_context():
    o = make_orchestrator(
        ask_optional_fields=False,
        ready_when_core_complete_and_no_conflict=True,
    )
    s = o.new_session("T4")

    # Missing core fields -> capacity question first.
    o.handle_message("64 clients, sequential", s)
    # Fill capacity manually via next message so planner eventually asks HA.
    o.handle_message("100 TiB", s)
    assert s.pending_question is not None
    assert s.pending_question.target_field == "ha_required"

    r = o.handle_message("yes", s)
    assert s.get("ha_required").value is True
    assert r.ready_for_final_validation is True


def test_new_verified_contradiction_creates_conflict():
    o = make_orchestrator(
        ask_optional_fields=False,
        ready_when_core_complete_and_no_conflict=True,
    )
    s = o.new_session("T5")

    o.handle_message(
        "100 TiB, 64 clients, HA mandatory, sequential",
        s,
    )
    r = o.handle_message("access is random", s)

    assert r.conflicts
    assert r.conflicts[0].field == "access_type"
    assert s.get("access_type").state == FieldState.CONFLICT
    assert r.pending_question is not None


def test_verified_value_is_not_degraded_by_abstention():
    class AbstainExtractor:
        domain = "abstain"
        def extract(self, text, *, message_id, pending_question=None):
            return [FieldObservation(
                "client_count", None, FieldState.UNRESOLVED,
                "ABSTAIN", Evidence(), message_id
            )]

    o = InputOrchestrator(
        router=ExtractionRouter([ScriptedExtractor(), AbstainExtractor()]),
        policy=OrchestrationPolicy(
            ask_optional_fields=False,
            ready_when_core_complete_and_no_conflict=True,
        ),
    )
    s = o.new_session("T6")
    o.handle_message("64 clients", s)

    assert s.get("client_count").state == FieldState.VERIFIED
    assert s.get("client_count").value == 64


def test_decline_optional_field_can_advance_collection():
    o = make_orchestrator(
        ask_optional_fields=True,
        ready_when_core_complete_and_no_conflict=False,
    )
    s = o.new_session("T7")

    o.handle_message(
        "100 TiB, 64 clients, HA mandatory, sequential",
        s,
    )
    assert s.pending_question is not None
    target = s.pending_question.target_field
    assert target == "target_read_gbps" or target == "read_write_ratio"

    o.handle_message("skip", s)
    assert s.get(target).state == FieldState.DECLINED
