from __future__ import annotations

import json

from .models import Evidence, FieldObservation, FieldState
from .orchestrator import InputOrchestrator
from .policies import OrchestrationPolicy
from .router import ExtractionRouter


class DemoExtractor:
    domain = "demo"

    def extract(self, text, *, message_id, pending_question=None):
        lower = text.lower()
        out = []

        if "100 tib" in lower:
            out.append(FieldObservation(
                "requested_usable_capacity_tib", 100.0,
                FieldState.VERIFIED, "DEMO_QUANTITY",
                Evidence("100 TiB", "DEMO_QUANTITY"), message_id
            ))
        if "64 clients" in lower:
            out.append(FieldObservation(
                "client_count", 64,
                FieldState.VERIFIED, "DEMO_QUANTITY",
                Evidence("64 clients", "DEMO_QUANTITY"), message_id
            ))
        if "sequential" in lower:
            out.append(FieldObservation(
                "access_type", "sequential",
                FieldState.VERIFIED, "DEMO_CATEGORICAL",
                Evidence("sequential", "DEMO_CATEGORICAL"), message_id
            ))
        if "ha is mandatory" in lower:
            out.append(FieldObservation(
                "ha_required", True,
                FieldState.VERIFIED, "DEMO_CATEGORICAL",
                Evidence("HA is mandatory", "DEMO_CATEGORICAL"), message_id
            ))
        if "20%" in lower:
            out.append(FieldObservation(
                "annual_growth_percent", 20.0,
                FieldState.VERIFIED, "DEMO_QUANTITY",
                Evidence("20%", "DEMO_QUANTITY"), message_id
            ))
        return out


def main() -> None:
    orchestrator = InputOrchestrator(
        router=ExtractionRouter([DemoExtractor()]),
        policy=OrchestrationPolicy(
            ask_optional_fields=False,
            ready_when_core_complete_and_no_conflict=True,
        ),
    )
    session = orchestrator.new_session("SMOKE")

    first = orchestrator.handle_message(
        (
            "We need 100 TiB for 64 clients. "
            "HA is mandatory and access is sequential. "
            "Growth is 20% per year."
        ),
        session,
    )

    assert first.pending_question is not None
    assert first.pending_question.target_field == "planning_horizon_years"

    second = orchestrator.handle_message("3", session)

    assert second.ready_for_final_validation is True
    assert session.get("planning_horizon_years").value == 3

    print(json.dumps({
        "step": "6.1",
        "status": "INPUT_ORCHESTRATOR_SMOKE_PASS",
        "first_turn": first.to_dict(),
        "second_turn": second.to_dict(),
        "session": session.snapshot(),
    }, indent=2))


if __name__ == "__main__":
    main()
