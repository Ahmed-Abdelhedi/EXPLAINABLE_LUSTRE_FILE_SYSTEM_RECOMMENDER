from __future__ import annotations

from .collection_gate import collection_gap_report
from .models import FieldState
from .policies import OrchestrationPolicy
from .session_state import WorkingSessionState


def is_ready_for_final_validation(
    session: WorkingSessionState,
    policy: OrchestrationPolicy,
) -> bool:
    # Legacy relaxed mode exists only for the already-frozen Step 6.1B
    # wiring smoke. It still respects conditional dependencies.
    if policy.ready_when_core_complete_and_no_conflict:
        if any(
            record.state == FieldState.CONFLICT
            for record in session.fields.values()
        ):
            return False

        if not all(
            session.get(field).state == FieldState.VERIFIED
            for field in policy.core_required_fields
        ):
            return False

        growth = session.get("annual_growth_percent")
        horizon = session.get("planning_horizon_years")

        if (
            policy.require_horizon_when_growth_present
            and growth.state == FieldState.VERIFIED
            and growth.value not in {None, 0, 0.0}
            and horizon.state != FieldState.VERIFIED
        ):
            return False

        return True

    # Strict production mode:
    # no raw Requirement field may remain missing/unresolved.
    return collection_gap_report(
        session,
        policy,
    ).complete
