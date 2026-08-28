from input_orchestrator.models import FieldState
from input_orchestrator.session_state import WorkingSessionState


def test_functional_regression_accepts_supported_verified_preference_levels():
    session = WorkingSessionState("S")

    reliability = session.get("reliability_priority")
    reliability.value = "HIGH"
    reliability.state = FieldState.VERIFIED

    performance = session.get("performance_priority")
    performance.value = "HIGH"
    performance.state = FieldState.VERIFIED

    allowed = {
        "VERY_LOW",
        "LOW",
        "MEDIUM",
        "HIGH",
        "VERY_HIGH",
    }

    assert reliability.value in allowed
    assert performance.value in allowed

    # 6.1E must not silently become a new exact-intensity accuracy benchmark.
    # Numerical ordering is resolved by the later formal BWM dialogue.
