from requirement_state import (
    DeterministicRequirementValidator,
    FinalRequirementState,
)


def _codes(state):
    DeterministicRequirementValidator().validate(state)
    return {item["code"] for item in state.validation_issues}


def test_average_cannot_exceed_max():
    state = FinalRequirementState(
        average_file_size_gb=20,
        max_file_size_gb=10,
    )
    assert "AVERAGE_FILE_SIZE_EXCEEDS_MAX" in _codes(state)


def test_growth_requires_horizon():
    state = FinalRequirementState(
        annual_growth_percent=20,
        planning_horizon_years=None,
    )
    assert "GROWTH_REQUIRES_HORIZON" in _codes(state)


def test_negative_throughput_is_rejected():
    state = FinalRequirementState(
        target_read_gbps=-1,
    )
    assert "NEGATIVE_VALUE" in _codes(state)
