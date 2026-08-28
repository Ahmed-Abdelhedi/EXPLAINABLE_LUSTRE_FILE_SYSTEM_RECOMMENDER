from requirement_state import (
    DeterministicRequirementValidator,
    FinalRequirementState,
)


def _codes(state):
    DeterministicRequirementValidator().validate(
        state
    )
    return {
        issue["code"]
        for issue in state.validation_issues
    }


def test_valid_structured_ratio_passes_ratio_checks():
    state = FinalRequirementState(
        read_write_ratio={
            "read_percent": 20.0,
            "write_percent": 80.0,
        }
    )

    codes = _codes(state)

    assert "INVALID_READ_WRITE_RATIO_STRUCTURE" not in codes
    assert "INVALID_READ_WRITE_RATIO_VALUE" not in codes
    assert "READ_WRITE_RATIO_NOT_NORMALIZED" not in codes


def test_scalar_ratio_is_rejected():
    state = FinalRequirementState(
        read_write_ratio=20
    )

    assert (
        "INVALID_READ_WRITE_RATIO_STRUCTURE"
        in _codes(state)
    )
