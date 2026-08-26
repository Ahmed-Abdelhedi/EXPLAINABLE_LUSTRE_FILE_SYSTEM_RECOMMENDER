from preference_extractor.evaluation.layer2_residual_validator import (
    RESIDUAL_VALIDATOR_VERSION,
    validate_residual_prediction,
)


def _resolved(level, evidence):
    return {
        "status": "RESOLVED",
        "level": level,
        "evidence": evidence,
        "accepted": True,
        "validation_error": None,
    }


def test_validator_version():
    assert RESIDUAL_VALIDATOR_VERSION


def test_canonicalizes_bottom_priority_to_low():
    text = "high sustained perfromance sits near the bottom of our priorities"
    result = validate_residual_prediction(
        text=text,
        dimension="performance",
        prediction=_resolved("VERY_LOW", text),
    )
    assert result.prediction["accepted"] is True
    assert result.prediction["level"] == "LOW"
    assert result.action == "CANONICALIZED_RESOLVED_LEVEL"


def test_rejects_resolved_without_absolute_intensity_support():
    evidence = "la fiabilté avant l'électricité"
    result = validate_residual_prediction(
        text=evidence,
        dimension="reliability",
        prediction=_resolved("VERY_HIGH", evidence),
    )
    assert result.prediction["accepted"] is False
    assert result.prediction["status"] == "UNRESOLVED"


def test_supports_french_elision_very_high():
    evidence = "rien n'est plus important qu'une haute disponibilité"
    result = validate_residual_prediction(
        text=evidence,
        dimension="reliability",
        prediction=_resolved("VERY_HIGH", evidence),
    )
    assert result.prediction["accepted"] is True
    assert result.prediction["level"] == "VERY_HIGH"


def test_supports_relative_only_with_outranks():
    evidence = "throughput outranks relibility"
    prediction = {
        "status": "RELATIVE_ONLY",
        "level": None,
        "evidence": evidence,
        "accepted": True,
        "validation_error": None,
    }
    result = validate_residual_prediction(
        text=evidence,
        dimension="reliability",
        prediction=prediction,
    )
    assert result.prediction["accepted"] is True
    assert result.prediction["status"] == "RELATIVE_ONLY"


def test_supports_relative_only_with_french_avant():
    evidence = "la fiabilté avant l'électricité"
    prediction = {
        "status": "RELATIVE_ONLY",
        "level": None,
        "evidence": evidence,
        "accepted": True,
        "validation_error": None,
    }
    result = validate_residual_prediction(
        text=evidence,
        dimension="reliability",
        prediction=prediction,
    )
    assert result.prediction["accepted"] is True


def test_residual_no_signal_is_conservatively_rejected():
    prediction = {
        "status": "NO_SIGNAL",
        "level": None,
        "evidence": None,
        "accepted": True,
        "validation_error": None,
    }
    result = validate_residual_prediction(
        text="The wording is unfamiliar.",
        dimension="cost",
        prediction=prediction,
    )
    assert result.prediction["accepted"] is False
    assert result.prediction["status"] == "UNRESOLVED"


def test_keeps_existing_abstention():
    prediction = {
        "status": "UNRESOLVED",
        "level": None,
        "evidence": None,
        "accepted": False,
        "validation_error": None,
    }
    result = validate_residual_prediction(
        text="ambiguous",
        dimension="cost",
        prediction=prediction,
    )
    assert result.prediction["accepted"] is False
    assert result.action == "KEEP_ABSTENTION"
