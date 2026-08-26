from __future__ import annotations

import json

from preference_extractor.evaluation.layer2_fallback_common import (
    parse_llm_response,
)
from preference_extractor.evaluation.layer2_hybrid_guard import (
    Layer2DeterministicSemanticGuard,
)
from preference_extractor.evaluation.layer2_residual_validator import (
    validate_residual_prediction,
)


def test_simple_absolute_en():
    guard = Layer2DeterministicSemanticGuard()
    d = guard.resolve_dimension(
        text="Cost is a top priority for this design.",
        dimension="cost",
    )
    assert d is not None
    assert d.status == "RESOLVED"
    assert d.level == "VERY_HIGH"


def test_simple_absolute_fr():
    guard = Layer2DeterministicSemanticGuard()
    d = guard.resolve_dimension(
        text=(
            "La consommation électrique est souhaitable "
            "sans être prioritaire."
        ),
        dimension="power",
    )
    assert d is not None
    assert d.status == "RESOLVED"
    assert d.level == "LOW"


def test_comparison_never_invents_absolute_level():
    guard = Layer2DeterministicSemanticGuard()
    text = "Performance is more important than cost."
    for dimension in ("performance", "cost"):
        d = guard.resolve_dimension(text=text, dimension=dimension)
        assert d is not None
        assert d.status == "RELATIVE_ONLY"
        assert d.level is None


def test_no_signal_is_not_very_low():
    guard = Layer2DeterministicSemanticGuard()
    d = guard.resolve_dimension(
        text="No preference; the system supports 10 GB/s throughput.",
        dimension="performance",
    )
    assert d is not None
    assert d.status == "NO_SIGNAL"
    assert d.level is None


def test_explicit_near_indifference_is_very_low():
    guard = Layer2DeterministicSemanticGuard()
    d = guard.resolve_dimension(
        text="Keeping wattage modest can be largely ignored.",
        dimension="power",
    )
    assert d is not None
    assert d.status == "RESOLVED"
    assert d.level == "VERY_LOW"


def test_residual_rejects_absolute_from_pure_comparison():
    text = "Reliability is more important than performance."
    r = validate_residual_prediction(
        text=text,
        dimension="reliability",
        prediction={
            "status": "RESOLVED",
            "level": "VERY_HIGH",
            "evidence": "Reliability is more important than performance",
            "accepted": True,
            "validation_error": None,
        },
    )
    assert r.prediction["status"] == "UNRESOLVED"
    assert r.prediction["accepted"] is False


def test_residual_accepts_relative_only_with_support():
    text = "Reliability is more important than performance."
    r = validate_residual_prediction(
        text=text,
        dimension="reliability",
        prediction={
            "status": "RELATIVE_ONLY",
            "level": None,
            "evidence": "Reliability is more important than performance",
            "accepted": True,
            "validation_error": None,
        },
    )
    assert r.prediction["status"] == "RELATIVE_ONLY"
    assert r.prediction["accepted"] is True


def test_residual_canonicalizes_low():
    text = "Cost sits near the bottom of our priorities."
    r = validate_residual_prediction(
        text=text,
        dimension="cost",
        prediction={
            "status": "RESOLVED",
            "level": "VERY_LOW",
            "evidence": "Cost sits near the bottom of our priorities",
            "accepted": True,
            "validation_error": None,
        },
    )
    assert r.prediction["status"] == "RESOLVED"
    assert r.prediction["level"] == "LOW"
    assert r.prediction["accepted"] is True


def test_residual_no_signal_is_not_auto_accepted():
    r = validate_residual_prediction(
        text="The system has 80 clients.",
        dimension="cost",
        prediction={
            "status": "NO_SIGNAL",
            "level": None,
            "evidence": None,
            "accepted": True,
            "validation_error": None,
        },
    )
    assert r.prediction["status"] == "UNRESOLVED"
    assert r.prediction["accepted"] is False


def test_parser_blocks_unrequested_dimensions():
    p = parse_llm_response(
        raw_text=json.dumps(
            {
                "dimensions": {
                    "cost": {
                        "status": "UNRESOLVED",
                        "level": None,
                        "evidence": None,
                    },
                    "power": {
                        "status": "UNRESOLVED",
                        "level": None,
                        "evidence": None,
                    },
                }
            }
        ),
        requested_dimensions=["cost"],
        user_text="Cost is mentioned.",
    )
    assert p["valid"] is False
    assert any(
        str(v).startswith("UNREQUESTED_DIMENSIONS:")
        for v in p["violations"]
    )


def test_parser_blocks_unsupported_evidence():
    p = parse_llm_response(
        raw_text=json.dumps(
            {
                "dimensions": {
                    "reliability": {
                        "status": "RESOLVED",
                        "level": "HIGH",
                        "evidence": "mission critical reliability",
                    }
                }
            }
        ),
        requested_dimensions=["reliability"],
        user_text="Reliability matters.",
    )
    pred = p["dimensions"]["reliability"]
    assert pred["accepted"] is False
    assert pred["status"] == "UNRESOLVED"
