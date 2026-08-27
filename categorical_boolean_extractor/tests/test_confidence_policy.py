from categorical_boolean_extractor.semantic.confidence import CalibratedConfidencePolicy, ClassThreshold
from categorical_boolean_extractor.semantic.schemas import SemanticHeadOutput

def out(label,top,second):
    return SemanticHeadOutput(
        probabilities={label:top,"OTHER":second},
        top_label=label,
        top_probability=top,
        second_probability=second,
        margin=top-second,
    )

def test_missing_threshold_means_abstain():
    p=CalibratedConfidencePolicy(
        thresholds={"ha":{}},
        target_precision=0.99,
        calibration_version="test",
    )
    assert not p.decide(head="ha",output=out("HA_REQUIRED",0.99,0.01)).accepted

def test_probability_and_margin_both_required():
    p=CalibratedConfidencePolicy(
        thresholds={"ha":{"HA_REQUIRED":ClassThreshold(
            min_top_probability=0.90,
            min_margin=0.50,
            validation_precision=1.0,
            validation_accepted=100,
        )}},
        target_precision=0.99,
        calibration_version="test",
    )
    assert p.decide(head="ha",output=out("HA_REQUIRED",0.95,0.02)).accepted
    assert not p.decide(head="ha",output=out("HA_REQUIRED",0.95,0.50)).accepted
    assert not p.decide(head="ha",output=out("HA_REQUIRED",0.89,0.01)).accepted
