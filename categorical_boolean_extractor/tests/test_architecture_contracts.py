from __future__ import annotations
import json
from categorical_boolean_extractor.explicit import AccessTypeExplicitResolver, HAExplicitResolver
from categorical_boolean_extractor.final_validator import FinalSemanticValidator
from categorical_boolean_extractor.llm_fallback import CategoricalBooleanLLMFallback
from categorical_boolean_extractor.models import FieldStatus
from categorical_boolean_extractor.runtime import CategoricalBooleanExtractor
from categorical_boolean_extractor.semantic.schemas import (
    SemanticDecision, SemanticDecisionStatus, SemanticHeadOutput,
)

def accepted(label, top=0.99, second=0.01):
    return SemanticDecision(
        status=SemanticDecisionStatus.ACCEPTED,
        label=label,
        head_output=SemanticHeadOutput(
            probabilities={label:top,"OTHER":second},
            top_label=label,
            top_probability=top,
            second_probability=second,
            margin=top-second,
        ),
        probability_threshold=0.9,
        margin_threshold=0.5,
        reason="TEST_ACCEPT",
    )

def abstain(label="UNKNOWN", top=0.55, second=0.45):
    return SemanticDecision(
        status=SemanticDecisionStatus.ABSTAIN,
        label=None,
        head_output=SemanticHeadOutput(
            probabilities={label:top,"OTHER":second},
            top_label=label,
            top_probability=top,
            second_probability=second,
            margin=top-second,
        ),
        probability_threshold=0.9,
        margin_threshold=0.5,
        reason="TEST_ABSTAIN",
    )

class FakeSemantic:
    def __init__(self, ha_decision, access_decision):
        self.ha_decision=ha_decision
        self.access_decision=access_decision
        self.ha_calls=[]
        self.access_calls=[]
    def verify_ha(self,text):
        self.ha_calls.append(text)
        return self.ha_decision(text) if callable(self.ha_decision) else self.ha_decision
    def verify_access(self,text):
        self.access_calls.append(text)
        return self.access_decision(text) if callable(self.access_decision) else self.access_decision

class StubLLM(CategoricalBooleanLLMFallback):
    def __init__(self,response):
        super().__init__(enabled=True)
        self.response=response
    def _call(self,prompt):
        self.call_count += 1
        self.last_raw_response=self.response
        return self.response

def test_explicit_ha_required():
    r=HAExplicitResolver().resolve("High availability is mandatory.")
    assert r.status==FieldStatus.VERIFIED and r.value is True

def test_explicit_ha_not_required():
    r=HAExplicitResolver().resolve("HA is not required.")
    assert r.status==FieldStatus.VERIFIED and r.value is False

def test_ha_matters_not_auto_true():
    r=HAExplicitResolver().resolve("HA matters for this design.")
    assert r.value is None and r.status==FieldStatus.UNRESOLVED

def test_parallel_alone_not_access_class():
    r=AccessTypeExplicitResolver().resolve("64 clients perform I/O in parallel.")
    assert r.value is None and r.status==FieldStatus.UNRESOLVED

def test_streaming_is_sequential():
    assert AccessTypeExplicitResolver().resolve("Continuous streaming workload.").value=="sequential"

def test_semantic_ha_required_maps_true():
    semantic=FakeSemantic(accepted("HA_REQUIRED"),accepted("NO_SUPPORTED_ACCESS_CLASS"))
    ex=CategoricalBooleanExtractor(
        semantic_verifier=semantic,
        llm_fallback=CategoricalBooleanLLMFallback(enabled=False),
    )
    r=ex.extract("The service must survive failures.")
    assert r.ha_required.value is True
    assert r.ha_required.source.value=="SEMANTIC_MODEL"

def test_semantic_ha_mention_maps_no_value():
    semantic=FakeSemantic(accepted("HA_MENTION_NO_COMMITMENT"),accepted("NO_SUPPORTED_ACCESS_CLASS"))
    ex=CategoricalBooleanExtractor(
        semantic_verifier=semantic,
        llm_fallback=CategoricalBooleanLLMFallback(enabled=False),
    )
    r=ex.extract("HA matters.")
    assert r.ha_required.status==FieldStatus.NO_EVIDENCE
    assert r.ha_required.value is None

def test_semantic_access_parallel_negative_does_not_guess():
    semantic=FakeSemantic(accepted("HA_NO_EVIDENCE"),accepted("NO_SUPPORTED_ACCESS_CLASS"))
    ex=CategoricalBooleanExtractor(
        semantic_verifier=semantic,
        llm_fallback=CategoricalBooleanLLMFallback(enabled=False),
    )
    r=ex.extract("Clients perform I/O in parallel.")
    assert r.access_type.status==FieldStatus.UNRESOLVED
    assert r.access_type.value is None
    assert not r.llm_fallback_used

def test_only_abstained_field_reaches_llm():
    semantic=FakeSemantic(abstain("HA_MENTION_NO_COMMITMENT"),accepted("NO_SUPPORTED_ACCESS_CLASS"))
    llm=StubLLM(json.dumps({
        "fields":{"ha_required":{"status":"UNRESOLVED","label":None,"evidence":None}}
    }))
    ex=CategoricalBooleanExtractor(semantic_verifier=semantic,llm_fallback=llm)
    r=ex.extract("HA matters.")
    assert llm.call_count==1
    assert llm.last_requested_fields==["ha_required"]
    assert r.llm_fallback_used

def test_llm_ha_matters_true_is_rejected_by_semantic_recheck():
    # Full message is uncertain -> routes to Qwen.
    # Qwen copies the narrower evidence "HA matters" and wrongly proposes
    # HA_REQUIRED. Evidence-only semantic recheck recognizes it as
    # HA_MENTION_NO_COMMITMENT and rejects the unsafe boolean.
    def ha_decision(text):
        if text=="HA matters":
            return accepted("HA_MENTION_NO_COMMITMENT")
        return abstain("HA_MENTION_NO_COMMITMENT")
    semantic=FakeSemantic(ha_decision,accepted("NO_SUPPORTED_ACCESS_CLASS"))
    llm=StubLLM(json.dumps({
        "fields":{"ha_required":{
            "status":"VERIFIED","label":"HA_REQUIRED","evidence":"HA matters"
        }}
    }))
    ex=CategoricalBooleanExtractor(semantic_verifier=semantic,llm_fallback=llm)
    r=ex.extract("HA matters for our design")
    assert llm.call_count==1
    assert r.ha_required.status==FieldStatus.UNRESOLVED
    assert r.ha_required.value is None

def test_llm_valid_ha_evidence_can_pass_recheck():
    def ha_decision(text):
        return accepted("HA_REQUIRED") if text=="must survive node failures" else abstain("HA_REQUIRED")
    semantic=FakeSemantic(ha_decision,accepted("NO_SUPPORTED_ACCESS_CLASS"))
    llm=StubLLM(json.dumps({
        "fields":{"ha_required":{
            "status":"VERIFIED","label":"HA_REQUIRED","evidence":"must survive node failures"
        }}
    }))
    ex=CategoricalBooleanExtractor(semantic_verifier=semantic,llm_fallback=llm)
    r=ex.extract("The service must survive node failures.")
    assert r.ha_required.status==FieldStatus.VERIFIED
    assert r.ha_required.value is True

def test_llm_parallel_only_access_rejected():
    semantic=FakeSemantic(accepted("HA_NO_EVIDENCE"),accepted("RANDOM"))
    validator=FinalSemanticValidator()
    r=validator.validate_access(
        text="Many clients operate in parallel.",
        proposal={
            "status":"VERIFIED",
            "label":"RANDOM",
            "evidence":"clients operate in parallel",
        },
        semantic_verifier=semantic,
    )
    assert r.status==FieldStatus.UNRESOLVED

def test_explicit_verified_bypasses_semantic_for_that_field():
    semantic=FakeSemantic(accepted("HA_NOT_REQUIRED"),accepted("NO_SUPPORTED_ACCESS_CLASS"))
    ex=CategoricalBooleanExtractor(
        semantic_verifier=semantic,
        llm_fallback=CategoricalBooleanLLMFallback(enabled=False),
    )
    r=ex.extract("HA is mandatory. Clients operate in parallel.")
    assert r.ha_required.value is True
    assert semantic.ha_calls==[]

def test_semantic_random_maps_canonical_access():
    semantic=FakeSemantic(accepted("HA_NO_EVIDENCE"),accepted("RANDOM"))
    ex=CategoricalBooleanExtractor(
        semantic_verifier=semantic,
        llm_fallback=CategoricalBooleanLLMFallback(enabled=False),
    )
    r=ex.extract("Requests jump between unrelated offsets.")
    assert r.access_type.value=="random"

def test_confident_negative_semantics_prevent_unnecessary_llm():
    semantic=FakeSemantic(accepted("HA_MENTION_NO_COMMITMENT"),accepted("NO_SUPPORTED_ACCESS_CLASS"))
    llm=StubLLM("{}")
    ex=CategoricalBooleanExtractor(semantic_verifier=semantic,llm_fallback=llm)
    ex.extract("HA matters; I/O is parallel.")
    assert llm.call_count==0
