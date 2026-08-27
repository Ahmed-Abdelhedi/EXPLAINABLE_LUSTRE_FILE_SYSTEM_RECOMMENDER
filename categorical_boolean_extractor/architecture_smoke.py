from __future__ import annotations
import json
from .explicit import AccessTypeExplicitResolver, HAExplicitResolver
from .semantic.labels import ACCESS_LABELS, HA_LABELS

def main():
    ha=HAExplicitResolver(); access=AccessTypeExplicitResolver()
    checks={
        "ha_required_explicit_true":ha.resolve("HA is mandatory.").value is True,
        "ha_required_explicit_false":ha.resolve("HA is not required.").value is False,
        "ha_matters_not_auto_true":ha.resolve("HA matters.").value is None,
        "parallel_not_access_type":access.resolve("Clients perform I/O in parallel.").value is None,
        "streaming_to_sequential":access.resolve("Streaming workload.").value=="sequential",
        "ha_semantic_ontology_exact":HA_LABELS==[
            "HA_REQUIRED","HA_NOT_REQUIRED","HA_MENTION_NO_COMMITMENT","HA_NO_EVIDENCE"
        ],
        "access_semantic_ontology_exact":ACCESS_LABELS==[
            "SEQUENTIAL","RANDOM","MIXED","NO_SUPPORTED_ACCESS_CLASS"
        ],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report={
        "step":"5.3",
        "status":"SEMANTIC_ARCHITECTURE_SMOKE_PASS",
        "scope":"architecture contract smoke before single heavy training",
        "checks":checks,
        "heavy_model_loaded":False,
        "training_started":False,
    }
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    main()
