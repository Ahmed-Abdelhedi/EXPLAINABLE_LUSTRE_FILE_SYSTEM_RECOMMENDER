#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

OUTCOMES={"READY_COHERENT","READY_AMBIGUOUS","BLOCKED_PLAUSIBILITY","CLARIFICATION_REQUIRED"}
FIELDS={"requested_usable_capacity_tib","client_count","average_file_size_gb","max_file_size_gb","total_file_count","read_write_ratio","access_type","target_read_gbps","target_write_gbps","ha_required","max_budget_usd","max_power_w","annual_growth_percent"}

def validate(path: Path):
    data=json.loads(path.read_text(encoding="utf-8"))
    scenarios=data.get("scenarios")
    assert isinstance(scenarios,list) and len(scenarios)==30
    ids=[s["id"] for s in scenarios]
    assert len(ids)==len(set(ids))
    counts=Counter()
    for s in scenarios:
        sid=s["id"]; outcome=s["pipeline_outcome"]; counts[outcome]+=1
        assert outcome in OUTCOMES, f"{sid}: bad outcome"
        assert isinstance(s.get("turns"),list) and s["turns"], f"{sid}: no turns"
        fr=s["expected_final_requirements"]
        assert set(fr)==FIELDS, f"{sid}: bad field set"
        e=s["expected"]
        if outcome=="READY_COHERENT":
            assert e["plausibility_status"]=="COHERENT" and e["should_block_recommendation"] is False
            assert not e["blocking_issue_codes"] and not e["warning_issue_codes"]
        elif outcome=="READY_AMBIGUOUS":
            assert e["plausibility_status"]=="AMBIGUOUS" and e["should_block_recommendation"] is False
            assert e["warning_issue_codes"] and not e["blocking_issue_codes"]
        elif outcome=="BLOCKED_PLAUSIBILITY":
            assert e["plausibility_status"]=="INCOHERENT" and e["should_block_recommendation"] is True
            assert e["blocking_issue_codes"]
        else:
            assert e["plausibility_status"] is None and e["should_block_recommendation"] is None
            assert e["clarification_fields"]
            for f in e["clarification_fields"]:
                assert f in FIELDS and fr[f] is None
    assert counts==Counter({"READY_COHERENT":7,"READY_AMBIGUOUS":8,"BLOCKED_PLAUSIBILITY":5,"CLARIFICATION_REQUIRED":10})
    multi=sum(len(s["turns"])>1 for s in scenarios)
    assert multi>=6
    print("Dataset validation: OK")
    print("Scenarios:",len(scenarios))
    print("Outcome distribution:",dict(counts))
    print("Multi-turn scenarios:",multi)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("dataset",type=Path); a=p.parse_args(); validate(a.dataset)