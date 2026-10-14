from __future__ import annotations
import argparse, json
from pathlib import Path
from ..semantic.confidence import CalibratedConfidencePolicy
from ..semantic.schemas import SemanticHeadOutput
from .predict import predict_split
from .train_multitask import macro_f1

def top_stats(probabilities):
    ordered=sorted(probabilities.items(),key=lambda x:x[1],reverse=True)
    return ordered[0][0],ordered[0][1],ordered[1][1]

def eval_head(rows, *, head, probability_key, gold_key, policy, labels):
    gold=[]; pred=[]; accepted=0; correct=0
    for row in rows:
        label,top,second=top_stats(row[probability_key])
        gold.append(labels.index(row[gold_key]))
        pred.append(labels.index(label))
        decision=policy.decide(
            head=head,
            output=SemanticHeadOutput(
                probabilities=row[probability_key],
                top_label=label,
                top_probability=float(top),
                second_probability=float(second),
                margin=float(top-second),
            ),
        )
        if decision.accepted:
            accepted += 1
            correct += int(decision.label == row[gold_key])
    return {
        "raw_macro_f1":macro_f1(gold,pred,len(labels)),
        "accepted":accepted,
        "accepted_precision":correct/accepted if accepted else None,
        "abstained":len(rows)-accepted,
        "coverage":accepted/len(rows),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir",required=True)
    p.add_argument("--test",required=True)
    p.add_argument("--batch-size",type=int,default=32)
    args=p.parse_args()
    checkpoint=Path(args.checkpoint_dir)
    calibration=checkpoint/"calibration.json"
    if not calibration.exists():
        raise FileNotFoundError("Calibration must be frozen before TEST.")
    policy=CalibratedConfidencePolicy.from_json(calibration)
    rows=predict_split(
        checkpoint_dir=checkpoint,dataset_path=Path(args.test),batch_size=args.batch_size
    )
    from ..semantic.labels import ACCESS_LABELS, HA_LABELS
    report={
        "status":"FROZEN_TEST_EVALUATION_COMPLETE",
        "rows":len(rows),
        "ha":eval_head(rows,head="ha",probability_key="ha_probabilities",gold_key="ha_gold",policy=policy,labels=HA_LABELS),
        "access":eval_head(rows,head="access",probability_key="access_probabilities",gold_key="access_gold",policy=policy,labels=ACCESS_LABELS),
        "calibration_modified_after_test":False,
        "final_holdout_used":False,
    }
    (checkpoint/"frozen_test_metrics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
