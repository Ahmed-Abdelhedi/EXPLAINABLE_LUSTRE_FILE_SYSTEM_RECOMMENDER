from __future__ import annotations
import argparse, json
from pathlib import Path
from .predict import predict_split

CALIBRATION_VERSION="precision_constrained_per_class_v1_20260826"

def top_stats(probabilities):
    ordered=sorted(probabilities.items(),key=lambda x:x[1],reverse=True)
    return ordered[0][0],float(ordered[0][1]),float(ordered[0][1]-ordered[1][1])

def calibrate_one_class(rows, *, probability_key, gold_key, predicted_label, target_precision):
    candidates=[]
    for row in rows:
        label,top,margin=top_stats(row[probability_key])
        if label==predicted_label:
            candidates.append({"top":top,"margin":margin,"correct":row[gold_key]==predicted_label})
    if not candidates:
        return None

    best=None
    for p_threshold in sorted({x["top"] for x in candidates},reverse=True):
        subset=[x for x in candidates if x["top"]>=p_threshold]
        for m_threshold in sorted({x["margin"] for x in subset},reverse=True):
            accepted=[x for x in subset if x["margin"]>=m_threshold]
            correct=sum(1 for x in accepted if x["correct"])
            precision=correct/len(accepted)
            if precision < target_precision:
                continue
            candidate={
                "min_top_probability":float(p_threshold),
                "min_margin":float(m_threshold),
                "validation_precision":float(precision),
                "validation_accepted":len(accepted),
            }
            if best is None or candidate["validation_accepted"] > best["validation_accepted"]:
                best=candidate
            elif (
                candidate["validation_accepted"] == best["validation_accepted"]
                and (candidate["min_top_probability"],candidate["min_margin"])
                < (best["min_top_probability"],best["min_margin"])
            ):
                best=candidate
    return best

def calibrate_head(rows, *, probability_key, gold_key, labels, target_precision):
    output={}
    for label in labels:
        calibrated=calibrate_one_class(
            rows,
            probability_key=probability_key,
            gold_key=gold_key,
            predicted_label=label,
            target_precision=target_precision,
        )
        if calibrated is not None:
            output[label]=calibrated
    return output

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir",required=True)
    p.add_argument("--validation",required=True)
    p.add_argument("--target-precision",type=float,default=0.99)
    p.add_argument("--batch-size",type=int,default=32)
    args=p.parse_args()
    if not 0.0 < args.target_precision <= 1.0:
        raise ValueError("target_precision must be in (0,1]")

    from ..semantic.labels import ACCESS_LABELS, HA_LABELS
    checkpoint=Path(args.checkpoint_dir)
    rows=predict_split(
        checkpoint_dir=checkpoint,
        dataset_path=Path(args.validation),
        batch_size=args.batch_size,
    )
    payload={
        "status":"CALIBRATED_ON_VALIDATION_BEFORE_TEST",
        "calibration_version":CALIBRATION_VERSION,
        "target_precision":args.target_precision,
        "thresholds":{
            "ha":calibrate_head(
                rows,probability_key="ha_probabilities",gold_key="ha_gold",
                labels=HA_LABELS,target_precision=args.target_precision
            ),
            "access":calibrate_head(
                rows,probability_key="access_probabilities",gold_key="access_gold",
                labels=ACCESS_LABELS,target_precision=args.target_precision
            ),
        },
        "validation_rows":len(rows),
        "test_used":False,
        "final_holdout_used":False,
    }
    (checkpoint/"calibration.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()
