from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from .dataset_schema import TrainingRecord

def load(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(TrainingRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records

def exact_text_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--final-holdout", required=True)
    args = parser.parse_args()

    datasets = {
        "train": load(Path(args.train)),
        "validation": load(Path(args.validation)),
        "test": load(Path(args.test)),
        "final_holdout": load(Path(args.final_holdout)),
    }

    errors = []
    registries = {
        "sample_id": {},
        "exact_text": {},
        "structure_fingerprint": {},
        "template_id": {},
    }
    summary = {}

    for split, records in datasets.items():
        summary[split] = {
            "rows": len(records),
            "language": dict(Counter(r.language for r in records)),
            "ha_label": dict(Counter(r.ha_label for r in records)),
            "access_label": dict(Counter(r.access_label for r in records)),
        }
        for r in records:
            values = {
                "sample_id": r.sample_id,
                "exact_text": exact_text_hash(r.text),
                "structure_fingerprint": r.structure_fingerprint,
                "template_id": r.template_id,
            }
            for kind, value in values.items():
                previous = registries[kind].get(value)
                if previous is not None and previous != split:
                    errors.append(f"CROSS_SPLIT_{kind.upper()}:{value}:{previous}->{split}")
                else:
                    registries[kind][value] = split

    report = {
        "status": "PASS" if not errors else "FAIL",
        "splits": summary,
        "cross_split_error_count": len(errors),
        "cross_split_errors": errors[:100],
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
