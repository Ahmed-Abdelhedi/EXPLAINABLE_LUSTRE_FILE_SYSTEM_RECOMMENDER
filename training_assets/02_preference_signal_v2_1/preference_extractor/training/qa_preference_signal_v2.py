from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


EXPECTED_SIZES = {
    "train_v2": 26000,
    "val_v2": 5200,
    "independent_holdout_v2": 1200,
}

EXPECTED_HARD_PER_FAMILY = {
    "train_v2": 600,
    "val_v2": 120,
    "independent_holdout_v2": 120,
}

HARD_FAMILIES = (
    "hard_conditional_choice",
    "hard_question_tradeoff",
    "hard_long_context_final_choice",
    "hard_ranked_telegraphic",
    "hard_counterfactual_tradeoff",
    "hard_third_party_rejected",
    "hard_negated_report",
    "hard_system_capability",
    "hard_rejected_draft",
    "hard_lexical_priority_trap",
)


def norm(text):
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    script = Path(__file__).resolve()
    repo = script.parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo / "preference_extractor" / "training" / "data" / "data_layer1_v2",
    )
    parser.add_argument(
        "--known-stress",
        type=Path,
        default=repo / "preference_extractor" / "evaluation" / "datasets" / "preference_signal_test_v1.jsonl",
    )
    args = parser.parse_args()

    paths = {
        "train_v2": args.data_dir / "preference_signal_train_v2.jsonl",
        "val_v2": args.data_dir / "preference_signal_val_v2.jsonl",
        "independent_holdout_v2": args.data_dir / "preference_signal_independent_holdout_v2.jsonl",
    }

    data = {name: read_jsonl(path) for name, path in paths.items()}
    checks = []

    def add(name, ok, observed, expected):
        checks.append({
            "name": name,
            "pass": bool(ok),
            "observed": observed,
            "expected": expected,
        })
        if not ok:
            raise AssertionError(f"{name}: observed={observed!r}, expected={expected!r}")

    required = {
        "sample_id", "group_id", "text", "label", "label_id", "language",
        "stress_family", "signal_dimensions", "template_family", "template_id",
        "source_dataset", "split",
    }

    for name, rows in data.items():
        add(f"{name}_size", len(rows) == EXPECTED_SIZES[name], len(rows), EXPECTED_SIZES[name])
        add(
            f"{name}_required_fields",
            all(required.issubset(row) for row in rows),
            True if all(required.issubset(row) for row in rows) else False,
            True,
        )
        add(
            f"{name}_unique_sample_id",
            len({x["sample_id"] for x in rows}) == len(rows),
            len({x["sample_id"] for x in rows}),
            len(rows),
        )
        add(
            f"{name}_unique_group_id",
            len({x["group_id"] for x in rows}) == len(rows),
            len({x["group_id"] for x in rows}),
            len(rows),
        )
        add(
            f"{name}_unique_text",
            len({norm(x["text"]) for x in rows}) == len(rows),
            len({norm(x["text"]) for x in rows}),
            len(rows),
        )

        labels = Counter(int(x["label_id"]) for x in rows)
        add(f"{name}_balanced_labels", labels[0] == labels[1], dict(labels), "50/50")

        langs = Counter(x["language"] for x in rows)
        expected_langs = {
            "en": int(len(rows) * 0.4),
            "fr": int(len(rows) * 0.4),
            "mixed": int(len(rows) * 0.2),
        }
        add(f"{name}_language_40_40_20", dict(langs) == expected_langs, dict(langs), expected_langs)

        hard = [x for x in rows if x["source_dataset"] == "hard_v2"]
        fam = Counter(x["stress_family"] for x in hard)
        expected_fam = {family: EXPECTED_HARD_PER_FAMILY[name] for family in HARD_FAMILIES}
        add(f"{name}_hard_family_counts", dict(fam) == expected_fam, dict(fam), expected_fam)

    pairs = (
        ("train_v2", "val_v2"),
        ("train_v2", "independent_holdout_v2"),
        ("val_v2", "independent_holdout_v2"),
    )

    for a, b in pairs:
        text_overlap = (
            {norm(x["text"]) for x in data[a]}
            & {norm(x["text"]) for x in data[b]}
        )
        group_overlap = (
            {x["group_id"] for x in data[a]}
            & {x["group_id"] for x in data[b]}
        )
        add(f"{a}_{b}_text_overlap_zero", not text_overlap, len(text_overlap), 0)
        add(f"{a}_{b}_group_overlap_zero", not group_overlap, len(group_overlap), 0)

        hard_a = [x for x in data[a] if x["source_dataset"] == "hard_v2"]
        hard_b = [x for x in data[b] if x["source_dataset"] == "hard_v2"]
        tf_overlap = (
            {x["template_family"] for x in hard_a}
            & {x["template_family"] for x in hard_b}
        )
        tid_overlap = (
            {x["template_id"] for x in hard_a}
            & {x["template_id"] for x in hard_b}
        )
        add(f"{a}_{b}_hard_template_family_overlap_zero", not tf_overlap, len(tf_overlap), 0)
        add(f"{a}_{b}_hard_template_id_overlap_zero", not tid_overlap, len(tid_overlap), 0)

    known_overlap = None
    if args.known_stress.exists():
        known = {norm(x["text"]) for x in read_jsonl(args.known_stress)}
        known_overlap = {}
        for name, rows in data.items():
            overlap = known & {norm(x["text"]) for x in rows}
            known_overlap[name] = len(overlap)
            add(f"{name}_known_500_exact_text_overlap_zero", not overlap, len(overlap), 0)

    report = {
        "dataset": "preference_signal_layer1_v2",
        "all_checks_pass": all(x["pass"] for x in checks),
        "checks_passed": sum(int(x["pass"]) for x in checks),
        "checks_total": len(checks),
        "known_500_exact_text_overlap": known_overlap,
        "checks": checks,
    }

    output = args.data_dir / "preference_signal_v2_QA_REPORT.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
