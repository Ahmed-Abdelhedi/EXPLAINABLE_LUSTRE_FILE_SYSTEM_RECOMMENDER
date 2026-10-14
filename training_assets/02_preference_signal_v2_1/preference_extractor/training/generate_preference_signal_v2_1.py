from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


SEED = 20260823 + 41

DATA_V2_DIR = Path(
    "preference_extractor/training/data/data_layer1_v2"
)

DEFAULT_OUTPUT_DIR = Path(
    "preference_extractor/training/data/data_layer1_v2_1"
)

V2_TRAIN = DATA_V2_DIR / "preference_signal_train_v2.jsonl"
V2_VAL = DATA_V2_DIR / "preference_signal_val_v2.jsonl"
V2_OLD_HOLDOUT = (
    DATA_V2_DIR
    / "preference_signal_independent_holdout_v2.jsonl"
)
V2_LEGACY_TEST = (
    DATA_V2_DIR
    / "preference_signal_legacy_test_v1_4000.jsonl"
)

KNOWN_500 = Path(
    "preference_extractor/evaluation/datasets/"
    "preference_signal_test_v1.jsonl"
)

TARGETED_TRAIN_SIZE = 2400
TARGETED_VAL_SIZE = 600

TARGETED_TRAIN_LABELS = {
    1: 1200,
    0: 1200,
}

TARGETED_VAL_LABELS = {
    1: 300,
    0: 300,
}

TARGETED_TRAIN_LANGS = {
    "en": 1600,
    "mixed": 800,
}

TARGETED_VAL_LANGS = {
    "en": 400,
    "mixed": 200,
}

POSITIVE_FAMILY = "v21_long_context_final_choice"

NEGATIVE_FAMILIES = (
    "v21_long_context_rejected_choice",
    "v21_long_context_third_party_choice",
    "v21_long_context_capability_choice",
    "v21_long_context_no_preference",
)

DIMENSIONS = (
    "performance",
    "cost",
    "power",
    "reliability",
)

TERMS_EN = {
    "performance": (
        "I/O latency",
        "application throughput",
        "storage response time",
        "read performance",
        "write performance",
        "data access speed",
        "small-file responsiveness",
        "aggregate bandwidth",
    ),
    "cost": (
        "acquisition cost",
        "lifecycle cost",
        "operating expense",
        "infrastructure cost",
        "budget efficiency",
        "hardware spend",
        "support cost",
        "total ownership cost",
    ),
    "power": (
        "power consumption",
        "energy demand",
        "electrical draw",
        "energy efficiency",
        "rack power",
        "cooling burden",
        "operating wattage",
        "energy use",
    ),
    "reliability": (
        "service continuity",
        "fault tolerance",
        "availability",
        "resilience",
        "failure recovery",
        "redundancy",
        "operational robustness",
        "failure resistance",
    ),
}

TERMS_FR = {
    "performance": (
        "la latence E/S",
        "le débit applicatif",
        "le temps de réponse du stockage",
        "les performances en lecture",
        "les performances en écriture",
        "la vitesse d'accès aux données",
        "la réactivité sur les petits fichiers",
        "la bande passante agrégée",
    ),
    "cost": (
        "le coût d'acquisition",
        "le coût du cycle de vie",
        "les dépenses d'exploitation",
        "le coût d'infrastructure",
        "l'efficacité budgétaire",
        "les dépenses matérielles",
        "le coût du support",
        "le coût total de possession",
    ),
    "power": (
        "la consommation électrique",
        "la demande énergétique",
        "la puissance électrique",
        "l'efficacité énergétique",
        "la puissance par baie",
        "la charge de refroidissement",
        "la puissance en fonctionnement",
        "la consommation d'énergie",
    ),
    "reliability": (
        "la continuité de service",
        "la tolérance aux pannes",
        "la disponibilité",
        "la résilience",
        "la reprise après panne",
        "la redondance",
        "la robustesse opérationnelle",
        "la résistance aux pannes",
    ),
}

FACTS_EN = (
    "The workload is sized for 800 TiB, 300 clients and 60 GB/s of reads",
    "Capacity, client count, file count and throughput are already fixed",
    "Every shortlisted design satisfies the mandatory capacity requirement",
    "The technical review confirms that all candidates meet the throughput floor",
    "The sizing pass already validated capacity, growth horizon and client count",
    "All remaining architectures satisfy the hard technical constraints",
    "The procurement shortlist contains only technically compliant systems",
    "The engineering team has already checked capacity and throughput feasibility",
    "The baseline workload parameters no longer separate the surviving candidates",
    "All mandatory Lustre sizing constraints are already satisfied",
)

FACTS_MIXED = (
    "The workload est dimensionné for 800 TiB, 300 clients et 60 GB/s reads",
    "Capacity, client count, file count et throughput sont already fixed",
    "Every shortlisted design respecte the mandatory capacity requirement",
    "The technical review confirme que all candidates meet the throughput floor",
    "The sizing pass a déjà validated capacity, growth horizon et client count",
    "All remaining architectures satisfont les hard technical constraints",
    "The procurement shortlist contient only technically compliant systems",
    "The engineering team a déjà checked capacity et throughput feasibility",
    "The baseline workload parameters ne séparent plus the surviving candidates",
    "All mandatory Lustre sizing constraints sont already satisfied",
)

BRIDGES_EN = (
    "Those quantitative values no longer distinguish the candidates",
    "The decision therefore moves to the preference layer",
    "The remaining choice is no longer about feasibility",
    "At this point the hard constraints are not the differentiator",
    "The final recommendation must now use a soft preference",
    "Only the decision criteria remain to separate the alternatives",
)

BRIDGES_MIXED = (
    "Those quantitative values ne distinguent plus les candidates",
    "The decision therefore passe to the preference layer",
    "The remaining choice n'est plus about feasibility",
    "At this point les hard constraints are not the differentiator",
    "The final recommendation doit now use a soft preference",
    "Only les decision criteria restent to separate the alternatives",
)

POSITIVE_TEMPLATES = {
    "train": {
        "en": (
            "{fact}. {bridge}. For the final choice, put {a} ahead of {b}.",
            "{fact}. {bridge}. The deciding factor should be {a}, while {b} remains secondary.",
            "{fact}. {bridge}. When selecting among the survivors, favor {a} before {b}.",
            "{fact}. {bridge}. Resolve the final tie according to {a} rather than {b}.",
            "{fact}. {bridge}. The preferred architecture is the one that improves {a}, even if {b} is not maximized.",
            "{fact}. {bridge}. Treat {a} as the primary soft criterion and {b} as the fallback criterion.",
            "{fact}. {bridge}. If the candidates stay equivalent on hard constraints, choose on {a} first and {b} second.",
            "{fact}. {bridge}. The recommendation should now optimize for {a}; {b} has lower priority.",
        ),
        "mixed": (
            "{fact}. {bridge}. For the final choix, put {a} ahead of {b}.",
            "{fact}. {bridge}. The deciding factor doit être {a}, while {b} reste secondary.",
            "{fact}. {bridge}. When selecting among les survivors, favor {a} before {b}.",
            "{fact}. {bridge}. Resolve the final tie selon {a} rather than {b}.",
            "{fact}. {bridge}. The preferred architecture est celle that improves {a}, even if {b} is not maximized.",
            "{fact}. {bridge}. Treat {a} comme the primary soft criterion and {b} as fallback.",
            "{fact}. {bridge}. If les candidates stay equivalent on hard constraints, choose on {a} first et {b} second.",
            "{fact}. {bridge}. The recommendation should now optimize for {a}; {b} a lower priority.",
        ),
    },
    "val": {
        "en": (
            "{fact}. {bridge}. From now on, rank the compliant options by {a}; use {b} only after that.",
            "{fact}. {bridge}. The last selection step must privilege {a} over {b}.",
            "{fact}. {bridge}. Among feasible designs, the winning criterion is {a}, not {b}.",
            "{fact}. {bridge}. Make the final recommendation on the basis of {a}, with {b} subordinate.",
        ),
        "mixed": (
            "{fact}. {bridge}. From now on, rank les compliant options by {a}; use {b} only after that.",
            "{fact}. {bridge}. The last selection step doit privilege {a} over {b}.",
            "{fact}. {bridge}. Among feasible designs, le winning criterion is {a}, not {b}.",
            "{fact}. {bridge}. Make the final recommendation sur la base de {a}, with {b} subordinate.",
        ),
    },
}

NEGATIVE_TEMPLATES = {
    "v21_long_context_rejected_choice": {
        "train": {
            "en": (
                "{fact}. {bridge}. An early draft proposed choosing on {a}, but that draft was rejected and does not define the active requirement.",
                "{fact}. {bridge}. A discarded scenario ranked {a} first; the approved requirements explicitly supersede that scenario.",
                "{fact}. {bridge}. The team previously considered making {a} decisive, but the idea was removed before approval.",
                "{fact}. {bridge}. One obsolete design note favored {a}; it is historical context only and is not an active preference.",
            ),
            "mixed": (
                "{fact}. {bridge}. An early draft proposed choosing on {a}, mais ce draft was rejected and does not define the active requirement.",
                "{fact}. {bridge}. A discarded scenario ranked {a} first; les approved requirements supersede that scenario.",
                "{fact}. {bridge}. The team previously considered making {a} decisive, mais the idea was removed before approval.",
                "{fact}. {bridge}. One obsolete design note favored {a}; c'est historical context only and not an active preference.",
            ),
        },
        "val": {
            "en": (
                "{fact}. {bridge}. A superseded proposal used {a} as the tie-breaker, but the current request contains no such preference.",
                "{fact}. {bridge}. A retired draft placed {a} first; that language is not part of the signed requirements.",
            ),
            "mixed": (
                "{fact}. {bridge}. A superseded proposal used {a} as tie-breaker, mais the current request contains no such preference.",
                "{fact}. {bridge}. A retired draft placed {a} first; ce language is not part of the signed requirements.",
            ),
        },
    },
    "v21_long_context_third_party_choice": {
        "train": {
            "en": (
                "{fact}. {bridge}. The vendor recommends prioritizing {a}; this is the vendor's preference, not ours.",
                "{fact}. {bridge}. A consultant ranks {a} first, but the statement only records the consultant's view.",
                "{fact}. {bridge}. The benchmark report argues for {a}; it does not establish a requirement for our design.",
                "{fact}. {bridge}. The supplier case study favors {a}. That external recommendation is not our selection preference.",
            ),
            "mixed": (
                "{fact}. {bridge}. The vendor recommends prioritizing {a}; c'est the vendor's preference, not ours.",
                "{fact}. {bridge}. A consultant ranks {a} first, mais the statement only records the consultant's view.",
                "{fact}. {bridge}. The benchmark report argues for {a}; il ne establishes pas a requirement for our design.",
                "{fact}. {bridge}. The supplier case study favors {a}. That external recommendation n'est pas our selection preference.",
            ),
        },
        "val": {
            "en": (
                "{fact}. {bridge}. An external study prefers {a}; the current requirement merely cites that study and adopts no preference.",
                "{fact}. {bridge}. The integrator suggests selecting on {a}, but that suggestion is not an instruction from us.",
            ),
            "mixed": (
                "{fact}. {bridge}. An external study prefers {a}; the current requirement cite that study and adopts no preference.",
                "{fact}. {bridge}. The integrator suggests selecting on {a}, mais that suggestion is not an instruction from us.",
            ),
        },
    },
    "v21_long_context_capability_choice": {
        "train": {
            "en": (
                "{fact}. {bridge}. The controller can rank candidates using {a}; this describes a software capability, not a requested preference.",
                "{fact}. {bridge}. The orchestration layer supports choosing by {a}. No user preference is stated.",
                "{fact}. {bridge}. The system is able to optimize {a} during selection; capability does not imply that we want it prioritized.",
                "{fact}. {bridge}. The policy engine can use {a} as a ranking key, but the request does not tell it to do so.",
            ),
            "mixed": (
                "{fact}. {bridge}. The controller can rank candidates using {a}; cela describes a software capability, not a requested preference.",
                "{fact}. {bridge}. The orchestration layer supports choosing by {a}. Aucune user preference is stated.",
                "{fact}. {bridge}. The system is able to optimize {a} during selection; capability ne veut pas dire we want it prioritized.",
                "{fact}. {bridge}. The policy engine can use {a} as a ranking key, mais the request does not tell it to do so.",
            ),
        },
        "val": {
            "en": (
                "{fact}. {bridge}. The recommender can break ties with {a}; the sentence only documents supported functionality.",
                "{fact}. {bridge}. The software permits {a}-based ranking, but no active preference selects that option.",
            ),
            "mixed": (
                "{fact}. {bridge}. The recommender can break ties with {a}; la sentence only documents supported functionality.",
                "{fact}. {bridge}. The software permits {a}-based ranking, mais no active preference selects that option.",
            ),
        },
    },
    "v21_long_context_no_preference": {
        "train": {
            "en": (
                "{fact}. {bridge}. The approved requirement deliberately leaves the final choice neutral; no criterion is preferred.",
                "{fact}. {bridge}. No decision has been made to favor {a} or any other soft criterion.",
                "{fact}. {bridge}. The requirement set records no ordering between cost, power, reliability and performance.",
                "{fact}. {bridge}. Selection remains neutral after the hard constraints; there is no stated preference.",
            ),
            "mixed": (
                "{fact}. {bridge}. The approved requirement laisse deliberately the final choice neutral; no criterion is preferred.",
                "{fact}. {bridge}. No decision has been made to favor {a} ou any other soft criterion.",
                "{fact}. {bridge}. The requirement set records no ordering between cost, power, reliability et performance.",
                "{fact}. {bridge}. Selection remains neutral after les hard constraints; there is no stated preference.",
            ),
        },
        "val": {
            "en": (
                "{fact}. {bridge}. The compliant alternatives are intentionally left unordered because no preference has been approved.",
                "{fact}. {bridge}. The final ranking criterion is unspecified; the request stays neutral.",
            ),
            "mixed": (
                "{fact}. {bridge}. The compliant alternatives are intentionally left unordered parce que no preference has been approved.",
                "{fact}. {bridge}. The final ranking criterion is unspecified; la request stays neutral.",
            ),
        },
    },
}


def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text).strip().casefold(),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def stable_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256(
        "||".join(parts).encode(
            "utf-8"
        )
    ).digest()

    seed = int.from_bytes(
        digest[:8],
        "big",
    ) ^ SEED

    return random.Random(seed)


def choose_terms(
    language: str,
    rng: random.Random,
):
    primary = rng.choice(
        DIMENSIONS
    )

    secondary = rng.choice(
        [
            dimension
            for dimension in DIMENSIONS
            if dimension != primary
        ]
    )

    if language == "en":
        table = TERMS_EN
    else:
        table = TERMS_FR

    return (
        primary,
        secondary,
        rng.choice(
            table[primary]
        ),
        rng.choice(
            table[secondary]
        ),
    )


def facts_for(language: str):
    return (
        FACTS_EN
        if language == "en"
        else FACTS_MIXED
    )


def bridges_for(language: str):
    return (
        BRIDGES_EN
        if language == "en"
        else BRIDGES_MIXED
    )


def make_row(
    *,
    split: str,
    language: str,
    family: str,
    template_index: int,
    text: str,
    label_id: int,
    sample_number: int,
) -> dict[str, Any]:
    label = (
        "PREFERENCE_SIGNAL"
        if label_id == 1
        else "NO_PREFERENCE_SIGNAL"
    )

    return {
        "sample_id": (
            f"v21_{split}_{sample_number:06d}"
        ),
        "group_id": (
            f"v21_{split}_{family}_{sample_number:06d}"
        ),
        "text": text,
        "label": label,
        "label_id": label_id,
        "language": language,
        "source_dataset": (
            "preference_signal_targeted_v2_1"
        ),
        "split": split,
        "hard_family": family,
        "stress_family": family,
        "template_family": (
            f"v21::{split}::{family}"
        ),
        "template_id": (
            f"v21::{split}::{family}::t{template_index:02d}"
        ),
        "is_targeted_v21": True,
    }


def generate_unique_rows(
    *,
    split: str,
    family: str,
    language: str,
    count: int,
    label_id: int,
    templates: tuple[str, ...],
    start_number: int,
) -> list[dict[str, Any]]:
    rng = stable_rng(
        split,
        family,
        language,
    )

    unique: dict[str, dict[str, Any]] = {}

    attempts = 0

    while len(unique) < count:
        (
            _primary,
            _secondary,
            a,
            b,
        ) = choose_terms(
            language,
            rng,
        )

        template_index = rng.randrange(
            len(templates)
        )

        template = templates[
            template_index
        ]

        text = template.format(
            fact=rng.choice(
                facts_for(language)
            ),
            bridge=rng.choice(
                bridges_for(language)
            ),
            a=a,
            b=b,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        key = normalize(text)

        if key not in unique:
            sample_number = (
                start_number
                + len(unique)
            )

            unique[key] = make_row(
                split=split,
                language=language,
                family=family,
                template_index=template_index,
                text=text,
                label_id=label_id,
                sample_number=sample_number,
            )

        attempts += 1

        if attempts > count * 500:
            raise RuntimeError(
                "Could not generate enough unique examples "
                f"for {split}/{family}/{language}. "
                f"Generated {len(unique)} / {count}."
            )

    return list(
        unique.values()
    )


def build_targeted_split(
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_number = 1

    if split == "train":
        positive_quota = {
            "en": 800,
            "mixed": 400,
        }

        negative_family_quota = {
            family: {
                "en": 200,
                "mixed": 100,
            }
            for family in NEGATIVE_FAMILIES
        }

    elif split == "val":
        positive_quota = {
            "en": 200,
            "mixed": 100,
        }

        negative_family_quota = {
            family: {
                "en": 50,
                "mixed": 25,
            }
            for family in NEGATIVE_FAMILIES
        }

    else:
        raise ValueError(split)

    for language, count in positive_quota.items():
        templates = POSITIVE_TEMPLATES[
            split
        ][
            language
        ]

        generated = generate_unique_rows(
            split=split,
            family=POSITIVE_FAMILY,
            language=language,
            count=count,
            label_id=1,
            templates=templates,
            start_number=sample_number,
        )

        rows.extend(generated)
        sample_number += len(
            generated
        )

    for family in NEGATIVE_FAMILIES:
        for language, count in (
            negative_family_quota[
                family
            ].items()
        ):
            templates = NEGATIVE_TEMPLATES[
                family
            ][
                split
            ][
                language
            ]

            generated = generate_unique_rows(
                split=split,
                family=family,
                language=language,
                count=count,
                label_id=0,
                templates=templates,
                start_number=sample_number,
            )

            rows.extend(generated)
            sample_number += len(
                generated
            )

    rng = stable_rng(
        "shuffle",
        split,
    )

    rng.shuffle(rows)

    return rows


def assert_unique(
    rows: list[dict[str, Any]],
    field: str,
    expected: int,
) -> None:
    observed = len(
        {
            str(row[field])
            for row in rows
        }
    )

    if observed != expected:
        raise AssertionError(
            f"{field}: expected {expected} unique, got {observed}"
        )


def validate_targeted(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(train_rows) != TARGETED_TRAIN_SIZE:
        raise AssertionError(
            len(train_rows)
        )

    if len(val_rows) != TARGETED_VAL_SIZE:
        raise AssertionError(
            len(val_rows)
        )

    for rows, expected, name in (
        (
            train_rows,
            TARGETED_TRAIN_SIZE,
            "train",
        ),
        (
            val_rows,
            TARGETED_VAL_SIZE,
            "val",
        ),
    ):
        assert_unique(
            rows,
            "sample_id",
            expected,
        )

        assert_unique(
            rows,
            "group_id",
            expected,
        )

        normalized = {
            normalize(
                row["text"]
            )
            for row in rows
        }

        if len(normalized) != expected:
            raise AssertionError(
                f"{name}: duplicate texts"
            )

    train_labels = Counter(
        int(
            row[
                "label_id"
            ]
        )
        for row in train_rows
    )

    val_labels = Counter(
        int(
            row[
                "label_id"
            ]
        )
        for row in val_rows
    )

    if dict(
        train_labels
    ) != TARGETED_TRAIN_LABELS:
        raise AssertionError(
            (
                train_labels,
                TARGETED_TRAIN_LABELS,
            )
        )

    if dict(
        val_labels
    ) != TARGETED_VAL_LABELS:
        raise AssertionError(
            (
                val_labels,
                TARGETED_VAL_LABELS,
            )
        )

    train_langs = Counter(
        row[
            "language"
        ]
        for row in train_rows
    )

    val_langs = Counter(
        row[
            "language"
        ]
        for row in val_rows
    )

    if dict(
        train_langs
    ) != TARGETED_TRAIN_LANGS:
        raise AssertionError(
            (
                train_langs,
                TARGETED_TRAIN_LANGS,
            )
        )

    if dict(
        val_langs
    ) != TARGETED_VAL_LANGS:
        raise AssertionError(
            (
                val_langs,
                TARGETED_VAL_LANGS,
            )
        )

    train_text = {
        normalize(
            row[
                "text"
            ]
        )
        for row in train_rows
    }

    val_text = {
        normalize(
            row[
                "text"
            ]
        )
        for row in val_rows
    }

    if train_text & val_text:
        raise AssertionError(
            "targeted train/val text overlap"
        )

    train_template_family = {
        row[
            "template_family"
        ]
        for row in train_rows
    }

    val_template_family = {
        row[
            "template_family"
        ]
        for row in val_rows
    }

    if (
        train_template_family
        & val_template_family
    ):
        raise AssertionError(
            "targeted train/val template-family overlap"
        )

    train_template_id = {
        row[
            "template_id"
        ]
        for row in train_rows
    }

    val_template_id = {
        row[
            "template_id"
        ]
        for row in val_rows
    }

    if train_template_id & val_template_id:
        raise AssertionError(
            "targeted train/val template-id overlap"
        )

    return {
        "targeted_train_size": len(
            train_rows
        ),
        "targeted_val_size": len(
            val_rows
        ),
        "targeted_train_label_counts": dict(
            train_labels
        ),
        "targeted_val_label_counts": dict(
            val_labels
        ),
        "targeted_train_language_counts": dict(
            train_langs
        ),
        "targeted_val_language_counts": dict(
            val_langs
        ),
        "targeted_train_val_text_overlap": 0,
        "targeted_train_val_template_family_overlap": 0,
        "targeted_train_val_template_id_overlap": 0,
    }


def exact_overlap_count(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> int:
    candidate = {
        normalize(
            row[
                "text"
            ]
        )
        for row in candidate_rows
    }

    reference = {
        normalize(
            row[
                "text"
            ]
        )
        for row in reference_rows
    }

    return len(
        candidate
        & reference
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate V2.1 targeted long-context augmentation. "
            "The 72 known V2 validation FN are NOT copied into training. "
            "New template families are generated instead."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--targeted-only",
        action="store_true",
        help=(
            "Generate only targeted train/val. "
            "Used for package self-tests when V2 files are unavailable."
        ),
    )

    args = parser.parse_args()

    output_dir = args.output_dir

    targeted_train = build_targeted_split(
        "train"
    )

    targeted_val = build_targeted_split(
        "val"
    )

    targeted_summary = validate_targeted(
        targeted_train,
        targeted_val,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    targeted_train_path = (
        output_dir
        / "preference_signal_targeted_train_v2_1.jsonl"
    )

    targeted_val_path = (
        output_dir
        / "preference_signal_targeted_val_v2_1.jsonl"
    )

    write_jsonl(
        targeted_train_path,
        targeted_train,
    )

    write_jsonl(
        targeted_val_path,
        targeted_val,
    )

    metadata: dict[str, Any] = {
        "task": (
            "binary preference-signal detection"
        ),
        "version": "v2.1",
        "seed": SEED,
        "motivation": {
            "observed_v2_validation_false_negatives": 72,
            "all_false_negatives_family": (
                "hard_long_context_final_choice"
            ),
            "false_negatives_by_language": {
                "en": 48,
                "mixed": 24,
                "fr": 0,
            },
            "design_decision": (
                "target EN and mixed long-context final-choice "
                "semantics with matched hard negatives"
            ),
        },
        "targeted_design": {
            "positive_family": POSITIVE_FAMILY,
            "negative_families": list(
                NEGATIVE_FAMILIES
            ),
            "train_size": TARGETED_TRAIN_SIZE,
            "val_size": TARGETED_VAL_SIZE,
            "train_label_counts": TARGETED_TRAIN_LABELS,
            "val_label_counts": TARGETED_VAL_LABELS,
            "train_language_counts": TARGETED_TRAIN_LANGS,
            "val_language_counts": TARGETED_VAL_LANGS,
            "template_family_isolation": True,
            "known_72_validation_fn_copied_to_training": False,
        },
        **targeted_summary,
    }

    if args.targeted_only:
        metadata[
            "mode"
        ] = "targeted_only"

        metadata_path = (
            output_dir
            / "preference_signal_v2_1_metadata.json"
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            )
        )

        print(
            f"\nSaved targeted train: {targeted_train_path}"
        )

        print(
            f"Saved targeted val:   {targeted_val_path}"
        )

        return

    required = (
        V2_TRAIN,
        V2_VAL,
        V2_OLD_HOLDOUT,
        V2_LEGACY_TEST,
        KNOWN_500,
    )

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required V2/reference files:\n- "
            + "\n- ".join(
                missing
            )
        )

    old_train = read_jsonl(
        V2_TRAIN
    )

    old_val = read_jsonl(
        V2_VAL
    )

    old_holdout = read_jsonl(
        V2_OLD_HOLDOUT
    )

    legacy_test = read_jsonl(
        V2_LEGACY_TEST
    )

    known_500 = read_jsonl(
        KNOWN_500
    )

    reference_sets = {
        "v2_train": old_train,
        "v2_val": old_val,
        "v2_old_holdout_regression": old_holdout,
        "known_500_regression": known_500,
    }

    overlap_report = {}

    for name, reference_rows in (
        reference_sets.items()
    ):
        overlap_report[
            f"targeted_train_vs_{name}"
        ] = exact_overlap_count(
            targeted_train,
            reference_rows,
        )

        overlap_report[
            f"targeted_val_vs_{name}"
        ] = exact_overlap_count(
            targeted_val,
            reference_rows,
        )

    if any(
        value != 0
        for value in overlap_report.values()
    ):
        raise AssertionError(
            {
                key: value
                for key, value
                in overlap_report.items()
                if value != 0
            }
        )

    train_v21 = (
        old_train
        + targeted_train
    )

    val_v21 = (
        old_val
        + targeted_val
    )

    expected_train = 28400
    expected_val = 5800

    if len(train_v21) != expected_train:
        raise AssertionError(
            len(train_v21)
        )

    if len(val_v21) != expected_val:
        raise AssertionError(
            len(val_v21)
        )

    train_text = {
        normalize(
            row[
                "text"
            ]
        )
        for row in train_v21
    }

    val_text = {
        normalize(
            row[
                "text"
            ]
        )
        for row in val_v21
    }

    if len(train_text) != expected_train:
        raise AssertionError(
            "combined train contains duplicate text"
        )

    if len(val_text) != expected_val:
        raise AssertionError(
            "combined val contains duplicate text"
        )

    if train_text & val_text:
        raise AssertionError(
            f"combined train/val exact-text overlap: "
            f"{len(train_text & val_text)}"
        )

    train_group_ids = {
        str(
            row[
                "group_id"
            ]
        )
        for row in train_v21
    }

    val_group_ids = {
        str(
            row[
                "group_id"
            ]
        )
        for row in val_v21
    }

    if train_group_ids & val_group_ids:
        raise AssertionError(
            f"combined train/val group overlap: "
            f"{len(train_group_ids & val_group_ids)}"
        )

    train_path = (
        output_dir
        / "preference_signal_train_v2_1.jsonl"
    )

    val_path = (
        output_dir
        / "preference_signal_val_v2_1.jsonl"
    )

    write_jsonl(
        train_path,
        train_v21,
    )

    write_jsonl(
        val_path,
        val_v21,
    )

    # Convenience copies for later regression only.
    regression_val_path = (
        output_dir
        / "preference_signal_regression_val_v2_5200.jsonl"
    )

    regression_holdout_path = (
        output_dir
        / "preference_signal_regression_holdout_v2_1200.jsonl"
    )

    legacy_path = (
        output_dir
        / "preference_signal_legacy_test_v1_4000.jsonl"
    )

    write_jsonl(
        regression_val_path,
        old_val,
    )

    write_jsonl(
        regression_holdout_path,
        old_holdout,
    )

    write_jsonl(
        legacy_path,
        legacy_test,
    )

    combined_train_labels = Counter(
        int(
            row[
                "label_id"
            ]
        )
        for row in train_v21
    )

    combined_val_labels = Counter(
        int(
            row[
                "label_id"
            ]
        )
        for row in val_v21
    )

    if dict(
        combined_train_labels
    ) != {
        0: 14200,
        1: 14200,
    }:
        raise AssertionError(
            combined_train_labels
        )

    if dict(
        combined_val_labels
    ) != {
        0: 2900,
        1: 2900,
    }:
        raise AssertionError(
            combined_val_labels
        )

    metadata.update(
        {
            "mode": "full",
            "combined_sizes": {
                "train_v2_1": len(
                    train_v21
                ),
                "val_v2_1": len(
                    val_v21
                ),
                "regression_val_v2": len(
                    old_val
                ),
                "regression_holdout_v2": len(
                    old_holdout
                ),
                "legacy_test_v1": len(
                    legacy_test
                ),
            },
            "combined_label_counts": {
                "train_v2_1": dict(
                    combined_train_labels
                ),
                "val_v2_1": dict(
                    combined_val_labels
                ),
            },
            "exact_text_overlap_checks": overlap_report,
            "combined_train_val_text_overlap": 0,
            "combined_train_val_group_overlap": 0,
            "protocol": {
                "old_v2_validation": (
                    "development/regression only; it has been inspected"
                ),
                "old_v2_holdout": (
                    "regression only; it has been inspected"
                ),
                "known_500": (
                    "regression only; never merged into training"
                ),
                "fresh_final_holdout_v3": (
                    "still untouched; do not generate/evaluate until "
                    "V2.1 checkpoint and final threshold are frozen"
                ),
            },
        }
    )

    metadata_path = (
        output_dir
        / "preference_signal_v2_1_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\nSaved combined train: {train_path}"
    )

    print(
        f"Saved combined val:   {val_path}"
    )

    print(
        f"Saved metadata:       {metadata_path}"
    )


if __name__ == "__main__":
    main()
