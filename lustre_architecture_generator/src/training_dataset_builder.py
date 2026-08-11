r"""Construit les datasets Learning-to-Rank MDT et OST pour Kaggle.

Les Top-10 déterministes existants sont conservés et utilisés comme
référence de validation. Ce script génère de nouveaux fichiers contenant
tous les candidats pré-RAID faisables.

Entrées par défaut :
- output/lustre_architecture_dataset.json
- data/catalogue_drives_ready_final.json
- output/mdt_drive_candidates_dataset.json
- output/ost_drive_candidates_dataset.json

Sorties par défaut :
- output/training/mdt_training_dataset.csv.gz
- output/training/ost_training_dataset.csv.gz
- output/training/case_splits.json
- output/training/training_dataset_manifest.json
- output/training/lustre_ranker_training_data_kaggle.zip

Exécution depuis la racine du projet :

    python .\src\training_dataset_builder.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mdt_candidate_generator as mdt_generator  # noqa: E402
import ost_candidate_generator as ost_generator  # noqa: E402


DEFAULT_ARCHITECTURES = (
    BASE_DIR / "output" / "lustre_architecture_dataset.json"
)
DEFAULT_CATALOG = (
    BASE_DIR / "data" / "catalogue_drives_ready_final.json"
)
DEFAULT_MDT_REFERENCE = (
    BASE_DIR / "output" / "mdt_drive_candidates_dataset.json"
)
DEFAULT_OST_REFERENCE = (
    BASE_DIR / "output" / "ost_drive_candidates_dataset.json"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "training"

BUILDER_VERSION = "1.0"
DEFAULT_SEED = 42

SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


MDT_IDENTIFIER_COLUMNS = [
    "split",
    "case_id",
    "drive_id",
    "drive_name",
    "manufacturer",
    "series",
]

MDT_CATEGORICAL_FEATURES = [
    "mdt_priority",
    "latency_requirement",
    "endurance_requirement",
    "reliability_requirement",
    "drive_media_type",
    "drive_protocol",
    "drive_form_factor",
    "drive_pcie_gen",
    "drive_latency_class",
]

MDT_NUMERIC_FEATURES = [
    "required_total_iops",
    "required_read_iops",
    "required_write_iops",
    "required_metadata_capacity_tib",
    "ha_required",
    "max_budget_usd",
    "max_power_w",
    "performance_priority",
    "cost_priority",
    "power_priority",
    "reliability_priority",
    "drive_pcie_lanes",
    "drive_capacity_tib",
    "drive_random_read_iops_4k",
    "drive_random_write_iops_4k",
    "drive_endurance_dwpd",
    "drive_mtbf_hours",
    "drive_warranty_years",
    "drive_price_usd",
    "drive_power_w",
    "count_by_capacity",
    "count_by_read_iops",
    "count_by_write_iops",
    "raw_minimum_drive_count",
    "raw_provided_capacity_tib",
    "raw_provided_read_iops",
    "raw_provided_write_iops",
    "raw_drive_cost_usd",
    "raw_drive_power_w",
    "capacity_requirement_to_drive_ratio",
    "read_requirement_to_drive_ratio",
    "write_requirement_to_drive_ratio",
    "capacity_headroom_ratio",
    "read_iops_headroom_ratio",
    "write_iops_headroom_ratio",
    "budget_fraction",
    "power_fraction",
    "endurance_margin_ratio",
    "reliability_margin_ratio",
    "latency_margin",
]

MDT_LABEL_COLUMNS = [
    "group_size",
    "teacher_rank",
    "teacher_score",
    "relevance_grade",
    "is_teacher_top1",
    "is_teacher_top5",
    "is_teacher_top10",
]

MDT_COLUMNS = (
    MDT_IDENTIFIER_COLUMNS
    + MDT_CATEGORICAL_FEATURES
    + MDT_NUMERIC_FEATURES
    + MDT_LABEL_COLUMNS
)


OST_IDENTIFIER_COLUMNS = [
    "split",
    "case_id",
    "drive_id",
    "drive_name",
    "manufacturer",
    "series",
]

OST_CATEGORICAL_FEATURES = [
    "ost_priority",
    "throughput_requirement",
    "access_pattern",
    "file_size_class",
    "reliability_requirement",
    "drive_media_type",
    "drive_protocol",
    "drive_form_factor",
    "drive_pcie_gen",
    "drive_recording_technology",
]

OST_NUMERIC_FEATURES = [
    "required_usable_capacity_tib",
    "required_read_bandwidth_gbps",
    "required_write_bandwidth_gbps",
    "required_total_bandwidth_gbps",
    "ha_required",
    "max_budget_usd",
    "max_power_w",
    "performance_priority",
    "cost_priority",
    "power_priority",
    "reliability_priority",
    "drive_pcie_lanes",
    "drive_capacity_tib",
    "drive_seq_read_mb_s",
    "drive_seq_write_mb_s",
    "drive_read_bandwidth_gbps",
    "drive_write_bandwidth_gbps",
    "drive_mtbf_hours",
    "drive_warranty_years",
    "drive_rpm",
    "drive_workload_rating_tb_per_year",
    "drive_price_usd",
    "drive_power_w",
    "count_by_capacity",
    "count_by_read_bandwidth",
    "count_by_write_bandwidth",
    "raw_minimum_drive_count",
    "raw_provided_capacity_tib",
    "raw_provided_read_bandwidth_gbps",
    "raw_provided_write_bandwidth_gbps",
    "raw_provided_total_bandwidth_gbps",
    "raw_drive_cost_usd",
    "raw_drive_power_w",
    "capacity_requirement_to_drive_ratio",
    "read_requirement_to_drive_ratio",
    "write_requirement_to_drive_ratio",
    "capacity_headroom_ratio",
    "read_bandwidth_headroom_ratio",
    "write_bandwidth_headroom_ratio",
    "budget_fraction",
    "power_fraction",
    "reliability_margin_ratio",
]

OST_LABEL_COLUMNS = [
    "group_size",
    "teacher_rank",
    "teacher_score",
    "relevance_grade",
    "is_teacher_top1",
    "is_teacher_top5",
    "is_teacher_top10",
]

OST_COLUMNS = (
    OST_IDENTIFIER_COLUMNS
    + OST_CATEGORICAL_FEATURES
    + OST_NUMERIC_FEATURES
    + OST_LABEL_COLUMNS
)


class TrainingDatasetError(ValueError):
    """Erreur de construction ou de validation des datasets ML."""


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{text}".encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)


def create_case_splits(
    architectures: list[dict[str, Any]],
    seed: int,
) -> dict[str, str]:
    """Séparation déterministe stratifiée par type de workload."""

    strata: dict[str, list[str]] = defaultdict(list)

    for case in architectures:
        case_id = str(case["case_id"])
        workload_type = str(
            case["workload_analysis"]["workload_type"]
        )
        strata[workload_type].append(case_id)

    split_by_case: dict[str, str] = {}

    for stratum, case_ids in sorted(strata.items()):
        case_ids = sorted(case_ids)
        rng = random.Random(stable_seed(stratum, seed))
        rng.shuffle(case_ids)

        count = len(case_ids)
        train_count = round(
            count * SPLIT_RATIOS["train"]
        )
        validation_count = round(
            count * SPLIT_RATIOS["validation"]
        )

        if train_count + validation_count > count:
            validation_count = count - train_count

        train_ids = case_ids[:train_count]
        validation_ids = case_ids[
            train_count:
            train_count + validation_count
        ]
        test_ids = case_ids[
            train_count + validation_count:
        ]

        for case_id in train_ids:
            split_by_case[case_id] = "train"

        for case_id in validation_ids:
            split_by_case[case_id] = "validation"

        for case_id in test_ids:
            split_by_case[case_id] = "test"

    if len(split_by_case) != len(architectures):
        raise TrainingDatasetError(
            "Tous les case_id n'ont pas reçu un split."
        )

    return split_by_case


def safe_ratio(
    numerator: float,
    denominator: float,
    *,
    zero_denominator_value: float = 0.0,
) -> float:
    if denominator == 0:
        return zero_denominator_value

    return numerator / denominator


def relevance_grade(rank: int, group_size: int) -> int:
    """Label ordinal centré sur la qualité du futur Top-K."""

    if rank == 1:
        return 4

    if rank <= 3:
        return 3

    if rank <= 10:
        return 2

    if rank <= max(10, math.ceil(0.40 * group_size)):
        return 1

    return 0


def categorical(value: Any) -> str:
    if value is None or value == "":
        return "NONE"

    return str(value)


def numeric(value: Any) -> Any:
    if value is None:
        return ""

    if isinstance(value, bool):
        return int(value)

    return value


def exact_top_k_validation(
    generated: list[dict[str, Any]],
    reference_case: dict[str, Any],
    case_id: str,
    role: str,
    top_k: int = 10,
) -> None:
    expected = generated[:top_k]
    reference = reference_case["candidates"]

    if len(reference) != min(top_k, len(generated)):
        raise TrainingDatasetError(
            f"{case_id}/{role}: taille du Top-K de référence "
            "incohérente."
        )

    for rank, candidate in enumerate(expected, start=1):
        candidate_with_rank = dict(candidate)
        candidate_with_rank["rank"] = rank

        if candidate_with_rank != reference[rank - 1]:
            raise TrainingDatasetError(
                f"{case_id}/{role}: le candidat de rang {rank} "
                "diffère du Top-K déterministe validé."
            )


def write_rows(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with gzip.open(
        path,
        mode="wt",
        encoding="utf-8",
        newline="",
        compresslevel=6,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)
            count += 1

    return count


def build_mdt_rows(
    architectures: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    split_by_case: dict[str, str],
    statistics: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    row_count_by_split: Counter[str] = Counter()
    grade_distribution: Counter[int] = Counter()
    group_sizes: list[int] = []

    for case_index, architecture in enumerate(
        architectures,
        start=1,
    ):
        case_id = str(architecture["case_id"])
        requirement = architecture["MDT_requirement"]
        constraints = architecture["constraints"]
        preferences = architecture["preferences"]

        feasible: list[dict[str, Any]] = []

        for drive in catalog:
            # Le ranker MDT n'évalue que les disques déjà marqués
            # éligibles. Les HDD gardent légitimement une endurance
            # DWPD à null et ne doivent pas atteindre evaluate_drive().
            if not bool(drive["mdt_eligible"]):
                continue

            candidate, reasons = mdt_generator.evaluate_drive(
                drive,
                requirement,
                constraints,
                preferences,
            )

            if not reasons:
                feasible.append(candidate)

        feasible.sort(key=mdt_generator.candidate_sort_key)

        if not feasible:
            raise TrainingDatasetError(
                f"{case_id}/MDT: aucun candidat faisable."
            )

        exact_top_k_validation(
            feasible,
            references[case_id],
            case_id,
            "MDT",
        )

        group_size = len(feasible)
        group_sizes.append(group_size)

        minimum_dwpd = (
            mdt_generator.ENDURANCE_MIN_DWPD[
                requirement["endurance_requirement"]
            ]
        )
        minimum_mtbf = (
            mdt_generator.RELIABILITY_MIN_MTBF_HOURS[
                requirement["reliability_requirement"]
            ]
        )

        required_latency_order = (
            mdt_generator.LATENCY_ORDER[
                requirement["latency_requirement"]
            ]
        )

        drive_by_id = {
            drive["drive_id"]: drive
            for drive in catalog
        }

        for rank, candidate in enumerate(feasible, start=1):
            drive = drive_by_id[candidate["drive_id"]]
            split = split_by_case[case_id]
            grade = relevance_grade(rank, group_size)

            required_capacity = float(
                requirement["required_metadata_capacity_tib"]
            )
            required_read = float(
                requirement["required_read_iops"]
            )
            required_write = float(
                requirement["required_write_iops"]
            )

            row = {
                "split": split,
                "case_id": case_id,
                "drive_id": drive["drive_id"],
                "drive_name": drive["name"],
                "manufacturer": drive["manufacturer"],
                "series": drive["series"],

                "mdt_priority": categorical(
                    requirement["priority"]
                ),
                "latency_requirement": categorical(
                    requirement["latency_requirement"]
                ),
                "endurance_requirement": categorical(
                    requirement["endurance_requirement"]
                ),
                "reliability_requirement": categorical(
                    requirement["reliability_requirement"]
                ),
                "drive_media_type": categorical(
                    drive["media_type"]
                ),
                "drive_protocol": categorical(
                    drive["protocol"]
                ),
                "drive_form_factor": categorical(
                    drive["drive_form_factor_standard"]
                ),
                "drive_pcie_gen": categorical(
                    drive["pcie_gen_required"]
                ),
                "drive_latency_class": categorical(
                    drive["latency_class"]
                ),

                "required_total_iops": requirement[
                    "required_total_iops"
                ],
                "required_read_iops": required_read,
                "required_write_iops": required_write,
                "required_metadata_capacity_tib":
                    required_capacity,
                "ha_required": int(
                    bool(constraints["ha_required"])
                ),
                "max_budget_usd": constraints[
                    "max_budget_usd"
                ],
                "max_power_w": constraints["max_power_w"],
                "performance_priority": preferences[
                    "performance_priority"
                ],
                "cost_priority": preferences["cost_priority"],
                "power_priority": preferences[
                    "power_priority"
                ],
                "reliability_priority": preferences[
                    "reliability_priority"
                ],
                "drive_pcie_lanes": numeric(
                    drive["pcie_lanes_required"]
                ),
                "drive_capacity_tib": drive["capacity_tib"],
                "drive_random_read_iops_4k": drive[
                    "random_read_iops_4k"
                ],
                "drive_random_write_iops_4k": drive[
                    "random_write_iops_4k"
                ],
                "drive_endurance_dwpd": drive[
                    "endurance_dwpd_numeric"
                ],
                "drive_mtbf_hours": drive["mtbf_hours"],
                "drive_warranty_years": drive[
                    "warranty_years"
                ],
                "drive_price_usd": drive[
                    "price_en_dollars"
                ],
                "drive_power_w": drive[
                    "power_consumption_en_w"
                ],
                "count_by_capacity": candidate[
                    "count_by_capacity"
                ],
                "count_by_read_iops": candidate[
                    "count_by_read_iops"
                ],
                "count_by_write_iops": candidate[
                    "count_by_write_iops"
                ],
                "raw_minimum_drive_count": candidate[
                    "raw_minimum_drive_count"
                ],
                "raw_provided_capacity_tib": candidate[
                    "raw_provided_capacity_tib"
                ],
                "raw_provided_read_iops": candidate[
                    "raw_provided_read_iops"
                ],
                "raw_provided_write_iops": candidate[
                    "raw_provided_write_iops"
                ],
                "raw_drive_cost_usd": candidate[
                    "raw_drive_cost_usd"
                ],
                "raw_drive_power_w": candidate[
                    "raw_drive_power_w"
                ],
                "capacity_requirement_to_drive_ratio":
                    round(
                        safe_ratio(
                            required_capacity,
                            float(drive["capacity_tib"]),
                        ),
                        8,
                ),
                "read_requirement_to_drive_ratio":
                    round(
                        safe_ratio(
                            required_read,
                            float(
                                drive[
                                    "random_read_iops_4k"
                                ]
                            ),
                        ),
                        8,
                ),
                "write_requirement_to_drive_ratio":
                    round(
                        safe_ratio(
                            required_write,
                            float(
                                drive[
                                    "random_write_iops_4k"
                                ]
                            ),
                        ),
                        8,
                ),
                "capacity_headroom_ratio": round(
                    safe_ratio(
                        float(
                            candidate[
                                "raw_provided_capacity_tib"
                            ]
                        ),
                        required_capacity,
                        zero_denominator_value=1.0,
                    ),
                    8,
                ),
                "read_iops_headroom_ratio": round(
                    safe_ratio(
                        float(
                            candidate[
                                "raw_provided_read_iops"
                            ]
                        ),
                        required_read,
                        zero_denominator_value=1.0,
                    ),
                    8,
                ),
                "write_iops_headroom_ratio": round(
                    safe_ratio(
                        float(
                            candidate[
                                "raw_provided_write_iops"
                            ]
                        ),
                        required_write,
                        zero_denominator_value=1.0,
                    ),
                    8,
                ),
                "budget_fraction": round(
                    safe_ratio(
                        float(
                            candidate[
                                "raw_drive_cost_usd"
                            ]
                        ),
                        float(
                            constraints["max_budget_usd"]
                        ),
                    ),
                    8,
                ),
                "power_fraction": round(
                    safe_ratio(
                        float(
                            candidate[
                                "raw_drive_power_w"
                            ]
                        ),
                        float(constraints["max_power_w"]),
                    ),
                    8,
                ),
                "endurance_margin_ratio": round(
                    safe_ratio(
                        float(
                            drive[
                                "endurance_dwpd_numeric"
                            ]
                        ),
                        float(minimum_dwpd),
                        zero_denominator_value=1.0,
                    ),
                    8,
                ),
                "reliability_margin_ratio": round(
                    safe_ratio(
                        float(drive["mtbf_hours"]),
                        float(
                            minimum_mtbf
                            if minimum_mtbf > 0
                            else 2_500_000
                        ),
                    ),
                    8,
                ),
                "latency_margin": (
                    mdt_generator.LATENCY_ORDER[
                        drive["latency_class"]
                    ]
                    - required_latency_order
                ),

                "group_size": group_size,
                "teacher_rank": rank,
                "teacher_score": candidate["score"],
                "relevance_grade": grade,
                "is_teacher_top1": int(rank == 1),
                "is_teacher_top5": int(rank <= 5),
                "is_teacher_top10": int(rank <= 10),
            }

            row_count_by_split[split] += 1
            grade_distribution[grade] += 1
            yield row

        if case_index % 100 == 0:
            print(
                f"Dataset MDT : {case_index}/"
                f"{len(architectures)}",
                end="\r",
            )

    print(
        f"Dataset MDT : {len(architectures)}/"
        f"{len(architectures)}"
    )

    statistics["mdt"] = {
        "rows_by_split": dict(row_count_by_split),
        "grade_distribution": {
            str(key): value
            for key, value in sorted(
                grade_distribution.items()
            )
        },
        "minimum_group_size": min(group_sizes),
        "maximum_group_size": max(group_sizes),
        "mean_group_size": round(
            sum(group_sizes) / len(group_sizes),
            6,
        ),
    }


def build_ost_rows(
    architectures: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    split_by_case: dict[str, str],
    statistics: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    row_count_by_split: Counter[str] = Counter()
    grade_distribution: Counter[int] = Counter()
    group_sizes: list[int] = []

    drive_by_id = {
        drive["drive_id"]: drive
        for drive in catalog
    }

    for case_index, architecture in enumerate(
        architectures,
        start=1,
    ):
        case_id = str(architecture["case_id"])
        requirement = architecture["OST_requirement"]
        constraints = architecture["constraints"]
        preferences = architecture["preferences"]

        feasible: list[dict[str, Any]] = []

        for drive in catalog:
            candidate, reasons = ost_generator.evaluate_drive(
                drive,
                requirement,
                constraints,
                preferences,
            )

            if not reasons:
                feasible.append(candidate)

        feasible.sort(key=ost_generator.candidate_sort_key)

        if not feasible:
            raise TrainingDatasetError(
                f"{case_id}/OST: aucun candidat faisable."
            )

        exact_top_k_validation(
            feasible,
            references[case_id],
            case_id,
            "OST",
        )

        group_size = len(feasible)
        group_sizes.append(group_size)

        minimum_mtbf = (
            ost_generator.RELIABILITY_MIN_MTBF_HOURS[
                requirement["reliability_requirement"]
            ]
        )

        required_capacity = float(
            requirement["required_usable_capacity_tib"]
        )
        required_read = float(
            requirement["required_read_bandwidth_gbps"]
        )
        required_write = float(
            requirement["required_write_bandwidth_gbps"]
        )

        for rank, candidate in enumerate(feasible, start=1):
            drive = drive_by_id[candidate["drive_id"]]
            split = split_by_case[case_id]
            grade = relevance_grade(rank, group_size)

            row = {
                "split": split,
                "case_id": case_id,
                "drive_id": drive["drive_id"],
                "drive_name": drive["name"],
                "manufacturer": drive["manufacturer"],
                "series": drive["series"],

                "ost_priority": categorical(
                    requirement["priority"]
                ),
                "throughput_requirement": categorical(
                    requirement["throughput_requirement"]
                ),
                "access_pattern": categorical(
                    requirement["access_pattern"]
                ),
                "file_size_class": categorical(
                    requirement["file_size_class"]
                ),
                "reliability_requirement": categorical(
                    requirement["reliability_requirement"]
                ),
                "drive_media_type": categorical(
                    drive["media_type"]
                ),
                "drive_protocol": categorical(
                    drive["protocol"]
                ),
                "drive_form_factor": categorical(
                    drive["drive_form_factor_standard"]
                ),
                "drive_pcie_gen": categorical(
                    drive["pcie_gen_required"]
                ),
                "drive_recording_technology": categorical(
                    drive["recording_technology"]
                ),

                "required_usable_capacity_tib":
                    required_capacity,
                "required_read_bandwidth_gbps":
                    required_read,
                "required_write_bandwidth_gbps":
                    required_write,
                "required_total_bandwidth_gbps":
                    requirement[
                        "required_total_bandwidth_gbps"
                ],
                "ha_required": int(
                    bool(constraints["ha_required"])
                ),
                "max_budget_usd": constraints[
                    "max_budget_usd"
                ],
                "max_power_w": constraints["max_power_w"],
                "performance_priority": preferences[
                    "performance_priority"
                ],
                "cost_priority": preferences["cost_priority"],
                "power_priority": preferences[
                    "power_priority"
                ],
                "reliability_priority": preferences[
                    "reliability_priority"
                ],
                "drive_pcie_lanes": numeric(
                    drive["pcie_lanes_required"]
                ),
                "drive_capacity_tib": drive["capacity_tib"],
                "drive_seq_read_mb_s": drive[
                    "seq_read_mb_s"
                ],
                "drive_seq_write_mb_s": drive[
                    "seq_write_mb_s"
                ],
                "drive_read_bandwidth_gbps": candidate[
                    "drive_read_bandwidth_gbps"
                ],
                "drive_write_bandwidth_gbps": candidate[
                    "drive_write_bandwidth_gbps"
                ],
                "drive_mtbf_hours": drive["mtbf_hours"],
                "drive_warranty_years": drive[
                    "warranty_years"
                ],
                "drive_rpm": numeric(drive["rpm"]),
                "drive_workload_rating_tb_per_year":
                    numeric(
                        drive[
                            "workload_rating_tb_per_year"
                        ]
                ),
                "drive_price_usd": drive[
                    "price_en_dollars"
                ],
                "drive_power_w": drive[
                    "power_consumption_en_w"
                ],
                "count_by_capacity": candidate[
                    "count_by_capacity"
                ],
                "count_by_read_bandwidth": candidate[
                    "count_by_read_bandwidth"
                ],
                "count_by_write_bandwidth": candidate[
                    "count_by_write_bandwidth"
                ],
                "raw_minimum_drive_count": candidate[
                    "raw_minimum_drive_count"
                ],
                "raw_provided_capacity_tib": candidate[
                    "raw_provided_capacity_tib"
                ],
                "raw_provided_read_bandwidth_gbps":
                    candidate[
                        "raw_provided_read_bandwidth_gbps"
                ],
                "raw_provided_write_bandwidth_gbps":
                    candidate[
                        "raw_provided_write_bandwidth_gbps"
                ],
                "raw_provided_total_bandwidth_gbps":
                    candidate[
                        "raw_provided_total_bandwidth_gbps"
                ],
                "raw_drive_cost_usd": candidate[
                    "raw_drive_cost_usd"
                ],
                "raw_drive_power_w": candidate[
                    "raw_drive_power_w"
                ],
                "capacity_requirement_to_drive_ratio":
                    round(
                        safe_ratio(
                            required_capacity,
                            float(drive["capacity_tib"]),
                        ),
                        8,
                ),
                "read_requirement_to_drive_ratio":
                    round(
                        safe_ratio(
                            required_read,
                            float(
                                candidate[
                                    "drive_read_bandwidth_gbps"
                                ]
                            ),
                        ),
                        8,
                ),
                "write_requirement_to_drive_ratio":
                    round(
                        safe_ratio(
                            required_write,
                            float(
                                candidate[
                                    "drive_write_bandwidth_gbps"
                                ]
                            ),
                        ),
                        8,
                ),
                "capacity_headroom_ratio": round(
                    safe_ratio(
                        float(
                            candidate[
                                "raw_provided_capacity_tib"
                            ]
                        ),
                        required_capacity,
                    ),
                    8,
                ),
                "read_bandwidth_headroom_ratio": round(
                    safe_ratio(
                        float(
                            candidate[
                                "raw_provided_read_bandwidth_gbps"
                            ]
                        ),
                        required_read,
                        zero_denominator_value=1.0,
                    ),
                    8,
                ),
                "write_bandwidth_headroom_ratio": round(
                    safe_ratio(
                        float(
                            candidate[
                                "raw_provided_write_bandwidth_gbps"
                            ]
                        ),
                        required_write,
                        zero_denominator_value=1.0,
                    ),
                    8,
                ),
                "budget_fraction": round(
                    safe_ratio(
                        float(
                            candidate[
                                "raw_drive_cost_usd"
                            ]
                        ),
                        float(
                            constraints["max_budget_usd"]
                        ),
                    ),
                    8,
                ),
                "power_fraction": round(
                    safe_ratio(
                        float(
                            candidate[
                                "raw_drive_power_w"
                            ]
                        ),
                        float(constraints["max_power_w"]),
                    ),
                    8,
                ),
                "reliability_margin_ratio": round(
                    safe_ratio(
                        float(drive["mtbf_hours"]),
                        float(
                            minimum_mtbf
                            if minimum_mtbf > 0
                            else 2_500_000
                        ),
                    ),
                    8,
                ),

                "group_size": group_size,
                "teacher_rank": rank,
                "teacher_score": candidate["score"],
                "relevance_grade": grade,
                "is_teacher_top1": int(rank == 1),
                "is_teacher_top5": int(rank <= 5),
                "is_teacher_top10": int(rank <= 10),
            }

            row_count_by_split[split] += 1
            grade_distribution[grade] += 1
            yield row

        if case_index % 100 == 0:
            print(
                f"Dataset OST : {case_index}/"
                f"{len(architectures)}",
                end="\r",
            )

    print(
        f"Dataset OST : {len(architectures)}/"
        f"{len(architectures)}"
    )

    statistics["ost"] = {
        "rows_by_split": dict(row_count_by_split),
        "grade_distribution": {
            str(key): value
            for key, value in sorted(
                grade_distribution.items()
            )
        },
        "minimum_group_size": min(group_sizes),
        "maximum_group_size": max(group_sizes),
        "mean_group_size": round(
            sum(group_sizes) / len(group_sizes),
            6,
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def create_zip(
    zip_path: Path,
    files: list[Path],
) -> None:
    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for path in files:
            archive.write(path, arcname=path.name)


def validate_references(
    architectures: list[dict[str, Any]],
    references: list[dict[str, Any]],
    role: str,
) -> dict[str, dict[str, Any]]:
    if len(architectures) != len(references):
        raise TrainingDatasetError(
            f"{role}: nombre de cas différent entre "
            "architectures et références."
        )

    reference_by_id = {
        str(case["case_id"]): case
        for case in references
    }

    architecture_ids = {
        str(case["case_id"])
        for case in architectures
    }

    if set(reference_by_id) != architecture_ids:
        raise TrainingDatasetError(
            f"{role}: ensemble de case_id incohérent."
        )

    return reference_by_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construit les datasets Learning-to-Rank MDT et OST."
        )
    )
    parser.add_argument(
        "--architectures",
        type=Path,
        default=DEFAULT_ARCHITECTURES,
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
    )
    parser.add_argument(
        "--mdt-reference",
        type=Path,
        default=DEFAULT_MDT_REFERENCE,
    )
    parser.add_argument(
        "--ost-reference",
        type=Path,
        default=DEFAULT_OST_REFERENCE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    architectures = load_json(args.architectures)
    catalog = load_json(args.catalog)
    mdt_reference = load_json(args.mdt_reference)
    ost_reference = load_json(args.ost_reference)

    if not isinstance(architectures, list) or not architectures:
        raise TrainingDatasetError(
            "Le dataset architectural doit être une liste non vide."
        )

    if not isinstance(catalog, list) or not catalog:
        raise TrainingDatasetError(
            "Le catalogue doit être une liste non vide."
        )

    mdt_generator.validate_catalog(catalog)
    ost_generator.validate_catalog(catalog)

    mdt_reference_by_id = validate_references(
        architectures,
        mdt_reference,
        "MDT",
    )
    ost_reference_by_id = validate_references(
        architectures,
        ost_reference,
        "OST",
    )

    split_by_case = create_case_splits(
        architectures,
        args.seed,
    )

    case_ids_by_split = {
        split: sorted(
            case_id
            for case_id, assigned in split_by_case.items()
            if assigned == split
        )
        for split in SPLIT_RATIOS
    }

    workload_distribution: dict[str, Counter[str]] = {
        split: Counter()
        for split in SPLIT_RATIOS
    }

    architecture_by_id = {
        str(case["case_id"]): case
        for case in architectures
    }

    for case_id, split in split_by_case.items():
        workload_distribution[split].update(
            [
                architecture_by_id[case_id][
                    "workload_analysis"
                ]["workload_type"]
            ]
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mdt_path = output_dir / "mdt_training_dataset.csv.gz"
    ost_path = output_dir / "ost_training_dataset.csv.gz"
    split_path = output_dir / "case_splits.json"
    manifest_path = (
        output_dir / "training_dataset_manifest.json"
    )
    zip_path = (
        output_dir
        / "lustre_ranker_training_data_kaggle.zip"
    )
    readme_path = output_dir / "KAGGLE_README.txt"

    statistics: dict[str, Any] = {}

    mdt_row_count = write_rows(
        mdt_path,
        MDT_COLUMNS,
        build_mdt_rows(
            architectures,
            catalog,
            mdt_reference_by_id,
            split_by_case,
            statistics,
        ),
    )

    ost_row_count = write_rows(
        ost_path,
        OST_COLUMNS,
        build_ost_rows(
            architectures,
            catalog,
            ost_reference_by_id,
            split_by_case,
            statistics,
        ),
    )

    split_document = {
        "schema_version": "1.0",
        "seed": args.seed,
        "strategy": (
            "case_id_group_split_stratified_by_workload_type"
        ),
        "ratios": SPLIT_RATIOS,
        "case_counts": {
            split: len(case_ids)
            for split, case_ids in case_ids_by_split.items()
        },
        "workload_distribution": {
            split: dict(sorted(counter.items()))
            for split, counter in workload_distribution.items()
        },
        "case_ids": case_ids_by_split,
    }
    save_json(split_path, split_document)

    manifest = {
        "schema_version": "1.0",
        "builder_version": BUILDER_VERSION,
        "description": (
            "Datasets de ranking pré-RAID. Une ligne représente "
            "une requête et un modèle de disque faisable."
        ),
        "training_contract": {
            "group_column": "case_id",
            "split_column": "split",
            "ranking_label": "relevance_grade",
            "continuous_teacher_label": "teacher_score",
            "teacher_rank_column": "teacher_rank",
            "all_rows_are_pre_raid_feasible": True,
            "hard_constraints_applied_before_ml": True,
            "split_must_never_be_redone_by_row": True,
            "top10_deterministic_references_verified": True,
        },
        "case_split": split_document,
        "mdt": {
            "file": mdt_path.name,
            "row_count": mdt_row_count,
            "identifier_columns": MDT_IDENTIFIER_COLUMNS,
            "categorical_features": MDT_CATEGORICAL_FEATURES,
            "numeric_features": MDT_NUMERIC_FEATURES,
            "model_feature_columns": (
                MDT_CATEGORICAL_FEATURES
                + MDT_NUMERIC_FEATURES
            ),
            "label_columns": MDT_LABEL_COLUMNS,
            "statistics": statistics["mdt"],
        },
        "ost": {
            "file": ost_path.name,
            "row_count": ost_row_count,
            "identifier_columns": OST_IDENTIFIER_COLUMNS,
            "categorical_features": OST_CATEGORICAL_FEATURES,
            "numeric_features": OST_NUMERIC_FEATURES,
            "model_feature_columns": (
                OST_CATEGORICAL_FEATURES
                + OST_NUMERIC_FEATURES
            ),
            "label_columns": OST_LABEL_COLUMNS,
            "statistics": statistics["ost"],
        },
    }

    save_json(manifest_path, manifest)

    manifest["checksums_sha256"] = {
        mdt_path.name: sha256_file(mdt_path),
        ost_path.name: sha256_file(ost_path),
        split_path.name: sha256_file(split_path),
    }
    save_json(manifest_path, manifest)

    readme_path.write_text(
        "LUSTRE MDT/OST LEARNING-TO-RANK DATA\n"
        "====================================\n\n"
        "Use case_id as the ranking group identifier.\n"
        "Use relevance_grade as the primary ranking label.\n"
        "Never use teacher_score or teacher_rank as input features.\n"
        "Keep the provided split column; never split rows randomly.\n"
        "Use model_feature_columns from the JSON manifest.\n"
        "The deterministic hard filter must remain before ML.\n",
        encoding="utf-8",
    )

    create_zip(
        zip_path,
        [
            mdt_path,
            ost_path,
            split_path,
            manifest_path,
            readme_path,
        ],
    )

    print("\nConstruction des datasets ML")
    print("----------------------------")
    print(
        f"Cas train                  : "
        f"{len(case_ids_by_split['train'])}"
    )
    print(
        f"Cas validation             : "
        f"{len(case_ids_by_split['validation'])}"
    )
    print(
        f"Cas test                   : "
        f"{len(case_ids_by_split['test'])}"
    )
    print(f"Lignes MDT                 : {mdt_row_count}")
    print(f"Lignes OST                 : {ost_row_count}")
    print(
        "Top-10 MDT vérifiés        : "
        f"{len(architectures)}"
    )
    print(
        "Top-10 OST vérifiés        : "
        f"{len(architectures)}"
    )
    print(f"Dataset MDT                : {mdt_path}")
    print(f"Dataset OST                : {ost_path}")
    print(f"Package Kaggle             : {zip_path}")
    print("\nSTATUT : VALIDÉ")


if __name__ == "__main__":
    main()
