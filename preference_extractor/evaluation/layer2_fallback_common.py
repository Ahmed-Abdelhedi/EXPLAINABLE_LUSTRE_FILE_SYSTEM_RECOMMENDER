from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DIMENSIONS: Tuple[str, ...] = (
    "cost",
    "power",
    "performance",
    "reliability",
)

LEVELS: Tuple[str, ...] = (
    "VERY_LOW",
    "LOW",
    "MEDIUM",
    "HIGH",
    "VERY_HIGH",
)

EXPECTED_ARTIFACT_SHA256 = (
    "cdbb6f6544d4b4d96578e0901a00f46c68916ad66547800ffc4d232095298890"
)

EXPECTED_DATASET_ZIP_SHA256 = (
    "4a37c5bc61ddfad2bcf2785c5c63cb3c93b40e745edac3f4530d3a5db2b26ad5"
)

EXPECTED_DATASET_VERSION = "2026-08-24-r3-kaggle-fresh"

DATASET_FILENAMES = {
    "train": "preference_layer2_r3_train.jsonl",
    "validation": "preference_layer2_r3_validation.jsonl",
    "test": "preference_layer2_r3_test.jsonl",
    "final_holdout": "preference_layer2_r3_final_holdout.jsonl",
}


PROMPT_VERSION = "layer2_qwen_fallback_v3_compact_20260825"

PROMPT_POLICY = r"""You are the guarded Layer-2 fallback for a Lustre preference extractor.

INPUT
- CURRENT_USER_MESSAGE
- REQUESTED_DIMENSIONS: subset of cost, power, performance, reliability

OUTPUT FOR EACH REQUESTED DIMENSION
status = RESOLVED | NO_SIGNAL | RELATIVE_ONLY | UNRESOLVED
level  = VERY_LOW | LOW | MEDIUM | HIGH | VERY_HIGH | null
evidence = exact substring from CURRENT_USER_MESSAGE or null

NEVER output an unrequested dimension.

CORE RULE 1 — NO_SIGNAL is NOT VERY_LOW
NO_SIGNAL means no current user preference for that dimension.
A technical measurement, limit, capability, log/API field, or unadopted third-party opinion is NO_SIGNAL.
Examples: "Budget cap is 100000 USD." -> cost NO_SIGNAL; "Maximum rack power is 15 kW." -> power NO_SIGNAL; "Need 83 GB/s read and 119 GB/s write." -> performance NO_SIGNAL; "The engine can optimize cost." -> cost NO_SIGNAL.
Explicit near-indifference is still a preference and is VERY_LOW: can be largely ignored; close to irrelevant; matters almost not at all; barely matters; peut être largement ignoré; proche d'être sans intérêt; compte presque pas.

CORE RULE 2 — PURE COMPARISON => RELATIVE_ONLY
If a requested dimension appears only in an ordering/comparison, do NOT invent an absolute level.
Comparison cues: A outranks B; A is more important than B; prioritize A over B; A before B; A first, then B; A > B > C; A prime sur B; A est prioritaire sur B; retenez A avant B; A d'abord, puis B.
For "Performance is more important than cost": performance = RELATIVE_ONLY; cost = RELATIVE_ONLY.
For "Can you prioritize performance over cost for our design": performance = RELATIVE_ONLY; cost = RELATIVE_ONLY.
If one dimension also has an independent absolute cue, use that absolute cue for that dimension and keep comparison-only dimensions RELATIVE_ONLY.

CORE RULE 3 — EXACT ABSOLUTE SCALE
Use the wording itself. Do not make the level stronger or weaker.
VERY_LOW: can be largely ignored; close to irrelevant; matters almost not at all; barely matters; negligible; peut être largement ignoré; proche d'être sans intérêt; compte presque pas.
LOW: weakly prioritized; secondary concern; low priority; nice to have rather than central; desirable but not a priority; sits near the bottom of our priorities; faible priorité; secondaire; souhaitable sans être prioritaire; se situe vers le bas de nos priorités.
MEDIUM: medium weight; moderate importance; meaningful but balanced role; matters, though it is not dominant; importance moyenne; rôle réel mais équilibré; compte sans être dominant.
HIGH: strong importance; high importance; very important; strongly prioritized; should weigh heavily in the decision; forte importance; importance élevée; très important; doit peser fortement dans la décision.
VERY_HIGH: critical; essential; non-negotiable; first-order priority; absolute/top priority; nothing matters more than; decisive criterion; cannot compromise; critique; essentiel; non négociable; priorité de premier ordre; priorité absolue; rien n'est plus important que; critère décisif.

EXACT CALIBRATION REMINDERS
weakly prioritized -> LOW, not VERY_LOW
sits near the bottom of our priorities -> LOW, not VERY_LOW
nice to have rather than central -> LOW, not MEDIUM
matters, though it is not dominant -> MEDIUM, not LOW
strong importance -> HIGH, not VERY_HIGH
high importance -> HIGH, not VERY_HIGH
first-order priority -> VERY_HIGH

CORE RULE 4 — TECHNICAL ADJECTIVE != PRIORITY LEVEL
"low wattage" describes wattage; "high availability" describes availability; "high throughput" describes throughput. Use the importance phrase, not the technical adjective.
Examples: "keeping wattage low carries strong importance" -> power HIGH; "high availability sits near the bottom of our priorities" -> reliability LOW; "high sustained performance matters almost not at all" -> performance VERY_LOW.

CORE RULE 5 — CURRENT CHOICE WINS
Use only the current/final/latest user choice. Superseded or historical preferences do not override the final choice. Unadopted vendor/third-party opinions are not the user's preference.

DECISION ORDER FOR EACH REQUESTED DIMENSION
1. Only measurement/capability/unadopted external statement? -> NO_SIGNAL
2. Only comparison/order? -> RELATIVE_ONLY
3. Independent absolute preference phrase? -> RESOLVED with exact scale above
4. Both absolute cue and comparison? -> absolute cue wins for that dimension
5. Truly ambiguous after these rules? -> UNRESOLVED

EVIDENCE
- RESOLVED: exact substring containing the absolute cue.
- RELATIVE_ONLY: exact substring containing the comparison.
- NO_SIGNAL: normally null.
- Never paraphrase evidence.

OUTPUT JSON ONLY
{
  "dimensions": {
    "<requested dimension>": {
      "status": "RESOLVED|NO_SIGNAL|RELATIVE_ONLY|UNRESOLVED",
      "level": "VERY_LOW|LOW|MEDIUM|HIGH|VERY_HIGH|null",
      "evidence": "exact substring|null"
    }
  }
}

STRICT CHECK BEFORE OUTPUT
- every requested dimension exactly once
- no unrequested dimensions
- RESOLVED => non-null level
- other statuses => null level
- evidence is copied exactly""".strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def prompt_policy_sha256() -> str:
    payload = (
        PROMPT_VERSION
        + "\n"
        + PROMPT_POLICY
    ).encode("utf-8")

    return sha256_bytes(payload)


def percentile(
    values: Sequence[float],
    probability: float,
) -> Optional[float]:
    if not values:
        return None

    ordered = sorted(
        float(value)
        for value in values
    )

    if len(ordered) == 1:
        return ordered[0]

    position = (
        probability
        * (
            len(ordered)
            - 1
        )
    )

    lower = int(
        math.floor(position)
    )

    upper = int(
        math.ceil(position)
    )

    if lower == upper:
        return ordered[lower]

    weight = (
        position
        - lower
    )

    return (
        ordered[lower]
        * (
            1.0
            - weight
        )
        + ordered[upper]
        * weight
    )


def load_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:
    rows: List[
        Dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path}:{line_number}"
                ) from exc

    return rows


def save_jsonl(
    path: Path,
    rows: Iterable[
        Mapping[str, Any]
    ],
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


def extract_zip_cached(
    zip_path: Path,
    namespace: str,
) -> Path:
    digest = sha256_file(
        zip_path
    )[:16]

    root = (
        Path(
            tempfile.gettempdir()
        )
        / "explainable_lustre_recommender"
        / namespace
        / digest
    )

    marker = (
        root
        / ".complete"
    )

    if marker.exists():
        return root

    if root.exists():
        import shutil

        shutil.rmtree(
            root
        )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:
        archive.extractall(
            root
        )

    marker.write_text(
        "ok\n",
        encoding="utf-8",
    )

    return root


def verify_artifact_zip(
    artifact_zip: Path,
    expected_sha256: Optional[str] = (
        EXPECTED_ARTIFACT_SHA256
    ),
) -> str:
    actual = sha256_file(
        artifact_zip
    )

    if (
        expected_sha256
        and actual
        != expected_sha256
    ):
        raise RuntimeError(
            "Wrong Layer-2 artifact ZIP.\n"
            f"Expected SHA256: {expected_sha256}\n"
            f"Actual SHA256:   {actual}"
        )

    return actual


def verify_dataset_zip(
    dataset_zip: Path,
    expected_sha256: Optional[str] = (
        EXPECTED_DATASET_ZIP_SHA256
    ),
) -> str:
    actual = sha256_file(
        dataset_zip
    )

    if (
        expected_sha256
        and actual
        != expected_sha256
    ):
        raise RuntimeError(
            "Wrong Layer-2 dataset ZIP.\n"
            f"Expected SHA256: {expected_sha256}\n"
            f"Actual SHA256:   {actual}"
        )

    return actual


def open_dataset_split(
    dataset_zip: Path,
    split: str,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    if split not in DATASET_FILENAMES:
        raise ValueError(
            f"Unsupported split: {split}"
        )

    verify_dataset_zip(
        dataset_zip
    )

    root = extract_zip_cached(
        dataset_zip,
        "layer2_dataset_r3",
    )

    metadata_path = (
        root
        / "dataset_metadata.json"
    )

    if not metadata_path.exists():
        raise RuntimeError(
            "dataset_metadata.json missing."
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8",
        )
    )

    if (
        metadata.get(
            "version"
        )
        != EXPECTED_DATASET_VERSION
    ):
        raise RuntimeError(
            "Wrong dataset metadata version.\n"
            f"Expected: {EXPECTED_DATASET_VERSION}\n"
            f"Actual:   {metadata.get('version')}"
        )

    path = (
        root
        / "data"
        / DATASET_FILENAMES[
            split
        ]
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing split file: {path}"
        )

    return (
        load_jsonl(
            path
        ),
        metadata,
    )


def gold_expected_for_dimension(
    row: Mapping[str, Any],
    dimension: str,
) -> Dict[str, Any]:
    label = row[
        "labels"
    ][
        dimension
    ]

    if (
        int(
            label[
                "presence"
            ]
        )
        == 0
    ):
        return {
            "status":
                "NO_SIGNAL",
            "level":
                None,
        }

    if (
        int(
            label[
                "intensity_mask"
            ]
        )
        == 1
    ):
        return {
            "status":
                "RESOLVED",
            "level":
                str(
                    label[
                        "intensity_label"
                    ]
                ),
        }

    return {
        "status":
            "RELATIVE_ONLY",
        "level":
            None,
    }


def ordinal_class_probabilities(
    cumulative_probabilities: Sequence[
        float
    ],
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
]:
    if len(
        cumulative_probabilities
    ) != 4:
        raise ValueError(
            "Expected 4 cumulative probabilities."
        )

    q = [
        min(
            1.0,
            max(
                0.0,
                float(value),
            ),
        )
        for value
        in cumulative_probabilities
    ]

    for index in range(
        1,
        4,
    ):
        q[index] = min(
            q[index],
            q[
                index
                - 1
            ],
        )

    raw = [
        1.0 - q[0],
        q[0] - q[1],
        q[1] - q[2],
        q[2] - q[3],
        q[3],
    ]

    raw = [
        max(
            0.0,
            value,
        )
        for value
        in raw
    ]

    total = sum(
        raw
    )

    if total <= 0:
        return (
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        )

    return tuple(
        value
        / total
        for value
        in raw
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class RouteDecision:
    route: str
    fallback_reason: Optional[str]
    direct_status: Optional[str]
    direct_level: Optional[str]
    presence_probability: float
    intensity_confidence: float
    transformer_level: str


def route_dimension(
    *,
    dimension: str,
    presence_probability: float,
    ordinal_probabilities: Sequence[
        float
    ],
    calibration: Mapping[str, Any],
) -> RouteDecision:
    policy = calibration[
        "dimensions"
    ][
        dimension
    ]

    presence_policy = policy[
        "presence"
    ]

    intensity_policy = policy[
        "intensity"
    ]

    class_probabilities = (
        ordinal_class_probabilities(
            ordinal_probabilities
        )
    )

    level_id = max(
        range(
            len(
                class_probabilities
            )
        ),
        key=lambda index:
            class_probabilities[
                index
            ],
    )

    transformer_level = (
        LEVELS[
            level_id
        ]
    )

    intensity_confidence = float(
        class_probabilities[
            level_id
        ]
    )

    probability = float(
        presence_probability
    )

    if (
        probability
        <= float(
            presence_policy[
                "negative_max"
            ]
        )
    ):
        return RouteDecision(
            route=
                "TRANSFORMER_DIRECT",
            fallback_reason=
                None,
            direct_status=
                "NO_SIGNAL",
            direct_level=
                None,
            presence_probability=
                probability,
            intensity_confidence=
                intensity_confidence,
            transformer_level=
                transformer_level,
        )

    if (
        probability
        < float(
            presence_policy[
                "positive_min"
            ]
        )
    ):
        return RouteDecision(
            route=
                "LLM_FALLBACK",
            fallback_reason=
                "PRESENCE_ABSTENTION",
            direct_status=
                None,
            direct_level=
                None,
            presence_probability=
                probability,
            intensity_confidence=
                intensity_confidence,
            transformer_level=
                transformer_level,
        )

    if (
        intensity_confidence
        < float(
            intensity_policy[
                "min_confidence"
            ]
        )
    ):
        return RouteDecision(
            route=
                "LLM_FALLBACK",
            fallback_reason=
                "INTENSITY_ABSTENTION",
            direct_status=
                None,
            direct_level=
                None,
            presence_probability=
                probability,
            intensity_confidence=
                intensity_confidence,
            transformer_level=
                transformer_level,
        )

    return RouteDecision(
        route=
            "TRANSFORMER_DIRECT",
        fallback_reason=
            None,
        direct_status=
            "RESOLVED",
        direct_level=
            transformer_level,
        presence_probability=
            probability,
        intensity_confidence=
            intensity_confidence,
        transformer_level=
            transformer_level,
    )


def build_prompt(
    text: str,
    requested_dimensions: Sequence[
        str
    ],
) -> str:
    dimensions = [
        str(
            dimension
        )
        for dimension
        in requested_dimensions
    ]

    return (
        PROMPT_POLICY
        + "\n\nREQUESTED_DIMENSIONS\n"
        + json.dumps(
            dimensions,
            ensure_ascii=False,
        )
        + "\n\nCURRENT_USER_MESSAGE\n"
        + text
    )


def _clean_json_candidate(
    raw: str,
) -> str:
    value = (
        raw
        or ""
    ).strip()

    if value.startswith(
        "```"
    ):
        value = re.sub(
            r"^```(?:json)?\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )

        value = re.sub(
            r"\s*```$",
            "",
            value,
        )

    first = value.find(
        "{"
    )

    last = value.rfind(
        "}"
    )

    if (
        first >= 0
        and last
        >= first
    ):
        value = value[
            first:
            last
            + 1
        ]

    return value


def parse_llm_response(
    *,
    raw_text: str,
    requested_dimensions: Sequence[
        str
    ],
    user_text: str,
) -> Dict[str, Any]:
    requested = tuple(
        requested_dimensions
    )

    requested_set = set(
        requested
    )

    violations: List[
        str
    ] = []

    try:
        payload = json.loads(
            _clean_json_candidate(
                raw_text
            )
        )
    except Exception:
        return {
            "valid":
                False,
            "violations": [
                "INVALID_JSON",
            ],
            "dimensions": {
                dimension: {
                    "status":
                        "UNRESOLVED",
                    "level":
                        None,
                    "evidence":
                        None,
                    "accepted":
                        False,
                    "validation_error":
                        "INVALID_JSON",
                }
                for dimension
                in requested
            },
        }

    raw_dimensions = payload.get(
        "dimensions"
    )

    if not isinstance(
        raw_dimensions,
        dict,
    ):
        raw_dimensions = {}

        violations.append(
            "MISSING_DIMENSIONS_OBJECT"
        )

    extras = sorted(
        set(
            raw_dimensions
        )
        - requested_set
    )

    if extras:
        violations.append(
            "UNREQUESTED_DIMENSIONS:"
            + ",".join(
                extras
            )
        )

    output: Dict[
        str,
        Dict[str, Any]
    ] = {}

    allowed_status = {
        "RESOLVED",
        "NO_SIGNAL",
        "RELATIVE_ONLY",
        "UNRESOLVED",
    }

    for dimension in requested:
        item = raw_dimensions.get(
            dimension
        )

        errors: List[
            str
        ] = []

        if not isinstance(
            item,
            dict,
        ):
            output[
                dimension
            ] = {
                "status":
                    "UNRESOLVED",
                "level":
                    None,
                "evidence":
                    None,
                "accepted":
                    False,
                "validation_error":
                    "MISSING_REQUESTED_DIMENSION",
            }

            violations.append(
                f"MISSING:{dimension}"
            )

            continue

        status = str(
            item.get(
                "status",
                "UNRESOLVED",
            )
        ).strip().upper()

        level_raw = item.get(
            "level"
        )

        level = (
            str(
                level_raw
            ).strip().upper()
            if level_raw
            is not None
            else None
        )

        evidence_raw = item.get(
            "evidence"
        )

        evidence = (
            str(
                evidence_raw
            )
            if evidence_raw
            is not None
            else None
        )

        if status not in allowed_status:
            errors.append(
                "INVALID_STATUS"
            )

            status = (
                "UNRESOLVED"
            )

            level = None

        if (
            status
            == "RESOLVED"
        ):
            if level not in LEVELS:
                errors.append(
                    "INVALID_RESOLVED_LEVEL"
                )

                status = (
                    "UNRESOLVED"
                )

                level = None

            if (
                not evidence
                or evidence
                not in user_text
            ):
                errors.append(
                    "UNSUPPORTED_EVIDENCE"
                )

                status = (
                    "UNRESOLVED"
                )

                level = None

        else:
            if level is not None:
                errors.append(
                    "LEVEL_MUST_BE_NULL"
                )

                status = (
                    "UNRESOLVED"
                )

                level = None

            if (
                status
                == "RELATIVE_ONLY"
                and (
                    not evidence
                    or evidence
                    not in user_text
                )
            ):
                errors.append(
                    "UNSUPPORTED_EVIDENCE"
                )

                status = (
                    "UNRESOLVED"
                )

            if (
                status
                == "NO_SIGNAL"
                and evidence
                is not None
                and evidence
                not in user_text
            ):
                errors.append(
                    "UNSUPPORTED_EVIDENCE"
                )

                status = (
                    "UNRESOLVED"
                )

        accepted = (
            not errors
            and status
            in {
                "RESOLVED",
                "NO_SIGNAL",
                "RELATIVE_ONLY",
            }
        )

        output[
            dimension
        ] = {
            "status":
                status,
            "level":
                level,
            "evidence":
                evidence,
            "accepted":
                accepted,
            "validation_error":
                (
                    ";".join(
                        errors
                    )
                    if errors
                    else None
                ),
        }

        if errors:
            violations.extend(
                f"{dimension}:{error}"
                for error
                in errors
            )

    return {
        "valid":
            not violations,
        "violations":
            violations,
        "dimensions":
            output,
    }


def prediction_matches_gold(
    prediction: Mapping[
        str,
        Any
    ],
    gold: Mapping[
        str,
        Any
    ],
) -> bool:
    if (
        prediction.get(
            "status"
        )
        != gold.get(
            "status"
        )
    ):
        return False

    if (
        gold.get(
            "status"
        )
        == "RESOLVED"
    ):
        return (
            prediction.get(
                "level"
            )
            == gold.get(
                "level"
            )
        )

    return (
        prediction.get(
            "level"
        )
        is None
    )


def stratified_sample_rows(
    rows: Sequence[
        Dict[str, Any]
    ],
    sample_size: int,
    seed: int,
) -> List[
    Dict[str, Any]
]:
    if (
        sample_size
        <= 0
        or sample_size
        >= len(
            rows
        )
    ):
        return list(
            rows
        )

    import random

    rng = random.Random(
        seed
    )

    buckets: Dict[
        Tuple[
            str,
            str,
            str,
        ],
        List[
            Dict[str, Any]
        ],
    ] = {}

    for row in rows:
        reasons = sorted(
            {
                item[
                    "fallback_reason"
                ]
                for item
                in row[
                    "fallback_dimensions"
                ]
            }
        )

        reason_key = (
            "+".join(
                reasons
            )
            or "NONE"
        )

        key = (
            str(
                row.get(
                    "language",
                    "unknown",
                )
            ),
            str(
                row.get(
                    "semantic_family",
                    "unknown",
                )
            ),
            reason_key,
        )

        buckets.setdefault(
            key,
            [],
        ).append(
            row
        )

    for bucket in buckets.values():
        rng.shuffle(
            bucket
        )

    keys = list(
        buckets
    )

    rng.shuffle(
        keys
    )

    selected: List[
        Dict[str, Any]
    ] = []

    # First guarantee one example from as many strata as possible.
    for key in keys:
        if len(
            selected
        ) >= sample_size:
            break

        if buckets[
            key
        ]:
            selected.append(
                buckets[
                    key
                ].pop()
            )

    remaining_pool = [
        item
        for key in keys
        for item
        in buckets[
            key
        ]
    ]

    rng.shuffle(
        remaining_pool
    )

    selected.extend(
        remaining_pool[
            : max(
                0,
                sample_size
                - len(
                    selected
                ),
            )
        ]
    )

    rng.shuffle(
        selected
    )

    return selected[
        :sample_size
    ]


def latency_summary(
    values: Sequence[
        float
    ],
) -> Dict[
    str,
    Optional[
        float
    ],
]:
    if not values:
        return {
            "mean_s":
                None,
            "median_s":
                None,
            "p95_s":
                None,
            "max_s":
                None,
        }

    numeric = [
        float(
            value
        )
        for value
        in values
    ]

    return {
        "mean_s":
            float(
                statistics.mean(
                    numeric
                )
            ),
        "median_s":
            float(
                statistics.median(
                    numeric
                )
            ),
        "p95_s":
            percentile(
                numeric,
                0.95,
            ),
        "max_s":
            max(
                numeric
            ),
    }
