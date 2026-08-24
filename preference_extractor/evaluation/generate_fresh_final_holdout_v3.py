from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


SEED = 20260823 + 103

FAMILIES = (
    "fresh_conditional_choice",
    "fresh_question_tradeoff",
    "fresh_long_context_choice",
    "fresh_ranked_preference",
    "fresh_counterfactual_tradeoff",
    "fresh_third_party_rejected",
    "fresh_negated_record",
    "fresh_system_capability",
    "fresh_superseded_scenario",
    "fresh_priority_lexical_trap",
)

POSITIVE = frozenset(FAMILIES[:5])

LANG_QUOTA = {
    "en": 48,
    "fr": 48,
    "mixed": 24,
}

DIMS = (
    "performance",
    "cost",
    "power",
    "reliability",
)

TERMS = {
    "en": {
        "performance": (
            "faster I/O response",
            "higher aggregate throughput",
            "lower storage latency",
            "better application performance",
            "faster data access",
            "stronger read/write performance",
        ),
        "cost": (
            "lower lifecycle cost",
            "lower acquisition cost",
            "reduced operating expense",
            "better budget efficiency",
            "lower total ownership cost",
            "smaller infrastructure spend",
        ),
        "power": (
            "lower power consumption",
            "better energy efficiency",
            "reduced electrical draw",
            "lower rack power",
            "smaller cooling burden",
            "lower operating wattage",
        ),
        "reliability": (
            "higher availability",
            "stronger fault tolerance",
            "better service continuity",
            "higher resilience",
            "better failure recovery",
            "stronger operational robustness",
        ),
    },
    "fr": {
        "performance": (
            "une réponse E/S plus rapide",
            "un débit agrégé plus élevé",
            "une latence de stockage plus faible",
            "de meilleures performances applicatives",
            "un accès aux données plus rapide",
            "de meilleures performances lecture/écriture",
        ),
        "cost": (
            "un coût de cycle de vie plus faible",
            "un coût d'acquisition réduit",
            "des dépenses d'exploitation plus faibles",
            "une meilleure efficacité budgétaire",
            "un coût total de possession plus faible",
            "une dépense d'infrastructure réduite",
        ),
        "power": (
            "une consommation électrique plus faible",
            "une meilleure efficacité énergétique",
            "une puissance électrique réduite",
            "une puissance par baie plus faible",
            "une charge de refroidissement réduite",
            "une puissance de fonctionnement plus faible",
        ),
        "reliability": (
            "une disponibilité plus élevée",
            "une tolérance aux pannes renforcée",
            "une meilleure continuité de service",
            "une résilience plus élevée",
            "une meilleure reprise après panne",
            "une robustesse opérationnelle renforcée",
        ),
    },
}

CONTEXT = {
    "en": (
        "After capacity and throughput constraints are satisfied",
        "Once the mandatory technical checks are complete",
        "Among the designs that remain fully compliant",
        "At the final architecture selection stage",
        "After sizing and feasibility no longer separate the options",
        "For the production recommendation",
        "When the shortlist contains only feasible systems",
        "Once the hard Lustre constraints are already met",
    ),
    "fr": (
        "Après satisfaction des contraintes de capacité et de débit",
        "Une fois les contrôles techniques obligatoires terminés",
        "Parmi les conceptions qui restent pleinement conformes",
        "À l'étape finale de sélection de l'architecture",
        "Lorsque le dimensionnement et la faisabilité ne départagent plus les options",
        "Pour la recommandation de production",
        "Lorsque la liste finale ne contient que des systèmes réalisables",
        "Une fois les contraintes Lustre strictes déjà satisfaites",
    ),
    "mixed": (
        "After les contraintes de capacity et throughput are satisfied",
        "Une fois the mandatory technical checks terminés",
        "Among les designs that remain fully compliant",
        "At l'étape finale de architecture selection",
        "After sizing et feasibility ne séparent plus the options",
        "Pour the production recommendation",
        "When la shortlist contains only feasible systems",
        "Une fois the hard Lustre constraints are already met",
    ),
}

FACTS = {
    "en": (
        "The workload sheet specifies 800 TiB, 300 clients and 60 GB/s reads",
        "The sizing model already fixes capacity, client count and growth horizon",
        "All surviving candidates satisfy the required throughput floor",
        "The quantitative workload requirements are already validated",
        "The technical shortlist contains only compliant architectures",
        "Capacity, file count and bandwidth are no longer differentiators",
    ),
    "fr": (
        "La fiche de charge indique 800 TiB, 300 clients et 60 GB/s en lecture",
        "Le modèle de dimensionnement fixe déjà la capacité, le nombre de clients et l'horizon de croissance",
        "Tous les candidats restants respectent le débit minimal requis",
        "Les exigences quantitatives de charge sont déjà validées",
        "La liste technique ne contient plus que des architectures conformes",
        "La capacité, le nombre de fichiers et la bande passante ne départagent plus les candidats",
    ),
    "mixed": (
        "The workload sheet indique 800 TiB, 300 clients et 60 GB/s reads",
        "Le sizing model fixe already capacity, client count et growth horizon",
        "All surviving candidates respectent the required throughput floor",
        "Les quantitative workload requirements are already validated",
        "The technical shortlist ne contient que des compliant architectures",
        "Capacity, file count et bandwidth ne sont plus differentiators",
    ),
}

TAILS = {
    "en": (
        "",
        " Treat this as the active decision statement.",
        " This sentence is part of the current requirement.",
        " The statement applies to the current selection.",
        " This is the criterion for the final recommendation.",
        " The wording refers to the active design decision.",
    ),
    "fr": (
        "",
        " Considérez ceci comme la décision active.",
        " Cette phrase fait partie de l'exigence actuelle.",
        " Cette formulation s'applique à la sélection actuelle.",
        " Il s'agit du critère de la recommandation finale.",
        " Cette formulation concerne la décision de conception active.",
    ),
    "mixed": (
        "",
        " Treat ceci as the active decision statement.",
        " Cette phrase is part of the current requirement.",
        " This wording s'applique à la sélection actuelle.",
        " Il s'agit du criterion for the final recommendation.",
        " Cette formulation refers to the active design decision.",
    ),
}

TEMPLATES = {
    "fresh_conditional_choice": {
        "en": (
            "{ctx}, if two candidates are otherwise equivalent, select the one with {a} rather than {b}.",
            "{ctx}, break any remaining tie in favor of {a}, not {b}.",
        ),
        "fr": (
            "{ctx}, si deux candidats sont par ailleurs équivalents, sélectionnez celui qui offre {a} plutôt que {b}.",
            "{ctx}, départagez les candidats restants en faveur de {a}, pas de {b}.",
        ),
        "mixed": (
            "{ctx}, if two candidates sont otherwise equivalent, select celui with {a} rather than {b}.",
            "{ctx}, break any remaining tie en faveur de {a}, not {b}.",
        ),
    },
    "fresh_question_tradeoff": {
        "en": (
            "{ctx}, should we prefer {a} even if that means accepting less {b}?",
            "{ctx}, would an architecture with {a} be preferable when {b} becomes weaker?",
        ),
        "fr": (
            "{ctx}, devons-nous préférer {a} même si cela implique moins de {b} ?",
            "{ctx}, une architecture offrant {a} serait-elle préférable lorsque {b} devient plus faible ?",
        ),
        "mixed": (
            "{ctx}, should we prefer {a} même si that means accepting less {b}?",
            "{ctx}, une architecture with {a} serait-elle preferable when {b} becomes weaker?",
        ),
    },
    "fresh_long_context_choice": {
        "en": (
            "{fact}. {ctx}. The final selection should favor {a}; keep {b} as a secondary criterion.",
            "{fact}. {ctx}. Rank the compliant designs first by {a} and only then by {b}.",
        ),
        "fr": (
            "{fact}. {ctx}. La sélection finale doit favoriser {a} ; gardez {b} comme critère secondaire.",
            "{fact}. {ctx}. Classez d'abord les conceptions conformes selon {a}, puis selon {b}.",
        ),
        "mixed": (
            "{fact}. {ctx}. The final selection doit favoriser {a}; keep {b} comme secondary criterion.",
            "{fact}. {ctx}. Rank les compliant designs first by {a} puis by {b}.",
        ),
    },
    "fresh_ranked_preference": {
        "en": (
            "{ctx}. Selection order: {a} first, {b} second.",
            "{ctx}. Primary soft criterion: {a}; secondary soft criterion: {b}.",
        ),
        "fr": (
            "{ctx}. Ordre de sélection : {a} d'abord, {b} ensuite.",
            "{ctx}. Critère souple principal : {a} ; critère secondaire : {b}.",
        ),
        "mixed": (
            "{ctx}. Selection order: {a} first, {b} ensuite.",
            "{ctx}. Critère soft principal: {a}; secondary criterion: {b}.",
        ),
    },
    "fresh_counterfactual_tradeoff": {
        "en": (
            "{ctx}, accept some loss in {b} when it produces a meaningful gain in {a}.",
            "{ctx}, a reduction in {b} is acceptable if {a} improves materially.",
        ),
        "fr": (
            "{ctx}, acceptez une certaine baisse de {b} si elle apporte un gain significatif en {a}.",
            "{ctx}, une réduction de {b} est acceptable si {a} s'améliore nettement.",
        ),
        "mixed": (
            "{ctx}, accept some loss in {b} si it produces a meaningful gain in {a}.",
            "{ctx}, une reduction de {b} is acceptable if {a} improves materially.",
        ),
    },
    "fresh_third_party_rejected": {
        "en": (
            "{fact}. A vendor report prefers {a}, but that is the vendor's view and not our requirement.",
            "{fact}. An external consultant recommends {a}; the current request does not adopt that preference.",
        ),
        "fr": (
            "{fact}. Un rapport fournisseur préfère {a}, mais il s'agit de l'avis du fournisseur et non de notre exigence.",
            "{fact}. Un consultant externe recommande {a} ; la demande actuelle n'adopte pas cette préférence.",
        ),
        "mixed": (
            "{fact}. A vendor report préfère {a}, but that is the vendor's view et not our requirement.",
            "{fact}. Un external consultant recommends {a}; la demande actuelle does not adopt that preference.",
        ),
    },
    "fresh_negated_record": {
        "en": (
            "{fact}. The approved requirement explicitly states that no preference for {a} has been established.",
            "{fact}. The current specification contains no decision that favors {a}.",
        ),
        "fr": (
            "{fact}. L'exigence approuvée indique explicitement qu'aucune préférence pour {a} n'a été établie.",
            "{fact}. La spécification actuelle ne contient aucune décision favorisant {a}.",
        ),
        "mixed": (
            "{fact}. The approved requirement indique explicitement qu'aucune preference for {a} has been established.",
            "{fact}. La current specification contient no decision that favors {a}.",
        ),
    },
    "fresh_system_capability": {
        "en": (
            "{fact}. The software can rank candidates using {a}; this documents a capability, not a requested preference.",
            "{fact}. The controller supports optimization of {a}, but the request does not tell it to prioritize that criterion.",
        ),
        "fr": (
            "{fact}. Le logiciel peut classer les candidats selon {a} ; cela décrit une capacité, pas une préférence demandée.",
            "{fact}. Le contrôleur prend en charge l'optimisation de {a}, mais la demande ne lui impose pas de prioriser ce critère.",
        ),
        "mixed": (
            "{fact}. The software peut rank candidates using {a}; cela documents a capability, not a requested preference.",
            "{fact}. Le controller supports optimization de {a}, but the request does not tell it to prioritize that criterion.",
        ),
    },
    "fresh_superseded_scenario": {
        "en": (
            "{fact}. A retired draft placed {a} first, but that draft has been superseded and is not active.",
            "{fact}. An obsolete scenario favored {a}; the current requirement replaced that scenario.",
        ),
        "fr": (
            "{fact}. Un ancien brouillon plaçait {a} en premier, mais ce brouillon a été remplacé et n'est plus actif.",
            "{fact}. Un scénario obsolète favorisait {a} ; l'exigence actuelle a remplacé ce scénario.",
        ),
        "mixed": (
            "{fact}. A retired draft plaçait {a} first, but that draft has been superseded et n'est plus active.",
            "{fact}. Un obsolete scenario favored {a}; la current requirement replaced that scenario.",
        ),
    },
    "fresh_priority_lexical_trap": {
        "en": (
            "{fact}. The telemetry process for {a} runs with high operating-system priority.",
            "{fact}. A priority-class field is attached to the monitoring task that measures {a}.",
        ),
        "fr": (
            "{fact}. Le processus de télémétrie pour {a} s'exécute avec une priorité système élevée.",
            "{fact}. Un champ de classe de priorité est associé à la tâche de supervision qui mesure {a}.",
        ),
        "mixed": (
            "{fact}. The telemetry process for {a} s'exécute with high operating-system priority.",
            "{fact}. Un priority-class field is attached to la monitoring task that measures {a}.",
        ),
    },
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def stable_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256(
        "||".join(parts).encode("utf-8")
    ).digest()

    return random.Random(
        int.from_bytes(digest[:8], "big") ^ SEED
    )


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def terms_for(language: str):
    if language == "en":
        return TERMS["en"]
    return TERMS["fr"]


def generate_row(
    family: str,
    language: str,
    rng: random.Random,
):
    label_id = 1 if family in POSITIVE else 0

    primary = rng.choice(DIMS)
    secondary = rng.choice(
        [
            dim
            for dim in DIMS
            if dim != primary
        ]
    )

    table = terms_for(language)

    a = rng.choice(table[primary])
    b = rng.choice(table[secondary])

    template = rng.choice(
        TEMPLATES[family][language]
    )

    text = template.format(
        a=a,
        b=b,
        ctx=rng.choice(CONTEXT[language]),
        fact=rng.choice(FACTS[language]),
    )

    text += rng.choice(TAILS[language])
    text = re.sub(r"\s+", " ", text).strip()

    return {
        "text": text,
        "label": (
            "PREFERENCE_SIGNAL"
            if label_id == 1
            else "NO_PREFERENCE_SIGNAL"
        ),
        "label_id": label_id,
        "language": language,
        "stress_family": family,
        "hard_family": family,
        "template_family": f"fresh_v3::{family}",
        "source_dataset": "preference_signal_fresh_final_holdout_v3",
        "split": "fresh_final_holdout_v3",
    }


def generate_family(
    family: str,
):
    rows = []

    for language, count in LANG_QUOTA.items():
        rng = stable_rng(
            family,
            language,
        )
        unique = {}
        attempts = 0

        while len(unique) < count:
            row = generate_row(
                family=family,
                language=language,
                rng=rng,
            )

            unique.setdefault(
                norm(row["text"]),
                row,
            )

            attempts += 1

            if attempts > count * 1000:
                raise RuntimeError(
                    f"Could not generate enough unique rows for "
                    f"{family}/{language}: {len(unique)} / {count}"
                )

        rows.extend(
            list(unique.values())[:count]
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the fresh final V3 holdout AFTER the V2.1 threshold "
            "has been frozen. Do not inspect individual messages before FIRST_RUN."
        )
    )

    parser.add_argument(
        "--frozen-threshold",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/results/"
            "v2_1_FINAL_threshold_FROZEN.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/datasets/"
            "preference_signal_fresh_final_holdout_v3.jsonl"
        ),
    )

    parser.add_argument(
        "--reference",
        action="append",
        default=[
            "preference_extractor/training/data/data_layer1_v2_1/"
            "preference_signal_train_v2_1.jsonl",
            "preference_extractor/training/data/data_layer1_v2_1/"
            "preference_signal_val_v2_1.jsonl",
            "preference_extractor/training/data/data_layer1_v2_1/"
            "preference_signal_regression_holdout_v2_1200.jsonl",
            "preference_extractor/evaluation/datasets/"
            "preference_signal_test_v1.jsonl",
        ],
    )

    parser.add_argument(
        "--skip-reference-overlap-check",
        action="store_true",
        help="Package self-test only. Do not use in the real project.",
    )

    args = parser.parse_args()

    frozen = json.loads(
        args.frozen_threshold.read_text(encoding="utf-8")
    )

    if frozen.get("status") != "FROZEN_BEFORE_FRESH_HOLDOUT":
        raise AssertionError(
            "Threshold is not frozen before fresh-holdout generation."
        )

    if frozen.get("fresh_holdout_seen_before_freeze") is not False:
        raise AssertionError(
            "Frozen record does not prove holdout isolation."
        )

    if args.output.exists():
        raise RuntimeError(
            f"Fresh holdout already exists: {args.output}. "
            "Do not regenerate or overwrite it."
        )

    rows = []

    for family in FAMILIES:
        family_rows = generate_family(family)

        if len(family_rows) != 120:
            raise AssertionError(
                (family, len(family_rows), 120)
            )

        rows.extend(family_rows)

    rng = stable_rng("full_fresh_holdout_v3")
    rng.shuffle(rows)

    for index, row in enumerate(rows, start=1):
        row["sample_id"] = f"freshv3_{index:06d}"
        row["group_id"] = (
            f"freshv3_{row['stress_family']}_"
            f"{row['language']}_{index:06d}"
        )
        row["template_id"] = (
            f"{row['template_family']}::"
            f"{row['language']}::{index:06d}"
        )

    normalized = {
        norm(row["text"])
        for row in rows
    }

    if len(rows) != 1200:
        raise AssertionError(len(rows))

    if len(normalized) != 1200:
        raise AssertionError(
            "Duplicate text inside fresh holdout."
        )

    label_counts = Counter(
        int(row["label_id"])
        for row in rows
    )

    language_counts = Counter(
        row["language"]
        for row in rows
    )

    family_counts = Counter(
        row["stress_family"]
        for row in rows
    )

    if label_counts != Counter({0: 600, 1: 600}):
        raise AssertionError(label_counts)

    if language_counts != Counter(
        {"en": 480, "fr": 480, "mixed": 240}
    ):
        raise AssertionError(language_counts)

    if set(family_counts.values()) != {120}:
        raise AssertionError(family_counts)

    overlap_report = {}

    if not args.skip_reference_overlap_check:
        for raw_path in args.reference:
            path = Path(raw_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Reference file not found: {path}"
                )

            reference_texts = {
                norm(row["text"])
                for row in read_jsonl(path)
            }

            overlap = len(
                normalized
                & reference_texts
            )

            overlap_report[str(path)] = overlap

            if overlap != 0:
                raise AssertionError(
                    f"Fresh holdout overlaps {path}: {overlap} texts"
                )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
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

    metadata = {
        "dataset": "preference_signal_fresh_final_holdout_v3",
        "seed": SEED,
        "samples": len(rows),
        "label_counts": dict(label_counts),
        "language_counts": dict(language_counts),
        "family_counts": dict(family_counts),
        "exact_text_unique": True,
        "reference_exact_text_overlap": overlap_report,
        "threshold_frozen_before_generation": True,
        "frozen_threshold_file": str(
            args.frozen_threshold
        ),
        "protocol": (
            "Do not inspect individual messages before FIRST_RUN. "
            "Any system change after seeing FIRST_RUN converts this dataset "
            "to regression-only status."
        ),
    }

    metadata_path = args.output.with_suffix(
        ".metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.output}")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()
