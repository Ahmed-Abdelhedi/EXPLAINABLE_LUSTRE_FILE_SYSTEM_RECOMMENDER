from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


SEED = 20260823 + 31

FAMILIES = (
    "final_conditional_choice",
    "final_question_tradeoff",
    "final_long_context_choice",
    "final_ranked_preference",
    "final_counterfactual_tradeoff",
    "final_third_party_rejected",
    "final_negated_record",
    "final_system_capability",
    "final_superseded_scenario",
    "final_priority_lexical_trap",
)

POSITIVE = frozenset(
    FAMILIES[:5]
)

PER_FAMILY = 120

LANG_QUOTA = {
    "en": 48,
    "fr": 48,
    "mixed": 24,
}

TERMS = {
    "en": {
        "performance": (
            "fast I/O response",
            "higher application throughput",
            "lower I/O latency",
            "stronger storage performance",
            "faster data access",
        ),
        "cost": (
            "lower lifecycle cost",
            "smaller acquisition spend",
            "reduced operating expense",
            "lower infrastructure cost",
            "budget efficiency",
        ),
        "power": (
            "lower energy demand",
            "better energy efficiency",
            "reduced electrical draw",
            "lower power consumption",
            "energy savings",
        ),
        "reliability": (
            "higher resilience",
            "better service continuity",
            "stronger fault tolerance",
            "higher availability",
            "failure robustness",
        ),
    },
    "fr": {
        "performance": (
            "une réponse E/S plus rapide",
            "un débit applicatif plus élevé",
            "une latence E/S plus faible",
            "de meilleures performances de stockage",
            "un accès aux données plus rapide",
        ),
        "cost": (
            "un coût de cycle de vie plus faible",
            "une dépense d'acquisition réduite",
            "des frais d'exploitation plus faibles",
            "un coût d'infrastructure réduit",
            "une meilleure efficacité budgétaire",
        ),
        "power": (
            "une demande énergétique plus faible",
            "une meilleure efficacité énergétique",
            "une consommation électrique réduite",
            "une puissance consommée plus faible",
            "des économies d'énergie",
        ),
        "reliability": (
            "une résilience plus élevée",
            "une meilleure continuité de service",
            "une tolérance aux pannes renforcée",
            "une disponibilité plus élevée",
            "une meilleure robustesse aux pannes",
        ),
    },
}

DIMS = tuple(
    TERMS["en"]
)

CONTEXT = {
    "en": (
        "After the mandatory sizing checks pass",
        "Once every shortlisted design meets the hard constraints",
        "For the production recommendation",
        "At the architecture decision point",
        "After capacity and throughput are satisfied",
        "For the final procurement decision",
    ),
    "fr": (
        "Après validation des contraintes de dimensionnement",
        "Une fois que toutes les conceptions présélectionnées respectent les contraintes",
        "Pour la recommandation de production",
        "Au moment de décider l'architecture",
        "Après satisfaction de la capacité et du débit",
        "Pour la décision finale d'acquisition",
    ),
    "mixed": (
        "After les mandatory sizing checks pass",
        "Une fois every shortlisted design meets les hard constraints",
        "For la production recommendation",
        "At le architecture decision point",
        "After capacity et throughput are satisfied",
        "Pour the final procurement decision",
    ),
}

FACTS = {
    "en": (
        "The workload summary contains 800 TiB, 300 clients and 60 GB/s reads",
        "The sizing sheet already fixes capacity, file count and throughput",
        "All shortlisted systems meet the minimum technical requirements",
        "Capacity, client count and growth horizon are already fixed",
    ),
    "fr": (
        "Le résumé de charge contient 800 TiB, 300 clients et 60 GB/s en lecture",
        "La fiche de dimensionnement fixe déjà la capacité, le nombre de fichiers et le débit",
        "Tous les systèmes présélectionnés respectent les exigences techniques minimales",
        "La capacité, le nombre de clients et l'horizon de croissance sont déjà fixés",
    ),
    "mixed": (
        "The workload summary contient 800 TiB, 300 clients et 60 GB/s reads",
        "La sizing sheet already fixe capacity, file count et throughput",
        "All shortlisted systems respectent les minimum technical requirements",
        "Capacity, client count et growth horizon sont already fixed",
    ),
}


TAILS = {
    "en": (
        "",
        " The sentence is included for traceability.",
        " This note is descriptive only.",
        " No additional requirement is implied.",
        " It is kept only as background context.",
        " This does not alter the active requirement set.",
        " The statement is recorded for documentation.",
        " It is not a new design instruction.",
        " This line is informational.",
        " The active request remains unchanged.",
    ),
    "fr": (
        "",
        " Cette phrase est conservée pour la traçabilité.",
        " Cette note est uniquement descriptive.",
        " Aucune exigence supplémentaire n'est implicite.",
        " Elle est gardée uniquement comme contexte.",
        " Cela ne modifie pas les exigences actives.",
        " La phrase est enregistrée pour documentation.",
        " Il ne s'agit pas d'une nouvelle instruction de conception.",
        " Cette ligne est informative.",
        " La demande active reste inchangée.",
    ),
    "mixed": (
        "",
        " Cette phrase is included for traceability.",
        " This note est uniquement descriptive.",
        " No additional exigence is implied.",
        " Elle est kept only as background context.",
        " This does not modifier les active requirements.",
        " La phrase is recorded for documentation.",
        " It is not une nouvelle design instruction.",
        " Cette line is informational.",
        " The active request reste unchanged.",
    ),
}


TEMPLATES = {
    "final_conditional_choice": {
        "en": (
            "If nothing else separates compliant candidates, make {a} the deciding criterion instead of {b}.",
            "Where two candidates are equally compliant, break the tie in favor of {a}, not {b}.",
        ),
        "fr": (
            "Si rien d'autre ne départage les candidats conformes, faites de {a} le critère décisif plutôt que {b}.",
            "Lorsque deux candidats sont également conformes, départagez-les en faveur de {a}, pas de {b}.",
        ),
        "mixed": (
            "If rien d'autre separates compliant candidates, make {a} le deciding criterion instead of {b}.",
            "Lorsque two candidates are equally compliant, break the tie en faveur de {a}, not {b}.",
        ),
    },
    "final_question_tradeoff": {
        "en": (
            "Would choosing {a} be preferable if the consequence is less {b}?",
            "Should the architecture favor {a} when doing so weakens {b}?",
        ),
        "fr": (
            "Le choix de {a} serait-il préférable si la conséquence est une baisse de {b} ?",
            "L'architecture doit-elle favoriser {a} même si cela affaiblit {b} ?",
        ),
        "mixed": (
            "Would choosing {a} be préférable if the consequence is less {b}?",
            "L'architecture should favor {a} even si cela weakens {b}?",
        ),
    },
    "final_long_context_choice": {
        "en": (
            "{fact}. Those numbers no longer distinguish the candidates. The deciding factor should now be {a}; {b} is secondary.",
            "{fact}. Treat the quantitative requirements as already satisfied. Choose between the remaining designs according to {a} before {b}.",
        ),
        "fr": (
            "{fact}. Ces chiffres ne distinguent plus les candidats. Le facteur décisif doit désormais être {a} ; {b} est secondaire.",
            "{fact}. Considérez les exigences quantitatives comme satisfaites. Choisissez entre les conceptions restantes selon {a} avant {b}.",
        ),
        "mixed": (
            "{fact}. Those numbers ne distinguent plus les candidates. The deciding factor should now be {a}; {b} est secondary.",
            "{fact}. Treat les quantitative requirements comme satisfied. Choose entre remaining designs according to {a} before {b}.",
        ),
    },
    "final_ranked_preference": {
        "en": (
            "Decision order — primary criterion: {a}; fallback criterion: {b}.",
            "Selection hierarchy: {a} has precedence, with {b} behind it.",
        ),
        "fr": (
            "Ordre de décision — critère principal : {a} ; critère de repli : {b}.",
            "Hiérarchie de sélection : {a} est prioritaire, avec {b} derrière.",
        ),
        "mixed": (
            "Decision order — critère principal: {a}; fallback criterion: {b}.",
            "Selection hierarchy: {a} est prioritaire, with {b} behind it.",
        ),
    },
    "final_counterfactual_tradeoff": {
        "en": (
            "{ctx}, a deterioration in {b} is acceptable when it buys a meaningful improvement in {a}.",
            "{ctx}, accept losing some {b} provided that {a} improves materially.",
        ),
        "fr": (
            "{ctx}, une dégradation de {b} est acceptable si elle permet une amélioration significative de {a}.",
            "{ctx}, acceptez de perdre un peu de {b} à condition que {a} s'améliore nettement.",
        ),
        "mixed": (
            "{ctx}, une deterioration de {b} is acceptable when it buys a meaningful improvement in {a}.",
            "{ctx}, accept losing some {b} à condition que {a} improves materially.",
        ),
    },
    "final_third_party_rejected": {
        "en": (
            "According to the vendor case study, {a} ranks first. That statement expresses the publisher's preference, not ours.",
            "An external consultant recommends {a}. This records the consultant's view and does not define our requirement.",
        ),
        "fr": (
            "Selon l'étude de cas du fournisseur, {a} arrive en tête. Cette phrase exprime la préférence de l'éditeur, pas la nôtre.",
            "Un consultant externe recommande {a}. Cela consigne son avis et ne définit pas notre exigence.",
        ),
        "mixed": (
            "According to l'étude du vendor, {a} ranks first. That statement expresses la préférence de l'éditeur, not ours.",
            "Un external consultant recommends {a}. This records son avis and does not define notre requirement.",
        ),
    },
    "final_negated_record": {
        "en": (
            "The approved requirements contain no decision that favors {a}.",
            "The signed review explicitly records that no preference was established for {a}.",
        ),
        "fr": (
            "Les exigences approuvées ne contiennent aucune décision favorisant {a}.",
            "La revue signée indique explicitement qu'aucune préférence n'a été établie pour {a}.",
        ),
        "mixed": (
            "The approved requirements ne contiennent aucune decision that favors {a}.",
            "La signed review explicitly records qu'aucune préférence was established for {a}.",
        ),
    },
    "final_system_capability": {
        "en": (
            "The storage software is able to adapt {a} dynamically; this describes functionality, not a requested preference.",
            "The control plane supports automatic adjustment of {a}. No selection preference is stated.",
        ),
        "fr": (
            "Le logiciel de stockage peut adapter dynamiquement {a} ; cela décrit une fonctionnalité, pas une préférence demandée.",
            "Le plan de contrôle permet l'ajustement automatique de {a}. Aucune préférence de sélection n'est exprimée.",
        ),
        "mixed": (
            "The storage software peut adapt {a} dynamically; cela describes functionality, not a requested preference.",
            "Le control plane supports automatic adjustment of {a}. Aucune selection preference is stated.",
        ),
    },
    "final_superseded_scenario": {
        "en": (
            "A retired design scenario put {a} first. The scenario has been superseded and is not part of the current request.",
            "An obsolete requirements draft emphasized {a}; the active requirements replaced that draft.",
        ),
        "fr": (
            "Un scénario de conception retiré plaçait {a} en premier. Il a été remplacé et ne fait pas partie de la demande actuelle.",
            "Un ancien brouillon d'exigences mettait l'accent sur {a} ; les exigences actives l'ont remplacé.",
        ),
        "mixed": (
            "A retired design scenario mettait {a} first. The scenario has been superseded et n'est pas part of current request.",
            "Un obsolete requirements draft emphasized {a}; les active requirements replaced that draft.",
        ),
    },
    "final_priority_lexical_trap": {
        "en": (
            "The monitoring daemon for {a} executes with elevated process priority.",
            "A priority-class field is attached to the telemetry task measuring {a}.",
        ),
        "fr": (
            "Le démon de supervision de {a} s'exécute avec une priorité de processus élevée.",
            "Un champ de classe de priorité est associé à la tâche de télémétrie qui mesure {a}.",
        ),
        "mixed": (
            "The monitoring daemon for {a} s'exécute with elevated process priority.",
            "Un priority-class field is attached to la telemetry task measuring {a}.",
        ),
    },
}


def norm(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text).strip().casefold(),
    )


def rng_for(*parts: str) -> random.Random:
    digest = hashlib.sha256(
        "||".join(
            parts
        ).encode(
            "utf-8"
        )
    ).digest()

    return random.Random(
        int.from_bytes(
            digest[:8],
            "big",
        )
        ^ SEED
    )


def read_jsonl(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def generate_one(
    family: str,
    language: str,
    rng: random.Random,
):
    label_id = (
        1
        if family in POSITIVE
        else 0
    )

    primary = rng.choice(
        DIMS
    )

    secondary = None

    if label_id == 1:
        secondary = rng.choice(
            [
                dimension
                for dimension in DIMS
                if dimension
                != primary
            ]
        )

    term_language = (
        "fr"
        if language
        in {
            "fr",
            "mixed",
        }
        else "en"
    )

    a = rng.choice(
        TERMS[
            term_language
        ][primary]
    )

    b = (
        ""
        if secondary is None
        else rng.choice(
            TERMS[
                term_language
            ][secondary]
        )
    )

    template = rng.choice(
        TEMPLATES[
            family
        ][language]
    )

    text = template.format(
        a=a,
        b=b,
        ctx=rng.choice(
            CONTEXT[
                language
            ]
        ),
        fact=rng.choice(
            FACTS[
                language
            ]
        ),
    )

    text = (
        text
        + rng.choice(
            TAILS[
                language
            ]
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return {
        "text": text,
        "label": (
            "PREFERENCE_SIGNAL"
            if label_id
            else "NO_PREFERENCE_SIGNAL"
        ),
        "label_id": label_id,
        "language": language,
        "stress_family": family,
        "signal_dimensions": (
            [
                primary,
                secondary,
            ]
            if label_id
            else []
        ),
        "template_family": (
            f"final_v3::{family}"
        ),
        "source_dataset": (
            "final_holdout_v3"
        ),
        "split": (
            "final_holdout_v3"
        ),
    }


def generate_family(
    family: str,
):
    rows = []

    for language, count in LANG_QUOTA.items():
        rng = rng_for(
            family,
            language,
        )

        unique = {}

        attempts = 0

        while (
            len(
                unique
            )
            < count
        ):
            row = generate_one(
                family=family,
                language=language,
                rng=rng,
            )

            unique.setdefault(
                norm(
                    row[
                        "text"
                    ]
                ),
                row,
            )

            attempts += 1

            if attempts > (
                count * 300
            ):
                raise RuntimeError(
                    f"Could not generate enough unique rows "
                    f"for {family}/{language}"
                )

        rows.extend(
            list(
                unique.values()
            )[:count]
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a FRESH Step 3.1C final holdout. "
            "Do not inspect its individual texts before the "
            "threshold candidate is frozen."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/datasets/"
            "preference_signal_final_holdout_v3.jsonl"
        ),
    )

    parser.add_argument(
        "--reference",
        action="append",
        default=[
            "preference_extractor/training/data/data_layer1_v2/"
            "preference_signal_train_v2.jsonl",
            "preference_extractor/training/data/data_layer1_v2/"
            "preference_signal_val_v2.jsonl",
            "preference_extractor/training/data/data_layer1_v2/"
            "preference_signal_independent_holdout_v2.jsonl",
            "preference_extractor/evaluation/datasets/"
            "preference_signal_test_v1.jsonl",
        ],
        help=(
            "JSONL file that must have zero exact-text overlap "
            "with the fresh holdout. May be repeated."
        ),
    )

    parser.add_argument(
        "--skip-reference-overlap-check",
        action="store_true",
        help=(
            "For package self-tests only. Do not use in the real repo."
        ),
    )

    args = parser.parse_args()

    rows = []

    for family in FAMILIES:
        family_rows = generate_family(
            family
        )

        if len(
            family_rows
        ) != PER_FAMILY:
            raise AssertionError(
                (
                    family,
                    len(
                        family_rows
                    ),
                    PER_FAMILY,
                )
            )

        rows.extend(
            family_rows
        )

    rng = rng_for(
        "full_final_holdout"
    )

    rng.shuffle(
        rows
    )

    for index, row in enumerate(
        rows,
        start=1,
    ):
        row[
            "sample_id"
        ] = (
            f"f3c_{index:06d}"
        )

        row[
            "group_id"
        ] = (
            f"f3c_{row['stress_family']}_"
            f"{row['language']}_{index:06d}"
        )

    texts = {
        norm(
            row[
                "text"
            ]
        )
        for row in rows
    }

    if len(
        texts
    ) != len(
        rows
    ):
        raise AssertionError(
            "Duplicate text inside final holdout"
        )

    if not args.skip_reference_overlap_check:
        reference_texts = set()

        for raw_path in args.reference:
            path = Path(
                raw_path
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Reference file not found: {path}"
                )

            reference_texts.update(
                norm(
                    row[
                        "text"
                    ]
                )
                for row in read_jsonl(
                    path
                )
            )

        overlap = (
            texts
            & reference_texts
        )

        if overlap:
            raise AssertionError(
                f"Fresh holdout overlaps references: "
                f"{len(overlap)} texts"
            )

    label_counts = Counter(
        int(
            row[
                "label_id"
            ]
        )
        for row in rows
    )

    language_counts = Counter(
        row[
            "language"
        ]
        for row in rows
    )

    family_counts = Counter(
        row[
            "stress_family"
        ]
        for row in rows
    )

    expected_labels = {
        0: 600,
        1: 600,
    }

    expected_languages = {
        "en": 480,
        "fr": 480,
        "mixed": 240,
    }

    expected_families = {
        family: 120
        for family in FAMILIES
    }

    if dict(
        label_counts
    ) != expected_labels:
        raise AssertionError(
            (
                dict(
                    label_counts
                ),
                expected_labels,
            )
        )

    if dict(
        language_counts
    ) != expected_languages:
        raise AssertionError(
            (
                dict(
                    language_counts
                ),
                expected_languages,
            )
        )

    if dict(
        family_counts
    ) != expected_families:
        raise AssertionError(
            (
                dict(
                    family_counts
                ),
                expected_families,
            )
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
        "dataset": (
            "preference_signal_final_holdout_v3"
        ),
        "seed": SEED,
        "samples": len(
            rows
        ),
        "label_counts": dict(
            label_counts
        ),
        "language_counts": dict(
            language_counts
        ),
        "family_counts": dict(
            family_counts
        ),
        "exact_text_unique": True,
        "reference_overlap_checked": (
            not args.skip_reference_overlap_check
        ),
        "protocol": (
            "Generate only after Step 3.1C calibration design is fixed. "
            "Do not inspect individual examples or tune from this dataset "
            "before FIRST_RUN evaluation."
        ),
    }

    metadata_path = (
        args.output.with_suffix(
            ".metadata.json"
        )
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
        f"\nSaved: {args.output}"
    )

    print(
        f"Saved: {metadata_path}"
    )


if __name__ == "__main__":
    main()
