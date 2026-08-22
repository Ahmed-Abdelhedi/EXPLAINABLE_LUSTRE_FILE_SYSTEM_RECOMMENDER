from __future__ import annotations

import json
from pathlib import Path


DATASET = (
    Path(__file__).resolve().parent
    / "datasets"
    / "preference_signal_test_v1.jsonl"
)

SEED_COUNT = 50
TARGET_COUNT = 500


DIMENSIONS = [
    {
        "en_a": "I/O responsiveness",
        "fr_a": "la réactivité des E/S",
        "en_b": "purchase cost",
        "fr_b": "le coût d'achat",
        "metric_en": "80 GB/s read throughput",
        "metric_fr": "un débit en lecture de 80 GB/s",
        "typo_en_a": "I/O responsivness",
        "typo_fr_a": "la réactivté des E/S",
    },
    {
        "en_a": "fault tolerance",
        "fr_a": "la tolérance aux pannes",
        "en_b": "raw speed",
        "fr_b": "la vitesse brute",
        "metric_en": "99.99 percent availability",
        "metric_fr": "une disponibilité de 99,99 %",
        "typo_en_a": "fault tolerence",
        "typo_fr_a": "la tolérance aux panes",
    },
    {
        "en_a": "operational cost",
        "fr_a": "le coût d'exploitation",
        "en_b": "peak throughput",
        "fr_b": "le débit de pointe",
        "metric_en": "a 100000 USD budget ceiling",
        "metric_fr": "un plafond budgétaire de 100000 USD",
        "typo_en_a": "operatonal cost",
        "typo_fr_a": "le coût d'explotation",
    },
    {
        "en_a": "energy efficiency",
        "fr_a": "l'efficacité énergétique",
        "en_b": "hardware density",
        "fr_b": "la densité matérielle",
        "metric_en": "15 kW maximum power",
        "metric_fr": "une puissance maximale de 15 kW",
        "typo_en_a": "energy eficiency",
        "typo_fr_a": "l'efficacite énergétique",
    },
    {
        "en_a": "operational simplicity",
        "fr_a": "la simplicité d'exploitation",
        "en_b": "feature breadth",
        "fr_b": "la richesse fonctionnelle",
        "metric_en": "500 TiB usable capacity",
        "metric_fr": "une capacité utile de 500 TiB",
        "typo_en_a": "operational simplicty",
        "typo_fr_a": "la simplicté d'exploitation",
    },
]


POSITIVE_FAMILIES = [
    (
        "stress_explicit_priority",
        "For this deployment, {en_a} is our highest priority.",
        "Pour ce déploiement, {fr_a} est notre priorité absolue.",
        "For ce déploiement, {fr_a} is our highest priority.",
    ),
    (
        "stress_comparative",
        "When forced to choose, {en_a} matters more to us than {en_b}.",
        "S'il faut choisir, {fr_a} compte davantage pour nous que {fr_b}.",
        "If we devons choisir, {fr_a} matters more than {en_b}.",
    ),
    (
        "stress_tradeoff",
        "We would accept worse {en_b} if that noticeably improved {en_a}.",
        "Nous accepterions de dégrader {fr_b} si cela améliorait nettement {fr_a}.",
        "We accepterions worse {en_b} pour améliorer nettement {fr_a}.",
    ),
    (
        "stress_deprioritization",
        "We do not care much about {en_b}; protect {en_a} first.",
        "Nous accordons peu d'importance à {fr_b} ; protégez d'abord {fr_a}.",
        "We ne care pas beaucoup about {en_b}; {fr_a} comes first.",
    ),
    (
        "stress_implicit_choice",
        "Between two compliant designs, we would choose the one with better {en_a}.",
        "Entre deux conceptions conformes, nous choisirions celle qui offre la meilleure {fr_a}.",
        "Entre two compliant designs, we choisirions the one with better {en_a}.",
    ),
    (
        "stress_soft_preference",
        "It would be nice if the final architecture favored {en_a}.",
        "Il serait souhaitable que l'architecture finale favorise {fr_a}.",
        "It serait préférable que the final architecture favorise {fr_a}.",
    ),
    (
        "stress_conditional_preference",
        "If both options meet capacity, select {en_a} over {en_b}.",
        "Si les deux options respectent la capacité, choisissez {fr_a} plutôt que {fr_b}.",
        "If both options respectent capacity, choose {fr_a} over {en_b}.",
    ),
    (
        "stress_question_preference",
        "Can we optimize for {en_a}, even at the expense of {en_b}?",
        "Pouvons-nous optimiser {fr_a}, même au détriment de {fr_b} ?",
        "Can we optimiser {fr_a}, even au détriment de {en_b}?",
    ),
    (
        "stress_correction",
        "Ignore my earlier focus on {en_b}; {en_a} is what matters now.",
        "Ignorez mon intérêt précédent pour {fr_b} ; c'est désormais {fr_a} qui compte.",
        "Ignore my earlier focus on {en_b}; maintenant, {fr_a} compte.",
    ),
    (
        "stress_mixed_requirement_preference",
        "We need {metric_en}, but among valid designs prioritize {en_a}.",
        "Il nous faut {metric_fr}, mais parmi les conceptions valides, privilégiez {fr_a}.",
        "We need {metric_en}, mais parmi valid designs, privilégiez {fr_a}.",
    ),
    (
        "stress_long_context_positive",
        "The sizing sheet lists {metric_en}, 200 clients, and ten million files. Several designs satisfy those figures. For the final choice, we strongly favor {en_a} over {en_b}.",
        "La fiche de dimensionnement indique {metric_fr}, 200 clients et dix millions de fichiers. Plusieurs conceptions respectent ces chiffres. Pour le choix final, nous privilégions clairement {fr_a} à {fr_b}.",
        "The sizing sheet lists {metric_en} and 200 clients. Plusieurs designs satisfy those values. For the final choice, nous privilégions {fr_a} over {en_b}.",
    ),
    (
        "stress_short_telegraphic",
        "Production choice: {en_a} first; {en_b} secondary.",
        "Choix production : {fr_a} d'abord ; {fr_b} ensuite.",
        "Production choice: {fr_a} first; {en_b} secondaire.",
    ),
    (
        "stress_double_negation",
        "It is not true that {en_a} is unimportant; we want it protected.",
        "Il est faux que {fr_a} soit sans importance ; nous voulons la préserver.",
        "It is not true que {fr_a} is unimportant; nous voulons la protéger.",
    ),
    (
        "stress_concession",
        "Although {en_b} matters, we are more concerned with {en_a}.",
        "Même si {fr_b} compte, nous sommes davantage préoccupés par {fr_a}.",
        "Although {en_b} matters, nous sommes more concerned with {fr_a}.",
    ),
    (
        "stress_typo_positive",
        "For us, {typo_en_a} remains the top prioritty.",
        "Pour nous, {typo_fr_a} reste la prioritté principale.",
        "For nous, {typo_fr_a} remains the top prioritty.",
    ),
]


NEGATIVE_FAMILIES = [
    (
        "stress_quantitative_only",
        "The requirement specifies {metric_en}.",
        "L'exigence spécifie {metric_fr}.",
        "The requirement spécifie {metric_fr}.",
    ),
    (
        "stress_concept_mention_only",
        "The meeting agenda contains a section about {en_a}.",
        "L'ordre du jour contient une section sur {fr_a}.",
        "The meeting agenda contient une section sur {fr_a}.",
    ),
    (
        "stress_descriptive_fact",
        "The monitoring dashboard records {en_a} every five minutes.",
        "Le tableau de supervision enregistre {fr_a} toutes les cinq minutes.",
        "The monitoring dashboard enregistre {fr_a} every five minutes.",
    ),
    (
        "stress_historical_observation",
        "During the previous deployment, {en_a} was measured weekly.",
        "Pendant le déploiement précédent, {fr_a} était mesurée chaque semaine.",
        "During le précédent deployment, {fr_a} was measured weekly.",
    ),
    (
        "stress_question_without_preference",
        "Where can I find yesterday's report about {en_a}?",
        "Où puis-je trouver le rapport d'hier sur {fr_a} ?",
        "Where puis-je trouver yesterday's report sur {fr_a}?",
    ),
    (
        "stress_quoted_third_party",
        "The vendor brochure says '{en_a} is our priority'; that quotation is not our requirement.",
        "La brochure du fournisseur dit « {fr_a} est notre priorité » ; cette citation n'est pas notre exigence.",
        "The vendor brochure dit « {fr_a} is our priority »; cette citation is not our requirement.",
    ),
    (
        "stress_negated_preference_report",
        "The minutes do not state that {en_a} is preferred.",
        "Le compte rendu ne dit pas que {fr_a} est privilégiée.",
        "The minutes ne disent pas que {fr_a} is preferred.",
    ),
    (
        "stress_measured_comparison",
        "Measured {en_a} is twelve percent higher on node A than on node B.",
        "La mesure de {fr_a} est supérieure de douze pour cent sur le nœud A.",
        "Measured {fr_a} is twelve percent higher sur node A.",
    ),
    (
        "stress_capability_statement",
        "The controller can tune {en_a} automatically.",
        "Le contrôleur peut régler automatiquement {fr_a}.",
        "The controller peut régler {fr_a} automatically.",
    ),
    (
        "stress_procedure_instruction",
        "Record {en_a} in the audit report before Friday.",
        "Consignez {fr_a} dans le rapport d'audit avant vendredi.",
        "Record {fr_a} dans the audit report before Friday.",
    ),
    (
        "stress_normative_requirement",
        "The platform must provide {metric_en}.",
        "La plateforme doit fournir {metric_fr}.",
        "The platform doit fournir {metric_fr}.",
    ),
    (
        "stress_lexical_priority_trap",
        "The priority queue schedules the job that measures {en_a}.",
        "La file de priorité planifie la tâche qui mesure {fr_a}.",
        "The priority queue planifie le job that measures {fr_a}.",
    ),
    (
        "stress_long_context_negative",
        "The planning notes list {metric_en}, 200 clients, and ten million files. The monitoring appendix also discusses {en_a}. The team will review these recorded values next week.",
        "Les notes de planification indiquent {metric_fr}, 200 clients et dix millions de fichiers. L'annexe de supervision traite aussi de {fr_a}. L'équipe examinera ces valeurs la semaine prochaine.",
        "The planning notes list {metric_en} and 200 clients. L'annexe de monitoring discusses {fr_a}. The team examinera these recorded values next week.",
    ),
    (
        "stress_typo_negative",
        "The report contains a measurment of {typo_en_a}.",
        "Le rapport contient une mesurre de {typo_fr_a}.",
        "The report contient une mesurre de {typo_fr_a}.",
    ),
    (
        "stress_rejected_hypothesis",
        "A rejected draft would have prioritized {en_a}; it does not describe the current request.",
        "Un brouillon rejeté aurait privilégié {fr_a} ; il ne décrit pas la demande actuelle.",
        "A rejected draft aurait privilégié {fr_a}; it does not describe la current request.",
    ),
]


def load_seed_samples() -> list[dict]:
    samples = []

    for raw_line in DATASET.read_text(encoding="utf-8-sig").splitlines():
        if not raw_line.strip():
            continue

        sample = json.loads(raw_line)
        sample_number = int(sample["id"].removeprefix("PS"))

        if sample_number <= SEED_COUNT:
            samples.append(sample)

    expected_ids = [f"PS{index:03d}" for index in range(1, SEED_COUNT + 1)]
    actual_ids = [sample["id"] for sample in samples]

    if actual_ids != expected_ids:
        raise ValueError("The original PS001-PS050 seed set is incomplete or reordered.")

    return samples


def build_generated_samples() -> list[dict]:
    generated = []
    next_id = SEED_COUNT + 1

    for label, families in (
        (1, POSITIVE_FAMILIES),
        (0, NEGATIVE_FAMILIES),
    ):
        for category, en_template, fr_template, mixed_template in families:
            templates = (
                ("en", en_template),
                ("fr", fr_template),
                ("mixed", mixed_template),
            )

            for language, template in templates:
                for dimension in DIMENSIONS:
                    generated.append(
                        {
                            "id": f"PS{next_id:03d}",
                            "text": template.format(**dimension),
                            "label": label,
                            "category": category,
                            "language": language,
                        }
                    )
                    next_id += 1

    return generated


def validate(samples: list[dict]) -> None:
    if len(samples) != TARGET_COUNT:
        raise ValueError(f"Expected {TARGET_COUNT} samples, found {len(samples)}.")

    ids = [sample["id"] for sample in samples]
    texts = [sample["text"].casefold() for sample in samples]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs detected.")

    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate sample texts detected.")

    if set(sample["label"] for sample in samples) != {0, 1}:
        raise ValueError("Labels must contain both classes 0 and 1.")


def main() -> None:
    samples = load_seed_samples() + build_generated_samples()
    validate(samples)

    payload = "\n".join(
        json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
        for sample in samples
    )

    DATASET.write_text(payload + "\n", encoding="utf-8")

    positives = sum(sample["label"] == 1 for sample in samples)
    negatives = sum(sample["label"] == 0 for sample in samples)
    print(f"Wrote {len(samples)} samples to {DATASET}")
    print(f"Class distribution: positive={positives}, negative={negatives}")
    print(f"Stress families: {len(POSITIVE_FAMILIES) + len(NEGATIVE_FAMILIES)}")


if __name__ == "__main__":
    main()
