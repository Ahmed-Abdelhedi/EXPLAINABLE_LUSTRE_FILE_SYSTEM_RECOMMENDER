from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence


SEED = 20260823

FAMILIES = (
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

POSITIVE = frozenset(FAMILIES[:5])

PER_FAMILY = {
    "train": 600,
    "val": 120,
    "holdout": 120,
}

LANG_RATIO = {
    "en": 0.40,
    "fr": 0.40,
    "mixed": 0.20,
}

TERMS: Dict[str, Dict[str, Sequence[str]]] = {
    "en": {
        "performance": (
            "I/O responsiveness",
            "storage throughput",
            "I/O speed",
            "low-latency I/O",
            "peak read/write performance",
        ),
        "cost": (
            "purchase cost",
            "operational cost",
            "acquisition cost",
            "overall cost",
            "infrastructure spend",
        ),
        "power": (
            "energy efficiency",
            "power efficiency",
            "lower power draw",
            "energy use",
            "electrical consumption",
        ),
        "reliability": (
            "fault tolerance",
            "service continuity",
            "failure resilience",
            "availability",
            "operational stability",
        ),
    },
    "fr": {
        "performance": (
            "la réactivité des E/S",
            "le débit de stockage",
            "la vitesse des E/S",
            "la faible latence E/S",
            "les performances de lecture/écriture",
        ),
        "cost": (
            "le coût d'achat",
            "le coût d'exploitation",
            "le coût d'acquisition",
            "le coût global",
            "les dépenses d'infrastructure",
        ),
        "power": (
            "l'efficacité énergétique",
            "la sobriété énergétique",
            "la consommation électrique",
            "la puissance consommée",
            "l'usage énergétique",
        ),
        "reliability": (
            "la tolérance aux pannes",
            "la continuité de service",
            "la résilience aux pannes",
            "la disponibilité",
            "la stabilité opérationnelle",
        ),
    },
}

DIMS = tuple(TERMS["en"])

CONTEXTS = {
    "en": (
        "For the production Lustre system",
        "For this storage architecture",
        "During the final design review",
        "For the planned shared filesystem",
        "For the current procurement decision",
        "When selecting the production design",
        "For the architecture recommendation",
        "In the final deployment decision",
    ),
    "fr": (
        "Pour le système Lustre de production",
        "Pour cette architecture de stockage",
        "Lors de la revue finale de conception",
        "Pour le système de fichiers partagé prévu",
        "Pour la décision d'achat actuelle",
        "Lors du choix de la conception de production",
        "Pour la recommandation d'architecture",
        "Dans la décision finale de déploiement",
    ),
    "mixed": (
        "Pour the production Lustre system",
        "During la revue finale d'architecture",
        "For la décision de déploiement",
        "Dans the final design review",
        "Pour the current storage design",
        "When selecting la production architecture",
    ),
}

CONSTRAINTS = {
    "en": (
        "the capacity target",
        "the required throughput",
        "the client-count requirement",
        "the mandatory storage constraints",
        "the sizing requirements",
        "the baseline technical constraints",
    ),
    "fr": (
        "l'objectif de capacité",
        "le débit requis",
        "l'exigence de nombre de clients",
        "les contraintes obligatoires de stockage",
        "les exigences de dimensionnement",
        "les contraintes techniques de base",
    ),
    "mixed": (
        "l'objectif de capacity",
        "the débit requis",
        "the client-count exigence",
        "les mandatory storage constraints",
        "the sizing exigences",
        "les baseline technical constraints",
    ),
}

FACTS = {
    "en": (
        "The sizing sheet lists 500 TiB usable capacity and 200 clients",
        "The workload requires 80 GB/s read throughput and 40 GB/s write throughput",
        "The design notes specify 10 million files and 30% annual growth",
        "The request includes a 100000 USD ceiling and 15 kW maximum power",
        "The baseline contains 500 TiB, 200 clients, and a 70/30 read/write ratio",
    ),
    "fr": (
        "La fiche de dimensionnement indique 500 TiB utiles et 200 clients",
        "La charge exige 80 GB/s en lecture et 40 GB/s en écriture",
        "Les notes indiquent 10 millions de fichiers et 30 % de croissance annuelle",
        "La demande comprend un plafond de 100000 USD et 15 kW de puissance maximale",
        "La base contient 500 TiB, 200 clients et un ratio lecture/écriture de 70/30",
    ),
    "mixed": (
        "The sizing sheet indique 500 TiB utiles et 200 clients",
        "La workload requires 80 GB/s read et 40 GB/s write throughput",
        "The design notes indiquent 10 million files et 30% annual growth",
        "La request includes a 100000 USD ceiling et 15 kW maximum power",
        "The baseline contient 500 TiB, 200 clients et a 70/30 read/write ratio",
    ),
}

SUFFIXES = {
    "en": (
        "",
        " for the final architecture",
        " in production",
        " for the deployment decision",
        " once mandatory requirements are met",
    ),
    "fr": (
        "",
        " pour l'architecture finale",
        " en production",
        " pour la décision de déploiement",
        " une fois les exigences obligatoires respectées",
    ),
    "mixed": (
        "",
        " pour the final architecture",
        " en production",
        " for la deployment decision",
        " once les mandatory requirements are met",
    ),
}


# Each split has distinct template IDs and distinct wording.
TEMPLATES = {
    "hard_conditional_choice": {
        "train": {
            "en": (
                ("cond_tr_a", "{ctx}: if both options satisfy {constraint}, choose {a} over {b}{suffix}."),
                ("cond_tr_b", "{ctx}: when both designs pass {constraint}, select {a} rather than {b}{suffix}."),
            ),
            "fr": (
                ("cond_tr_fr_a", "{ctx} : si les deux options respectent {constraint}, choisissez {a} plutôt que {b}{suffix}."),
                ("cond_tr_fr_b", "{ctx} : lorsque les deux conceptions satisfont {constraint}, privilégiez {a} par rapport à {b}{suffix}."),
            ),
            "mixed": (
                ("cond_tr_m_a", "{ctx}: if both options respectent {constraint}, choose {a} over {b}{suffix}."),
                ("cond_tr_m_b", "{ctx}: lorsque both designs satisfy {constraint}, privilégiez {a} rather than {b}{suffix}."),
            ),
        },
        "val": {
            "en": (("cond_va_a", "{ctx}: provided that each candidate meets {constraint}, our choice is {a} before {b}{suffix}."),),
            "fr": (("cond_va_fr_a", "{ctx} : à condition que chaque option respecte {constraint}, notre choix va à {a} avant {b}{suffix}."),),
            "mixed": (("cond_va_m_a", "{ctx}: provided que each option respecte {constraint}, notre choice is {a} before {b}{suffix}."),),
        },
        "holdout": {
            "en": (("cond_ho_a", "{ctx}: between otherwise compliant designs, let {a} decide instead of {b}{suffix}."),),
            "fr": (("cond_ho_fr_a", "{ctx} : entre des conceptions conformes, retenez celle qui favorise {a}, même si {b} est moins bon{suffix}."),),
            "mixed": (("cond_ho_m_a", "{ctx}: between des designs otherwise compliant, let {a} decide instead of {b}{suffix}."),),
        },
    },
    "hard_question_tradeoff": {
        "train": {
            "en": (
                ("q_tr_a", "{ctx}: can we optimize for {a}, even at the expense of {b}?"),
                ("q_tr_b", "{ctx}: could we prioritize {a} even if that reduces {b}?"),
            ),
            "fr": (
                ("q_tr_fr_a", "{ctx} : pouvons-nous optimiser {a}, même au détriment de {b} ?"),
                ("q_tr_fr_b", "{ctx} : peut-on prioriser {a} même si cela dégrade {b} ?"),
            ),
            "mixed": (
                ("q_tr_m_a", "{ctx}: can we optimiser {a}, even au détriment de {b}?"),
                ("q_tr_m_b", "{ctx}: peut-on prioritize {a} even if cela réduit {b}?"),
            ),
        },
        "val": {
            "en": (("q_va_a", "{ctx}: would it be acceptable to favor {a} if that means sacrificing {b}?"),),
            "fr": (("q_va_fr_a", "{ctx} : serait-il acceptable de favoriser {a} si cela implique de sacrifier {b} ?"),),
            "mixed": (("q_va_m_a", "{ctx}: would it be acceptable de favoriser {a} if cela sacrifie {b}?"),),
        },
        "holdout": {
            "en": (("q_ho_a", "{ctx}: should {a} take precedence when the trade-off is weaker {b}?"),),
            "fr": (("q_ho_fr_a", "{ctx} : {a} doit-il passer en priorité lorsque le compromis est une baisse de {b} ?"),),
            "mixed": (("q_ho_m_a", "{ctx}: should {a} passer en priorité when the trade-off is weaker {b}?"),),
        },
    },
    "hard_long_context_final_choice": {
        "train": {
            "en": (
                ("long_tr_a", "{fact}. Several architectures satisfy those values. For the final choice, we favor {a} over {b}{suffix}."),
                ("long_tr_b", "{fact}. Mandatory numbers are already covered. Among compliant designs, we prefer {a} to {b}{suffix}."),
            ),
            "fr": (
                ("long_tr_fr_a", "{fact}. Plusieurs architectures respectent ces valeurs. Pour le choix final, nous privilégions {a} plutôt que {b}{suffix}."),
                ("long_tr_fr_b", "{fact}. Les chiffres obligatoires sont couverts. Parmi les conceptions conformes, nous préférons {a} à {b}{suffix}."),
            ),
            "mixed": (
                ("long_tr_m_a", "{fact}. Plusieurs designs satisfy those values. For the final choice, nous privilégions {a} over {b}{suffix}."),
                ("long_tr_m_b", "{fact}. Mandatory numbers sont covered. Among compliant designs, nous préférons {a} to {b}{suffix}."),
            ),
        },
        "val": {
            "en": (("long_va_a", "{fact}. These figures do not separate the remaining candidates. As the tie-breaker, {a} matters more to us than {b}{suffix}."),),
            "fr": (("long_va_fr_a", "{fact}. Ces chiffres ne départagent pas les candidats restants. Pour les départager, {a} compte davantage pour nous que {b}{suffix}."),),
            "mixed": (("long_va_m_a", "{fact}. These figures ne départagent pas les remaining candidates. As tie-breaker, {a} compte more than {b}{suffix}."),),
        },
        "holdout": {
            "en": (("long_ho_a", "{fact}. Assume every shortlisted design complies. The recommendation should therefore be driven by {a}, not {b}{suffix}."),),
            "fr": (("long_ho_fr_a", "{fact}. Supposons que toutes les conceptions présélectionnées soient conformes. La recommandation doit être guidée par {a}, et non par {b}{suffix}."),),
            "mixed": (("long_ho_m_a", "{fact}. Assume every shortlisted design respecte les requirements. The recommendation should be driven by {a}, not {b}{suffix}."),),
        },
    },
    "hard_ranked_telegraphic": {
        "train": {
            "en": (
                ("rank_tr_a", "Production decision — {a} first; {b} second{suffix}."),
                ("rank_tr_b", "Architecture priority: 1) {a}; 2) {b}{suffix}."),
            ),
            "fr": (
                ("rank_tr_fr_a", "Choix production — {a} d'abord ; {b} ensuite{suffix}."),
                ("rank_tr_fr_b", "Priorités d'architecture : 1) {a} ; 2) {b}{suffix}."),
            ),
            "mixed": (
                ("rank_tr_m_a", "Production choice — {a} d'abord; {b} second{suffix}."),
                ("rank_tr_m_b", "Architecture priorities: 1) {a}; 2) {b} ensuite{suffix}."),
            ),
        },
        "val": {
            "en": (("rank_va_a", "Tie-break order for production: primary = {a}; secondary = {b}{suffix}."),),
            "fr": (("rank_va_fr_a", "Ordre de départage en production : principal = {a} ; secondaire = {b}{suffix}."),),
            "mixed": (("rank_va_m_a", "Tie-break order en production: principal = {a}; secondary = {b}{suffix}."),),
        },
        "holdout": {
            "en": (("rank_ho_a", "Final ranking: give precedence to {a}; {b} comes after{suffix}."),),
            "fr": (("rank_ho_fr_a", "Classement final : priorité à {a} ; {b} vient après{suffix}."),),
            "mixed": (("rank_ho_m_a", "Final ranking: priorité à {a}; {b} comes after{suffix}."),),
        },
    },
    "hard_counterfactual_tradeoff": {
        "train": {
            "en": (
                ("counter_tr_a", "{ctx}: even if {b} gets worse, we still want stronger {a}{suffix}."),
                ("counter_tr_b", "{ctx}: we can accept weaker {b} when it materially improves {a}{suffix}."),
            ),
            "fr": (
                ("counter_tr_fr_a", "{ctx} : même si {b} se dégrade, nous voulons toujours renforcer {a}{suffix}."),
                ("counter_tr_fr_b", "{ctx} : nous pouvons accepter une baisse de {b} si cela améliore nettement {a}{suffix}."),
            ),
            "mixed": (
                ("counter_tr_m_a", "{ctx}: même if {b} gets worse, nous voulons stronger {a}{suffix}."),
                ("counter_tr_m_b", "{ctx}: we can accepter weaker {b} si cela improves {a}{suffix}."),
            ),
        },
        "val": {
            "en": (("counter_va_a", "{ctx}: we are willing to give up some {b} to protect {a}{suffix}."),),
            "fr": (("counter_va_fr_a", "{ctx} : nous sommes prêts à céder sur {b} afin de préserver {a}{suffix}."),),
            "mixed": (("counter_va_m_a", "{ctx}: nous sommes willing to give up some {b} pour préserver {a}{suffix}."),),
        },
        "holdout": {
            "en": (("counter_ho_a", "{ctx}: a design may lose on {b}; that is acceptable if it improves {a}{suffix}."),),
            "fr": (("counter_ho_fr_a", "{ctx} : une conception peut être moins bonne sur {b} ; c'est acceptable si elle améliore {a}{suffix}."),),
            "mixed": (("counter_ho_m_a", "{ctx}: a design peut lose on {b}; c'est acceptable if it improves {a}{suffix}."),),
        },
    },
    "hard_third_party_rejected": {
        "train": {
            "en": (
                ("third_tr_a", "The vendor brochure says '{a} is our priority'; that quotation is not our requirement{suffix}."),
                ("third_tr_b", "The supplier proposal recommends {a}. We record the claim, not adopt it as our preference{suffix}."),
            ),
            "fr": (
                ("third_tr_fr_a", "La brochure du fournisseur dit « {a} est notre priorité » ; cette citation n'est pas notre exigence{suffix}."),
                ("third_tr_fr_b", "La proposition du fournisseur recommande {a}. Nous consignons cette affirmation sans l'adopter comme préférence{suffix}."),
            ),
            "mixed": (
                ("third_tr_m_a", "The vendor brochure dit « {a} is our priority »; cette citation is not our requirement{suffix}."),
                ("third_tr_m_b", "La supplier proposal recommends {a}. Nous record the claim, not l'adopter comme préférence{suffix}."),
            ),
        },
        "val": {
            "en": (("third_va_a", "An integrator memo favors {a}, but it represents the integrator's position and no such preference has been approved by us{suffix}."),),
            "fr": (("third_va_fr_a", "Une note de l'intégrateur favorise {a}, mais elle représente sa position et aucune préférence de ce type n'a été approuvée par nous{suffix}."),),
            "mixed": (("third_va_m_a", "An integrator memo favorise {a}, but it represents la position de l'intégrateur and no such preference has been approved by us{suffix}."),),
        },
        "holdout": {
            "en": (("third_ho_a", "A benchmark report praises {a}. That statement describes the publisher's priorities, not ours{suffix}."),),
            "fr": (("third_ho_fr_a", "Un rapport de benchmark vante {a}. Cette phrase décrit les priorités de l'éditeur, pas les nôtres{suffix}."),),
            "mixed": (("third_ho_m_a", "A benchmark report vante {a}. That statement describes les priorités de l'éditeur, not ours{suffix}."),),
        },
    },
    "hard_negated_report": {
        "train": {
            "en": (
                ("neg_tr_a", "The meeting minutes do not state that {a} is preferred{suffix}."),
                ("neg_tr_b", "The review notes do not record any decision to prioritize {a}{suffix}."),
            ),
            "fr": (
                ("neg_tr_fr_a", "Le compte rendu ne dit pas que {a} est privilégié{suffix}."),
                ("neg_tr_fr_b", "Les notes de revue ne consignent aucune décision donnant la priorité à {a}{suffix}."),
            ),
            "mixed": (
                ("neg_tr_m_a", "The minutes ne disent pas que {a} is preferred{suffix}."),
                ("neg_tr_m_b", "Les review notes do not record any decision to prioritize {a}{suffix}."),
            ),
        },
        "val": {
            "en": (("neg_va_a", "No preference for {a} was recorded in the signed requirement review{suffix}."),),
            "fr": (("neg_va_fr_a", "Aucune préférence pour {a} n'a été enregistrée dans la revue d'exigences signée{suffix}."),),
            "mixed": (("neg_va_m_a", "No preference for {a} n'a été recorded in the signed requirement review{suffix}."),),
        },
        "holdout": {
            "en": (("neg_ho_a", "The approved record explicitly says that the team made no choice concerning {a}{suffix}."),),
            "fr": (("neg_ho_fr_a", "Le document approuvé indique explicitement que l'équipe n'a fait aucun choix concernant {a}{suffix}."),),
            "mixed": (("neg_ho_m_a", "The approved record indique explicitly que l'équipe made no choice concerning {a}{suffix}."),),
        },
    },
    "hard_system_capability": {
        "train": {
            "en": (
                ("cap_tr_a", "The controller can automatically tune {a}{suffix}."),
                ("cap_tr_b", "The platform can adjust {a} automatically based on telemetry{suffix}."),
            ),
            "fr": (
                ("cap_tr_fr_a", "Le contrôleur peut régler automatiquement {a}{suffix}."),
                ("cap_tr_fr_b", "La plateforme peut ajuster automatiquement {a} à partir de la télémétrie{suffix}."),
            ),
            "mixed": (
                ("cap_tr_m_a", "The controller peut régler {a} automatically{suffix}."),
                ("cap_tr_m_b", "La platform can adjust {a} automatiquement based on telemetry{suffix}."),
            ),
        },
        "val": {
            "en": (("cap_va_a", "The management software supports dynamic optimization of {a}; this only describes a product capability{suffix}."),),
            "fr": (("cap_va_fr_a", "Le logiciel de gestion permet l'optimisation dynamique de {a} ; cette phrase décrit seulement une capacité du produit{suffix}."),),
            "mixed": (("cap_va_m_a", "The management software permet dynamic optimization of {a}; cette phrase only describes a product capability{suffix}."),),
        },
        "holdout": {
            "en": (("cap_ho_a", "The scheduler is capable of adapting {a} without operator input; no user choice is stated here{suffix}."),),
            "fr": (("cap_ho_fr_a", "L'ordonnanceur est capable d'adapter {a} sans intervention de l'opérateur ; aucun choix utilisateur n'est exprimé ici{suffix}."),),
            "mixed": (("cap_ho_m_a", "The scheduler est capable d'adapter {a} without operator input; aucun user choice is stated here{suffix}."),),
        },
    },
    "hard_rejected_draft": {
        "train": {
            "en": (
                ("draft_tr_a", "A rejected draft would have prioritized {a}; it does not describe the current request{suffix}."),
                ("draft_tr_b", "The discarded proposal favored {a}, but it is no longer part of the active requirement{suffix}."),
            ),
            "fr": (
                ("draft_tr_fr_a", "Un brouillon rejeté aurait privilégié {a} ; il ne décrit pas la demande actuelle{suffix}."),
                ("draft_tr_fr_b", "La proposition abandonnée favorisait {a}, mais elle ne fait plus partie de l'exigence active{suffix}."),
            ),
            "mixed": (
                ("draft_tr_m_a", "A rejected draft aurait privilégié {a}; it does not describe la current request{suffix}."),
                ("draft_tr_m_b", "La discarded proposal favored {a}, but elle ne fait plus partie de l'active requirement{suffix}."),
            ),
        },
        "val": {
            "en": (("draft_va_a", "An earlier option ranked {a} first, but it was formally withdrawn before current requirements were approved{suffix}."),),
            "fr": (("draft_va_fr_a", "Une ancienne option classait {a} en premier, mais elle a été retirée avant l'approbation des exigences actuelles{suffix}."),),
            "mixed": (("draft_va_m_a", "An earlier option classait {a} first, but it was withdrawn avant l'approbation des current requirements{suffix}."),),
        },
        "holdout": {
            "en": (("draft_ho_a", "A superseded scenario emphasized {a}; it is archived and must not be treated as the present request{suffix}."),),
            "fr": (("draft_ho_fr_a", "Un scénario remplacé mettait l'accent sur {a} ; il est archivé et ne doit pas être traité comme la demande présente{suffix}."),),
            "mixed": (("draft_ho_m_a", "A superseded scenario mettait l'accent sur {a}; it is archived and must not be treated as la demande présente{suffix}."),),
        },
    },
    "hard_lexical_priority_trap": {
        "train": {
            "en": (
                ("lex_tr_a", "The priority queue schedules the task that measures {a}{suffix}."),
                ("lex_tr_b", "A high-priority monitoring job records {a}; this is a scheduler attribute{suffix}."),
            ),
            "fr": (
                ("lex_tr_fr_a", "La file de priorité planifie la tâche qui mesure {a}{suffix}."),
                ("lex_tr_fr_b", "Une tâche de supervision à haute priorité enregistre {a} ; il s'agit d'un attribut de l'ordonnanceur{suffix}."),
            ),
            "mixed": (
                ("lex_tr_m_a", "The priority queue planifie la tâche qui measures {a}{suffix}."),
                ("lex_tr_m_b", "Une high-priority monitoring task enregistre {a}; this is un attribut de l'ordonnanceur{suffix}."),
            ),
        },
        "val": {
            "en": (("lex_va_a", "The telemetry record has a field named priority next to the metric for {a}; it is only a data-field name{suffix}."),),
            "fr": (("lex_va_fr_a", "L'enregistrement de télémétrie contient un champ nommé priorité à côté de la mesure de {a} ; c'est seulement un nom de champ{suffix}."),),
            "mixed": (("lex_va_m_a", "The telemetry record contient un field named priority next to la mesure de {a}; it is only un nom de champ{suffix}."),),
        },
        "holdout": {
            "en": (("lex_ho_a", "The daemon handling the {a} probe runs in the operating system's high-priority class{suffix}."),),
            "fr": (("lex_ho_fr_a", "Le démon qui exécute la sonde de {a} fonctionne dans la classe haute priorité du système d'exploitation{suffix}."),),
            "mixed": (("lex_ho_m_a", "The daemon handling la sonde de {a} runs in la high-priority class du système d'exploitation{suffix}."),),
        },
    },
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def rng_for(*parts: str) -> random.Random:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big") ^ SEED)


def quota(total: int) -> Dict[str, int]:
    en = int(total * 0.4)
    fr = int(total * 0.4)
    return {"en": en, "fr": fr, "mixed": total - en - fr}


def term_lang(language: str) -> str:
    return "fr" if language in {"fr", "mixed"} else "en"


def random_row(family: str, split: str, language: str, r: random.Random) -> dict:
    primary = r.choice(DIMS)
    secondary = None
    if family in POSITIVE:
        secondary = r.choice([d for d in DIMS if d != primary])

    lang = term_lang(language)
    a = r.choice(TERMS[lang][primary])
    b = "" if secondary is None else r.choice(TERMS[lang][secondary])
    template_id, template = r.choice(TEMPLATES[family][split][language])

    ctx = r.choice(CONTEXTS[language])
    text = template.format(
        a=a,
        b=b,
        ctx=ctx,
        constraint=r.choice(CONSTRAINTS[language]),
        fact=r.choice(FACTS[language]),
        suffix=r.choice(SUFFIXES[language]),
    )

    # Negative families need enough lexical/contextual diversity to avoid
    # producing near-clone training rows. Prefixing a neutral deployment
    # context keeps the label unchanged while expanding the combination space.
    if family not in POSITIVE:
        text = f"{ctx}: {text}"

    text = re.sub(r"\s+", " ", text).strip()

    label_id = 1 if family in POSITIVE else 0

    return {
        "text": text,
        "label": "PREFERENCE_SIGNAL" if label_id else "NO_PREFERENCE_SIGNAL",
        "label_id": label_id,
        "language": language,
        "stress_family": family,
        "signal_dimensions": [primary, secondary] if label_id else [],
        "template_family": f"hard_v2::{split}::{family}::{template_id}",
        "template_id": template_id,
        "source_dataset": "hard_v2",
        "split": split,
    }


def generate_family(family: str, split: str, language: str, n: int) -> List[dict]:
    r = rng_for(family, split, language)
    by_text = {}
    attempts = 0
    max_attempts = n * 100

    while len(by_text) < n and attempts < max_attempts:
        row = random_row(family, split, language, r)
        by_text.setdefault(norm(row["text"]), row)
        attempts += 1

    if len(by_text) < n:
        raise RuntimeError(
            f"Not enough unique samples for {family}/{split}/{language}: "
            f"{len(by_text)} < {n}"
        )

    rows = list(by_text.values())[:n]
    r.shuffle(rows)
    return rows


def generate_split(split: str) -> List[dict]:
    total_per_family = PER_FAMILY[split]
    q = quota(total_per_family)
    rows = []

    for family in FAMILIES:
        for language, n in q.items():
            rows.extend(generate_family(family, split, language, n))

    r = rng_for("full", split)
    r.shuffle(rows)

    prefix = {"train": "h2tr", "val": "h2va", "holdout": "h2ho"}[split]
    for i, row in enumerate(rows, start=1):
        row["sample_id"] = f"{prefix}_{i:06d}"
        row["group_id"] = (
            f"{prefix}_{row['stress_family']}_{row['language']}_{i:06d}"
        )

    return rows


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def decorate_legacy(rows: Sequence[dict], split: str) -> List[dict]:
    out = []
    for row in rows:
        item = dict(row)
        family = item.get("stress_family", "unknown")
        item["template_family"] = f"legacy_v1::{family}"
        item["template_id"] = f"legacy_v1::{family}"
        item["source_dataset"] = "legacy_v1"
        item["split"] = split
        out.append(item)
    return out


def assert_no_text_overlap(a_name: str, a: Sequence[dict], b_name: str, b: Sequence[dict]):
    overlap = {norm(x["text"]) for x in a} & {norm(x["text"]) for x in b}
    if overlap:
        raise AssertionError(f"{a_name}/{b_name} text overlap: {len(overlap)}")


def build(v1_dir: Path, output_dir: Path, known_stress: Path | None, hard_only: bool) -> dict:
    hard_train = generate_split("train")
    hard_val = generate_split("val")
    holdout = generate_split("holdout")

    assert_no_text_overlap("hard_train", hard_train, "hard_val", hard_val)
    assert_no_text_overlap("hard_train", hard_train, "holdout", holdout)
    assert_no_text_overlap("hard_val", hard_val, "holdout", holdout)

    for a, b in ((hard_train, hard_val), (hard_train, holdout), (hard_val, holdout)):
        assert not (
            {x["template_family"] for x in a}
            & {x["template_family"] for x in b}
        )
        assert not (
            {x["template_id"] for x in a}
            & {x["template_id"] for x in b}
        )

    known_overlap = None
    if known_stress is not None and known_stress.exists():
        known = {norm(x["text"]) for x in read_jsonl(known_stress)}
        known_overlap = {
            "hard_train": len(known & {norm(x["text"]) for x in hard_train}),
            "hard_val": len(known & {norm(x["text"]) for x in hard_val}),
            "independent_holdout": len(known & {norm(x["text"]) for x in holdout}),
        }
        if any(known_overlap.values()):
            raise AssertionError(f"Overlap with known stress benchmark: {known_overlap}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "preference_signal_hard_train_v2.jsonl", hard_train)
    write_jsonl(output_dir / "preference_signal_hard_val_v2.jsonl", hard_val)
    write_jsonl(output_dir / "preference_signal_independent_holdout_v2.jsonl", holdout)

    metadata = {
        "task": "binary preference-signal detection",
        "version": "v2",
        "seed": SEED,
        "hard_families": list(FAMILIES),
        "hard_split_sizes": {
            "train": len(hard_train),
            "val": len(hard_val),
            "independent_holdout": len(holdout),
        },
        "hard_per_family": dict(PER_FAMILY),
        "language_ratios": dict(LANG_RATIO),
        "template_family_isolation": True,
        "known_500_stress_exact_text_overlap": known_overlap,
        "protocol": {
            "known_500_case_benchmark": "regression only; never merged into training",
            "independent_holdout": (
                "template families disjoint from hard train and hard validation; "
                "do not inspect before checkpoint and threshold freeze"
            ),
        },
    }

    if not hard_only:
        train_v1 = decorate_legacy(
            read_jsonl(v1_dir / "preference_signal_train_v1.jsonl"),
            "train",
        )
        val_v1 = decorate_legacy(
            read_jsonl(v1_dir / "preference_signal_val_v1.jsonl"),
            "val",
        )
        legacy_test_v1 = decorate_legacy(
            read_jsonl(v1_dir / "preference_signal_test_v1.jsonl"),
            "legacy_test",
        )

        train_v2 = train_v1 + hard_train
        val_v2 = val_v1 + hard_val
        rng_for("combined_train").shuffle(train_v2)
        rng_for("combined_val").shuffle(val_v2)

        assert_no_text_overlap("train_v2", train_v2, "val_v2", val_v2)
        assert_no_text_overlap("train_v2", train_v2, "holdout", holdout)
        assert_no_text_overlap("val_v2", val_v2, "holdout", holdout)

        write_jsonl(output_dir / "preference_signal_train_v2.jsonl", train_v2)
        write_jsonl(output_dir / "preference_signal_val_v2.jsonl", val_v2)
        # Give the old 4,000-case in-distribution TEST a unique filename.
        # The separate 500-case stress benchmark also uses the basename
        # preference_signal_test_v1.jsonl in another directory.
        write_jsonl(
            output_dir / "preference_signal_legacy_test_v1_4000.jsonl",
            legacy_test_v1,
        )

        metadata["combined_sizes"] = {
            "train_v2": len(train_v2),
            "val_v2": len(val_v2),
            "independent_holdout_v2": len(holdout),
            "legacy_test_v1_4000": len(legacy_test_v1),
        }
        metadata["combined_label_counts"] = {
            "train_v2": dict(Counter(x["label"] for x in train_v2)),
            "val_v2": dict(Counter(x["label"] for x in val_v2)),
        }
        metadata["combined_language_counts"] = {
            "train_v2": dict(Counter(x["language"] for x in train_v2)),
            "val_v2": dict(Counter(x["language"] for x in val_v2)),
        }

    (output_dir / "preference_signal_v2_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata


def main() -> None:
    script = Path(__file__).resolve()
    repo = script.parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=repo / "preference_extractor" / "training" / "data" / "data_layer1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "preference_extractor" / "training" / "data" / "data_layer1_v2",
    )
    parser.add_argument(
        "--known-stress",
        type=Path,
        default=repo / "preference_extractor" / "evaluation" / "datasets" / "preference_signal_test_v1.jsonl",
    )
    parser.add_argument("--hard-only", action="store_true")
    args = parser.parse_args()

    metadata = build(
        v1_dir=args.v1_dir,
        output_dir=args.output_dir,
        known_stress=args.known_stress,
        hard_only=args.hard_only,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
