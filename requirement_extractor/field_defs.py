from __future__ import annotations

from typing import Dict, List, Set

from .models import ParamName


REQUIRED_FIELDS: List[ParamName] = [
    ParamName.requested_usable_capacity_tib,
    ParamName.client_count,
    ParamName.average_file_size_gb,
    ParamName.max_file_size_gb,
    ParamName.total_file_count,
    ParamName.read_write_ratio,
    ParamName.access_type,
    ParamName.target_read_gbps,
    ParamName.target_write_gbps,
    ParamName.ha_required,
    ParamName.max_budget_usd,
    ParamName.max_power_w,
    ParamName.annual_growth_percent,
]


ALLOWED_ACCESS_TYPES: Set[str] = {
    "sequential",
    "random",
    "parallel",
    "streaming",
    "mixed",
}


LOW_CONFIDENCE_THRESHOLD = 0.70


FIELD_QUESTIONS: Dict[ParamName, str] = {
    ParamName.requested_usable_capacity_tib:
        "Quelle capacité utile demandée voulez-vous, en TiB ?",

    ParamName.client_count:
        "Combien de clients ou nœuds de calcul utiliseront le système Lustre ?",

    ParamName.average_file_size_gb:
        "Quelle est la taille moyenne des fichiers, en GB ?",

    ParamName.max_file_size_gb:
        "Quelle est la taille maximale typique des fichiers, en GB ?",

    ParamName.total_file_count:
        "Combien de fichiers environ seront stockés ?",

    ParamName.read_write_ratio:
        "Quel est le ratio lecture/écriture ? Exemple : 70/30.",

    ParamName.access_type:
        "Quel est le type d'accès principal : sequential, random, parallel, streaming ou mixed ?",

    ParamName.target_read_gbps:
        "Quel débit de lecture cible voulez-vous, en GB/s ?",

    ParamName.target_write_gbps:
        "Quel débit d'écriture cible voulez-vous, en GB/s ?",

    ParamName.ha_required:
        "La haute disponibilité est-elle obligatoire ? Répondez oui ou non.",

    ParamName.max_budget_usd:
        "Quel est le budget maximal en USD ?",

    ParamName.max_power_w:
        "Quelle est la puissance maximale autorisée en W ou kW ?",

    ParamName.annual_growth_percent:
        "Quel taux de croissance annuelle faut-il prévoir, en % ?",
}


TARGET_UNITS: Dict[ParamName, str | None] = {
    ParamName.requested_usable_capacity_tib: "TiB",
    ParamName.client_count: None,
    ParamName.average_file_size_gb: "GB",
    ParamName.max_file_size_gb: "GB",
    ParamName.total_file_count: None,
    ParamName.read_write_ratio: "%",
    ParamName.access_type: None,
    ParamName.target_read_gbps: "GB/s",
    ParamName.target_write_gbps: "GB/s",
    ParamName.ha_required: None,
    ParamName.max_budget_usd: "USD",
    ParamName.max_power_w: "W",
    ParamName.annual_growth_percent: "%",
}