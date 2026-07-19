from __future__ import annotations

from typing import List, Set

from field_defs import REQUIRED_FIELDS
from llm_fallback_extractor import LLMFallbackExtractor
from models import ExtractedCandidate, ExtractionResult, ParamName
from rule_entity_extractor import RuleBasedEntityExtractor


class HybridExtractor:
    """
    Extracteur hybride rule-first.

    Principe :
    1. Les règles déterministes passent toujours en premier.
    2. Les champs déjà trouvés par les règles sont protégés.
    3. Le LLM fallback est autorisé seulement pour :
       - les champs encore absents ;
       - les champs ayant un signal textuel fort dans le message.
    4. Le LLM ne peut jamais modifier un champ déjà extrait par les règles.
    """

    def __init__(self):
        self.rule_extractor = RuleBasedEntityExtractor()
        self.llm_fallback = LLMFallbackExtractor()

    def _found_fields(
        self,
        candidates: List[ExtractedCandidate],
    ) -> Set[ParamName]:
        return {
            candidate.field
            for candidate in candidates
        }

    def _unresolved_fields(
        self,
        candidates: List[ExtractedCandidate],
    ) -> List[ParamName]:
        found_fields = self._found_fields(candidates)

        return [
            field
            for field in REQUIRED_FIELDS
            if field not in found_fields
        ]

    def _text_contains_signal_for_field(
        self,
        user_text: str,
        field: ParamName,
    ) -> bool:
        """
        Détermine si le texte contient un indice fort pour appeler le LLM.

        Exemple :
        - "les plus gros fichiers environ 80 GB"
          => signal pour max_file_size_gb

        - "HA required"
          => signal pour ha_required

        - champ manquant sans indice clair
          => pas d'appel LLM, on pose une question utilisateur.
        """

        text = user_text.lower()

        signals = {
            ParamName.requested_usable_capacity_tib: [
                "tib",
                "tb",
                "stockage",
                "storage",
                "capacité",
                "capacite",
                "capacity",
                "utile",
                "usable",
            ],
            ParamName.client_count: [
                "client",
                "clients",
                "noeud",
                "noeuds",
                "nœud",
                "nœuds",
                "node",
                "nodes",
                "compute",
            ],
            ParamName.average_file_size_gb: [
                "moyenne",
                "moyen",
                "fichiers font",
                "average file",
                "avg file",
                "average size",
                "file size",
            ],
            ParamName.max_file_size_gb: [
                "taille max",
                "taille maximale",
                "maximum",
                "max file",
                "maximum file",
                "largest file",
                "biggest file",
                "plus gros fichier",
                "plus gros fichiers",
                "gros fichiers",
            ],
            ParamName.total_file_count: [
                "fichiers",
                "files",
                "million",
                "millions",
                "milliard",
                "milliards",
                "file count",
                "number of files",
            ],
            ParamName.read_write_ratio: [
                "ratio",
                "lecture",
                "écriture",
                "ecriture",
                "read/write",
                "read write",
                "read",
                "write",
            ],
            ParamName.access_type: [
                "accès",
                "acces",
                "access",
                "profil",
                "pattern",
                "workload",
                "mixte",
                "mixed",
                "random",
                "aléatoire",
                "aleatoire",
                "sequential",
                "séquentiel",
                "sequentiel",
                "parallel",
                "parallèle",
                "parallele",
                "streaming",
            ],
            ParamName.target_read_gbps: [
                "lecture",
                "read",
                "read target",
                "target read",
                "débit lecture",
                "debit lecture",
                "gb/s",
            ],
            ParamName.target_write_gbps: [
                "écriture",
                "ecriture",
                "write",
                "write target",
                "target write",
                "débit écriture",
                "debit ecriture",
                "gb/s",
            ],
            ParamName.ha_required: [
                "ha",
                "haute disponibilité",
                "haute disponibilite",
                "high availability",
                "availability",
                "required",
                "obligatoire",
                "redondance",
                "redundancy",
                "fault tolerant",
                "sans ha",
                "no ha",
            ],
            ParamName.max_budget_usd: [
                "budget",
                "usd",
                "dollar",
                "dollars",
                "$",
                "maximum",
                "max",
                "ne pas dépasser",
                "ne pas depasser",
            ],
            ParamName.max_power_w: [
                "puissance",
                "power",
                "w",
                "kw",
                "watt",
                "watts",
                "maximum",
                "max",
                "limite",
                "limit",
            ],
            ParamName.annual_growth_percent: [
                "croissance",
                "growth",
                "annual",
                "annuelle",
                "%",
            ],
        }

        return any(
            signal in text
            for signal in signals.get(field, [])
        )

    def _fields_for_llm(
        self,
        user_text: str,
        unresolved_fields: List[ParamName],
        protected_fields: Set[ParamName],
    ) -> List[ParamName]:
        """
        Sélectionne les champs que le LLM a le droit de chercher.

        Conditions :
        - le champ est encore non résolu ;
        - le champ n'a pas été extrait par les règles ;
        - le texte contient un signal fort pour ce champ.
        """

        fields = []

        for field in unresolved_fields:
            if field in protected_fields:
                continue

            if self._text_contains_signal_for_field(user_text, field):
                fields.append(field)

        return fields

    def extract(
        self,
        user_text: str,
        turn_id: int,
    ) -> ExtractionResult:
        # 1. Extraction déterministe
        rule_result = self.rule_extractor.extract(
            user_text=user_text,
            turn_id=turn_id,
        )

        rule_candidates = rule_result.candidates
        protected_fields = self._found_fields(rule_candidates)

        # 2. Champs absents après règles
        unresolved_after_rules = self._unresolved_fields(rule_candidates)

        # 3. LLM seulement pour les champs absents avec signal fort
        llm_requested_fields = self._fields_for_llm(
            user_text=user_text,
            unresolved_fields=unresolved_after_rules,
            protected_fields=protected_fields,
        )

        fallback_result = self.llm_fallback.extract(
            user_text=user_text,
            unresolved_fields=llm_requested_fields,
            protected_fields=protected_fields,
            turn_id=turn_id,
        )

        # 4. Sécurité finale : le LLM ne touche jamais aux champs protégés
        safe_llm_candidates = [
            candidate
            for candidate in fallback_result.candidates
            if candidate.field not in protected_fields
        ]

        candidates = []
        candidates.extend(rule_candidates)
        candidates.extend(safe_llm_candidates)

        warnings = []
        warnings.extend(rule_result.warnings)
        warnings.extend(fallback_result.warnings)

        unresolved_after_fallback = self._unresolved_fields(candidates)

        return ExtractionResult(
            candidates=candidates,
            warnings=warnings,
            unresolved_fields=[
                field.value
                for field in unresolved_after_fallback
            ],
        )