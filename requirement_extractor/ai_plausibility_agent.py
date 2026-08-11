from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field as dataclass_field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ollama import Client

from .models import FinalFieldValue, ParamName


class AIPlausibilityStatus:
    OK = "OK"
    WARNING = "WARNING"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    BLOCKING = "BLOCKING"


@dataclass(frozen=True)
class AIPlausibilityIssue:
    issue_type: str
    field: ParamName
    severity: str
    message: str
    question: str
    confidence: float = 0.0
    suggested_correction: Optional[Dict[str, Any]] = None
    evidence_fields: Dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class AIPlausibilityReport:
    status: str
    issues: List[AIPlausibilityIssue] = dataclass_field(default_factory=list)
    raw_response: str = ""

    @property
    def needs_clarification(self) -> bool:
        return self.status in {
            AIPlausibilityStatus.NEEDS_CLARIFICATION,
            AIPlausibilityStatus.BLOCKING,
        }

    @property
    def ok(self) -> bool:
        return self.status == AIPlausibilityStatus.OK


class AIPlausibilityAgent:
    """
    Vérificateur de cohérence des besoins Lustre.

    Principe de sécurité :
    - les incohérences mathématiques directes sont vérifiées localement ;
    - les contraintes qui dépendent de l'architecture produisent seulement
      des avertissements ;
    - Ollama peut reformuler les avertissements, mais ne peut ni créer un
      problème, ni changer son champ, ni modifier le statut final ;
    - aucun champ manquant n'est rempli ;
    - aucune valeur extraite n'est modifiée.

    L'agent reçoit uniquement le ``final_json`` déjà produit et validé par
    les couches d'extraction.
    """

    DIRECT_POSITIVE_FIELDS: Tuple[ParamName, ...] = (
        ParamName.requested_usable_capacity_tib,
        ParamName.client_count,
        ParamName.average_file_size_gb,
        ParamName.max_file_size_gb,
        ParamName.total_file_count,
        ParamName.target_read_gbps,
        ParamName.target_write_gbps,
        ParamName.max_budget_usd,
        ParamName.max_power_w,
    )

    def __init__(self) -> None:
        self.enabled = self._env_bool(
            "ENABLE_AI_PLAUSIBILITY_AGENT",
            default=True,
        )

        self.host = os.getenv(
            "OLLAMA_HOST",
            os.getenv(
                "PLAUSIBILITY_AGENT_HOST",
                "http://localhost:11434",
            ),
        )

        self.model = os.getenv(
            "PLAUSIBILITY_AGENT_MODEL",
            os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        )

        self.temperature = self._env_float(
            "PLAUSIBILITY_AGENT_TEMPERATURE",
            default=0.0,
        )

        self.debug = self._env_bool(
            "PLAUSIBILITY_AGENT_DEBUG",
            default=False,
        )

        # ------------------------------------------------------------
        # Ollama safety limits
        # ------------------------------------------------------------
        # The enrichment output is intentionally tiny: one JSON object
        # containing only rewritten warning message/question fields.
        #
        # These limits prevent a local model from monopolising the GPU
        # for several minutes if generation becomes pathological.
        self.timeout_seconds = self._env_float(
            "PLAUSIBILITY_AGENT_TIMEOUT_SECONDS",
            default=60.0,
        )

        self.num_predict = self._env_int(
            "PLAUSIBILITY_AGENT_NUM_PREDICT",
            default=192,
        )

        self.keep_alive = os.getenv(
            "PLAUSIBILITY_AGENT_KEEP_ALIVE",
            "30s",
        ).strip() or "30s"

        # Le LLM est facultatif et non décisionnaire.
        # Il sert uniquement à reformuler les WARNING déjà démontrés
        # par les contrôles locaux.
        self.use_llm_enrichment = self._env_bool(
            "PLAUSIBILITY_AGENT_USE_LLM_ENRICHMENT",
            default=True,
        )

        # Seuils généraux. Ils peuvent être ajustés depuis .env sans
        # modifier le code.
        self.volume_blocking_ratio = self._env_float(
            "PLAUSIBILITY_VOLUME_BLOCKING_RATIO",
            default=1.25,
        )
        self.budget_warning_usd_per_tib = self._env_float(
            "PLAUSIBILITY_BUDGET_WARNING_USD_PER_TIB",
            default=25.0,
        )
        self.power_warning_w_per_tib = self._env_float(
            "PLAUSIBILITY_POWER_WARNING_W_PER_TIB",
            default=2.0,
        )
        self.throughput_warning_gbps_per_tib = self._env_float(
            "PLAUSIBILITY_THROUGHPUT_WARNING_GBPS_PER_TIB",
            default=0.30,
        )
        self.clients_warning_per_tib = self._env_float(
            "PLAUSIBILITY_CLIENTS_WARNING_PER_TIB",
            default=3.0,
        )
        self.growth_warning_percent = self._env_float(
            "PLAUSIBILITY_GROWTH_WARNING_PERCENT",
            default=100.0,
        )

        self.client = Client(
            host=self.host,
            timeout=self.timeout_seconds,
        )

    # ============================================================
    # Public API
    # ============================================================

    def analyze(
        self,
        final_json: Dict[str, Optional[FinalFieldValue]],
    ) -> AIPlausibilityReport:
        if not self.enabled:
            if self.debug:
                print("[AI PLAUSIBILITY] disabled")

            return AIPlausibilityReport(
                status=AIPlausibilityStatus.OK,
                issues=[],
                raw_response="AI plausibility agent disabled.",
            )

        payload = self._final_json_to_plain_dict(final_json)
        derived_facts = self._build_derived_facts(payload)

        if self.debug:
            print(
                f"[AI PLAUSIBILITY] enabled={self.enabled} "
                f"model={self.model} host={self.host} "
                f"llm_enrichment={self.use_llm_enrichment} "
                f"timeout={self.timeout_seconds}s "
                f"num_predict={self.num_predict} "
                f"keep_alive={self.keep_alive}"
            )
            print("[AI PLAUSIBILITY PAYLOAD]")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            print("[AI PLAUSIBILITY DERIVED FACTS]")
            print(
                json.dumps(
                    derived_facts,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        deterministic_report = self._run_guarded_checks(
            payload=payload,
            derived_facts=derived_facts,
        )

        # Une contradiction démontrable reste entièrement déterministe.
        if deterministic_report.status in {
            AIPlausibilityStatus.BLOCKING,
            AIPlausibilityStatus.NEEDS_CLARIFICATION,
        }:
            if self.debug:
                print(
                    "[AI PLAUSIBILITY] deterministic blocking "
                    f"issues={len(deterministic_report.issues)}"
                )

            return deterministic_report

        # Un état cohérent ne doit pas être perturbé par des suppositions
        # du LLM.
        if deterministic_report.status == AIPlausibilityStatus.OK:
            if self.debug:
                print("[AI PLAUSIBILITY] deterministic status=OK")

            return deterministic_report

        # Les cas AMBIGUOUS sont déjà identifiés localement. Ollama peut
        # seulement améliorer la formulation du message et de la question.
        if (
            deterministic_report.status == AIPlausibilityStatus.WARNING
            and self.use_llm_enrichment
        ):
            return self._enrich_warning_report(
                payload=payload,
                derived_facts=derived_facts,
                report=deterministic_report,
            )

        return deterministic_report

    # ============================================================
    # Deterministic guarded checks
    # ============================================================

    def _run_guarded_checks(
        self,
        payload: Dict[str, Any],
        derived_facts: Dict[str, Any],
    ) -> AIPlausibilityReport:
        blocking_issues: List[AIPlausibilityIssue] = []
        warnings: List[AIPlausibilityIssue] = []

        invalid_base_fields = self._check_non_positive_values(
            payload=payload,
            issues=blocking_issues,
        )

        self._check_growth(
            payload=payload,
            blocking_issues=blocking_issues,
            warnings=warnings,
        )

        # Le ratio lecture/écriture est indépendant des autres champs.
        self._check_read_write_ratio(
            payload=payload,
            issues=blocking_issues,
        )

        # ------------------------------------------------------------
        # Contrôles relationnels avec gestion des dépendances
        #
        # Un champ déjà invalide ne doit pas produire une seconde
        # incohérence qui n'est qu'une conséquence de sa valeur invalide.
        # ------------------------------------------------------------

        if not self._has_invalid_dependency(
            invalid_base_fields,
            {
                ParamName.average_file_size_gb,
                ParamName.max_file_size_gb,
            },
        ):
            self._check_file_size_order(
                payload=payload,
                issues=blocking_issues,
            )

        if not self._has_invalid_dependency(
            invalid_base_fields,
            {
                ParamName.max_file_size_gb,
                ParamName.requested_usable_capacity_tib,
            },
        ):
            self._check_max_file_vs_capacity(
                payload=payload,
                derived_facts=derived_facts,
                issues=blocking_issues,
            )

        if not self._has_invalid_dependency(
            invalid_base_fields,
            {
                ParamName.average_file_size_gb,
                ParamName.total_file_count,
                ParamName.requested_usable_capacity_tib,
            },
        ):
            self._check_estimated_volume_vs_capacity(
                payload=payload,
                derived_facts=derived_facts,
                issues=blocking_issues,
            )

        # Une incohérence mathématique directe est prioritaire.
        if blocking_issues:
            blocking_issues = self._deduplicate_issues(
                blocking_issues
            )

            return AIPlausibilityReport(
                status=AIPlausibilityStatus.BLOCKING,
                issues=blocking_issues,
                raw_response=self._deterministic_trace(
                    status=AIPlausibilityStatus.BLOCKING,
                    issues=blocking_issues,
                    derived_facts=derived_facts,
                ),
            )

        self._check_missing_fields(
            payload=payload,
            warnings=warnings,
        )
        self._check_budget_requires_architecture(
            payload=payload,
            derived_facts=derived_facts,
            warnings=warnings,
        )
        self._check_power_requires_architecture(
            payload=payload,
            derived_facts=derived_facts,
            warnings=warnings,
        )
        self._check_throughput_requires_architecture(
            payload=payload,
            derived_facts=derived_facts,
            warnings=warnings,
        )
        self._check_client_scale_requires_architecture(
            payload=payload,
            derived_facts=derived_facts,
            warnings=warnings,
        )

        warnings = self._deduplicate_issues(warnings)

        if warnings:
            return AIPlausibilityReport(
                status=AIPlausibilityStatus.WARNING,
                issues=warnings,
                raw_response=self._deterministic_trace(
                    status=AIPlausibilityStatus.WARNING,
                    issues=warnings,
                    derived_facts=derived_facts,
                ),
            )

        return AIPlausibilityReport(
            status=AIPlausibilityStatus.OK,
            issues=[],
            raw_response=self._deterministic_trace(
                status=AIPlausibilityStatus.OK,
                issues=[],
                derived_facts=derived_facts,
            ),
        )

    def _check_non_positive_values(
        self,
        payload: Dict[str, Any],
        issues: List[AIPlausibilityIssue],
    ) -> Set[ParamName]:
        invalid_fields: Set[ParamName] = set()

        for field in self.DIRECT_POSITIVE_FIELDS:
            value = self._as_float(payload.get(field.value))

            if value is None:
                continue

            if value <= 0:
                invalid_fields.add(field)

                issues.append(
                    self._make_issue(
                        issue_type="NON_POSITIVE_VALUE",
                        field=field,
                        severity="blocking",
                        message=(
                            f"La valeur du champ {field.value} doit être "
                            "strictement positive."
                        ),
                        question=(
                            f"Veuillez corriger {field.value} avec une "
                            "valeur strictement positive."
                        ),
                        confidence=1.0,
                        evidence_fields={
                            field.value: payload.get(field.value)
                        },
                    )
                )

        return invalid_fields

    def _check_growth(
        self,
        payload: Dict[str, Any],
        blocking_issues: List[AIPlausibilityIssue],
        warnings: List[AIPlausibilityIssue],
    ) -> None:
        growth = self._as_float(
            payload.get(ParamName.annual_growth_percent.value)
        )

        if growth is None:
            return

        if growth < 0:
            blocking_issues.append(
                self._make_issue(
                    issue_type="NEGATIVE_ANNUAL_GROWTH",
                    field=ParamName.annual_growth_percent,
                    severity="blocking",
                    message=(
                        "La croissance annuelle ne peut pas être négative."
                    ),
                    question=(
                        "Veuillez fournir une croissance annuelle "
                        "supérieure ou égale à 0 %."
                    ),
                    confidence=1.0,
                    evidence_fields={
                        ParamName.annual_growth_percent.value: growth
                    },
                )
            )
            return

        if growth > self.growth_warning_percent:
            warnings.append(
                self._make_issue(
                    issue_type="HIGH_GROWTH_REQUIRES_CONFIRMATION",
                    field=ParamName.annual_growth_percent,
                    severity="warning",
                    message=(
                        f"La croissance annuelle demandée est de "
                        f"{growth:g} %, ce qui mérite une confirmation."
                    ),
                    question=(
                        "Confirmez-vous cette croissance annuelle ?"
                    ),
                    confidence=0.95,
                    evidence_fields={
                        ParamName.annual_growth_percent.value: growth
                    },
                )
            )

    def _check_read_write_ratio(
        self,
        payload: Dict[str, Any],
        issues: List[AIPlausibilityIssue],
    ) -> None:
        ratio = payload.get(ParamName.read_write_ratio.value)

        if ratio is None:
            return

        if not isinstance(ratio, dict):
            issues.append(
                self._make_issue(
                    issue_type="INVALID_READ_WRITE_RATIO_FORMAT",
                    field=ParamName.read_write_ratio,
                    severity="blocking",
                    message=(
                        "Le ratio lecture/écriture doit contenir une part "
                        "de lecture et une part d'écriture."
                    ),
                    question=(
                        "Veuillez fournir le ratio sous la forme 70/30."
                    ),
                    confidence=1.0,
                    evidence_fields={
                        ParamName.read_write_ratio.value: ratio
                    },
                )
            )
            return

        read_percent = self._as_float(
            ratio.get("read_percent")
        )
        write_percent = self._as_float(
            ratio.get("write_percent")
        )

        if read_percent is None or write_percent is None:
            issues.append(
                self._make_issue(
                    issue_type="INCOMPLETE_READ_WRITE_RATIO",
                    field=ParamName.read_write_ratio,
                    severity="blocking",
                    message=(
                        "Le ratio lecture/écriture est incomplet."
                    ),
                    question=(
                        "Veuillez fournir les deux composantes du ratio."
                    ),
                    confidence=1.0,
                    evidence_fields={
                        ParamName.read_write_ratio.value: ratio
                    },
                )
            )
            return

        if (
            read_percent < 0
            or write_percent < 0
            or read_percent > 100
            or write_percent > 100
            or abs((read_percent + write_percent) - 100.0) > 1e-9
        ):
            issues.append(
                self._make_issue(
                    issue_type="READ_WRITE_RATIO_NOT_100",
                    field=ParamName.read_write_ratio,
                    severity="blocking",
                    message=(
                        f"Le ratio lecture/écriture vaut "
                        f"{read_percent:g}/{write_percent:g} et sa somme "
                        "n'est pas égale à 100."
                    ),
                    question=(
                        "Veuillez corriger le ratio, par exemple 70/30."
                    ),
                    confidence=1.0,
                    evidence_fields={
                        "read_percent": read_percent,
                        "write_percent": write_percent,
                        "ratio_sum": read_percent + write_percent,
                    },
                )
            )

    def _check_file_size_order(
        self,
        payload: Dict[str, Any],
        issues: List[AIPlausibilityIssue],
    ) -> None:
        average_file_size = self._as_float(
            payload.get(ParamName.average_file_size_gb.value)
        )
        max_file_size = self._as_float(
            payload.get(ParamName.max_file_size_gb.value)
        )

        if average_file_size is None or max_file_size is None:
            return

        if average_file_size > max_file_size:
            issues.append(
                self._make_issue(
                    issue_type="AVERAGE_FILE_EXCEEDS_MAXIMUM",
                    field=ParamName.average_file_size_gb,
                    severity="blocking",
                    message=(
                        f"La taille moyenne ({average_file_size:g} GB) "
                        f"dépasse la taille maximale "
                        f"({max_file_size:g} GB)."
                    ),
                    question=(
                        "Veuillez corriger la taille moyenne ou la taille "
                        "maximale des fichiers."
                    ),
                    confidence=1.0,
                    evidence_fields={
                        ParamName.average_file_size_gb.value:
                            average_file_size,
                        ParamName.max_file_size_gb.value:
                            max_file_size,
                    },
                )
            )

    def _check_max_file_vs_capacity(
        self,
        payload: Dict[str, Any],
        derived_facts: Dict[str, Any],
        issues: List[AIPlausibilityIssue],
    ) -> None:
        max_file_gb = self._as_float(
            payload.get(ParamName.max_file_size_gb.value)
        )
        capacity_gb = self._as_float(
            derived_facts.get("capacity_gb")
        )

        if max_file_gb is None or capacity_gb is None:
            return

        if max_file_gb > capacity_gb:
            issues.append(
                self._make_issue(
                    issue_type="MAX_FILE_EXCEEDS_TOTAL_CAPACITY",
                    field=ParamName.max_file_size_gb,
                    severity="blocking",
                    message=(
                        f"La taille maximale d'un fichier "
                        f"({max_file_gb:g} GB) dépasse la capacité totale "
                        f"({capacity_gb:g} GB)."
                    ),
                    question=(
                        "Veuillez corriger la taille maximale du fichier "
                        "ou la capacité demandée."
                    ),
                    confidence=1.0,
                    evidence_fields={
                        ParamName.max_file_size_gb.value: max_file_gb,
                        "capacity_gb": capacity_gb,
                    },
                )
            )

    def _check_estimated_volume_vs_capacity(
        self,
        payload: Dict[str, Any],
        derived_facts: Dict[str, Any],
        issues: List[AIPlausibilityIssue],
    ) -> None:
        estimated_volume_tib = self._as_float(
            derived_facts.get(
                "estimated_dataset_volume_tib_from_average_size"
            )
        )
        capacity_tib = self._as_float(
            payload.get(
                ParamName.requested_usable_capacity_tib.value
            )
        )
        volume_ratio = self._as_float(
            derived_facts.get(
                "estimated_volume_to_capacity_ratio"
            )
        )

        if (
            estimated_volume_tib is None
            or capacity_tib is None
            or volume_ratio is None
        ):
            return

        if volume_ratio > self.volume_blocking_ratio:
            issues.append(
                self._make_issue(
                    issue_type="DATASET_VOLUME_EXCEEDS_CAPACITY",
                    field=ParamName.requested_usable_capacity_tib,
                    severity="blocking",
                    message=(
                        f"Le volume estimé à partir de la taille moyenne "
                        f"et du nombre de fichiers est d'environ "
                        f"{estimated_volume_tib:.2f} TiB, alors que la "
                        f"capacité demandée est {capacity_tib:g} TiB."
                    ),
                    question=(
                        "Veuillez vérifier la capacité, la taille moyenne "
                        "des fichiers ou leur nombre total."
                    ),
                    confidence=1.0,
                    evidence_fields={
                        ParamName.requested_usable_capacity_tib.value:
                            capacity_tib,
                        ParamName.average_file_size_gb.value:
                            payload.get(
                                ParamName.average_file_size_gb.value
                            ),
                        ParamName.total_file_count.value:
                            payload.get(
                                ParamName.total_file_count.value
                            ),
                        "estimated_dataset_volume_tib":
                            estimated_volume_tib,
                        "estimated_volume_to_capacity_ratio":
                            volume_ratio,
                        "blocking_ratio":
                            self.volume_blocking_ratio,
                    },
                )
            )

    def _check_missing_fields(
        self,
        payload: Dict[str, Any],
        warnings: List[AIPlausibilityIssue],
    ) -> None:
        missing_fields = [
            field_name
            for field_name, value in payload.items()
            if value is None
        ]

        if not missing_fields:
            return

        first_field_name = missing_fields[0]

        try:
            first_field = ParamName(first_field_name)
        except Exception:
            first_field = ParamName.requested_usable_capacity_tib

        warnings.append(
            self._make_issue(
                issue_type="INSUFFICIENT_INFORMATION",
                field=first_field,
                severity="warning",
                message=(
                    "Les informations disponibles ne suffisent pas pour "
                    "vérifier toutes les relations de cohérence."
                ),
                question=(
                    "Veuillez compléter les champs manquants nécessaires "
                    "à l'analyse de cohérence."
                ),
                confidence=1.0,
                evidence_fields={
                    "missing_fields": missing_fields
                },
            )
        )

    def _check_budget_requires_architecture(
        self,
        payload: Dict[str, Any],
        derived_facts: Dict[str, Any],
        warnings: List[AIPlausibilityIssue],
    ) -> None:
        budget = self._as_float(
            payload.get(ParamName.max_budget_usd.value)
        )
        capacity = self._as_float(
            payload.get(
                ParamName.requested_usable_capacity_tib.value
            )
        )
        budget_per_tib = self._as_float(
            derived_facts.get("budget_per_tib_usd")
        )

        if (
            budget is None
            or capacity is None
            or budget_per_tib is None
        ):
            return

        if budget_per_tib < self.budget_warning_usd_per_tib:
            warnings.append(
                self._make_issue(
                    issue_type="BUDGET_REQUIRES_ARCHITECTURE",
                    field=ParamName.max_budget_usd,
                    severity="warning",
                    message=(
                        f"Le budget correspond à environ "
                        f"{budget_per_tib:.2f} USD/TiB. Sa faisabilité "
                        "ne peut être confirmée qu'après génération et "
                        "chiffrage de l'architecture."
                    ),
                    question=(
                        "Souhaitez-vous conserver cette limite budgétaire "
                        "pour la phase de génération ?"
                    ),
                    confidence=0.95,
                    evidence_fields={
                        ParamName.max_budget_usd.value: budget,
                        ParamName.requested_usable_capacity_tib.value:
                            capacity,
                        "budget_per_tib_usd": budget_per_tib,
                    },
                )
            )

    def _check_power_requires_architecture(
        self,
        payload: Dict[str, Any],
        derived_facts: Dict[str, Any],
        warnings: List[AIPlausibilityIssue],
    ) -> None:
        power = self._as_float(
            payload.get(ParamName.max_power_w.value)
        )
        capacity = self._as_float(
            payload.get(
                ParamName.requested_usable_capacity_tib.value
            )
        )
        power_per_tib = self._as_float(
            derived_facts.get("power_per_tib_w")
        )

        if (
            power is None
            or capacity is None
            or power_per_tib is None
        ):
            return

        if power_per_tib < self.power_warning_w_per_tib:
            warnings.append(
                self._make_issue(
                    issue_type="POWER_REQUIRES_ARCHITECTURE",
                    field=ParamName.max_power_w,
                    severity="warning",
                    message=(
                        f"La limite correspond à environ "
                        f"{power_per_tib:.2f} W/TiB. Elle semble faible, "
                        "mais sa faisabilité dépend des serveurs, disques, "
                        "contrôleurs et équipements réseau sélectionnés."
                    ),
                    question=(
                        "Souhaitez-vous conserver cette limite de puissance "
                        "pour la génération de l'architecture ?"
                    ),
                    confidence=0.95,
                    evidence_fields={
                        ParamName.max_power_w.value: power,
                        ParamName.requested_usable_capacity_tib.value:
                            capacity,
                        "power_per_tib_w": power_per_tib,
                    },
                )
            )

    def _check_throughput_requires_architecture(
        self,
        payload: Dict[str, Any],
        derived_facts: Dict[str, Any],
        warnings: List[AIPlausibilityIssue],
    ) -> None:
        throughput_per_tib = self._as_float(
            derived_facts.get("throughput_per_tib_gbps")
        )
        total_throughput = self._as_float(
            derived_facts.get("total_throughput_gbps")
        )
        capacity = self._as_float(
            payload.get(
                ParamName.requested_usable_capacity_tib.value
            )
        )

        if (
            throughput_per_tib is None
            or total_throughput is None
            or capacity is None
        ):
            return

        if (
            throughput_per_tib
            > self.throughput_warning_gbps_per_tib
        ):
            warnings.append(
                self._make_issue(
                    issue_type="THROUGHPUT_REQUIRES_ARCHITECTURE",
                    field=ParamName.target_read_gbps,
                    severity="warning",
                    message=(
                        f"Le débit total demandé est de "
                        f"{total_throughput:g} GB/s, soit environ "
                        f"{throughput_per_tib:.3f} GB/s par TiB. "
                        "Sa faisabilité dépend du nombre d'OSS/OST, "
                        "des médias et du réseau."
                    ),
                    question=(
                        "Souhaitez-vous conserver ces objectifs de débit "
                        "pour la génération de l'architecture ?"
                    ),
                    confidence=0.95,
                    evidence_fields={
                        ParamName.target_read_gbps.value:
                            payload.get(
                                ParamName.target_read_gbps.value
                            ),
                        ParamName.target_write_gbps.value:
                            payload.get(
                                ParamName.target_write_gbps.value
                            ),
                        ParamName.requested_usable_capacity_tib.value:
                            capacity,
                        "total_throughput_gbps":
                            total_throughput,
                        "throughput_per_tib_gbps":
                            throughput_per_tib,
                    },
                )
            )

    def _check_client_scale_requires_architecture(
        self,
        payload: Dict[str, Any],
        derived_facts: Dict[str, Any],
        warnings: List[AIPlausibilityIssue],
    ) -> None:
        clients_per_tib = self._as_float(
            derived_facts.get("clients_per_tib")
        )
        client_count = self._as_float(
            payload.get(ParamName.client_count.value)
        )
        capacity = self._as_float(
            payload.get(
                ParamName.requested_usable_capacity_tib.value
            )
        )

        if (
            clients_per_tib is None
            or client_count is None
            or capacity is None
        ):
            return

        if clients_per_tib > self.clients_warning_per_tib:
            warnings.append(
                self._make_issue(
                    issue_type="CLIENT_SCALE_REQUIRES_ARCHITECTURE",
                    field=ParamName.client_count,
                    severity="warning",
                    message=(
                        f"Le système doit servir {client_count:g} clients "
                        f"pour {capacity:g} TiB, soit environ "
                        f"{clients_per_tib:.2f} clients/TiB. "
                        "La faisabilité dépend de la topologie, des services "
                        "Lustre et du réseau."
                    ),
                    question=(
                        "Souhaitez-vous conserver ce nombre de clients pour "
                        "la génération de l'architecture ?"
                    ),
                    confidence=0.95,
                    evidence_fields={
                        ParamName.client_count.value: client_count,
                        ParamName.requested_usable_capacity_tib.value:
                            capacity,
                        "clients_per_tib": clients_per_tib,
                    },
                )
            )

    # ============================================================
    # Optional LLM enrichment
    # ============================================================

    def _enrich_warning_report(
        self,
        payload: Dict[str, Any],
        derived_facts: Dict[str, Any],
        report: AIPlausibilityReport,
    ) -> AIPlausibilityReport:
        prompt = self._build_enrichment_prompt(
            payload=payload,
            derived_facts=derived_facts,
            issues=report.issues,
        )

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._enrichment_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                format="json",
                stream=False,
                think=False,
                keep_alive=self.keep_alive,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                },
            )

            raw_text = response["message"]["content"]

            if self.debug:
                print("[AI PLAUSIBILITY ENRICHMENT RAW]")
                print(raw_text)

        except Exception as exc:
            if self.debug:
                print(
                    f"[AI PLAUSIBILITY ENRICHMENT ERROR] {exc}"
                )

            # L'échec d'Ollama ne doit jamais casser la décision locale.
            return AIPlausibilityReport(
                status=report.status,
                issues=report.issues,
                raw_response=(
                    "AI plausibility enrichment unavailable: "
                    f"{exc}\n{report.raw_response}"
                ),
            )

        parsed = self._parse_json_response(raw_text)

        if parsed is None:
            return AIPlausibilityReport(
                status=report.status,
                issues=report.issues,
                raw_response=raw_text,
            )

        enriched_issues = self._apply_enrichment(
            original_issues=report.issues,
            parsed=parsed,
        )

        return AIPlausibilityReport(
            # Le statut reste déterministe et ne peut pas être changé.
            status=report.status,
            issues=enriched_issues,
            raw_response=raw_text,
        )

    def _enrichment_system_prompt(self) -> str:
        return """
You are a French explanation assistant for an HPC Lustre plausibility checker.

The checker has ALREADY made the decision.
The warnings supplied to you are authoritative and immutable.

You are NOT allowed to:
- create a new issue;
- delete an issue;
- change an issue_type;
- change a field;
- change severity;
- change the final status;
- change any numeric value, unit, ratio or derived fact;
- propose or infer a missing value;
- recommend hardware, models, vendors or architectures;
- turn a warning into a blocking error;
- claim that feasibility is proven before architecture generation.

Your ONLY task is to rewrite the French wording of:
- message;
- question.

IMPORTANT ENRICHMENT REQUIREMENT:
- You MUST genuinely rewrite the message.
- You MAY keep the original question unchanged if it is already clear.
- Preserve exactly the same factual meaning.
- Copy every numeric fact faithfully: never recalculate, convert, round,
  replace or invent a number.
- Preserve units semantically and exactly.
- "GB/s" means GIGABYTES per second. NEVER rewrite "GB/s" as "Gbps",
  "Gb/s", "gigabits/s" or any bit-based unit.
- Do not change TiB values, USD/TiB values, W/TiB values or clients/TiB values.
- Decimal comma versus decimal point is acceptable only when the numeric
  value itself remains identical.
- Make the explanation clearer for a non-expert HPC user.
- Explain why the point remains architecture-dependent when relevant.
- Keep the message concise: normally 1 or 2 sentences.
- Keep the question concise and actionable.
- Never introduce a stronger technical conclusion than the original warning.

Return JSON only.
""".strip()

    def _build_enrichment_prompt(
        self,
        payload: Dict[str, Any],
        derived_facts: Dict[str, Any],
        issues: List[AIPlausibilityIssue],
    ) -> str:
        issue_payload = [
            {
                "issue_type": issue.issue_type,
                "field": issue.field.value,
                "severity": issue.severity,
                "message": issue.message,
                "question": issue.question,
            }
            for issue in issues
        ]

        return f"""
Reformulate the explanation of every warning below.

This is an EXPLANATION task, not a decision task.

For EACH warning:
1. Keep issue_type EXACTLY unchanged.
2. Keep field EXACTLY unchanged.
3. Rewrite message in clearer French for a non-expert.
4. Rewrite question only when a clearer formulation is useful; otherwise
   the original question may be kept.
5. The rewritten MESSAGE must be meaningfully different from the original.
6. Copy every number faithfully. Do not recompute any derived value.
7. Preserve units exactly. In particular, literal "GB/s" must stay "GB/s";
   it must NEVER become "Gbps", "Gb/s" or another bit-based unit.
8. Preserve the same level of uncertainty.
9. For architecture-dependent warnings, explicitly make clear that the value
   cannot be judged definitively before architecture generation/evaluation.
10. Do not propose a replacement value.
11. Do not add, remove or merge warnings.

Validated requirement JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Derived facts:
{json.dumps(derived_facts, ensure_ascii=False, indent=2)}

Warnings to reformulate:
{json.dumps(issue_payload, ensure_ascii=False, indent=2)}

Return exactly this JSON structure:
{{
  "issues": [
    {{
      "issue_type": "EXACT_ORIGINAL_ISSUE_TYPE",
      "field": "EXACT_ORIGINAL_FIELD",
      "message": "A genuinely reformulated French explanation.",
      "question": "A genuinely reformulated French question."
    }}
  ]
}}

The returned message must be meaningfully reformulated, while the question may remain unchanged when already clear.
""".strip()

    def _extract_numeric_facts(
        self,
        text: str,
    ) -> List[Decimal]:
        """
        Extrait les valeurs numériques d'un texte en normalisant
        uniquement le séparateur décimal français/anglais.

        10.00 et 10,00 sont donc considérés comme la même valeur.
        En revanche 1500 et 1536 restent différents.
        """
        if not text:
            return []

        tokens = re.findall(
            r"(?<![A-Za-z0-9_])[-+]?\d+(?:[.,]\d+)?",
            text,
        )

        values: List[Decimal] = []

        for token in tokens:
            normalized = token.replace(",", ".")

            try:
                values.append(Decimal(normalized))
            except InvalidOperation:
                continue

        return values

    def _extract_protected_units(
        self,
        text: str,
    ) -> List[str]:
        """
        Extrait uniquement les unités dont une modification changerait
        le sens métier du warning.

        Les variantes linguistiques équivalentes comme
        "W/TiB" et "W par TiB" sont regroupées.
        En revanche "GB/s" et "Gbps" restent volontairement distinctes :
        GB/s = gigabytes/s, Gbps = gigabits/s.
        """
        if not text:
            return []

        protected: List[str] = []

        patterns = (
            (
                r"\bUSD\s*(?:/|par)\s*TiB\b",
                "USD_PER_TIB",
            ),
            (
                r"\bW\s*(?:/|par)\s*TiB\b",
                "W_PER_TIB",
            ),
            (
                r"\bclients?\s*(?:/|par)\s*TiB\b",
                "CLIENTS_PER_TIB",
            ),
            (
                r"\bGB\s*/\s*s\b",
                "GIGABYTES_PER_SECOND",
            ),
            (
                r"\bGBps\b",
                "GIGABYTES_PER_SECOND",
            ),
            (
                r"\bGbps\b",
                "GIGABITS_PER_SECOND",
            ),
            (
                r"\bGb\s*/\s*s\b",
                "GIGABITS_PER_SECOND",
            ),
        )

        for pattern, canonical_name in patterns:
            count = len(
                re.findall(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                )
            )
            protected.extend(
                [canonical_name] * count
            )

        return sorted(protected)

    def _enrichment_text_preserves_facts(
        self,
        original: str,
        candidate: str,
    ) -> bool:
        """
        Autorise une reformulation seulement si les faits quantitatifs
        et les unités protégées restent inchangés.

        Cette garde est locale et déterministe : même si le LLM ignore
        le prompt, une sortie qui transforme par exemple GB/s en Gbps
        ou 1500 TiB en 1536 TiB n'est jamais appliquée.
        """
        if not candidate:
            return False

        original_numbers = sorted(
            self._extract_numeric_facts(original)
        )
        candidate_numbers = sorted(
            self._extract_numeric_facts(candidate)
        )

        if original_numbers != candidate_numbers:
            if self.debug:
                print(
                    "[AI PLAUSIBILITY ENRICHMENT REJECTED] "
                    "numeric facts changed"
                )
            return False

        original_units = self._extract_protected_units(
            original
        )
        candidate_units = self._extract_protected_units(
            candidate
        )

        if original_units != candidate_units:
            if self.debug:
                print(
                    "[AI PLAUSIBILITY ENRICHMENT REJECTED] "
                    "protected units changed"
                )
            return False

        return True

    def _apply_enrichment(
        self,
        original_issues: List[AIPlausibilityIssue],
        parsed: Dict[str, Any],
    ) -> List[AIPlausibilityIssue]:
        raw_issues = parsed.get("issues", [])

        if not isinstance(raw_issues, list):
            return original_issues

        enriched_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for raw_issue in raw_issues:
            if not isinstance(raw_issue, dict):
                continue

            issue_type = str(
                raw_issue.get("issue_type", "")
            ).strip()
            field_name = str(
                raw_issue.get("field", "")
            ).strip()

            if not issue_type or not field_name:
                continue

            enriched_by_key[(issue_type, field_name)] = raw_issue

        result: List[AIPlausibilityIssue] = []

        for issue in original_issues:
            key = (issue.issue_type, issue.field.value)
            enrichment = enriched_by_key.get(key)

            if enrichment is None:
                result.append(issue)
                continue

            message = str(
                enrichment.get("message", issue.message)
            ).strip()
            question = str(
                enrichment.get("question", issue.question)
            ).strip()

            if not message:
                message = issue.message

            if not question:
                question = issue.question

            # Safety gate: the LLM may rewrite prose, but it cannot alter
            # quantitative facts or protected units in the final report.
            if not self._enrichment_text_preserves_facts(
                original=issue.message,
                candidate=message,
            ):
                message = issue.message

            if not self._enrichment_text_preserves_facts(
                original=issue.question,
                candidate=question,
            ):
                question = issue.question

            result.append(
                AIPlausibilityIssue(
                    issue_type=issue.issue_type,
                    field=issue.field,
                    severity=issue.severity,
                    message=message,
                    question=question,
                    confidence=issue.confidence,
                    suggested_correction=None,
                    evidence_fields=issue.evidence_fields,
                )
            )

        return result

    # ============================================================
    # Derived facts
    # ============================================================

    def _build_derived_facts(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        capacity_tib = self._as_float(
            payload.get(
                ParamName.requested_usable_capacity_tib.value
            )
        )
        client_count = self._as_float(
            payload.get(ParamName.client_count.value)
        )
        average_file_size_gb = self._as_float(
            payload.get(ParamName.average_file_size_gb.value)
        )
        max_file_size_gb = self._as_float(
            payload.get(ParamName.max_file_size_gb.value)
        )
        total_file_count = self._as_float(
            payload.get(ParamName.total_file_count.value)
        )
        target_read_gbps = self._as_float(
            payload.get(ParamName.target_read_gbps.value)
        )
        target_write_gbps = self._as_float(
            payload.get(ParamName.target_write_gbps.value)
        )
        max_budget_usd = self._as_float(
            payload.get(ParamName.max_budget_usd.value)
        )
        max_power_w = self._as_float(
            payload.get(ParamName.max_power_w.value)
        )

        capacity_gb = (
            capacity_tib * 1024.0
            if capacity_tib is not None
            else None
        )

        estimated_dataset_volume_tib = None
        if (
            average_file_size_gb is not None
            and total_file_count is not None
            and average_file_size_gb >= 0
            and total_file_count >= 0
        ):
            estimated_dataset_volume_tib = (
                average_file_size_gb
                * total_file_count
                / 1024.0
            )

        total_throughput_gbps = None
        if (
            target_read_gbps is not None
            or target_write_gbps is not None
        ):
            total_throughput_gbps = (
                float(target_read_gbps or 0.0)
                + float(target_write_gbps or 0.0)
            )

        return {
            "capacity_gb": capacity_gb,
            "estimated_dataset_volume_tib_from_average_size":
                estimated_dataset_volume_tib,
            "estimated_volume_to_capacity_ratio":
                self._safe_divide(
                    estimated_dataset_volume_tib,
                    capacity_tib,
                ),
            "max_file_to_capacity_ratio":
                self._safe_divide(
                    max_file_size_gb,
                    capacity_gb,
                ),
            "total_throughput_gbps":
                total_throughput_gbps,
            "throughput_per_tib_gbps":
                self._safe_divide(
                    total_throughput_gbps,
                    capacity_tib,
                ),
            "clients_per_tib":
                self._safe_divide(
                    client_count,
                    capacity_tib,
                ),
            "power_per_tib_w":
                self._safe_divide(
                    max_power_w,
                    capacity_tib,
                ),
            "power_per_client_w":
                self._safe_divide(
                    max_power_w,
                    client_count,
                ),
            "budget_per_tib_usd":
                self._safe_divide(
                    max_budget_usd,
                    capacity_tib,
                ),
            "budget_per_client_usd":
                self._safe_divide(
                    max_budget_usd,
                    client_count,
                ),
            "missing_fields": [
                field_name
                for field_name, value in payload.items()
                if value is None
            ],
            "decision_thresholds": {
                "volume_blocking_ratio":
                    self.volume_blocking_ratio,
                "budget_warning_usd_per_tib":
                    self.budget_warning_usd_per_tib,
                "power_warning_w_per_tib":
                    self.power_warning_w_per_tib,
                "throughput_warning_gbps_per_tib":
                    self.throughput_warning_gbps_per_tib,
                "clients_warning_per_tib":
                    self.clients_warning_per_tib,
                "growth_warning_percent":
                    self.growth_warning_percent,
            },
        }

    # ============================================================
    # Utilities
    # ============================================================

    def _final_json_to_plain_dict(
        self,
        final_json: Dict[str, Optional[FinalFieldValue]],
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {}

        # Itérer sur ParamName garantit un schéma stable de 13 champs.
        for field in ParamName:
            item = final_json.get(field.value)

            if item is None:
                output[field.value] = None
            else:
                output[field.value] = item.value

        return output

    def _has_invalid_dependency(
        self,
        invalid_fields: Set[ParamName],
        dependencies: Set[ParamName],
    ) -> bool:
        """
        Retourne True lorsqu'un contrôle relationnel dépend d'au moins
        un champ déjà déclaré invalide par un contrôle de base.

        Exemple :
        - capacité <= 0 :
          on conserve NON_POSITIVE_VALUE,
          mais on ne teste pas max_file_size > capacité ni volume > capacité.
        - max_file_size <= 0 :
          on conserve NON_POSITIVE_VALUE,
          mais on ne teste pas average_file_size > max_file_size.

        Les contrôles indépendants restent exécutés, ce qui permet de
        conserver plusieurs vrais conflits dans un même scénario.
        """
        return bool(invalid_fields.intersection(dependencies))

    def _make_issue(
        self,
        issue_type: str,
        field: ParamName,
        severity: str,
        message: str,
        question: str,
        confidence: float,
        evidence_fields: Dict[str, Any],
    ) -> AIPlausibilityIssue:
        return AIPlausibilityIssue(
            issue_type=issue_type,
            field=field,
            severity=severity,
            message=message,
            question=question,
            confidence=max(0.0, min(1.0, confidence)),
            suggested_correction=None,
            evidence_fields=evidence_fields,
        )

    def _deduplicate_issues(
        self,
        issues: Iterable[AIPlausibilityIssue],
    ) -> List[AIPlausibilityIssue]:
        result: List[AIPlausibilityIssue] = []
        seen: set[Tuple[str, str]] = set()

        for issue in issues:
            key = (issue.issue_type, issue.field.value)

            if key in seen:
                continue

            seen.add(key)
            result.append(issue)

        return result

    def _deterministic_trace(
        self,
        status: str,
        issues: List[AIPlausibilityIssue],
        derived_facts: Dict[str, Any],
    ) -> str:
        return json.dumps(
            {
                "decision_source": "DETERMINISTIC_PLAUSIBILITY_GUARD",
                "status": status,
                "issues": [
                    {
                        "issue_type": issue.issue_type,
                        "field": issue.field.value,
                        "severity": issue.severity,
                        "message": issue.message,
                        "question": issue.question,
                        "confidence": issue.confidence,
                        "suggested_correction":
                            issue.suggested_correction,
                        "evidence_fields":
                            issue.evidence_fields,
                    }
                    for issue in issues
                ],
                "derived_facts": derived_facts,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _parse_json_response(
        self,
        raw_text: str,
    ) -> Optional[Dict[str, Any]]:
        text = (raw_text or "").strip()

        try:
            parsed = json.loads(text)

            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL,
        )

        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))

            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _as_float(
        self,
        value: Any,
    ) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_divide(
        self,
        numerator: Optional[float],
        denominator: Optional[float],
    ) -> Optional[float]:
        if (
            numerator is None
            or denominator is None
            or denominator == 0
        ):
            return None

        return numerator / denominator

    def _env_bool(
        self,
        name: str,
        default: bool,
    ) -> bool:
        raw_value = os.getenv(name)

        if raw_value is None:
            return default

        return raw_value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _env_float(
        self,
        name: str,
        default: float,
    ) -> float:
        raw_value = os.getenv(name)

        if raw_value is None:
            return default

        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default


    def _env_int(
        self,
        name: str,
        default: int,
    ) -> int:
        raw_value = os.getenv(name)

        if raw_value is None:
            return default

        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return default

        if value <= 0:
            return default

        return value