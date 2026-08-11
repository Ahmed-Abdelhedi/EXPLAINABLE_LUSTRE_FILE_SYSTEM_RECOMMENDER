from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


VALIDATION_DIR = Path(__file__).resolve().parent
REQUIREMENT_EXTRACTOR_DIR = VALIDATION_DIR.parent
REPOSITORY_ROOT = REQUIREMENT_EXTRACTOR_DIR.parent

DEFAULT_DATASET_PATH = (
    VALIDATION_DIR
    / "datasets"
    / "ai_plausibility"
    / "ai_plausibility_stress_dataset_v1.json"
)

DEFAULT_REPORTS_ROOT = VALIDATION_DIR / "reports" / "ai_plausibility"

FIELD_UNITS: Dict[str, Optional[str]] = {
    "requested_usable_capacity_tib": "TiB",
    "client_count": None,
    "average_file_size_gb": "GB",
    "max_file_size_gb": "GB",
    "total_file_count": None,
    "read_write_ratio": "%",
    "access_type": None,
    "target_read_gbps": "GB/s",
    "target_write_gbps": "GB/s",
    "ha_required": None,
    "max_budget_usd": "USD",
    "max_power_w": "W",
    "annual_growth_percent": "%",
}

EXPECTED_LABELS = {
    "COHERENT",
    "AMBIGUOUS",
    "INCOHERENT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exécute le dataset dédié à l'AI Plausibility Agent. "
            "Le test appelle directement l'agent sur des champs structurés "
            "et ne lance ni extraction, ni chatbot, ni calcul d'architecture."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Chemin du dataset AI Plausibility JSON.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Dossier de sortie des rapports.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre maximal de scénarios à exécuter.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Arrêter au premier échec technique.",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Surcharge facultative de OLLAMA_HOST.",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Surcharge facultative de PLAUSIBILITY_AGENT_MODEL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Surcharge facultative de PLAUSIBILITY_AGENT_TEMPERATURE.",
    )
    parser.add_argument(
        "--debug-agent",
        action="store_true",
        help="Active les traces internes de l'AI Plausibility Agent.",
    )
    parser.add_argument(
        "--omit-raw-response",
        action="store_true",
        help="N'enregistre pas la réponse brute du modèle dans le rapport.",
    )

    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> Dict[str, str]:
    """
    Charge le fichier .env puis active uniquement l'agent de plausibilité.

    L'import de l'agent est volontairement effectué après cette fonction,
    car sa configuration est lue lors de son initialisation.
    """

    load_dotenv(REPOSITORY_ROOT / ".env", override=False)

    os.environ["ENABLE_AI_PLAUSIBILITY_AGENT"] = "true"
    os.environ["ENABLE_LLM_FALLBACK"] = "false"

    if args.ollama_host:
        os.environ["OLLAMA_HOST"] = args.ollama_host

    if args.ollama_model:
        os.environ["PLAUSIBILITY_AGENT_MODEL"] = args.ollama_model

    if args.temperature is not None:
        os.environ["PLAUSIBILITY_AGENT_TEMPERATURE"] = str(
            args.temperature
        )

    if args.debug_agent:
        os.environ["PLAUSIBILITY_AGENT_DEBUG"] = "true"

    return {
        "ENABLE_AI_PLAUSIBILITY_AGENT": os.getenv(
            "ENABLE_AI_PLAUSIBILITY_AGENT",
            "",
        ),
        "ENABLE_LLM_FALLBACK": os.getenv(
            "ENABLE_LLM_FALLBACK",
            "",
        ),
        "OLLAMA_HOST": os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434",
        ),
        "PLAUSIBILITY_AGENT_MODEL": os.getenv(
            "PLAUSIBILITY_AGENT_MODEL",
            os.getenv("OLLAMA_MODEL", ""),
        ),
        "PLAUSIBILITY_AGENT_TEMPERATURE": os.getenv(
            "PLAUSIBILITY_AGENT_TEMPERATURE",
            "",
        ),
        "PLAUSIBILITY_AGENT_DEBUG": os.getenv(
            "PLAUSIBILITY_AGENT_DEBUG",
            "false",
        ),
    }


def load_dataset(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError(
            "La racine du dataset doit être un objet JSON."
        )

    scenarios = dataset.get("scenarios")

    if not isinstance(scenarios, list):
        raise ValueError(
            "Le dataset doit contenir une liste 'scenarios'."
        )

    declared_count = (
        dataset.get("metadata", {}).get("scenario_count")
    )

    if (
        declared_count is not None
        and int(declared_count) != len(scenarios)
    ):
        raise ValueError(
            "metadata.scenario_count ne correspond pas "
            "au nombre réel de scénarios."
        )

    seen_ids = set()

    for position, scenario in enumerate(scenarios, start=1):
        validate_scenario(
            scenario=scenario,
            position=position,
            seen_ids=seen_ids,
        )

    return dataset


def validate_scenario(
    scenario: Any,
    position: int,
    seen_ids: set[str],
) -> None:
    if not isinstance(scenario, dict):
        raise ValueError(
            f"Scénario {position} : objet JSON attendu."
        )

    scenario_id = scenario.get("id")

    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError(
            f"Scénario {position} : identifiant invalide."
        )

    if scenario_id in seen_ids:
        raise ValueError(
            f"Identifiant de scénario dupliqué : {scenario_id}"
        )

    seen_ids.add(scenario_id)

    label = scenario.get("label")

    if label not in EXPECTED_LABELS:
        raise ValueError(
            f"{scenario_id} : label invalide {label!r}."
        )

    fields = scenario.get("fields")

    if not isinstance(fields, dict):
        raise ValueError(
            f"{scenario_id} : 'fields' doit être un objet JSON."
        )

    missing_schema_fields = sorted(
        set(FIELD_UNITS) - set(fields)
    )
    extra_schema_fields = sorted(
        set(fields) - set(FIELD_UNITS)
    )

    if missing_schema_fields or extra_schema_fields:
        raise ValueError(
            f"{scenario_id} : schéma de champs invalide. "
            f"Manquants={missing_schema_fields}, "
            f"supplémentaires={extra_schema_fields}"
        )

    expected = scenario.get("expected")

    if not isinstance(expected, dict):
        raise ValueError(
            f"{scenario_id} : 'expected' doit être un objet JSON."
        )

    if expected.get("plausibility_status") != label:
        raise ValueError(
            f"{scenario_id} : expected.plausibility_status "
            "doit être identique à label."
        )

    for list_field in (
        "blocking_issues",
        "warnings",
        "checked_relations",
        "must_not_infer",
    ):
        if not isinstance(expected.get(list_field), list):
            raise ValueError(
                f"{scenario_id} : expected.{list_field} "
                "doit être une liste."
            )

    for field_name in expected["must_not_infer"]:
        if field_name not in FIELD_UNITS:
            raise ValueError(
                f"{scenario_id} : champ must_not_infer inconnu "
                f"{field_name!r}."
            )

        if fields.get(field_name) is not None:
            raise ValueError(
                f"{scenario_id} : {field_name} doit rester null "
                "car il figure dans must_not_infer."
            )


def make_final_json(
    plain_fields: Dict[str, Any],
    final_field_value_class: Any,
) -> Dict[str, Any]:
    """
    Convertit les valeurs simples du dataset vers le format attendu
    par AIPlausibilityAgent.analyze().

    Les champs null restent strictement à None.
    """

    final_json: Dict[str, Any] = {}

    for field_name in FIELD_UNITS:
        value = plain_fields.get(field_name)

        if value is None:
            final_json[field_name] = None
            continue

        final_json[field_name] = final_field_value_class(
            value=copy.deepcopy(value),
            unit=FIELD_UNITS[field_name],
            confidence=1.0,
            evidence="AI plausibility validation dataset",
            source="AI_PLAUSIBILITY_DATASET",
        )

    return final_json


def final_json_to_plain(
    final_json: Dict[str, Any],
) -> Dict[str, Any]:
    plain: Dict[str, Any] = {}

    for field_name in FIELD_UNITS:
        item = final_json.get(field_name)

        if item is None:
            plain[field_name] = None
            continue

        plain[field_name] = copy.deepcopy(
            getattr(item, "value", None)
        )

    return plain


def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def serialize_issue(issue: Any) -> Dict[str, Any]:
    field = enum_value(getattr(issue, "field", None))
    evidence_fields = copy.deepcopy(
        getattr(issue, "evidence_fields", {}) or {}
    )
    suggested_correction = copy.deepcopy(
        getattr(issue, "suggested_correction", None)
    )

    referenced_fields: List[str] = []

    if isinstance(field, str) and field:
        referenced_fields.append(field)

    if isinstance(evidence_fields, dict):
        for evidence_field in evidence_fields:
            if (
                evidence_field in FIELD_UNITS
                and evidence_field not in referenced_fields
            ):
                referenced_fields.append(evidence_field)

    return {
        "issue_type": str(
            getattr(issue, "issue_type", "")
        ),
        "field": field,
        "referenced_fields": referenced_fields,
        "severity": str(
            getattr(issue, "severity", "")
        ),
        "message": str(
            getattr(issue, "message", "")
        ),
        "question": str(
            getattr(issue, "question", "")
        ),
        "confidence": float(
            getattr(issue, "confidence", 0.0) or 0.0
        ),
        "suggested_correction": suggested_correction,
        "evidence_fields": evidence_fields,
    }


def predicted_label_from_report(
    report: Any,
    status_class: Any,
) -> str:
    status = str(enum_value(getattr(report, "status", ""))).upper()

    if status == str(status_class.OK).upper():
        return "COHERENT"

    if status == str(status_class.WARNING).upper():
        return "AMBIGUOUS"

    if status in {
        str(status_class.NEEDS_CLARIFICATION).upper(),
        str(status_class.BLOCKING).upper(),
    }:
        return "INCOHERENT"

    return "AMBIGUOUS"


def report_should_block(
    report: Any,
    status_class: Any,
) -> bool:
    status = str(enum_value(getattr(report, "status", ""))).upper()

    return status in {
        str(status_class.NEEDS_CLARIFICATION).upper(),
        str(status_class.BLOCKING).upper(),
    }


def detect_no_inference_violations(
    plain_input: Dict[str, Any],
    expected: Dict[str, Any],
    serialized_issues: List[Dict[str, Any]],
    input_mutated: bool,
) -> List[Dict[str, Any]]:
    """
    Détecte uniquement des violations observables :

    - mutation de l'entrée structurée ;
    - proposition d'une correction chiffrée pour un champ absent.

    Mentionner qu'un champ est absent dans un warning n'est pas une inférence.
    """

    violations: List[Dict[str, Any]] = []

    if input_mutated:
        violations.append(
            {
                "type": "INPUT_MUTATION",
                "field": None,
                "message": (
                    "L'agent a modifié l'objet final_json reçu en entrée."
                ),
            }
        )

    protected_missing_fields = set(
        expected.get("must_not_infer", [])
    )

    for issue in serialized_issues:
        field_name = issue.get("field")
        suggested = issue.get("suggested_correction")

        if (
            field_name in protected_missing_fields
            and plain_input.get(field_name) is None
            and isinstance(suggested, dict)
            and suggested.get("value") is not None
        ):
            violations.append(
                {
                    "type": "SUGGESTED_VALUE_FOR_MISSING_FIELD",
                    "field": field_name,
                    "message": (
                        "L'agent a proposé une valeur pour un champ "
                        "explicitement absent."
                    ),
                    "suggested_correction": suggested,
                }
            )

    return violations


def hidden_agent_failure(
    report: Any,
) -> Optional[Dict[str, str]]:
    """
    L'agent encapsule certaines erreurs Ollama dans un rapport WARNING.
    Le lanceur les requalifie en erreur technique pour ne pas fausser
    les métriques fonctionnelles.
    """

    raw_response = str(
        getattr(report, "raw_response", "") or ""
    )

    markers = (
        "AI plausibility agent unavailable:",
        "AI plausibility agent error:",
    )

    for marker in markers:
        if raw_response.startswith(marker):
            return {
                "type": "AIPlausibilityAgentUnavailable",
                "message": raw_response,
            }

    return None


def execute_scenario(
    agent: Any,
    status_class: Any,
    final_field_value_class: Any,
    scenario: Dict[str, Any],
    omit_raw_response: bool,
) -> Dict[str, Any]:
    scenario_id = scenario["id"]
    plain_fields = copy.deepcopy(scenario["fields"])
    expected = copy.deepcopy(scenario["expected"])

    final_json = make_final_json(
        plain_fields=plain_fields,
        final_field_value_class=final_field_value_class,
    )

    before_plain = final_json_to_plain(final_json)
    start = time.perf_counter()

    try:
        report = agent.analyze(final_json)
        latency_ms = (time.perf_counter() - start) * 1000.0

        hidden_failure = hidden_agent_failure(report)

        if hidden_failure is not None:
            return {
                "id": scenario_id,
                "label": scenario["label"],
                "category": scenario.get("category"),
                "difficulty": scenario.get("difficulty"),
                "language": scenario.get("language"),
                "user_text": scenario.get("user_text"),
                "fields": plain_fields,
                "derived_reference": scenario.get(
                    "derived_reference",
                    {},
                ),
                "expected": expected,
                "actual": None,
                "latency_ms": round(latency_ms, 3),
                "technical_error": hidden_failure,
            }

        after_plain = final_json_to_plain(final_json)
        input_mutated = before_plain != after_plain

        serialized_issues = [
            serialize_issue(issue)
            for issue in (getattr(report, "issues", None) or [])
        ]

        violations = detect_no_inference_violations(
            plain_input=before_plain,
            expected=expected,
            serialized_issues=serialized_issues,
            input_mutated=input_mutated,
        )

        actual = {
            "raw_status": str(
                enum_value(getattr(report, "status", ""))
            ),
            "predicted_label": predicted_label_from_report(
                report=report,
                status_class=status_class,
            ),
            "should_block_recommendation": report_should_block(
                report=report,
                status_class=status_class,
            ),
            "issues": serialized_issues,
            "issue_count": len(serialized_issues),
            "input_mutated": input_mutated,
            "input_after_analysis": after_plain,
            "no_inference_violations": violations,
        }

        if not omit_raw_response:
            actual["raw_response"] = str(
                getattr(report, "raw_response", "") or ""
            )

        return {
            "id": scenario_id,
            "label": scenario["label"],
            "category": scenario.get("category"),
            "difficulty": scenario.get("difficulty"),
            "language": scenario.get("language"),
            "user_text": scenario.get("user_text"),
            "fields": plain_fields,
            "derived_reference": scenario.get(
                "derived_reference",
                {},
            ),
            "expected": expected,
            "actual": actual,
            "latency_ms": round(latency_ms, 3),
            "technical_error": None,
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "id": scenario_id,
            "label": scenario["label"],
            "category": scenario.get("category"),
            "difficulty": scenario.get("difficulty"),
            "language": scenario.get("language"),
            "user_text": scenario.get("user_text"),
            "fields": plain_fields,
            "derived_reference": scenario.get(
                "derived_reference",
                {},
            ),
            "expected": expected,
            "actual": None,
            "latency_ms": round(latency_ms, 3),
            "technical_error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def run_validation(args: argparse.Namespace) -> Path:
    environment = configure_environment(args)

    # Imports tardifs : les variables d'environnement doivent être
    # positionnées avant l'initialisation du client Ollama.
    from requirement_extractor.ai_plausibility_agent import (
        AIPlausibilityAgent,
        AIPlausibilityStatus,
    )
    from requirement_extractor.models import FinalFieldValue

    dataset_path = args.dataset.resolve()
    dataset = load_dataset(dataset_path)
    scenarios: List[Dict[str, Any]] = dataset["scenarios"]

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit doit être strictement positif."
            )

        scenarios = scenarios[: args.limit]

    reports_root = args.reports_root.resolve()
    reports_root.mkdir(parents=True, exist_ok=True)

    results_path = (
        reports_root / "results_ai_plausibility.json"
    )

    agent = AIPlausibilityAgent()

    run_started = datetime.now(timezone.utc)
    global_start = time.perf_counter()

    results: List[Dict[str, Any]] = []
    technical_errors = 0
    label_counts: Dict[str, int] = {
        label: 0
        for label in sorted(EXPECTED_LABELS)
    }

    print("=" * 76)
    print("VALIDATION DE L'AI PLAUSIBILITY AGENT")
    print("=" * 76)
    print(f"Dataset    : {dataset_path}")
    print(f"Scénarios  : {len(scenarios)}")
    print(f"Modèle     : {environment['PLAUSIBILITY_AGENT_MODEL']}")
    print(f"Ollama     : {environment['OLLAMA_HOST']}")
    print(f"Rapport    : {results_path}")
    print("=" * 76)

    for position, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario["id"]
        expected_label = scenario["label"]
        label_counts[expected_label] += 1

        print(
            f"[{position:03d}/{len(scenarios):03d}] "
            f"{scenario_id} - "
            f"{scenario.get('category', 'unknown')} - "
            f"attendu={expected_label}"
        )

        result = execute_scenario(
            agent=agent,
            status_class=AIPlausibilityStatus,
            final_field_value_class=FinalFieldValue,
            scenario=scenario,
            omit_raw_response=args.omit_raw_response,
        )
        results.append(result)

        if result["technical_error"] is not None:
            technical_errors += 1
            error = result["technical_error"]

            print(
                f"  ERREUR TECHNIQUE : "
                f"{error['type']} - {error['message']}"
            )

            if args.fail_fast:
                break

        else:
            actual = result["actual"]
            predicted = actual["predicted_label"]
            issue_count = actual["issue_count"]
            violations = len(
                actual["no_inference_violations"]
            )

            print(
                f"  prédit={predicted} | "
                f"statut={actual['raw_status']} | "
                f"issues={issue_count} | "
                f"no-inference violations={violations} | "
                f"{result['latency_ms']:.3f} ms"
            )

    total_duration_s = time.perf_counter() - global_start
    latencies = [
        float(result["latency_ms"])
        for result in results
    ]

    average_latency_ms = (
        sum(latencies) / len(latencies)
        if latencies
        else 0.0
    )

    final_report = {
        "run": {
            "started_at_utc": run_started.isoformat(),
            "finished_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "environment": environment,
            "repository_root": str(REPOSITORY_ROOT),
            "dataset_path": str(dataset_path),
            "results_path": str(results_path),
        },
        "dataset_metadata": dataset.get("metadata", {}),
        "execution_summary": {
            "requested_scenarios": len(scenarios),
            "executed_scenarios": len(results),
            "technical_errors": technical_errors,
            "expected_label_distribution": label_counts,
            "total_duration_seconds": round(
                total_duration_s,
                3,
            ),
            "average_scenario_latency_ms": round(
                average_latency_ms,
                3,
            ),
        },
        "scenarios": results,
    }

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(
            final_report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 76)
    print("EXÉCUTION TERMINÉE")
    print("=" * 76)
    print(f"Scénarios exécutés : {len(results)}")
    print(f"Erreurs techniques : {technical_errors}")
    print(f"Durée totale       : {total_duration_s:.3f} s")
    print(f"Latence moyenne    : {average_latency_ms:.3f} ms")
    print(f"Résultats          : {results_path}")

    return results_path


def main() -> int:
    args = parse_args()

    try:
        run_validation(args)
        return 0

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        print(
            f"ERREUR DE VALIDATION : {exc}",
            file=sys.stderr,
        )
        return 2

    except KeyboardInterrupt:
        print(
            "\nExécution interrompue par l'utilisateur.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
