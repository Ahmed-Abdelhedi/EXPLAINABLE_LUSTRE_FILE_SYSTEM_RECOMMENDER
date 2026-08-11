from __future__ import annotations

import argparse
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
    VALIDATION_DIR / "datasets" / "stress_requests_v1.json"
)
DEFAULT_REPORTS_ROOT = VALIDATION_DIR / "reports"

SUPPORTED_MODES = {
    "deterministic": {
        "ENABLE_LLM_FALLBACK": "false",
        "ENABLE_AI_PLAUSIBILITY_AGENT": "false",
    },
    "llm_fallback": {
        "ENABLE_LLM_FALLBACK": "true",
        "ENABLE_AI_PLAUSIBILITY_AGENT": "false",
    },
    "hybrid": {
        "ENABLE_LLM_FALLBACK": "true",
        "ENABLE_AI_PLAUSIBILITY_AGENT": "true",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exécute le dataset de validation du Requirement Extractor "
            "et enregistre les sorties réelles sans modifier la logique métier."
        )
    )

    parser.add_argument(
        "--mode",
        choices=sorted(SUPPORTED_MODES),
        default="deterministic",
        help=(
            "Configuration à tester : deterministic, llm_fallback ou hybrid."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Chemin du dataset JSON.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Dossier racine des rapports.",
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
        help="Surcharge facultative de OLLAMA_MODEL.",
    )

    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> Dict[str, str]:
    """
    Charge .env puis applique explicitement le mode demandé.

    L'import de RequirementChatbot est effectué seulement après cette étape,
    car les composants LLM lisent les variables d'environnement à leur
    initialisation.
    """

    load_dotenv(REPOSITORY_ROOT / ".env", override=False)

    mode_config = SUPPORTED_MODES[args.mode]

    for name, value in mode_config.items():
        os.environ[name] = value

    if args.ollama_host:
        os.environ["OLLAMA_HOST"] = args.ollama_host

    if args.ollama_model:
        os.environ["OLLAMA_MODEL"] = args.ollama_model

    return {
        "mode": args.mode,
        "ENABLE_LLM_FALLBACK": os.environ["ENABLE_LLM_FALLBACK"],
        "ENABLE_AI_PLAUSIBILITY_AGENT": os.environ[
            "ENABLE_AI_PLAUSIBILITY_AGENT"
        ],
        "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
    }


def load_dataset(dataset_path: Path) -> Dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError("Le dataset doit être un objet JSON.")

    scenarios = dataset.get("scenarios")

    if not isinstance(scenarios, list):
        raise ValueError(
            "Le dataset doit contenir une liste sous la clé 'scenarios'."
        )

    seen_ids = set()

    for scenario_index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            raise ValueError(
                f"Le scénario {scenario_index} doit être un objet JSON."
            )

        scenario_id = scenario.get("id")

        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError(
                f"Le scénario {scenario_index} possède un id invalide."
            )

        if scenario_id in seen_ids:
            raise ValueError(f"Identifiant dupliqué : {scenario_id}")

        seen_ids.add(scenario_id)

        turns = scenario.get("turns")

        if not isinstance(turns, list) or not turns:
            raise ValueError(
                f"Le scénario {scenario_id} doit contenir au moins un tour."
            )

        for turn_index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                raise ValueError(
                    f"{scenario_id}, tour {turn_index} : objet JSON attendu."
                )

            if not isinstance(turn.get("user_text"), str):
                raise ValueError(
                    f"{scenario_id}, tour {turn_index} : "
                    "'user_text' doit être une chaîne."
                )

            if not isinstance(turn.get("expected"), dict):
                raise ValueError(
                    f"{scenario_id}, tour {turn_index} : "
                    "'expected' doit être un objet JSON."
                )

    return dataset


def enum_to_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value

    return value


def issue_field(issue: Any) -> Optional[str]:
    field = getattr(issue, "field", None)

    if field is None:
        return None

    return str(enum_to_value(field))


def infer_active_question_field(state: Any) -> Optional[str]:
    """
    Le chatbot place la question active dans le premier ValidationIssue.
    Cette information est enregistrée pour la future évaluation des
    clarifications.
    """

    questions = getattr(state, "questions", None) or []
    issues = getattr(state, "issues", None) or []

    if not questions or not issues:
        return None

    return issue_field(issues[0])


def state_to_dict(state: Any) -> Dict[str, Any]:
    if hasattr(state, "to_dict"):
        data = state.to_dict()

        if isinstance(data, dict):
            return data

    raise TypeError(
        "RequirementState doit fournir une méthode to_dict() retournant un dict."
    )


def build_actual_output(state: Any) -> Dict[str, Any]:
    state_data = state_to_dict(state)

    return {
        "status": state_data.get("status"),
        "stage": state_data.get("stage"),
        "fields": state_data.get("plain_final_json", {}),
        "missing_fields": state_data.get("missing_fields", []),
        "conflicting_fields": state_data.get("conflicting_fields", []),
        "invalid_fields": state_data.get("invalid_fields", []),
        "unsupported_fields": state_data.get("unsupported_fields", []),
        "question_field": infer_active_question_field(state),
        "questions": state_data.get("questions", []),
        "issues": state_data.get("issues", []),
        "calculation": state_data.get("calculation_result", {}),
        "extracted_candidates": state_data.get("extracted_candidates", []),
    }


def execute_turn(
    bot: Any,
    scenario_id: str,
    turn_index: int,
    turn: Dict[str, Any],
) -> Dict[str, Any]:
    user_text = turn["user_text"]
    expected = turn["expected"]

    start = time.perf_counter()

    try:
        state = bot.process_user_message(user_text)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "scenario_id": scenario_id,
            "turn_index": turn_index,
            "user_text": user_text,
            "expected": expected,
            "actual": build_actual_output(state),
            "latency_ms": round(latency_ms, 3),
            "technical_error": None,
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "scenario_id": scenario_id,
            "turn_index": turn_index,
            "user_text": user_text,
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

    # Import volontairement tardif : les variables d'environnement doivent
    # être configurées avant l'initialisation des composants LLM.
    from requirement_extractor.requirement_chatbot import RequirementChatbot

    dataset_path = args.dataset.resolve()
    dataset = load_dataset(dataset_path)
    scenarios: List[Dict[str, Any]] = dataset["scenarios"]

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit doit être strictement positif.")

        scenarios = scenarios[: args.limit]

    output_dir = args.reports_root.resolve() / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"results_{args.mode}.json"

    run_started = datetime.now(timezone.utc)
    global_start = time.perf_counter()

    scenario_results: List[Dict[str, Any]] = []
    technical_error_count = 0
    executed_turn_count = 0

    print("=" * 72)
    print("VALIDATION DU REQUIREMENT EXTRACTOR")
    print("=" * 72)
    print(f"Mode       : {args.mode}")
    print(f"Dataset    : {dataset_path}")
    print(f"Scénarios  : {len(scenarios)}")
    print(f"Rapport    : {results_path}")
    print("=" * 72)

    for scenario_position, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario["id"]
        bot = RequirementChatbot()
        turn_results: List[Dict[str, Any]] = []

        print(
            f"[{scenario_position:03d}/{len(scenarios):03d}] "
            f"{scenario_id} - {scenario.get('category', 'unknown')}"
        )

        for turn_index, turn in enumerate(scenario["turns"], start=1):
            turn_result = execute_turn(
                bot=bot,
                scenario_id=scenario_id,
                turn_index=turn_index,
                turn=turn,
            )

            executed_turn_count += 1
            turn_results.append(turn_result)

            if turn_result["technical_error"] is not None:
                technical_error_count += 1
                error = turn_result["technical_error"]

                print(
                    f"  Tour {turn_index}: ERREUR TECHNIQUE "
                    f"{error['type']} - {error['message']}"
                )

                if args.fail_fast:
                    break

        scenario_results.append(
            {
                "id": scenario_id,
                "category": scenario.get("category"),
                "difficulty": scenario.get("difficulty"),
                "language": scenario.get("language"),
                "tags": scenario.get("tags", []),
                "description": scenario.get("description"),
                "turns": turn_results,
                "technical_error_count": sum(
                    1
                    for item in turn_results
                    if item["technical_error"] is not None
                ),
            }
        )

        if args.fail_fast and technical_error_count:
            break

    total_duration_s = time.perf_counter() - global_start
    latencies = [
        turn["latency_ms"]
        for scenario in scenario_results
        for turn in scenario["turns"]
    ]

    average_latency_ms = (
        sum(latencies) / len(latencies)
        if latencies
        else 0.0
    )

    final_report = {
        "run": {
            "started_at_utc": run_started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "environment": environment,
            "repository_root": str(REPOSITORY_ROOT),
            "dataset_path": str(dataset_path),
            "results_path": str(results_path),
        },
        "dataset_metadata": dataset.get("metadata", {}),
        "execution_summary": {
            "requested_scenarios": len(scenarios),
            "executed_scenarios": len(scenario_results),
            "executed_turns": executed_turn_count,
            "technical_errors": technical_error_count,
            "total_duration_seconds": round(total_duration_s, 3),
            "average_turn_latency_ms": round(average_latency_ms, 3),
        },
        "scenarios": scenario_results,
    }

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(
            final_report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 72)
    print("EXÉCUTION TERMINÉE")
    print("=" * 72)
    print(f"Scénarios exécutés : {len(scenario_results)}")
    print(f"Tours exécutés     : {executed_turn_count}")
    print(f"Erreurs techniques : {technical_error_count}")
    print(f"Durée totale       : {total_duration_s:.3f} s")
    print(f"Latence moyenne    : {average_latency_ms:.3f} ms")
    print(f"Résultats          : {results_path}")

    return results_path


def main() -> int:
    args = parse_args()

    try:
        run_validation(args)
        return 0

    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERREUR DE VALIDATION : {exc}", file=sys.stderr)
        return 2

    except KeyboardInterrupt:
        print("\nExécution interrompue par l'utilisateur.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
