#!/usr/bin/env python3
"""
plausibility_full_agent_error_analyzer.py

Analyse les erreurs du benchmark final de l'AI Plausibility Agent complet (guard déterministe + LLM enrichment contrôlé).

Entrée par défaut :
    requirement_extractor/validation/reports/ai_plausibility/full_agent/
        results_ai_plausibility_full_agent.json

Sortie par défaut :
    requirement_extractor/validation/reports/ai_plausibility/full_agent/
        errors_ai_plausibility_full_agent.json

Le script détecte :
- mauvais label de plausibilité ;
- mauvais should_block_recommendation ;
- blocking issue manquante ;
- blocking issue supplémentaire ;
- warning manquant ;
- warning supplémentaire ;
- mauvaise référence de champ attendue ;
- mutation de l'entrée ;
- violation de non-inférence ;
- erreur technique.

Important :
Les champs contextuels supplémentaires dans evidence_fields ne sont PAS
considérés comme une erreur tant que tous les champs attendus sont référencés.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


BLOCKING_SEVERITIES = {"BLOCKING", "ERROR", "CRITICAL"}
WARNING_SEVERITIES = {"WARNING", "WARN"}


def normalize(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_upper(value: Any) -> str:
    return normalize(value).upper()


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Le JSON racine doit être un objet.")

    if not isinstance(data.get("scenarios"), list):
        raise ValueError("La clé 'scenarios' doit être une liste.")

    return data


def expected_label(scenario: Dict[str, Any]) -> str:
    expected = scenario.get("expected", {})
    return normalize_upper(
        expected.get("plausibility_status", scenario.get("label"))
    )


def scenario_actual(
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    value = scenario.get("actual", {})
    return value if isinstance(value, dict) else {}


def scenario_technical_error(
    scenario: Dict[str, Any],
) -> Any:
    return scenario_actual(
        scenario
    ).get("technical_error")


def scenario_enrichment_error(
    scenario: Dict[str, Any],
) -> Any:
    return scenario_actual(
        scenario
    ).get("enrichment_error")


def predicted_label(scenario: Dict[str, Any]) -> str:
    if scenario_technical_error(
        scenario
    ) not in (None, "", False):
        return "__TECHNICAL_ERROR__"

    return normalize_upper(
        scenario_actual(
            scenario
        ).get("predicted_label")
    )


def expected_should_block(scenario: Dict[str, Any]) -> bool:
    return bool(
        scenario.get("expected", {}).get(
            "should_block_recommendation", False
        )
    )


def actual_should_block(scenario: Dict[str, Any]) -> bool:
    return bool(
        scenario_actual(scenario).get(
            "should_block_recommendation", False
        )
    )


def expected_issues(
    scenario: Dict[str, Any],
    kind: str,
) -> List[Dict[str, Any]]:
    expected = scenario.get("expected", {})

    key = "blocking_issues" if kind == "blocking" else "warnings"

    return [
        item for item in as_list(expected.get(key))
        if isinstance(item, dict)
    ]


def actual_issues(
    scenario: Dict[str, Any],
    kind: str,
) -> List[Dict[str, Any]]:
    issues = [
        item
        for item in as_list(
            scenario_actual(scenario).get("issues")
        )
        if isinstance(item, dict)
    ]

    selected = []

    for item in issues:
        severity = normalize_upper(item.get("severity"))

        if kind == "blocking" and severity in BLOCKING_SEVERITIES:
            selected.append(item)

        if kind == "warning" and severity in WARNING_SEVERITIES:
            selected.append(item)

    return selected


def expected_code(issue: Dict[str, Any]) -> str:
    return normalize_upper(issue.get("code"))


def actual_code(issue: Dict[str, Any]) -> str:
    return normalize_upper(
        issue.get("issue_type", issue.get("code"))
    )


def expected_fields(issue: Dict[str, Any]) -> List[str]:
    return [
        normalize(field)
        for field in as_list(issue.get("fields"))
        if normalize(field)
    ]


def actual_referenced_fields(issue: Dict[str, Any]) -> List[str]:
    fields = set()

    direct_field = normalize(issue.get("field"))
    if direct_field:
        fields.add(direct_field)

    for field in as_list(issue.get("referenced_fields")):
        field = normalize(field)
        if field:
            fields.add(field)

    evidence = issue.get("evidence_fields")

    if isinstance(evidence, dict):
        ignored_context_keys = {
            "missing_fields",
            "ratio_sum",
            "capacity_gb",
            "estimated_dataset_volume_tib",
            "estimated_volume_to_capacity_ratio",
            "blocking_ratio",
            "total_throughput_gbps",
            "throughput_per_tib_gbps",
            "clients_per_tib",
            "budget_per_tib_usd",
            "power_per_tib_w",
        }

        for key in evidence:
            if key not in ignored_context_keys:
                key = normalize(key)
                if key:
                    fields.add(key)

        for field in as_list(evidence.get("missing_fields")):
            field = normalize(field)
            if field:
                fields.add(field)

    return sorted(fields)


def pair_by_code(
    expected: Sequence[Dict[str, Any]],
    actual: Sequence[Dict[str, Any]],
) -> Tuple[
    List[Tuple[Dict[str, Any], Dict[str, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Associe les issues attendues/réelles par code.
    En cas de doublon, choisit le meilleur recouvrement de champs.
    """

    unused_actual = list(actual)
    matched = []
    missing = []

    for exp in expected:
        code = expected_code(exp)
        exp_fields = set(expected_fields(exp))

        candidate_indexes = [
            i for i, act in enumerate(unused_actual)
            if actual_code(act) == code
        ]

        if not candidate_indexes:
            missing.append(exp)
            continue

        def score(i: int):
            act_fields = set(actual_referenced_fields(unused_actual[i]))
            overlap = len(exp_fields & act_fields)
            return overlap

        best_index = max(candidate_indexes, key=score)
        act = unused_actual.pop(best_index)

        matched.append((exp, act))

    extras = unused_actual
    return matched, missing, extras


def add_error(
    errors: List[Dict[str, Any]],
    scenario: Dict[str, Any],
    error_type: str,
    details: Dict[str, Any],
) -> None:
    errors.append({
        "scenario_id": scenario.get("id"),
        "label": scenario.get("label"),
        "category": scenario.get("category"),
        "difficulty": scenario.get("difficulty"),
        "language": scenario.get("language"),
        "error_type": error_type,
        "details": details,
    })


def analyze_scenario(
    scenario: Dict[str, Any]
) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []

    actual = scenario_actual(
        scenario
    )

    technical_error = scenario_technical_error(
        scenario
    )

    if technical_error not in (
        None,
        "",
        False,
    ):
        add_error(
            errors,
            scenario,
            "technical_error",
            {
                "technical_error": technical_error,
            },
        )
        return errors

    exp_label = expected_label(
        scenario
    )
    act_label = predicted_label(
        scenario
    )

    if exp_label != act_label:
        add_error(
            errors,
            scenario,
            "wrong_plausibility_label",
            {
                "expected": exp_label,
                "actual": act_label,
            },
        )

    exp_block = expected_should_block(
        scenario
    )
    act_block = actual_should_block(
        scenario
    )

    if exp_block != act_block:
        add_error(
            errors,
            scenario,
            "wrong_should_block_recommendation",
            {
                "expected": exp_block,
                "actual": act_block,
            },
        )

    for kind in (
        "blocking",
        "warning",
    ):
        exp_issues = expected_issues(
            scenario,
            kind,
        )
        act_issues = actual_issues(
            scenario,
            kind,
        )

        matched, missing, extras = pair_by_code(
            exp_issues,
            act_issues,
        )

        for issue in missing:
            add_error(
                errors,
                scenario,
                f"missing_{kind}_issue",
                {
                    "expected_code": expected_code(
                        issue
                    ),
                    "expected_fields": expected_fields(
                        issue
                    ),
                },
            )

        for issue in extras:
            add_error(
                errors,
                scenario,
                f"extra_{kind}_issue",
                {
                    "actual_code": actual_code(
                        issue
                    ),
                    "actual_field": issue.get(
                        "field"
                    ),
                    "actual_referenced_fields": (
                        actual_referenced_fields(
                            issue
                        )
                    ),
                },
            )

        for exp_issue, act_issue in matched:
            exp_fields = set(
                expected_fields(
                    exp_issue
                )
            )
            act_fields = set(
                actual_referenced_fields(
                    act_issue
                )
            )

            missing_fields = sorted(
                exp_fields - act_fields
            )

            if missing_fields:
                add_error(
                    errors,
                    scenario,
                    (
                        "missing_expected_"
                        "issue_field_reference"
                    ),
                    {
                        "issue_code": expected_code(
                            exp_issue
                        ),
                        "expected_fields": sorted(
                            exp_fields
                        ),
                        "actual_referenced_fields": sorted(
                            act_fields
                        ),
                        "missing_fields": (
                            missing_fields
                        ),
                    },
                )

    if bool(
        actual.get(
            "input_mutated",
            False,
        )
    ):
        add_error(
            errors,
            scenario,
            "input_mutation",
            {
                "input_after_analysis": (
                    actual.get(
                        "input_after_analysis"
                    )
                )
            },
        )

    violations = as_list(
        actual.get(
            "no_inference_violations"
        )
    )

    if violations:
        add_error(
            errors,
            scenario,
            "no_inference_violation",
            {
                "violations": violations,
            },
        )

    enrichment_error = (
        scenario_enrichment_error(
            scenario
        )
    )

    if enrichment_error not in (
        None,
        "",
        False,
    ):
        add_error(
            errors,
            scenario,
            "llm_enrichment_error",
            {
                "enrichment_error": (
                    enrichment_error
                ),
            },
        )

    # --------------------------------------------------------
    # LLM routing checks
    # --------------------------------------------------------
    expected_llm_path = (
        exp_label == "AMBIGUOUS"
    )

    reported_llm_path = bool(
        actual.get(
            "llm_enrichment_path_expected",
            False,
        )
    )

    observed_llm_output = bool(
        actual.get(
            "llm_output_observed",
            False,
        )
    )

    if (
        expected_llm_path
        != reported_llm_path
    ):
        add_error(
            errors,
            scenario,
            "wrong_llm_routing",
            {
                "expected_llm_path": (
                    expected_llm_path
                ),
                "reported_llm_path": (
                    reported_llm_path
                ),
            },
        )

    if (
        not expected_llm_path
        and observed_llm_output
    ):
        add_error(
            errors,
            scenario,
            "unexpected_llm_output",
            {
                "expected_label": exp_label,
                "observed_llm_output": (
                    observed_llm_output
                ),
            },
        )

    if (
        expected_llm_path
        and reported_llm_path
        and not observed_llm_output
        and enrichment_error
        in (
            None,
            "",
            False,
        )
    ):
        add_error(
            errors,
            scenario,
            "missing_expected_llm_output",
            {
                "expected_label": exp_label,
                "reported_llm_path": (
                    reported_llm_path
                ),
                "observed_llm_output": (
                    observed_llm_output
                ),
            },
        )

    return errors


def compute_error_report(data: Dict[str, Any]) -> Dict[str, Any]:
    scenarios = [
        s for s in data.get("scenarios", [])
        if isinstance(s, dict)
    ]

    all_errors: List[Dict[str, Any]] = []
    scenario_error_counts = Counter()

    for scenario in scenarios:
        scenario_errors = analyze_scenario(scenario)

        all_errors.extend(scenario_errors)

        if scenario_errors:
            scenario_error_counts[
                normalize(scenario.get("id"))
            ] = len(scenario_errors)

    by_type = Counter(
        error["error_type"]
        for error in all_errors
    )

    by_category = Counter(
        error.get("category")
        for error in all_errors
    )

    by_language = Counter(
        error.get("language")
        for error in all_errors
    )

    by_difficulty = Counter(
        error.get("difficulty")
        for error in all_errors
    )

    failed_scenarios = len(scenario_error_counts)

    return {
        "analyzer_version": "1.0.0",
        "evaluation_scope": "AI Plausibility full-agent benchmark with LLM enrichment enabled",
        "summary": {
            "scenario_count": len(scenarios),
            "successful_scenarios": len(scenarios) - failed_scenarios,
            "failed_scenarios": failed_scenarios,
            "total_detected_errors": len(all_errors),
            "scenario_success_rate": (
                (len(scenarios) - failed_scenarios) / len(scenarios)
                if scenarios
                else 0.0
            ),
        },
        "error_counts_by_type": dict(sorted(by_type.items())),
        "error_counts_by_category": dict(sorted(by_category.items())),
        "error_counts_by_language": dict(sorted(by_language.items())),
        "error_counts_by_difficulty": dict(sorted(by_difficulty.items())),
        "scenario_error_counts": dict(sorted(scenario_error_counts.items())),
        "errors": all_errors,
    }


def print_report(report: Dict[str, Any]) -> None:
    summary = report["summary"]

    print()
    print("=" * 72)
    print("AI PLAUSIBILITY FULL AGENT - ERROR ANALYZER")
    print("=" * 72)
    print(f"Scenarios              : {summary['scenario_count']}")
    print(f"Successful scenarios   : {summary['successful_scenarios']}")
    print(f"Failed scenarios       : {summary['failed_scenarios']}")
    print(f"Total detected errors  : {summary['total_detected_errors']}")
    print(
        "Scenario success rate  : "
        f"{100.0 * summary['scenario_success_rate']:.3f}%"
    )

    if report["error_counts_by_type"]:
        print()
        print("--- Errors by type ---")
        for key, value in report["error_counts_by_type"].items():
            print(f"{key:42s}: {value}")
    else:
        print()
        print("No functional errors detected.")

    print("=" * 72)
    print()


def repository_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_path() -> Path:
    root = repository_root_from_script()

    return (
        root
        / "requirement_extractor"
        / "validation"
        / "reports"
        / "ai_plausibility"
        / "full_agent"
        / "results_ai_plausibility_full_agent.json"
    )


def default_output_path() -> Path:
    root = repository_root_from_script()

    return (
        root
        / "requirement_extractor"
        / "validation"
        / "reports"
        / "ai_plausibility"
        / "full_agent"
        / "errors_ai_plausibility_full_agent.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse les erreurs du benchmark AI Plausibility."
        )
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Chemin vers results_ai_plausibility_full_agent.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Chemin vers errors_ai_plausibility_full_agent.json",
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help="N'écrit pas le JSON de sortie.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_path = (
        args.results.resolve()
        if args.results is not None
        else default_results_path()
    )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else default_output_path()
    )

    data = load_json(results_path)
    report = compute_error_report(data)

    print_report(report)

    if not args.no_write:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                indent=2,
            )

        print("Error analysis JSON written to:")
        print(output_path)


if __name__ == "__main__":
    main()
