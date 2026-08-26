from __future__ import annotations

import importlib.util
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = BASE_DIR / "evaluation" / "ranking" / "drive_filter_benchmark"
SCRIPT = BENCHMARK_DIR / "run_drive_filter_benchmark.py"
CASES = BENCHMARK_DIR / "filter_benchmark_cases.json"
CATALOG = BASE_DIR / "data" / "catalogue_drives_ready_final.json"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("drive_filter_benchmark", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_has_required_three_groups_for_both_roles() -> None:
    definition = json.loads(CASES.read_text(encoding="utf-8"))
    pairs = {(case["role"], case["group"]) for case in definition["cases"]}
    expected = {
        (role, group)
        for role in ("MDT", "OST")
        for group in ("clearly_feasible", "clearly_infeasible", "boundary")
    }
    assert pairs == expected


def test_benchmark_has_balanced_coverage() -> None:
    definition = json.loads(CASES.read_text(encoding="utf-8"))
    cases = definition["cases"]
    assert len(cases) == 24
    for role in ("MDT", "OST"):
        role_cases = [case for case in cases if case["role"] == role]
        assert len(role_cases) == 12
        for group in ("clearly_feasible", "clearly_infeasible", "boundary"):
            assert sum(case["group"] == group for case in role_cases) == 4


def test_filter_benchmark_is_fully_validated() -> None:
    module = load_benchmark_module()
    result = module.run_benchmark(CASES, CATALOG)
    metrics = result["metrics"]
    assert result["status"] == "VALIDATED"
    assert metrics["decision_accuracy_percent"] == 100.0
    assert metrics["false_feasible_rate_percent"] == 0.0
    assert metrics["false_infeasible_rate_percent"] == 0.0
    assert metrics["rejection_reason_accuracy_percent"] == 100.0


def test_ost_boundary_contract_550_mb_s_equals_point_55_gb_s() -> None:
    module = load_benchmark_module()
    definition = json.loads(CASES.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_id = {case["case_id"]: case for case in definition["cases"]}

    exact = module.evaluate_case(by_id["OST_B01_EXACT_THROUGHPUT"], catalog)
    above = module.evaluate_case(by_id["OST_B02_JUST_ABOVE_READ"], catalog)

    assert exact["predicted_feasible"] is True
    assert exact["feasible_drive_count"] == 1
    assert above["predicted_feasible"] is False
    assert above["rejection_counts"] == {"raw_ost_cost_exceeds_global_budget": 1}


def test_boundary_budget_is_inclusive_then_rejects_below_threshold() -> None:
    module = load_benchmark_module()
    definition = json.loads(CASES.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_id = {case["case_id"]: case for case in definition["cases"]}

    exact = module.evaluate_case(by_id["MDT_B01_EXACT_BUDGET"], catalog)
    below = module.evaluate_case(by_id["MDT_B02_JUST_BELOW_BUDGET"], catalog)

    assert exact["predicted_feasible"] is True
    assert below["predicted_feasible"] is False
    assert below["rejection_counts"] == {"raw_mdt_cost_exceeds_global_budget": 1}
