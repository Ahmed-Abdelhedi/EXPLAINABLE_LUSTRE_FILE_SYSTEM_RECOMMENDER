"""CLI wiring tests; Requirement conversation and downstream are substituted."""

import sys
import pytest

import main as cli
from e2e_pipeline import PipelineLimits, SearchOptions


def test_cli_defaults_to_beam(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])
    args = cli.parse_args()
    assert args.search_strategy == "beam"
    assert args.beam_width == 8
    assert args.top_k == 10
    assert args.max_paths_per_variant == 2
    assert args.max_role_options == 4
    assert args.max_architectures == 16


@pytest.mark.parametrize("strategy", ["beam", "exhaustive"])
def test_cli_propagates_strategy_width_limits_and_prints_the_correct_route(tmp_path, monkeypatch, capsys, strategy):
    requirement = tmp_path / "requirement.json"
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--device", "cpu", "--no-llm", "--no-auto-start-ollama",
        "--output", str(requirement), "--e2e-output", str(output),
        "--search-strategy", strategy, "--beam-width", "13", "--top-k", "7",
        "--max-paths-per-variant", "3", "--max-role-options", "9", "--max-architectures", "99",
    ])
    production_calls, downstream_calls = [], []
    def production(**kwargs):
        production_calls.append(kwargs)
        kwargs["output_path"].write_text('{"new_requirement": true}', encoding="utf-8")
        return 0
    def downstream(path, **kwargs):
        downstream_calls.append((path, kwargs))
        return {
            "status": "SUCCESS", "trace": {"output_path": str(output)},
            "best_architecture": {"architecture_id": "TEST_VALID", "score": {"score": 0.8}},
        }
    monkeypatch.setattr(cli, "run_production", production)
    monkeypatch.setattr(cli, "run_e2e_from_file", downstream)

    assert cli.main() == 0
    assert production_calls == [{"device": "cpu", "enable_llm_fallback": False, "auto_start_ollama": False, "output_path": requirement}]
    assert downstream_calls == [(requirement, {
        "output_path": output,
        "limits": PipelineLimits(top_k=7, max_paths_per_variant=3, max_role_options_per_role=9, max_architectures=99),
        "search_options": SearchOptions(strategy=strategy, beam_width=13),
    })]
    printed = capsys.readouterr().out
    route = "Beam V2.1" if strategy == "beam" else "H8"
    assert f"Requirement -> S10 -> Ranking -> {route} -> H9 -> H10" in printed
    assert f"[SEARCH STRATEGY] {strategy}" in printed
    assert ("[BEAM WIDTH] 13" in printed) is (strategy == "beam")
    assert "[BEST ARCHITECTURE] TEST_VALID" in printed
    assert "[H9 SCORE] 0.800000" in printed
    assert "[H10 VALID] true" in printed


@pytest.mark.parametrize("status,expected_exit", [("NO_VALID_ARCHITECTURE_WITHIN_SEARCH_DOMAIN", 0), ("PIPELINE_ERROR", 4)])
def test_cli_preserves_engineering_outcome_vs_runtime_error_exit_codes(tmp_path, monkeypatch, status, expected_exit):
    path = tmp_path / "requirement.json"
    monkeypatch.setattr(sys, "argv", ["main.py", "--output", str(path)])
    def production(**kwargs):
        path.write_text("{}", encoding="utf-8")
        return 0
    def downstream(*args, **kwargs):
        assert kwargs["search_options"] == SearchOptions()
        return {"status": status, "failure_stage": "BEAM_SEARCH", "message": "test", "trace": {}}
    monkeypatch.setattr(cli, "run_production", production)
    monkeypatch.setattr(cli, "run_e2e_from_file", downstream)
    assert cli.main() == expected_exit


@pytest.mark.parametrize("existing", [False, True])
def test_cli_never_consumes_a_stale_requirement_after_quit(tmp_path, monkeypatch, existing):
    path = tmp_path / "requirement.json"
    if existing:
        path.write_text('{"old": true}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["main.py", "--output", str(path)])
    monkeypatch.setattr(cli, "run_production", lambda **kwargs: 0)
    def unexpected(*args, **kwargs):
        pytest.fail("Downstream must not consume a missing or stale Requirement.")
    monkeypatch.setattr(cli, "run_e2e_from_file", unexpected)
    assert cli.main() == 0


@pytest.mark.parametrize("requirement_only,requirement_status", [(True, 0), (False, 3)])
def test_cli_preserves_requirement_only_and_upstream_failure(tmp_path, monkeypatch, requirement_only, requirement_status):
    path = tmp_path / "requirement.json"
    argv = ["main.py", "--output", str(path)]
    if requirement_only:
        argv.append("--requirement-only")
    monkeypatch.setattr(sys, "argv", argv)
    def production(**kwargs):
        path.write_text("{}", encoding="utf-8")
        return requirement_status
    monkeypatch.setattr(cli, "run_production", production)
    def unexpected(*args, **kwargs):
        pytest.fail("Downstream must not run here.")
    monkeypatch.setattr(cli, "run_e2e_from_file", unexpected)
    assert cli.main() == requirement_status


def test_cli_validates_beam_width_before_starting_the_conversation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--beam-width", "0"])
    def unexpected(**kwargs):
        pytest.fail("Invalid search options must be rejected before conversation.")
    monkeypatch.setattr(cli, "run_production", unexpected)
    with pytest.raises(ValueError, match="beam_width"):
        cli.main()


def test_cli_rejects_unknown_search_strategy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--search-strategy", "unknown"])
    with pytest.raises(SystemExit) as error:
        cli.parse_args()
    assert error.value.code == 2
