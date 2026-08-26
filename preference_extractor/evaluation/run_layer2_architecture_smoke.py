from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import zipfile
from pathlib import Path
from typing import Any, Dict

from preference_extractor.evaluation.layer2_fallback_common import (
    EXPECTED_ARTIFACT_SHA256,
    parse_llm_response,
    sha256_file,
)
from preference_extractor.evaluation.layer2_hybrid_guard import (
    GUARD_VERSION,
    Layer2DeterministicSemanticGuard,
)
from preference_extractor.evaluation.layer2_residual_validator import (
    RESIDUAL_VALIDATOR_VERSION,
    validate_residual_prediction,
)
from preference_extractor.evaluation.run_layer2_hybrid_fallback_benchmark import (
    HYBRID_PROMPT,
    HYBRID_PROMPT_VERSION,
    _call_ollama,
    _prompt,
)


EXPECTED_FREEZE_STATUS = (
    "LAYER2_HYBRID_GUARD_RESIDUAL_VALIDATOR_LLM_FROZEN_BEFORE_TEST"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _decision_dict(decision) -> Dict[str, Any] | None:
    return None if decision is None else decision.to_dict()


def run_contract_smoke() -> Dict[str, Any]:
    checks = []
    guard = Layer2DeterministicSemanticGuard()

    def check(name: str, condition: bool, details: Any = None) -> None:
        _require(condition, name)
        checks.append(
            {
                "name": name,
                "status": "PASS",
                "details": details,
            }
        )

    # 1) Absolute EN.
    d = guard.resolve_dimension(
        text="Cost is a top priority for this design.",
        dimension="cost",
    )
    check(
        "guard_absolute_en",
        d is not None
        and d.status == "RESOLVED"
        and d.level == "VERY_HIGH",
        _decision_dict(d),
    )

    # 2) Absolute FR.
    d = guard.resolve_dimension(
        text=(
            "La consommation électrique est souhaitable "
            "sans être prioritaire."
        ),
        dimension="power",
    )
    check(
        "guard_absolute_fr",
        d is not None
        and d.status == "RESOLVED"
        and d.level == "LOW",
        _decision_dict(d),
    )

    # 3) Comparison must stay relative: no invented absolute intensity.
    text = "Performance is more important than cost."
    perf = guard.resolve_dimension(text=text, dimension="performance")
    cost = guard.resolve_dimension(text=text, dimension="cost")
    check(
        "guard_comparison_relative_only",
        perf is not None
        and cost is not None
        and perf.status == "RELATIVE_ONLY"
        and cost.status == "RELATIVE_ONLY"
        and perf.level is None
        and cost.level is None,
        {
            "performance": _decision_dict(perf),
            "cost": _decision_dict(cost),
        },
    )

    # 4) Hard negative must not become VERY_LOW.
    d = guard.resolve_dimension(
        text="No preference; the system supports 10 GB/s throughput.",
        dimension="performance",
    )
    check(
        "guard_no_signal_not_very_low",
        d is not None
        and d.status == "NO_SIGNAL"
        and d.level is None,
        _decision_dict(d),
    )

    # 5) Explicit near-indifference is VERY_LOW, distinct from NO_SIGNAL.
    d = guard.resolve_dimension(
        text="Keeping wattage modest can be largely ignored.",
        dimension="power",
    )
    check(
        "guard_explicit_very_low",
        d is not None
        and d.status == "RESOLVED"
        and d.level == "VERY_LOW",
        _decision_dict(d),
    )

    # 6) Residual validator rejects absolute intensity invented from comparison.
    text = "Reliability is more important than performance."
    result = validate_residual_prediction(
        text=text,
        dimension="reliability",
        prediction={
            "status": "RESOLVED",
            "level": "VERY_HIGH",
            "evidence": "Reliability is more important than performance",
            "accepted": True,
            "validation_error": None,
        },
    )
    check(
        "residual_rejects_absolute_from_comparison",
        result.prediction["status"] == "UNRESOLVED"
        and result.prediction["accepted"] is False,
        result.to_dict(),
    )

    # 7) Residual validator accepts supported RELATIVE_ONLY.
    result = validate_residual_prediction(
        text=text,
        dimension="reliability",
        prediction={
            "status": "RELATIVE_ONLY",
            "level": None,
            "evidence": "Reliability is more important than performance",
            "accepted": True,
            "validation_error": None,
        },
    )
    check(
        "residual_accepts_supported_relative_only",
        result.prediction["status"] == "RELATIVE_ONLY"
        and result.prediction["accepted"] is True,
        result.to_dict(),
    )

    # 8) Residual validator canonicalizes explicit LOW cue.
    text = "Cost sits near the bottom of our priorities."
    result = validate_residual_prediction(
        text=text,
        dimension="cost",
        prediction={
            "status": "RESOLVED",
            "level": "VERY_LOW",
            "evidence": "Cost sits near the bottom of our priorities",
            "accepted": True,
            "validation_error": None,
        },
    )
    check(
        "residual_canonicalizes_level",
        result.prediction["status"] == "RESOLVED"
        and result.prediction["level"] == "LOW"
        and result.prediction["accepted"] is True,
        result.to_dict(),
    )

    # 9) Residual NO_SIGNAL cannot be auto-accepted.
    result = validate_residual_prediction(
        text="The system has 80 clients.",
        dimension="cost",
        prediction={
            "status": "NO_SIGNAL",
            "level": None,
            "evidence": None,
            "accepted": True,
            "validation_error": None,
        },
    )
    check(
        "residual_no_signal_is_safety_abstention",
        result.prediction["status"] == "UNRESOLVED"
        and result.prediction["accepted"] is False,
        result.to_dict(),
    )

    # 10) Parser rejects an unrequested dimension globally.
    parsed = parse_llm_response(
        raw_text=json.dumps(
            {
                "dimensions": {
                    "cost": {
                        "status": "UNRESOLVED",
                        "level": None,
                        "evidence": None,
                    },
                    "power": {
                        "status": "UNRESOLVED",
                        "level": None,
                        "evidence": None,
                    },
                }
            }
        ),
        requested_dimensions=["cost"],
        user_text="Cost is mentioned.",
    )
    check(
        "parser_flags_unrequested_dimensions",
        parsed["valid"] is False
        and any(
            str(v).startswith("UNREQUESTED_DIMENSIONS:")
            for v in parsed["violations"]
        ),
        parsed,
    )

    # 11) Unsupported evidence cannot be accepted.
    parsed = parse_llm_response(
        raw_text=json.dumps(
            {
                "dimensions": {
                    "reliability": {
                        "status": "RESOLVED",
                        "level": "HIGH",
                        "evidence": "mission critical reliability",
                    }
                }
            }
        ),
        requested_dimensions=["reliability"],
        user_text="Reliability matters.",
    )
    pred = parsed["dimensions"]["reliability"]
    check(
        "parser_rejects_unsupported_evidence",
        pred["accepted"] is False
        and pred["status"] == "UNRESOLVED",
        parsed,
    )

    return {
        "status": "PASS",
        "checks_passed": len(checks),
        "checks": checks,
    }


def run_llm_liveness(
    *,
    host: str,
    model: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    # This is a liveness/safety smoke only, NOT an accuracy benchmark.
    text = (
        "For the final design, reliability matters to us, "
        "but I cannot assign it a clear absolute level."
    )
    requested = ["reliability"]

    raw, latency = _call_ollama(
        host=host,
        model=model,
        prompt=_prompt(text=text, dimensions=requested),
        timeout_seconds=timeout_seconds,
    )

    parsed = parse_llm_response(
        raw_text=raw,
        requested_dimensions=requested,
        user_text=text,
    )

    pred = parsed["dimensions"]["reliability"]
    validated = validate_residual_prediction(
        text=text,
        dimension="reliability",
        prediction=pred,
    )

    # Safety/liveness contract:
    # - endpoint answered,
    # - requested dimension exists after parsing,
    # - any accepted output passed the deterministic residual validator.
    _require(bool(raw.strip()), "LLM returned an empty response.")
    _require(
        set(parsed["dimensions"]) == {"reliability"},
        "Parser did not constrain output to requested dimensions.",
    )

    final_pred = validated.prediction
    if final_pred.get("accepted", False):
        _require(
            final_pred.get("status") in {"RESOLVED", "RELATIVE_ONLY"},
            "Unsafe residual status was auto-accepted.",
        )

    return {
        "status": "PASS",
        "scope": "liveness_and_safety_only_not_accuracy",
        "model": model,
        "latency_s": latency,
        "raw_response": raw,
        "parsed": parsed,
        "residual_validation": validated.to_dict(),
        "final_prediction": final_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast Layer-2 architecture acceptance smoke. "
            "No TRAIN/VALIDATION/TEST/FINAL_HOLDOUT data are used."
        )
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/"
            "layer2_hybrid_policy_FROZEN.json"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("preference_layer2_xlmr_v1_FINAL.zip"),
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Make exactly one short Ollama call.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/"
            "layer2_architecture_smoke_report.json"
        ),
    )
    parser.add_argument(
        "--freeze-output",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/"
            "layer2_architecture_FROZEN.json"
        ),
    )
    args = parser.parse_args()

    if not args.policy.exists():
        raise RuntimeError(f"Frozen policy not found: {args.policy}")

    policy = json.loads(args.policy.read_text(encoding="utf-8"))

    if policy.get("status") != EXPECTED_FREEZE_STATUS:
        raise RuntimeError(
            "Unexpected frozen policy status: "
            f"{policy.get('status')!r}"
        )

    prompt_sha = _sha256_text(
        HYBRID_PROMPT_VERSION + "\n" + HYBRID_PROMPT
    )

    if policy.get("guard_version") != GUARD_VERSION:
        raise RuntimeError("Guard version differs from frozen policy.")

    if (
        policy.get("residual_validator_version")
        != RESIDUAL_VALIDATOR_VERSION
    ):
        raise RuntimeError(
            "Residual validator version differs from frozen policy."
        )

    if policy.get("hybrid_prompt_version") != HYBRID_PROMPT_VERSION:
        raise RuntimeError("Prompt version differs from frozen policy.")

    if policy.get("hybrid_prompt_sha256") != prompt_sha:
        raise RuntimeError("Prompt SHA256 differs from frozen policy.")

    if not args.artifact.exists():
        raise RuntimeError(f"Layer-2 artifact not found: {args.artifact}")

    artifact_sha = sha256_file(args.artifact)
    if artifact_sha != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError(
            "Wrong Layer-2 artifact SHA256.\n"
            f"Expected: {EXPECTED_ARTIFACT_SHA256}\n"
            f"Actual:   {artifact_sha}"
        )

    with zipfile.ZipFile(args.artifact, "r") as archive:
        names = archive.namelist()

    final_holdout_files = [
        name
        for name in names
        if name.endswith("final_holdout_metrics.json")
    ]

    if final_holdout_files:
        raise RuntimeError(
            "FINAL_HOLDOUT metrics are already present in artifact; "
            "this smoke expects the untouched holdout state."
        )

    contracts = run_contract_smoke()

    llm_smoke = {
        "status": "SKIPPED",
        "reason": "Run with --with-llm for one short Ollama liveness call.",
    }

    if args.with_llm:
        try:
            llm_smoke = run_llm_liveness(
                host=args.host,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
            )
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as exc:
            raise RuntimeError(
                f"Ollama liveness smoke failed: {exc}"
            ) from exc

    report = {
        "step": "4.3",
        "status": "LAYER2_ARCHITECTURE_SMOKE_PASS",
        "scope": (
            "functional architecture + safety contracts; "
            "not a statistical accuracy claim"
        ),
        "frozen_policy": str(args.policy),
        "frozen_policy_sha256": sha256_file(args.policy),
        "artifact": str(args.artifact),
        "artifact_sha256": artifact_sha,
        "guard_version": GUARD_VERSION,
        "residual_validator_version": RESIDUAL_VALIDATOR_VERSION,
        "hybrid_prompt_version": HYBRID_PROMPT_VERSION,
        "hybrid_prompt_sha256": prompt_sha,
        "model": args.model,
        "contracts": contracts,
        "llm_smoke": llm_smoke,
        "test_or_holdout_data_used_by_this_smoke": False,
        "final_holdout_seen": False,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    freeze = {
        "step": "4.3",
        "status": "LAYER2_ARCHITECTURE_FROZEN_FUNCTIONAL_BASELINE",
        "scope": (
            "Layer-2 architecture/integration/safety contracts are frozen. "
            "This does NOT claim final statistical generalization."
        ),
        "components": {
            "encoder": "XLM-R Base shared multilingual encoder",
            "heads": "4 presence + 4 cumulative ordinal intensity heads",
            "dimensions": [
                "cost",
                "power",
                "performance",
                "reliability",
            ],
            "output_vocabulary": [
                "NO_SIGNAL",
                "VERY_LOW",
                "LOW",
                "MEDIUM",
                "HIGH",
                "VERY_HIGH",
                "RELATIVE_ONLY",
                "UNRESOLVED",
            ],
            "fallback": (
                "selective deterministic guard -> residual "
                "qwen2.5-coder:7b -> residual evidence validator"
            ),
        },
        "frozen_policy_sha256": report["frozen_policy_sha256"],
        "artifact_sha256": artifact_sha,
        "guard_version": GUARD_VERSION,
        "residual_validator_version": RESIDUAL_VALIDATOR_VERSION,
        "hybrid_prompt_version": HYBRID_PROMPT_VERSION,
        "hybrid_prompt_sha256": prompt_sha,
        "model": args.model,
        "architecture_smoke_report": str(args.report),
        "architecture_smoke_status": report["status"],
        "architecture_smoke_checks_passed": contracts["checks_passed"],
        "llm_liveness_checked": bool(args.with_llm),
        "known_evaluation_note": (
            "The previously started frozen TEST diagnostic exposed weaker "
            "fallback generalization than VALIDATION. No TEST-driven tuning "
            "is performed. This freeze is therefore a functional baseline, "
            "not a claim of final accuracy."
        ),
        "fine_tuning_performed_after_test": False,
        "final_holdout_seen": False,
        "next_project_step": (
            "categorical/boolean extractor -> merged Requirement State -> sizing"
        ),
    }

    args.freeze_output.write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nFREEZE FILE:")
    print(args.freeze_output)
    print(
        "\nSTATUS:",
        freeze["status"],
    )


if __name__ == "__main__":
    main()
