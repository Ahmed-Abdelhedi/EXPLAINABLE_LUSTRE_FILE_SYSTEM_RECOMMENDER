from __future__ import annotations

import argparse
import json

from .policies import OrchestrationPolicy
from .production_wiring import (
    build_frozen_production_orchestrator,
    production_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 6.1B real production wiring smoke. "
            "No TRAIN/VALIDATION/TEST data are used."
        )
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help=(
            "Enable selective Qwen fallbacks. "
            "Default smoke keeps them disabled."
        ),
    )
    args = parser.parse_args()

    preflight = production_preflight()
    if not preflight["ready"]:
        raise RuntimeError(
            "Production preflight failed:\n"
            + json.dumps(preflight, indent=2)
        )

    orchestrator = build_frozen_production_orchestrator(
        device=args.device,
        enable_llm_fallback=args.with_llm,
        policy=OrchestrationPolicy(
            ask_optional_fields=False,
            ready_when_core_complete_and_no_conflict=True,
        ),
    )
    session = orchestrator.new_session(
        "STEP6_1B_REAL_SMOKE"
    )

    text = (
        "We need 100 TiB of usable capacity for 64 clients. "
        "High availability is mandatory. "
        "The workload is sequential. "
        "Reliability is absolutely critical and "
        "performance is very important."
    )

    response = orchestrator.handle_message(
        text,
        session,
    )

    expected = {
        "requested_usable_capacity_tib": 100.0,
        "client_count": 64,
        "ha_required": True,
        "access_type": "sequential",
        "reliability_priority": "VERY_HIGH",
        "performance_priority": "HIGH",
    }

    actual = {
        field: session.get(field).value
        for field in expected
    }

    mismatches = {
        field: {
            "expected": expected[field],
            "actual": actual[field],
            "state": session.get(field).state.value,
            "source": session.get(field).source,
        }
        for field in expected
        if actual[field] != expected[field]
    }

    if mismatches:
        raise RuntimeError(
            "Real production wiring smoke mismatch:\n"
            + json.dumps(mismatches, indent=2)
        )

    components = orchestrator.production_components
    preference_runtime = components["preference_runtime"]

    report = {
        "step": "6.1B",
        "status": (
            "INPUT_ORCHESTRATOR_"
            "REAL_PRODUCTION_WIRING_PASS"
        ),
        "scope": (
            "real frozen runtimes + one user message; "
            "not a statistical benchmark"
        ),
        "device": args.device,
        "llm_fallback_enabled": args.with_llm,
        "preflight": preflight,
        "input": text,
        "expected": expected,
        "actual": actual,
        "layer2_loaded_after_layer1_yes": (
            preference_runtime
            .layer2_loader
            ._runtime
            is not None
        ),
        "layer2_artifact_sha256": (
            preference_runtime
            .layer2_loader
            .artifact_sha256
        ),
        "response": response.to_dict(),
        "session": session.snapshot(),
    }

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print(
        "STATUS: "
        "INPUT_ORCHESTRATOR_"
        "REAL_PRODUCTION_WIRING_PASS"
    )


if __name__ == "__main__":
    main()
