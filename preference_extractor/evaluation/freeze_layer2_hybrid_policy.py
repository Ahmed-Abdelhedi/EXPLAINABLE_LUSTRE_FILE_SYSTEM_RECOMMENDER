from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .layer2_hybrid_guard import GUARD_VERSION
from .layer2_residual_validator import RESIDUAL_VALIDATOR_VERSION
from .run_layer2_hybrid_fallback_benchmark import (
    HYBRID_PROMPT,
    HYBRID_PROMPT_VERSION,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-benchmark",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
    )
    args = parser.parse_args()

    report = json.loads(
        args.validation_benchmark.read_text(encoding="utf-8")
    )

    if report.get("guard_only"):
        raise RuntimeError(
            "Cannot freeze from guard-only output. Run the residual-LLM "
            "validation benchmark first."
        )

    if report.get("step") != "4.2E":
        raise RuntimeError(
            "Freeze requires a Step 4.2E validation benchmark produced "
            "with the residual validator."
        )

    if report.get("guard_version") != GUARD_VERSION:
        raise RuntimeError("Guard version mismatch.")

    if (
        report.get("residual_validator_version")
        != RESIDUAL_VALIDATOR_VERSION
    ):
        raise RuntimeError("Residual validator version mismatch.")

    if report.get("hybrid_prompt_version") != HYBRID_PROMPT_VERSION:
        raise RuntimeError("Residual prompt version mismatch.")

    if report.get("model") != args.model:
        raise RuntimeError("Model mismatch.")

    overall = report.get("overall", {})

    freeze = {
        "step": "4.2E",
        "status": (
            "LAYER2_HYBRID_GUARD_RESIDUAL_VALIDATOR_LLM_"
            "FROZEN_BEFORE_TEST"
        ),
        "guard_version": GUARD_VERSION,
        "residual_validator_version": RESIDUAL_VALIDATOR_VERSION,
        "hybrid_prompt_version": HYBRID_PROMPT_VERSION,
        "hybrid_prompt_sha256": _sha256_text(
            HYBRID_PROMPT_VERSION + "\n" + HYBRID_PROMPT
        ),
        "model": args.model,
        "validation_benchmark": str(args.validation_benchmark),
        "validation_overall": overall,
        "test_hybrid_seen_before_freeze": False,
        "final_holdout_seen": False,
        "protocol": (
            "After this freeze, do not modify guard rules, residual "
            "validator rules, residual prompt, model, parser, or decoding "
            "based on TEST results."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
