from __future__ import annotations

import argparse
import json
from pathlib import Path

from .layer2_fallback_common import (
    PROMPT_VERSION,
    prompt_policy_sha256,
)


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

    benchmark = json.loads(
        args.validation_benchmark.read_text(
            encoding="utf-8"
        )
    )

    if (
        benchmark.get(
            "prompt_policy_sha256"
        )
        != prompt_policy_sha256()
    ):
        raise RuntimeError(
            "Validation benchmark was not generated with the current prompt."
        )

    if (
        benchmark.get(
            "model"
        )
        != args.model
    ):
        raise RuntimeError(
            "Validation benchmark model does not match the model being frozen."
        )

    freeze = {
        "step":
            "4.2A",
        "status":
            "LAYER2_LLM_PROMPT_FROZEN_BEFORE_TEST_FALLBACK_BENCHMARK",
        "model":
            args.model,
        "prompt_version":
            PROMPT_VERSION,
        "prompt_policy_sha256":
            prompt_policy_sha256(),
        "validation_benchmark":
            str(
                args.validation_benchmark
            ),
        "validation_overall_metrics":
            benchmark.get(
                "overall"
            ),
        "test_fallback_seen_before_prompt_freeze":
            False,
        "protocol":
            (
                "The prompt/model pair is frozen after validation fallback "
                "evaluation and before running the TEST fallback benchmark. "
                "Do not change prompt rules, parser rules, model, or decoding "
                "after TEST results are observed."
            ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            freeze,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            freeze,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
