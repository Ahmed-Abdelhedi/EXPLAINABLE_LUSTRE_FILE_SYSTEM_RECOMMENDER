from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from .confidence import (
    ConfidencePolicy,
    EXPECTED_LAYER2_ARTIFACT_SHA256,
    FrozenCalibratedConfidencePolicy,
)
from .labels import (
    PreferenceDimension,
    ResolutionStatus,
)
from .llm_fallback import (
    HYBRID_PROMPT,
    HYBRID_PROMPT_VERSION,
    PreferenceLLMFallback,
)
from .residual_validator import (
    RESIDUAL_VALIDATOR_VERSION,
)
from .runtime import (
    Layer2PreferenceExtractor,
)
from .semantic_guard import (
    GUARD_VERSION,
)


EXPECTED_PROMPT_SHA256 = (
    "04e60847fc4739e3ece178cbc2c37fefabd1854314a457636747cfcd535df137"
)


class _FakeTokenizer:
    def __call__(self, *args, **kwargs):
        del args, kwargs
        return {
            "input_ids": torch.tensor(
                [[1, 2, 3]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1]],
                dtype=torch.long,
            ),
        }


class _FakeModel(torch.nn.Module):
    """
    Force only PERFORMANCE into the fallback band.
    Other dimensions are high-confidence NO_SIGNAL.

    DIMENSIONS order:
      cost, power, performance, reliability
    """
    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        del input_ids, attention_mask

        probabilities = torch.tensor(
            [[0.01, 0.01, 0.50, 0.01]],
            dtype=torch.float32,
        )

        presence_logits = torch.logit(
            probabilities.clamp(
                1e-6,
                1 - 1e-6,
            )
        )

        intensity_probabilities = torch.tensor(
            [
                [
                    [0.90, 0.80, 0.70, 0.60],
                    [0.90, 0.80, 0.70, 0.60],
                    [0.90, 0.80, 0.70, 0.60],
                    [0.90, 0.80, 0.70, 0.60],
                ]
            ],
            dtype=torch.float32,
        )

        intensity_logits = torch.logit(
            intensity_probabilities.clamp(
                1e-6,
                1 - 1e-6,
            )
        )

        return SimpleNamespace(
            presence_logits=presence_logits,
            intensity_logits=intensity_logits,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast production Layer-2 runtime smoke. "
            "Uses the real production runtime and exactly one "
            "Ollama call; no dataset benchmark."
        )
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "preference_layer2_xlmr_v1_FINAL.zip"
        ),
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
        "--timeout",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "preference_extractor/layer2/"
            "layer2_production_runtime_smoke.json"
        ),
    )
    parser.add_argument(
        "--freeze-output",
        type=Path,
        default=Path(
            "preference_extractor/layer2/"
            "LAYER2_PRODUCTION_RUNTIME_FROZEN.json"
        ),
    )
    args = parser.parse_args()

    if not args.artifact.exists():
        raise RuntimeError(
            f"Artifact not found: {args.artifact}"
        )

    artifact_sha = _sha256_file(
        args.artifact
    )

    if (
        artifact_sha
        != EXPECTED_LAYER2_ARTIFACT_SHA256
    ):
        raise RuntimeError(
            "Wrong Layer-2 artifact SHA256.\n"
            f"Expected: {EXPECTED_LAYER2_ARTIFACT_SHA256}\n"
            f"Actual:   {artifact_sha}"
        )

    frozen_policy = (
        FrozenCalibratedConfidencePolicy
        .from_artifact_zip(
            args.artifact
        )
    )

    prompt_sha = hashlib.sha256(
        (
            HYBRID_PROMPT_VERSION
            + "\n"
            + HYBRID_PROMPT
        ).encode("utf-8")
    ).hexdigest()

    if prompt_sha != EXPECTED_PROMPT_SHA256:
        raise RuntimeError(
            "Production prompt differs from frozen prompt."
        )

    # 1) Production deterministic guard path.
    guard_extractor = Layer2PreferenceExtractor(
        model=cast(Any, _FakeModel()),
        tokenizer=_FakeTokenizer(),
        policy=ConfidencePolicy(),
        device="cpu",
        llm_fallback=PreferenceLLMFallback(
            enabled=True,
            host=args.host,
            model=args.model,
            timeout_seconds=args.timeout,
        ),
    )

    # This text contains a recognized explicit PERFORMANCE cue.
    # Fake model routes PERFORMANCE to fallback; production guard must catch it
    # before Qwen, so call_count must remain zero.
    guard_result = guard_extractor.extract(
        "Performance is a top priority for this design."
    )

    performance = guard_result.dimensions[
        PreferenceDimension.PERFORMANCE
    ]

    if not (
        performance.status
        == ResolutionStatus.RESOLVED
        and performance.level is not None
        and performance.level.value
        == "VERY_HIGH"
        and guard_result.deterministic_guard_used
        and not guard_result.llm_fallback_used
        and guard_extractor.llm_fallback.call_count
        == 0
    ):
        raise RuntimeError(
            "Production deterministic guard smoke failed."
        )

    # 2) Real production residual LLM liveness.
    llm_fallback = PreferenceLLMFallback(
        enabled=True,
        host=args.host,
        model=args.model,
        timeout_seconds=args.timeout,
    )

    llm_extractor = Layer2PreferenceExtractor(
        model=cast(Any, _FakeModel()),
        tokenizer=_FakeTokenizer(),
        policy=ConfidencePolicy(),
        device="cpu",
        llm_fallback=llm_fallback,
    )

    ambiguous_text = (
        "For the final design, performance should influence our choice."
    )

    llm_result = llm_extractor.extract(
        ambiguous_text
    )

    if llm_fallback.call_count != 1:
        raise RuntimeError(
            "Production runtime did not make exactly one Ollama call."
        )

    if not llm_result.llm_fallback_used:
        raise RuntimeError(
            "Production result did not record the real LLM call."
        )

    if (
        PreferenceDimension.PERFORMANCE
        not in llm_result.llm_fallback_dimensions
    ):
        raise RuntimeError(
            "Production result did not record PERFORMANCE as residual LLM input."
        )

    # Whatever Qwen returns, final runtime must remain schema-safe.
    if (
        set(llm_result.dimensions)
        != set(PreferenceDimension)
    ):
        raise RuntimeError(
            "Production runtime did not return exactly four dimensions."
        )

    report = {
        "step": "4.4",
        "status":
            "LAYER2_PRODUCTION_RUNTIME_SMOKE_PASS",
        "scope":
            (
                "functional production integration smoke; "
                "not a statistical accuracy benchmark"
            ),
        "artifact": str(args.artifact),
        "artifact_sha256": artifact_sha,
        "frozen_calibration_status":
            frozen_policy.calibration_status,
        "frozen_calibration_max_length":
            frozen_policy.max_length,
        "guard_version": GUARD_VERSION,
        "residual_validator_version":
            RESIDUAL_VALIDATOR_VERSION,
        "hybrid_prompt_version":
            HYBRID_PROMPT_VERSION,
        "hybrid_prompt_sha256": prompt_sha,
        "ollama_model": args.model,
        "checks": {
            "production_guard_intercepts_before_llm":
                True,
            "production_guard_llm_call_count":
                guard_extractor.llm_fallback.call_count,
            "production_residual_llm_called":
                True,
            "production_residual_llm_call_count":
                llm_fallback.call_count,
            "production_residual_requested_dimensions":
                llm_fallback.last_requested_dimensions,
            "production_residual_response_valid":
                llm_fallback.last_response_valid,
            "production_residual_response_violations":
                llm_fallback.last_response_violations,
            "production_residual_latency_seconds":
                llm_fallback.last_latency_seconds,
            "production_residual_accepted_dimensions":
                llm_fallback.last_accepted_dimensions,
            "final_four_dimension_contract":
                True,
        },
        "guard_example":
            guard_result.to_dict(),
        "llm_liveness_example":
            llm_result.to_dict(),
        "test_dataset_used": False,
        "final_holdout_used": False,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    production_files = [
        Path(
            "preference_extractor/layer2/"
            "confidence.py"
        ),
        Path(
            "preference_extractor/layer2/"
            "labels.py"
        ),
        Path(
            "preference_extractor/layer2/"
            "schemas.py"
        ),
        Path(
            "preference_extractor/layer2/"
            "semantic_guard.py"
        ),
        Path(
            "preference_extractor/layer2/"
            "residual_validator.py"
        ),
        Path(
            "preference_extractor/layer2/"
            "llm_fallback.py"
        ),
        Path(
            "preference_extractor/layer2/"
            "runtime.py"
        ),
    ]

    file_hashes = {
        str(path):
            _sha256_file(path)
        for path in production_files
    }

    freeze = {
        "step": "4.4",
        "status":
            "LAYER2_PRODUCTION_RUNTIME_FROZEN",
        "meaning":
            (
                "The real preference_extractor/layer2 runtime "
                "is integrated and functionally frozen. "
                "This is not a claim of final statistical generalization."
            ),
        "artifact_sha256": artifact_sha,
        "guard_version": GUARD_VERSION,
        "residual_validator_version":
            RESIDUAL_VALIDATOR_VERSION,
        "hybrid_prompt_version":
            HYBRID_PROMPT_VERSION,
        "hybrid_prompt_sha256": prompt_sha,
        "ollama_model": args.model,
        "production_runtime_smoke":
            str(args.output),
        "production_runtime_smoke_status":
            report["status"],
        "production_file_sha256":
            file_hashes,
        "fine_tuning_performed": False,
        "test_dataset_used_by_smoke": False,
        "final_holdout_seen": False,
        "known_limitation":
            (
                "A previous large frozen TEST diagnostic showed "
                "weaker fallback generalization than validation. "
                "No TEST-driven tuning is incorporated here."
            ),
    }

    args.freeze_output.write_text(
        json.dumps(
            freeze,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print(
        "STATUS:",
        freeze["status"],
    )
    print(
        "FREEZE FILE:",
        args.freeze_output,
    )


if __name__ == "__main__":
    main()
