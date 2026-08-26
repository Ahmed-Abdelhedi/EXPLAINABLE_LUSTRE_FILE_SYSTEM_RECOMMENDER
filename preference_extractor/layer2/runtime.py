from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from .confidence import (
    RuntimeConfidencePolicy,
    model_max_length,
    policy_for_dimension,
    selective_decision,
)
from .labels import (
    DIMENSIONS,
    PreferenceDimension,
    PreferenceLevel,
    ResolutionSource,
    ResolutionStatus,
)
from .llm_fallback import PreferenceLLMFallback
from .relation_resolver import ComparativeRelationResolver
from .schemas import (
    DimensionPreferenceResult,
    PreferenceExtractionResult,
)
from .semantic_guard import (
    GUARD_VERSION,
    Layer2DeterministicSemanticGuard,
)
from .validator import PreferenceOutputValidator


if TYPE_CHECKING:
    from .model import XLMRPreferenceMultiTaskModel


class Layer2PreferenceExtractor:
    """
    Production Layer 2.

    Runtime order:
        XLM-R Base multi-task extractor
        -> frozen/selective confidence gate
        -> comparative relation preservation
        -> deterministic semantic guard on Transformer abstentions
        -> residual local Qwen fallback only where guard also abstains
        -> deterministic residual evidence validator inside LLM fallback
        -> final deterministic output validator

    Call only when Layer 1 returned YES.
    """

    guard_version = GUARD_VERSION

    def __init__(
        self,
        *,
        model: "XLMRPreferenceMultiTaskModel",
        tokenizer,
        policy: RuntimeConfidencePolicy,
        device: str | None = None,
        llm_fallback: Optional[
            PreferenceLLMFallback
        ] = None,
        semantic_guard: Optional[
            Layer2DeterministicSemanticGuard
        ] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.policy = policy
        self.policy.validate()

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model.to(
            self.device
        )
        self.model.eval()

        self.relation_resolver = (
            ComparativeRelationResolver()
        )

        self.semantic_guard = (
            semantic_guard
            if semantic_guard is not None
            else Layer2DeterministicSemanticGuard()
        )

        self.llm_fallback = (
            llm_fallback
            if llm_fallback is not None
            else PreferenceLLMFallback()
        )

        self.validator = (
            PreferenceOutputValidator()
        )

    @torch.no_grad()
    def _predict_probabilities(
        self,
        text: str,
    ):
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=model_max_length(
                self.policy
            ),
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        output = self.model(
            input_ids=inputs[
                "input_ids"
            ],
            attention_mask=inputs[
                "attention_mask"
            ],
        )

        presence = torch.sigmoid(
            output.presence_logits
        )[0]

        intensity = torch.sigmoid(
            output.intensity_logits
        )[0]

        return (
            presence.detach().cpu(),
            intensity.detach().cpu(),
        )

    @staticmethod
    def _guard_to_result(
        *,
        dimension: PreferenceDimension,
        decision,
        previous: DimensionPreferenceResult,
    ) -> DimensionPreferenceResult:
        try:
            status = ResolutionStatus(
                decision.status
            )
        except ValueError as exc:
            raise ValueError(
                "Unsupported deterministic guard status: "
                f"{decision.status}"
            ) from exc

        level = None

        if (
            status
            == ResolutionStatus.RESOLVED
        ):
            if decision.level is None:
                raise ValueError(
                    "Guard RESOLVED result requires a level."
                )

            level = PreferenceLevel(
                decision.level
            )

        return DimensionPreferenceResult(
            dimension=dimension,
            status=status,
            source=(
                ResolutionSource.DETERMINISTIC_GUARD
            ),
            level=level,
            presence_probability=(
                previous.presence_probability
            ),
            intensity_confidence=(
                previous.intensity_confidence
            ),
            evidence=decision.evidence,
            reason=decision.reason,
        )

    def extract(
        self,
        text: str,
    ) -> PreferenceExtractionResult:
        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "Layer-2 input must be a string."
            )

        relation_analysis = (
            self.relation_resolver.resolve(
                text
            )
        )

        presence, intensity = (
            self._predict_probabilities(
                text
            )
        )

        dimensions: Dict[
            PreferenceDimension,
            DimensionPreferenceResult,
        ] = {}

        transformer_unresolved: List[
            PreferenceDimension
        ] = []

        for index, dimension in enumerate(
            DIMENSIONS
        ):
            local_policy = (
                policy_for_dimension(
                    self.policy,
                    dimension,
                )
            )

            (
                status,
                level,
                confidence,
                reason,
            ) = selective_decision(
                presence_probability=float(
                    presence[index].item()
                ),
                cumulative_intensity_probabilities=[
                    float(value)
                    for value
                    in intensity[
                        index
                    ].tolist()
                ],
                policy=local_policy,
                relative_only=(
                    dimension
                    in relation_analysis
                    .relative_only_dimensions
                ),
            )

            source = (
                ResolutionSource.RELATION_RESOLVER
                if status
                == ResolutionStatus.RELATIVE_ONLY
                else ResolutionSource.TRANSFORMER
            )

            dimensions[dimension] = (
                DimensionPreferenceResult(
                    dimension=dimension,
                    status=status,
                    source=source,
                    level=level,
                    presence_probability=float(
                        presence[
                            index
                        ].item()
                    ),
                    intensity_confidence=confidence,
                    reason=reason,
                )
            )

            if (
                status
                == ResolutionStatus.NEEDS_FALLBACK
            ):
                transformer_unresolved.append(
                    dimension
                )

        # Step 1 after Transformer abstention:
        # frozen deterministic semantic guard.
        guarded_dimensions: List[
            PreferenceDimension
        ] = []

        guard_results = (
            self.semantic_guard.resolve_many(
                text=text,
                dimensions=[
                    dimension.value
                    for dimension
                    in transformer_unresolved
                ],
            )
        )

        residual_dimensions: List[
            PreferenceDimension
        ] = []

        for dimension in transformer_unresolved:
            guard_decision = guard_results.get(
                dimension.value
            )

            if guard_decision is None:
                residual_dimensions.append(
                    dimension
                )
                continue

            dimensions[dimension] = (
                self._guard_to_result(
                    dimension=dimension,
                    decision=guard_decision,
                    previous=dimensions[
                        dimension
                    ],
                )
            )
            guarded_dimensions.append(
                dimension
            )

        # Step 2:
        # only guard abstentions may reach Qwen.
        llm_calls_before = (
            self.llm_fallback.call_count
        )

        fallback_results = (
            self.llm_fallback.resolve(
                text=text,
                unresolved_dimensions=(
                    residual_dimensions
                ),
                relations=(
                    relation_analysis.relations
                ),
            )
        )

        llm_called = (
            self.llm_fallback.call_count
            > llm_calls_before
        )

        for dimension in residual_dimensions:
            if dimension in fallback_results:
                llm_result = fallback_results[
                    dimension
                ]

                # Keep Transformer probabilities for traceability.
                previous = dimensions[
                    dimension
                ]

                llm_result.presence_probability = (
                    previous.presence_probability
                )
                llm_result.intensity_confidence = (
                    previous.intensity_confidence
                )

                dimensions[
                    dimension
                ] = llm_result
            else:
                previous = dimensions[
                    dimension
                ]

                dimensions[
                    dimension
                ] = DimensionPreferenceResult(
                    dimension=dimension,
                    status=(
                        ResolutionStatus.UNRESOLVED
                    ),
                    source=(
                        ResolutionSource.NONE
                    ),
                    presence_probability=(
                        previous
                        .presence_probability
                    ),
                    intensity_confidence=(
                        previous
                        .intensity_confidence
                    ),
                    reason=(
                        "Transformer abstained, deterministic guard "
                        "abstained, and residual LLM did not produce "
                        "an accepted verified result."
                    ),
                )

        self.validator.validate_all(
            dimensions=dimensions,
            relations=(
                relation_analysis.relations
            ),
            text=text,
        )

        return PreferenceExtractionResult(
            text=text,
            dimensions=dimensions,
            relations=(
                relation_analysis.relations
            ),
            deterministic_guard_used=bool(
                guarded_dimensions
            ),
            deterministic_guard_dimensions=(
                guarded_dimensions
            ),
            llm_fallback_used=llm_called,
            llm_fallback_dimensions=(
                list(
                    residual_dimensions
                )
                if llm_called
                else []
            ),
        )
