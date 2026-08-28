from __future__ import annotations

from pathlib import Path
from typing import Optional

from .bwm_coordinator import BWMCoordinator
from .orchestrator import InputOrchestrator
from .policies import (
    DEFAULT_POLICY,
    OrchestrationPolicy,
)
from .preference_runtime import FrozenPreferenceRuntime
from .production_adapters import (
    CategoricalProductionAdapter,
    PreferenceProductionAdapter,
    QuantityProductionAdapter,
)
from .router import ExtractionRouter


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def production_preflight(
    repo_root: str | Path | None = None,
) -> dict:
    root = (
        default_repo_root()
        if repo_root is None
        else Path(repo_root).expanduser().resolve()
    )

    paths = {
        "preference_layer1":
            root / "preference_signal_detector_v2_2.zip",
        "preference_layer2":
            root / "preference_layer2_xlmr_v1_FINAL.zip",
        "categorical":
            root
            / "categorical_boolean_extractor"
            / "artifacts"
            / "categorical_boolean_xlmr_v1_FROZEN.zip",
    }

    result = {
        "repo_root": str(root),
        "artifacts": {},
        "ready": True,
    }

    for name, path in paths.items():
        exists = path.is_file()
        result["artifacts"][name] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": (
                path.stat().st_size
                if exists
                else None
            ),
        }
        if not exists:
            result["ready"] = False

    return result


def build_frozen_production_orchestrator(
    *,
    repo_root: str | Path | None = None,
    device: Optional[str] = None,
    enable_llm_fallback: Optional[bool] = None,
    policy: OrchestrationPolicy = DEFAULT_POLICY,
) -> InputOrchestrator:
    root = (
        default_repo_root()
        if repo_root is None
        else Path(repo_root).expanduser().resolve()
    )

    preflight = production_preflight(root)
    if not preflight["ready"]:
        missing = [
            item["path"]
            for item in preflight["artifacts"].values()
            if not item["exists"]
        ]
        raise FileNotFoundError(
            "Production artifacts missing:\n"
            + "\n".join(missing)
        )

    from requirement_extractor_v2.llm_fallback_extractor import (
        LLMFallbackExtractor,
    )
    from requirement_extractor_v2.selective_cascade import (
        SelectiveCascade,
    )
    from requirement_extractor_v2.verified_pipeline import (
        VerifiedRequirementPipeline,
    )
    from categorical_boolean_extractor.llm_fallback import (
        CategoricalBooleanLLMFallback,
    )
    from categorical_boolean_extractor.runtime import (
        CategoricalBooleanExtractor,
    )

    quantity = QuantityProductionAdapter(
        pipeline=VerifiedRequirementPipeline(
            cascade=SelectiveCascade(
                llm_fallback=LLMFallbackExtractor(
                    enabled=enable_llm_fallback,
                )
            )
        )
    )

    preference_runtime = FrozenPreferenceRuntime(
        repo_root=root,
        device=device,
        enable_llm_fallback=enable_llm_fallback,
    )
    preference = PreferenceProductionAdapter(
        signal_detector=preference_runtime.signal_detector,
        layer2_provider=preference_runtime.layer2,
    )

    categorical_runtime = (
        CategoricalBooleanExtractor
        .from_default_artifact(
            device=device,
            llm_fallback=(
                CategoricalBooleanLLMFallback(
                    enabled=enable_llm_fallback
                )
            ),
        )
    )
    categorical = CategoricalProductionAdapter(
        categorical_runtime
    )

    orchestrator = InputOrchestrator(
        router=ExtractionRouter(
            [
                quantity,
                preference,
                categorical,
            ]
        ),
        policy=policy,
        bwm=BWMCoordinator.from_frozen_layer(),
    )

    orchestrator.production_components = {
        "quantity": quantity,
        "preference": preference,
        "preference_runtime": preference_runtime,
        "categorical": categorical,
    }

    return orchestrator


def build_orchestrator(
    *,
    repo_root: str | Path | None = None,
    device: Optional[str] = None,
    enable_llm_fallback: Optional[bool] = None,
    policy: OrchestrationPolicy = DEFAULT_POLICY,
) -> InputOrchestrator:
    return build_frozen_production_orchestrator(
        repo_root=repo_root,
        device=device,
        enable_llm_fallback=enable_llm_fallback,
        policy=policy,
    )
