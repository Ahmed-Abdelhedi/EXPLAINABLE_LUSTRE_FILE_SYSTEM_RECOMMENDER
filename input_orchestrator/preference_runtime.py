from __future__ import annotations

from pathlib import Path
from typing import Optional

from .layer2_artifact_loader import (
    EXPECTED_LAYER2_ARTIFACT_FILENAME,
    PreferenceLayer2ArtifactLoader,
)


class FrozenPreferenceRuntime:
    """
    Layer 1 is loaded at startup.
    Layer 2 is loaded lazily only when Layer 1 detects a preference signal.
    """

    def __init__(
        self,
        *,
        repo_root: str | Path,
        device: Optional[str] = None,
        enable_llm_fallback: Optional[bool] = None,
    ) -> None:
        from preference_extractor.signal_detector.runtime import (
            PreferenceSignalDetector,
        )

        self.repo_root = (
            Path(repo_root)
            .expanduser()
            .resolve()
        )

        self.signal_detector = PreferenceSignalDetector(
            artifact_path=(
                self.repo_root
                / "preference_signal_detector_v2_2.zip"
            ),
            device=device,
        )

        self.layer2_loader = PreferenceLayer2ArtifactLoader(
            self.repo_root
            / EXPECTED_LAYER2_ARTIFACT_FILENAME,
            device=device,
            enable_llm_fallback=enable_llm_fallback,
        )

    def layer2(self):
        return self.layer2_loader.load()
