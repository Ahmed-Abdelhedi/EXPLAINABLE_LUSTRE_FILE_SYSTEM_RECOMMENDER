from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .context_guard import PreferenceContextGuard, PreferenceGuardDecision
from .schemas import PreferenceSignalResult


_PRODUCTION_CONFIG_PATH = Path(__file__).with_name("production_config.json")


class PreferenceSignalDetector:
    """
    Final production Layer-1 preference signal detector.

    Architecture:
        text
          -> multilingual DistilBERT V2.2
          -> high-precision PreferenceContextGuard
          -> final binary decision

    Layer 1 answers only whether the CURRENT user message contains a
    preference signal. Preference dimensions and weights belong to Layer 2.

    The production model is stored as the Git-LFS ZIP
    ``preference_signal_detector_v2_2.zip`` at repository root. The ZIP is
    extracted once into the operating-system temporary cache and reused.
    """

    def __init__(
        self,
        artifact_path: str | Path | None = None,
        device: str | None = None,
        use_context_guard: bool = True,
        context_guard: Optional[PreferenceContextGuard] = None,
    ) -> None:
        self.production_config = self._load_production_config()
        self.model_version = str(self.production_config["model_version"])

        if artifact_path is None:
            artifact_path = self._default_artifact_path()

        self.artifact_source = Path(artifact_path).expanduser().resolve()

        if not self.artifact_source.exists():
            raise FileNotFoundError(
                f"Preference detector artifact not found: "
                f"{self.artifact_source}"
            )

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.use_context_guard = bool(use_context_guard)
        self.context_guard = (
            context_guard
            if context_guard is not None
            else PreferenceContextGuard()
        )

        self.artifact_path = self._resolve_artifact_root(
            self.artifact_source
        )

        self._load_artifacts()

    @staticmethod
    def _load_production_config() -> dict:
        if not _PRODUCTION_CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Layer-1 production config not found: "
                f"{_PRODUCTION_CONFIG_PATH}"
            )

        return json.loads(
            _PRODUCTION_CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )

    def _default_artifact_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]

        return (
            repo_root
            / str(
                self.production_config["artifact_filename"]
            )
        )

    @staticmethod
    def _looks_like_git_lfs_pointer(path: Path) -> bool:
        try:
            if path.stat().st_size > 4096:
                return False

            prefix = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )[:256]

        except OSError:
            return False

        return prefix.startswith(
            "version https://git-lfs.github.com/spec/v1"
        )

    def _resolve_artifact_root(
        self,
        source: Path,
    ) -> Path:
        if source.is_dir():
            return source

        if source.suffix.lower() != ".zip":
            raise ValueError(
                "Preference detector artifact must be an "
                "extracted directory or ZIP."
            )

        if self._looks_like_git_lfs_pointer(source):
            raise RuntimeError(
                "preference_signal_detector_v2_2.zip is still a "
                "Git LFS pointer. Run `git lfs pull` before "
                "starting the preference detector."
            )

        if not zipfile.is_zipfile(source):
            raise ValueError(
                f"Invalid preference detector ZIP: {source}"
            )

        cache_key = str(
            self.production_config["model_sha256"]
        )[:16]

        cache_base = (
            Path(tempfile.gettempdir())
            / "explainable_lustre_recommender"
            / (
                f"preference_signal_detector_"
                f"{self.model_version}_{cache_key}"
            )
        )

        payload_dir = cache_base / "payload"
        marker = cache_base / ".ready.json"

        if not marker.exists():
            self._extract_zip_to_cache(
                source=source,
                cache_base=cache_base,
                payload_dir=payload_dir,
                marker=marker,
            )

        root = self._find_extracted_root(
            payload_dir
        )

        if root is not None:
            return root

        # A stale or interrupted cache must never silently survive.
        self._extract_zip_to_cache(
            source=source,
            cache_base=cache_base,
            payload_dir=payload_dir,
            marker=marker,
        )

        root = self._find_extracted_root(
            payload_dir
        )

        if root is None:
            raise RuntimeError(
                "Could not locate extracted preference detector "
                f"root under {payload_dir}"
            )

        return root

    def _extract_zip_to_cache(
        self,
        source: Path,
        cache_base: Path,
        payload_dir: Path,
        marker: Path,
    ) -> None:
        if cache_base.exists():
            shutil.rmtree(
                cache_base
            )

        payload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with zipfile.ZipFile(
            source,
            "r",
        ) as archive:
            bad_member = archive.testzip()

            if bad_member is not None:
                raise RuntimeError(
                    "Corrupted preference detector ZIP member: "
                    f"{bad_member}"
                )

            archive.extractall(
                payload_dir
            )

        marker.write_text(
            json.dumps(
                {
                    "model_version": self.model_version,
                    "model_sha256": (
                        self.production_config[
                            "model_sha256"
                        ]
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _find_extracted_root(
        payload_dir: Path,
    ) -> Path | None:
        if not payload_dir.exists():
            return None

        candidates = [
            payload_dir
        ]

        candidates.extend(
            path
            for path in payload_dir.iterdir()
            if path.is_dir()
        )

        for candidate in candidates:
            if (
                (candidate / "model").is_dir()
                and (candidate / "tokenizer").is_dir()
                and (candidate / "labels.json").is_file()
                and (candidate / "thresholds.json").is_file()
            ):
                return candidate

        return None

    def _load_artifacts(self) -> None:
        model_path = self.artifact_path / "model"
        tokenizer_path = self.artifact_path / "tokenizer"
        labels_path = self.artifact_path / "labels.json"
        threshold_path = self.artifact_path / "thresholds.json"

        for required in (
            model_path,
            tokenizer_path,
            labels_path,
            threshold_path,
        ):
            if not required.exists():
                raise FileNotFoundError(
                    required
                )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_path
            )
        )

        self.model.to(
            self.device
        )
        self.model.eval()

        labels = json.loads(
            labels_path.read_text(
                encoding="utf-8"
            )
        )

        self.id2label = {
            int(key): value
            for key, value
            in labels["id2label"].items()
        }

        threshold_data = json.loads(
            threshold_path.read_text(
                encoding="utf-8"
            )
        )

        # Kept only for provenance. It was the raw Transformer candidate
        # exported by Kaggle before the final guarded local calibration.
        self.raw_artifact_threshold = float(
            threshold_data[
                "preference_signal_probability_threshold"
            ]
        )

        # Production decision threshold frozen before fresh V4 FIRST_RUN.
        self.threshold = float(
            self.production_config[
                "final_threshold"
            ]
        )

        self.threshold_policy = str(
            self.production_config[
                "threshold_policy"
            ]
        )

    @staticmethod
    def _validate_text(
        text: str,
    ) -> None:
        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "Input text must be a string"
            )

    @torch.no_grad()
    def _preference_probability(
        self,
        text: str,
    ) -> float:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
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

        outputs = self.model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        return float(
            probabilities[1]
            .detach()
            .cpu()
            .item()
        )

    def predict_model_only(
        self,
        text: str,
    ) -> PreferenceSignalResult:
        """Raw V2.2 Transformer decision using the FINAL threshold."""
        self._validate_text(
            text
        )

        probability = (
            self._preference_probability(
                text
            )
        )

        has_signal = (
            probability
            >= self.threshold
        )

        label_id = (
            1
            if has_signal
            else 0
        )

        return PreferenceSignalResult(
            has_preference_signal=has_signal,
            label=self.id2label[
                label_id
            ],
            probability=round(
                probability,
                6,
            ),
            threshold=self.threshold,
            decision_source="transformer",
            guard_decision=None,
            guard_reason=None,
            guard_evidence=None,
            transformer_has_preference_signal=(
                has_signal
            ),
        )

    def apply_context_guard(
        self,
        text: str,
        model_result: PreferenceSignalResult,
    ) -> PreferenceSignalResult:
        """
        Apply the deterministic discourse guard without recomputing
        Transformer inference.
        """
        self._validate_text(
            text
        )

        if not self.use_context_guard:
            return model_result

        guard = self.context_guard.resolve(
            text
        )

        if (
            guard.decision
            == PreferenceGuardDecision.PASS_TO_MODEL
        ):
            return replace(
                model_result,
                decision_source="transformer",
                guard_decision=(
                    guard.decision.value
                ),
                guard_reason=None,
                guard_evidence=None,
                transformer_has_preference_signal=(
                    model_result
                    .has_preference_signal
                ),
            )

        transformer_has_signal = (
            model_result
            .has_preference_signal
        )

        if (
            guard.decision
            == PreferenceGuardDecision.FORCE_SIGNAL
        ):
            final_has_signal = True
            final_label_id = 1
            source = (
                "context_guard_positive"
            )

        else:
            final_has_signal = False
            final_label_id = 0
            source = (
                "context_guard_negative"
            )

        return PreferenceSignalResult(
            has_preference_signal=(
                final_has_signal
            ),
            label=self.id2label[
                final_label_id
            ],
            probability=(
                model_result.probability
            ),
            threshold=(
                model_result.threshold
            ),
            decision_source=source,
            guard_decision=(
                guard.decision.value
            ),
            guard_reason=guard.reason,
            guard_evidence=guard.evidence,
            transformer_has_preference_signal=(
                transformer_has_signal
            ),
        )

    def predict(
        self,
        text: str,
    ) -> PreferenceSignalResult:
        """Final production Layer-1 prediction."""
        model_result = (
            self.predict_model_only(
                text
            )
        )

        return self.apply_context_guard(
            text=text,
            model_result=model_result,
        )

    def info(self) -> dict:
        return {
            "layer": 1,
            "name": (
                "PreferenceSignalDetector"
            ),
            "status": "CLOSED_FINAL",
            "model_version": (
                self.model_version
            ),
            "artifact_source": str(
                self.artifact_source
            ),
            "artifact_root": str(
                self.artifact_path
            ),
            "device": str(
                self.device
            ),
            "max_length": 128,
            "final_threshold": (
                self.threshold
            ),
            "raw_artifact_threshold": (
                self.raw_artifact_threshold
            ),
            "threshold_policy": (
                self.threshold_policy
            ),
            "context_guard_enabled": (
                self.use_context_guard
            ),
            "context_guard": (
                self.context_guard.info()
            ),
        }
