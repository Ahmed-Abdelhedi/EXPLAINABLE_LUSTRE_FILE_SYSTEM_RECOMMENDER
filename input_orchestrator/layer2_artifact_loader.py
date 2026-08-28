from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional


EXPECTED_LAYER2_ARTIFACT_FILENAME = (
    "preference_layer2_xlmr_v1_FINAL.zip"
)
EXPECTED_LAYER2_ARTIFACT_SHA256 = (
    "cdbb6f6544d4b4d96578e0901a00f46c68916ad66547800ffc4d232095298890"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


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


def _safe_extract(
    archive: zipfile.ZipFile,
    destination: Path,
) -> None:
    destination = destination.resolve()

    for member in archive.infolist():
        target = (
            destination
            / member.filename
        ).resolve()

        if (
            destination != target
            and destination not in target.parents
        ):
            raise RuntimeError(
                "Unsafe path inside Layer-2 artifact: "
                + member.filename
            )

    archive.extractall(destination)


class PreferenceLayer2ArtifactLoader:
    """
    Load the already-frozen Layer-2 artifact.

    No training, test-set use or recalibration occurs here.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        *,
        device: Optional[str] = None,
        enable_llm_fallback: Optional[bool] = None,
    ) -> None:
        self.artifact_path = (
            Path(artifact_path)
            .expanduser()
            .resolve()
        )
        self.device = device
        self.enable_llm_fallback = enable_llm_fallback
        self._runtime = None
        self.artifact_sha256 = None

    def load(self):
        if self._runtime is not None:
            return self._runtime

        self._preflight()

        actual_sha = sha256_file(
            self.artifact_path
        )
        self.artifact_sha256 = actual_sha

        if actual_sha != EXPECTED_LAYER2_ARTIFACT_SHA256:
            raise RuntimeError(
                "Wrong frozen Layer-2 artifact.\n"
                f"Expected: {EXPECTED_LAYER2_ARTIFACT_SHA256}\n"
                f"Actual:   {actual_sha}"
            )

        cache_root = (
            Path(tempfile.gettempdir())
            / "explainable_lustre_recommender"
            / (
                "preference_layer2_"
                + actual_sha[:16]
            )
        )
        marker = cache_root / ".ready.json"

        if not marker.exists():
            self._extract_to_cache(
                cache_root=cache_root,
                marker=marker,
            )

        required = {
            "model": cache_root / "model.safetensors",
            "tokenizer": cache_root / "tokenizer",
            "calibration": cache_root / "calibration.json",
            "architecture": cache_root / "architecture_config.json",
            "freeze": cache_root / "freeze_manifest.json",
        }

        missing = [
            str(path)
            for path in required.values()
            if not path.exists()
        ]

        if missing:
            self._extract_to_cache(
                cache_root=cache_root,
                marker=marker,
            )
            missing = [
                str(path)
                for path in required.values()
                if not path.exists()
            ]

        if missing:
            raise FileNotFoundError(
                "Layer-2 artifact is incomplete: "
                + ", ".join(missing)
            )

        architecture = json.loads(
            required["architecture"].read_text(
                encoding="utf-8"
            )
        )

        if architecture.get("base_model") != (
            "FacebookAI/xlm-roberta-base"
        ):
            raise RuntimeError(
                "Unexpected Layer-2 base model: "
                f"{architecture.get('base_model')!r}"
            )

        if int(
            architecture.get("max_length", 0)
        ) != 128:
            raise RuntimeError(
                "Unexpected Layer-2 max_length."
            )

        calibration = json.loads(
            required["calibration"].read_text(
                encoding="utf-8"
            )
        )

        if calibration.get("status") != "FROZEN_BEFORE_TEST":
            raise RuntimeError(
                "Layer-2 calibration is not frozen before TEST."
            )

        freeze = json.loads(
            required["freeze"].read_text(
                encoding="utf-8"
            )
        )

        model_sha = sha256_file(required["model"])
        frozen_model_sha = freeze.get("model_sha256")

        if frozen_model_sha and model_sha != frozen_model_sha:
            raise RuntimeError(
                "model.safetensors SHA mismatch against "
                "freeze_manifest.json."
            )

        from safetensors.torch import load_file
        from transformers import (
            AutoTokenizer,
            XLMRobertaConfig,
            XLMRobertaModel,
        )

        from preference_extractor.layer2.confidence import (
            FrozenCalibratedConfidencePolicy,
        )
        from preference_extractor.layer2.llm_fallback import (
            PreferenceLLMFallback,
        )
        from preference_extractor.layer2.model import (
            XLMRPreferenceMultiTaskModel,
        )
        from preference_extractor.layer2.runtime import (
            Layer2PreferenceExtractor,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            required["tokenizer"],
            local_files_only=True,
            use_fast=True,
        )

        encoder_config = XLMRobertaConfig(
            vocab_size=250002,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=514,
            type_vocab_size=1,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            use_cache=True,
        )
        encoder = XLMRobertaModel(encoder_config)

        model = XLMRPreferenceMultiTaskModel(
            encoder=encoder,
            dropout=0.10,
            monotonicity_weight=0.10,
        )

        state = load_file(
            str(required["model"]),
            device="cpu",
        )

        if state and all(
            key.startswith("module.")
            for key in state
        ):
            state = {
                key[len("module."):]: value
                for key, value in state.items()
            }

        model.load_state_dict(
            state,
            strict=True,
        )

        policy = (
            FrozenCalibratedConfidencePolicy
            .from_calibration_dict(calibration)
        )

        llm = PreferenceLLMFallback(
            enabled=self.enable_llm_fallback
        )

        self._runtime = Layer2PreferenceExtractor(
            model=model,
            tokenizer=tokenizer,
            policy=policy,
            device=self.device,
            llm_fallback=llm,
        )

        return self._runtime

    def _preflight(self) -> None:
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                "Frozen Layer-2 artifact not found: "
                + str(self.artifact_path)
            )

        if _looks_like_git_lfs_pointer(self.artifact_path):
            raise RuntimeError(
                "preference_layer2_xlmr_v1_FINAL.zip is still "
                "a Git LFS pointer. Run `git lfs pull` first."
            )

        if not zipfile.is_zipfile(self.artifact_path):
            raise RuntimeError(
                "Invalid Layer-2 ZIP: "
                + str(self.artifact_path)
            )

    def _extract_to_cache(
        self,
        *,
        cache_root: Path,
        marker: Path,
    ) -> None:
        if cache_root.exists():
            shutil.rmtree(cache_root)

        cache_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        with zipfile.ZipFile(
            self.artifact_path,
            "r",
        ) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(
                    "Corrupted Layer-2 ZIP member: "
                    + bad_member
                )

            _safe_extract(
                archive,
                cache_root,
            )

        marker.write_text(
            json.dumps(
                {
                    "artifact_sha256":
                        EXPECTED_LAYER2_ARTIFACT_SHA256,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
