from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import torch

from ..artifact_contract import FROZEN_ARTIFACT_CONTRACT
from .confidence import CalibratedConfidencePolicy
from .labels import ACCESS_LABELS, HA_LABELS
from .model import CategoricalBooleanMultiTaskXLMR
from .schemas import SemanticHeadOutput


def sha256_file(path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()

    for member in archive.infolist():
        target = (destination / member.filename).resolve()

        if destination != target and destination not in target.parents:
            raise ValueError(
                "Unsafe path inside frozen artifact ZIP: "
                + member.filename
            )

    archive.extractall(destination)


class SemanticVerifier:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        confidence_policy: CalibratedConfidencePolicy,
        max_length: int,
        device: str,
        artifact_sha256: Optional[str] = None,
        temp_dir=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.confidence_policy = confidence_policy
        self.max_length = int(max_length)
        self.device = str(device)
        self.artifact_sha256 = artifact_sha256
        self._temp_dir = temp_dir

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_artifact_zip(
        cls,
        artifact_zip,
        *,
        expected_sha256: Optional[str] = None,
        device: Optional[str] = None,
    ):
        artifact_zip = Path(artifact_zip)

        if not artifact_zip.is_file():
            raise FileNotFoundError(
                "Frozen categorical/boolean artifact not found: "
                + str(artifact_zip)
            )

        actual_sha = sha256_file(artifact_zip)

        if (
            expected_sha256
            and actual_sha.lower()
            != expected_sha256.lower()
        ):
            raise ValueError(
                "Artifact SHA256 mismatch: expected "
                f"{expected_sha256}, got {actual_sha}"
            )

        temp = tempfile.TemporaryDirectory(
            prefix="categorical_boolean_semantic_"
        )
        extracted = Path(temp.name)

        try:
            with zipfile.ZipFile(
                artifact_zip,
                "r",
            ) as archive:
                _safe_extract_zip(archive, extracted)

            config_path = extracted / "model_config.json"
            calibration_path = extracted / "calibration.json"
            weights_path = extracted / "model.pt"
            tokenizer_dir = extracted / "tokenizer"
            encoder_config_dir = extracted / "encoder_config"

            required = [
                config_path,
                calibration_path,
                weights_path,
                tokenizer_dir,
                encoder_config_dir,
            ]

            missing = [
                str(path)
                for path in required
                if not path.exists()
            ]

            if missing:
                raise FileNotFoundError(
                    "Frozen artifact is incomplete; missing: "
                    + ", ".join(missing)
                )

            config = json.loads(
                config_path.read_text(encoding="utf-8")
            )
            calibration_payload = json.loads(
                calibration_path.read_text(encoding="utf-8")
            )

            # Immutable production contract checks.
            FROZEN_ARTIFACT_CONTRACT.validate_model_config(
                config
            )
            FROZEN_ARTIFACT_CONTRACT.validate_calibration(
                calibration_payload
            )

            from transformers import (
                AutoConfig,
                AutoModel,
                AutoTokenizer,
            )

            # Local-only loading: no model download in production.
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                local_files_only=True,
            )

            encoder_config = AutoConfig.from_pretrained(
                encoder_config_dir,
                local_files_only=True,
            )
            encoder = AutoModel.from_config(
                encoder_config
            )

            model = CategoricalBooleanMultiTaskXLMR(
                base_model_name=config["base_model_name"],
                dropout=float(config["dropout"]),
                encoder=encoder,
            )

            state = torch.load(
                weights_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(
                state,
                strict=True,
            )

            policy = (
                CalibratedConfidencePolicy.from_json(
                    calibration_path
                )
            )

            resolved_device = device or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            return cls(
                model=model,
                tokenizer=tokenizer,
                confidence_policy=policy,
                max_length=int(config["max_length"]),
                device=resolved_device,
                artifact_sha256=actual_sha,
                temp_dir=temp,
            )

        except Exception:
            temp.cleanup()
            raise

    @staticmethod
    def _head_output(labels, probabilities):
        ordered = sorted(
            zip(labels, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        top_label, top_probability = ordered[0]
        second_probability = ordered[1][1]

        return SemanticHeadOutput(
            probabilities={
                label: float(probability)
                for label, probability
                in zip(labels, probabilities)
            },
            top_label=top_label,
            top_probability=float(top_probability),
            second_probability=float(second_probability),
            margin=float(
                top_probability
                - second_probability
            ),
        )

    def _forward(self, text: str):
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            output = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

        ha_probabilities = torch.softmax(
            output["ha_logits"][0],
            dim=-1,
        ).cpu().tolist()

        access_probabilities = torch.softmax(
            output["access_logits"][0],
            dim=-1,
        ).cpu().tolist()

        return (
            self._head_output(
                HA_LABELS,
                ha_probabilities,
            ),
            self._head_output(
                ACCESS_LABELS,
                access_probabilities,
            ),
        )

    def verify_ha(self, text: str):
        ha_output, _ = self._forward(text)

        return self.confidence_policy.decide(
            head="ha",
            output=ha_output,
        )

    def verify_access(self, text: str):
        _, access_output = self._forward(text)

        return self.confidence_policy.decide(
            head="access",
            output=access_output,
        )
