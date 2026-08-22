
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .schemas import PreferenceSignalResult


class PreferenceSignalDetector:
    """
    Layer 1:
    Detects whether a user message contains
    a preference signal.

    It does NOT extract the preference.
    Layer 2 will do that later.
    """

    def __init__(
        self,
        artifact_path: str | None = None,
        device: str | None = None,
    ):

        if artifact_path is None:
            artifact_path = (
                Path(__file__).parent
                / "artifacts"
                / "preference_signal_detector_v1"
            )

        self.artifact_path = Path(artifact_path)

        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Preference detector artifact not found: "
                f"{self.artifact_path}"
            )

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.device = torch.device(self.device)

        self._load_artifacts()


    def _load_artifacts(self):

        model_path = self.artifact_path / "model"
        tokenizer_path = self.artifact_path / "tokenizer"

        labels_path = self.artifact_path / "labels.json"
        threshold_path = self.artifact_path / "thresholds.json"


        if not model_path.exists():
            raise FileNotFoundError(model_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(tokenizer_path)


        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path
        )


        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )


        self.model.to(self.device)
        self.model.eval()


        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)


        self.id2label = {
            int(k): v
            for k, v in labels["id2label"].items()
        }


        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold_data = json.load(f)


        self.threshold = float(
            threshold_data[
                "preference_signal_probability_threshold"
            ]
        )


    @torch.no_grad()
    def predict(
        self,
        text: str,
    ) -> PreferenceSignalResult:
        """
        Predict if the text contains
        a preference signal.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Input text must be a string"
            )


        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )


        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }


        outputs = self.model(**inputs)


        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]


        preference_probability = (
            probabilities[1]
            .detach()
            .cpu()
            .item()
        )


        has_signal = (
            preference_probability
            >= self.threshold
        )


        label_id = 1 if has_signal else 0


        return PreferenceSignalResult(
            has_preference_signal=has_signal,
            label=self.id2label[label_id],
            probability=round(
                preference_probability,
                6
            ),
            threshold=self.threshold,
        )
    