from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .labels import DIMENSIONS


@dataclass
class Layer2ModelOutput:
    presence_logits: torch.Tensor
    intensity_logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    presence_loss: Optional[torch.Tensor] = None
    intensity_loss: Optional[torch.Tensor] = None
    monotonicity_loss: Optional[torch.Tensor] = None


class XLMRPreferenceMultiTaskModel(
    nn.Module
):
    """
    XLM-R Base + shared multilingual encoder + 8 lightweight heads.

    4 presence heads:
        cost, power, performance, reliability

    4 ordinal intensity heads:
        each predicts four cumulative logits for five ordered levels.

    NO_SIGNAL comes from the presence head and is not an intensity class.

    Step 6.1B adds only the optional `encoder=` constructor argument so the
    frozen full state dict can be loaded without downloading base weights.
    """

    def __init__(
        self,
        encoder_name_or_path: str = (
            "FacebookAI/xlm-roberta-base"
        ),
        dropout: float = 0.10,
        monotonicity_weight: float = 0.10,
        *,
        encoder=None,
    ) -> None:
        super().__init__()

        self.encoder_name_or_path = (
            encoder_name_or_path
        )

        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(
                encoder_name_or_path
            )
        )

        hidden_size = int(
            self.encoder.config.hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.presence_heads = nn.ModuleDict(
            {
                dimension.value: nn.Linear(
                    hidden_size,
                    1,
                )
                for dimension in DIMENSIONS
            }
        )

        self.intensity_heads = nn.ModuleDict(
            {
                dimension.value: nn.Linear(
                    hidden_size,
                    4,
                )
                for dimension in DIMENSIONS
            }
        )

        self.monotonicity_weight = float(
            monotonicity_weight
        )

    @staticmethod
    def _ordinal_targets(
        labels: torch.Tensor,
    ) -> torch.Tensor:
        thresholds = torch.arange(
            4,
            device=labels.device,
        )

        return (
            labels.unsqueeze(
                -1
            )
            > thresholds
        ).float()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        presence_labels: Optional[
            torch.Tensor
        ] = None,
        intensity_labels: Optional[
            torch.Tensor
        ] = None,
        intensity_mask: Optional[
            torch.Tensor
        ] = None,
    ) -> Layer2ModelOutput:
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = self.dropout(
            encoded.last_hidden_state[
                :,
                0,
                :,
            ]
        )

        presence_logits = torch.cat(
            [
                self.presence_heads[
                    dimension.value
                ](
                    pooled
                )
                for dimension
                in DIMENSIONS
            ],
            dim=1,
        )

        intensity_logits = torch.stack(
            [
                self.intensity_heads[
                    dimension.value
                ](
                    pooled
                )
                for dimension
                in DIMENSIONS
            ],
            dim=1,
        )

        total_loss = None
        presence_loss = None
        intensity_loss = None
        monotonicity_loss = None

        if (
            presence_labels
            is not None
        ):
            presence_loss = (
                F.binary_cross_entropy_with_logits(
                    presence_logits,
                    presence_labels.float(),
                )
            )

        if (
            intensity_labels
            is not None
        ):
            if intensity_mask is None:
                intensity_mask = (
                    intensity_labels
                    >= 0
                )

            safe_labels = (
                intensity_labels.clamp(
                    min=0,
                    max=4,
                )
            )

            targets = self._ordinal_targets(
                safe_labels
            )

            raw_loss = (
                F.binary_cross_entropy_with_logits(
                    intensity_logits,
                    targets,
                    reduction="none",
                )
            ).mean(
                dim=-1
            )

            mask = (
                intensity_mask.float()
            )

            denominator = mask.sum().clamp(
                min=1.0
            )

            intensity_loss = (
                raw_loss
                * mask
            ).sum() / denominator

            cumulative = torch.sigmoid(
                intensity_logits
            )

            monotonic_violations = F.relu(
                cumulative[
                    :,
                    :,
                    1:,
                ]
                - cumulative[
                    :,
                    :,
                    :-1,
                ]
            ).mean(
                dim=-1
            )

            monotonicity_loss = (
                monotonic_violations
                * mask
            ).sum() / denominator

        pieces = [
            value
            for value in (
                presence_loss,
                intensity_loss,
            )
            if value is not None
        ]

        if pieces:
            total_loss = pieces[0]
            for piece in pieces[1:]:
                total_loss = total_loss + piece

            if (
                monotonicity_loss
                is not None
            ):
                total_loss = (
                    total_loss
                    + self.monotonicity_weight
                    * monotonicity_loss
                )

        return Layer2ModelOutput(
            presence_logits=presence_logits,
            intensity_logits=intensity_logits,
            loss=total_loss,
            presence_loss=presence_loss,
            intensity_loss=intensity_loss,
            monotonicity_loss=monotonicity_loss,
        )
