from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .labels import ACCESS_LABELS, HA_LABELS


class CategoricalBooleanMultiTaskXLMR(nn.Module):
    """
    Frozen production architecture.

    One shared XLM-R encoder with two independent 4-class semantic heads:
      - HA semantic head
      - access-type semantic head

    The classifier uses last_hidden_state[:, 0, :], not the optional XLM-R
    pooler.  The pooler is therefore intentionally frozen, matching the final
    two-GPU training architecture.
    """

    def __init__(
        self,
        *,
        base_model_name: str = "FacebookAI/xlm-roberta-base",
        dropout: float = 0.1,
        encoder=None,
    ) -> None:
        super().__init__()

        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(base_model_name)

        self.encoder = encoder

        pooler = getattr(self.encoder, "pooler", None)
        if pooler is not None:
            for parameter in pooler.parameters():
                parameter.requires_grad = False

        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ha_head = nn.Linear(hidden_size, len(HA_LABELS))
        self.access_head = nn.Linear(hidden_size, len(ACCESS_LABELS))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ha_labels: Optional[torch.Tensor] = None,
        access_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls = self.dropout(encoded.last_hidden_state[:, 0, :])
        ha_logits = self.ha_head(cls)
        access_logits = self.access_head(cls)

        output = {
            "ha_logits": ha_logits,
            "access_logits": access_logits,
        }

        if ha_labels is not None and access_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            ha_loss = loss_fn(ha_logits, ha_labels)
            access_loss = loss_fn(access_logits, access_labels)
            output["ha_loss"] = ha_loss
            output["access_loss"] = access_loss
            output["loss"] = ha_loss + access_loss

        return output
