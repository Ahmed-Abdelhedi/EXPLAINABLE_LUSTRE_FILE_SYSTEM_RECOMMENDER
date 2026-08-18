from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..models import (
    ParamName,
    Quantity,
    SemanticLink,
    SemanticRole,
)

from .input_builder import SemanticLinkerInputBuilder


# =====================================================================
# RUNTIME RESULT
# =====================================================================


@dataclass(frozen=True)
class SemanticLinkerPrediction:
    """
    Result of one Semantic Linker inference.

    raw_field / raw_role:
        Best FIELD/ROLE pair predicted after compatibility constraints.

    accepted:
        True only when:
        - model confidence passes calibrated thresholds
        - production safety gate also accepts the mapping

    link:
        SemanticLink ready for the next pipeline stage when accepted.
        None when the Semantic Linker abstains.
    """

    quantity_id: str

    raw_field: str
    raw_role: str

    confidence: float
    margin: float

    accepted: bool

    final_field: str
    final_role: str

    link: Optional[SemanticLink] = None

    def to_dict(self) -> dict:
        return {
            "quantity_id": self.quantity_id,
            "raw_field": self.raw_field,
            "raw_role": self.raw_role,
            "confidence": self.confidence,
            "margin": self.margin,
            "accepted": self.accepted,
            "final_field": self.final_field,
            "final_role": self.final_role,
            "link": (
                None
                if self.link is None
                else self.link.to_dict()
            ),
        }


# =====================================================================
# XLM-R + TWO CLASSIFICATION HEADS
# =====================================================================


class _XLMRSemanticLinkerModel(nn.Module):
    """
    Runtime reconstruction of the trained Semantic Linker.

    Architecture:

        XLM-R encoder
             ↓
        first <s> token
             ↓
          Dropout
          /     \
       FIELD    ROLE
       head     head
    """

    def __init__(
        self,
        encoder_dir: Path,
        hidden_size: int,
        num_fields: int,
        num_roles: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            str(encoder_dir),
            local_files_only=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.field_head = nn.Linear(
            hidden_size,
            num_fields,
        )

        self.role_head = nn.Linear(
            hidden_size,
            num_roles,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Representation used during training:
        # first token <s>.
        hidden = output.last_hidden_state[:, 0, :]

        hidden = self.dropout(hidden)

        return (
            self.field_head(hidden),
            self.role_head(hidden),
        )


# =====================================================================
# SEMANTIC LINKER RUNTIME
# =====================================================================


class SemanticLinkerRuntime:
    """
    Local inference runtime for the fine-tuned XLM-R Semantic Linker.

    Pipeline:

        original text + Quantity
                ↓
        SemanticLinkerInputBuilder
                ↓
        [Q] target quantity [/Q]
                ↓
             XLM-R
                ↓
        FIELD logits + ROLE logits
                ↓
        compatibility-constrained decoding
                ↓
        confidence / margin
                ↓
        calibrated thresholds
                ↓
        production safety gate
             /            \
         ACCEPT          ABSTAIN
                           ↓
                     LLM fallback
                     (by cascade)

    Important:
        This component does NOT call the LLM itself.
    """

    REQUIRED_FILES = (
        "classifier_heads.pt",
        "labels.json",
        "compatibility.json",
        "thresholds.json",
        "training_config.json",
    )

    # =================================================================
    # PRODUCTION SAFETY POLICY
    # =================================================================

    # UNKNOWN means that QuantityScanner did not detect a physical unit
    # or other deterministic dimension.
    #
    # The trained model keeps UNKNOWN broad because this was part of the
    # training compatibility space. In production, however, directly
    # accepting UNKNOWN as power, throughput, capacity, file size,
    # budget, etc. is unsafe.
    #
    # Unitless UNKNOWN quantities may therefore be accepted directly
    # only for explicit count-like concepts.
    _UNKNOWN_DIRECT_FIELDS = frozenset(
        {
            "client_count",
            "total_file_count",
        }
    )

    _UNKNOWN_FIELD_ANCHORS = {
        "client_count": (
            "client",
            "clients",
            "host",
            "hosts",
            "node",
            "nodes",
            "endpoint",
            "endpoints",
            "machine",
            "machines",
            "server",
            "servers",
            "worker",
            "workers",
            "hôte",
            "hôtes",
            "noeud",
            "noeuds",
            "nœud",
            "nœuds",
            "serveur",
            "serveurs",
        ),

        "total_file_count": (
            "file",
            "files",
            "object",
            "objects",
            "entry",
            "entries",
            "inode",
            "inodes",
            "fichier",
            "fichiers",
            "objet",
            "objets",
            "entrée",
            "entrées",
        ),
    }

    # Number of characters taken around the target quantity when
    # checking semantic anchors.
    #
    # This avoids using a keyword associated with another quantity
    # elsewhere in a long multi-quantity message.
    _SAFETY_CONTEXT_WINDOW = 80

    # =================================================================
    # INITIALIZATION
    # =================================================================

    def __init__(
        self,
        artifact_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> None:

        # -------------------------------------------------------------
        # Artifact directory
        # -------------------------------------------------------------

        if artifact_dir is None:
            artifact_dir = self._discover_artifact_dir()

        self.artifact_dir = Path(
            artifact_dir
        ).resolve()

        self._validate_artifact_directory()

        # -------------------------------------------------------------
        # Device
        # -------------------------------------------------------------

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        # -------------------------------------------------------------
        # Metadata
        # -------------------------------------------------------------

        self.labels = self._read_json(
            "labels.json"
        )

        self.compatibility = self._read_json(
            "compatibility.json"
        )

        self.thresholds = self._read_json(
            "thresholds.json"
        )

        self.training_config = self._read_json(
            "training_config.json"
        )

        # -------------------------------------------------------------
        # Labels
        # -------------------------------------------------------------

        self.field_labels = list(
            self.labels["field_labels"]
        )

        self.role_labels = list(
            self.labels["role_labels"]
        )

        self.field_to_id = {
            label: index
            for index, label
            in enumerate(self.field_labels)
        }

        self.role_to_id = {
            label: index
            for index, label
            in enumerate(self.role_labels)
        }

        self.id_to_field = {
            index: label
            for index, label
            in enumerate(self.field_labels)
        }

        self.id_to_role = {
            index: label
            for index, label
            in enumerate(self.role_labels)
        }

        # -------------------------------------------------------------
        # Calibrated selective thresholds
        # -------------------------------------------------------------

        self.confidence_threshold = float(
            self.thresholds[
                "confidence_threshold"
            ]
        )

        self.margin_threshold = float(
            self.thresholds[
                "margin_threshold"
            ]
        )

        self.max_length = int(
            self.training_config.get(
                "max_length",
                96,
            )
        )

        # -------------------------------------------------------------
        # Tokenizer
        # -------------------------------------------------------------

        tokenizer_dir = (
            self.artifact_dir
            / "tokenizer"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True,
        )

        self._validate_special_tokens()

        # -------------------------------------------------------------
        # Classification heads
        # -------------------------------------------------------------

        heads_path = (
            self.artifact_dir
            / "classifier_heads.pt"
        )

        try:
            heads = torch.load(
                heads_path,
                map_location="cpu",
                weights_only=True,
            )

        except TypeError:
            # Compatibility with older PyTorch versions.
            heads = torch.load(
                heads_path,
                map_location="cpu",
            )

        hidden_size = int(
            heads["hidden_size"]
        )

        num_fields = int(
            heads["num_fields"]
        )

        num_roles = int(
            heads["num_roles"]
        )

        dropout = float(
            heads["dropout"]
        )

        # -------------------------------------------------------------
        # Contract validation
        # -------------------------------------------------------------

        if num_fields != len(
            self.field_labels
        ):
            raise RuntimeError(
                "FIELD contract mismatch: "
                f"heads={num_fields}, "
                f"labels={len(self.field_labels)}"
            )

        if num_roles != len(
            self.role_labels
        ):
            raise RuntimeError(
                "ROLE contract mismatch: "
                f"heads={num_roles}, "
                f"labels={len(self.role_labels)}"
            )

        # -------------------------------------------------------------
        # Rebuild model
        # -------------------------------------------------------------

        encoder_dir = (
            self.artifact_dir
            / "encoder"
        )

        self.model = _XLMRSemanticLinkerModel(
            encoder_dir=encoder_dir,
            hidden_size=hidden_size,
            num_fields=num_fields,
            num_roles=num_roles,
            dropout=dropout,
        )

        if (
            self.model.encoder.config.hidden_size
            != hidden_size
        ):
            raise RuntimeError(
                "Encoder/head hidden-size mismatch: "
                f"encoder="
                f"{self.model.encoder.config.hidden_size}, "
                f"heads={hidden_size}"
            )

        self.model.field_head.load_state_dict(
            heads["field_head"]
        )

        self.model.role_head.load_state_dict(
            heads["role_head"]
        )

        self.model.to(self.device)
        self.model.eval()

        # -------------------------------------------------------------
        # Input builder
        # -------------------------------------------------------------

        self.input_builder = (
            SemanticLinkerInputBuilder()
        )

    # =================================================================
    # ARTIFACT DISCOVERY
    # =================================================================

    @staticmethod
    def _discover_artifact_dir() -> Path:
        """
        Automatically discover the Semantic Linker export under:

            requirement_extractor_v2/artifacts/
        """

        package_root = (
            Path(__file__)
            .resolve()
            .parent
            .parent
        )

        artifacts_root = (
            package_root
            / "artifacts"
        )

        if not artifacts_root.exists():
            raise FileNotFoundError(
                f"Artifacts directory not found: "
                f"{artifacts_root}"
            )

        candidates = []

        for child in artifacts_root.iterdir():

            if not child.is_dir():
                continue

            if (
                (child / "encoder").is_dir()
                and
                (child / "tokenizer").is_dir()
                and
                (
                    child
                    / "classifier_heads.pt"
                ).is_file()
            ):
                candidates.append(child)

        if len(candidates) == 0:
            raise FileNotFoundError(
                "No Semantic Linker model found "
                f"inside {artifacts_root}"
            )

        if len(candidates) > 1:
            raise RuntimeError(
                "Several Semantic Linker models were found. "
                "Pass artifact_dir explicitly.\n"
                f"Candidates: {candidates}"
            )

        return candidates[0]

    def _validate_artifact_directory(
        self,
    ) -> None:

        if not self.artifact_dir.is_dir():
            raise FileNotFoundError(
                f"Artifact directory not found: "
                f"{self.artifact_dir}"
            )

        if not (
            self.artifact_dir
            / "encoder"
        ).is_dir():
            raise FileNotFoundError(
                "Missing encoder/ directory"
            )

        if not (
            self.artifact_dir
            / "tokenizer"
        ).is_dir():
            raise FileNotFoundError(
                "Missing tokenizer/ directory"
            )

        for filename in self.REQUIRED_FILES:

            path = (
                self.artifact_dir
                / filename
            )

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing model artifact: {path}"
                )

    def _read_json(
        self,
        filename: str,
    ) -> dict:

        path = (
            self.artifact_dir
            / filename
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(handle)

    def _validate_special_tokens(
        self,
    ) -> None:

        for token in (
            "[Q]",
            "[/Q]",
        ):
            token_id = (
                self.tokenizer
                .convert_tokens_to_ids(token)
            )

            if (
                token_id is None
                or
                token_id
                == self.tokenizer.unk_token_id
            ):
                raise RuntimeError(
                    "Tokenizer does not contain "
                    f"required special token {token!r}"
                )

    # =================================================================
    # COMPATIBILITY DECODER
    # =================================================================

    def _valid_pairs(
        self,
        dimension: str,
    ):
        """
        Build valid FIELD/ROLE pairs using compatibility.json.

        The exact exported FIELD and ROLE orders are preserved.
        """

        fields_by_dimension = (
            self.compatibility[
                "allowed_fields_by_dimension"
            ]
        )

        roles_by_field = (
            self.compatibility[
                "allowed_roles_by_field"
            ]
        )

        if dimension not in fields_by_dimension:
            raise ValueError(
                "Unsupported quantity dimension: "
                f"{dimension!r}"
            )

        allowed_fields = set(
            fields_by_dimension[
                dimension
            ]
        )

        pairs = []

        for field in self.field_labels:

            if field not in allowed_fields:
                continue

            allowed_roles = set(
                roles_by_field[field]
            )

            field_id = (
                self.field_to_id[field]
            )

            for role in self.role_labels:

                if role not in allowed_roles:
                    continue

                role_id = (
                    self.role_to_id[role]
                )

                pairs.append(
                    (
                        field_id,
                        role_id,
                    )
                )

        if not pairs:
            raise RuntimeError(
                "No compatible FIELD/ROLE pair "
                f"for dimension={dimension!r}"
            )

        return pairs

    def _decode_valid_pair(
        self,
        field_logits: torch.Tensor,
        role_logits: torch.Tensor,
        dimension: str,
    ):
        """
        Constrained decoding:

            pair_score =
                log P(FIELD)
                +
                log P(ROLE)

        Softmax is then applied only across valid FIELD/ROLE pairs.
        """

        field_log_probs = (
            torch.log_softmax(
                field_logits,
                dim=-1,
            )
        )

        role_log_probs = (
            torch.log_softmax(
                role_logits,
                dim=-1,
            )
        )

        pairs = self._valid_pairs(
            dimension
        )

        scores = torch.stack(
            [
                field_log_probs[
                    0,
                    field_id,
                ]
                +
                role_log_probs[
                    0,
                    role_id,
                ]

                for field_id, role_id
                in pairs
            ]
        )

        probabilities = torch.softmax(
            scores,
            dim=0,
        )

        order = torch.argsort(
            probabilities,
            descending=True,
        )

        best_index = int(
            order[0].item()
        )

        field_id, role_id = (
            pairs[best_index]
        )

        confidence = float(
            probabilities[
                best_index
            ].item()
        )

        if len(order) > 1:

            second_index = int(
                order[1].item()
            )

            second_probability = float(
                probabilities[
                    second_index
                ].item()
            )

        else:
            second_probability = 0.0

        margin = (
            confidence
            -
            second_probability
        )

        return (
            field_id,
            role_id,
            confidence,
            margin,
        )

    # =================================================================
    # PRODUCTION SAFETY GATE
    # =================================================================

    def _target_context(
        self,
        text: str,
        quantity: Quantity,
    ) -> str:
        """
        Return only a local context around the target quantity.

        This prevents a keyword associated with another quantity in the
        same sentence/message from validating the wrong target.
        """

        start = max(
            0,
            quantity.start
            - self._SAFETY_CONTEXT_WINDOW,
        )

        end = min(
            len(text),
            quantity.end
            + self._SAFETY_CONTEXT_WINDOW,
        )

        return text[start:end]

    @staticmethod
    def _contains_semantic_anchor(
        text: str,
        anchors,
    ) -> bool:
        """
        Check whether the local context contains explicit lexical
        evidence for a predicted count-like FIELD.

        This function never chooses a FIELD.
        It can only veto an unsupported prediction.
        """

        normalized = text.casefold()

        for anchor in anchors:

            pattern = (
                r"(?<!\w)"
                + re.escape(
                    anchor.casefold()
                )
                + r"(?!\w)"
            )

            if re.search(
                pattern,
                normalized,
            ):
                return True

        return False

    def _passes_production_safety_gate(
        self,
        text: str,
        quantity: Quantity,
        predicted_field: str,
    ) -> bool:
        """
        Conservative deterministic guard after ML decoding.

        For known dimensions:
            keep the normal trained behavior.

        For UNKNOWN/unitless quantities:
            only explicit count-like FIELD predictions may be accepted,
            and the local text must contain lexical support for that
            predicted FIELD.

        Otherwise Semantic Linker abstains and SelectiveCascade may call
        the LLM fallback.
        """

        if (
            quantity.dimension.value
            != "unknown"
        ):
            return True

        # Never directly accept a physical/non-count FIELD from an
        # UNKNOWN quantity.
        if (
            predicted_field
            not in self._UNKNOWN_DIRECT_FIELDS
        ):
            return False

        anchors = (
            self._UNKNOWN_FIELD_ANCHORS
            .get(
                predicted_field,
                (),
            )
        )

        local_context = (
            self._target_context(
                text=text,
                quantity=quantity,
            )
        )

        return self._contains_semantic_anchor(
            text=local_context,
            anchors=anchors,
        )

    # =================================================================
    # PUBLIC INFERENCE
    # =================================================================

    @torch.no_grad()
    def predict(
        self,
        text: str,
        quantity: Quantity,
        previous_question: Optional[str] = None,
    ) -> SemanticLinkerPrediction:
        """
        Predict FIELD + ROLE for exactly one already-detected Quantity.
        """

        # -------------------------------------------------------------
        # Build marked model input
        # -------------------------------------------------------------

        linker_input = (
            self.input_builder.build(
                text=text,
                quantity=quantity,
                previous_question=
                    previous_question,
            )
        )

        # -------------------------------------------------------------
        # Tokenization
        # -------------------------------------------------------------

        encoded = self.tokenizer(
            linker_input.marked_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = (
            encoded["input_ids"]
            .to(self.device)
        )

        attention_mask = (
            encoded["attention_mask"]
            .to(self.device)
        )

        # -------------------------------------------------------------
        # Model inference
        # -------------------------------------------------------------

        field_logits, role_logits = (
            self.model(
                input_ids=input_ids,
                attention_mask=
                    attention_mask,
            )
        )

        # -------------------------------------------------------------
        # Compatibility-constrained decode
        # -------------------------------------------------------------

        (
            field_id,
            role_id,
            confidence,
            margin,
        ) = self._decode_valid_pair(
            field_logits.float().cpu(),
            role_logits.float().cpu(),
            quantity.dimension.value,
        )

        raw_field = (
            self.id_to_field[
                field_id
            ]
        )

        raw_role = (
            self.id_to_role[
                role_id
            ]
        )

        # -------------------------------------------------------------
        # Original calibrated ML acceptance
        # -------------------------------------------------------------

        model_accepted = (
            raw_field
            != "__UNRESOLVED__"
            and
            confidence
            >= self.confidence_threshold
            and
            margin
            >= self.margin_threshold
        )

        # -------------------------------------------------------------
        # Additional deterministic production safety gate
        # -------------------------------------------------------------

        safety_gate_passed = (
            self._passes_production_safety_gate(
                text=text,
                quantity=quantity,
                predicted_field=
                    raw_field,
            )
        )

        accepted = (
            model_accepted
            and
            safety_gate_passed
        )

        # -------------------------------------------------------------
        # Abstention
        # -------------------------------------------------------------

        if not accepted:

            return SemanticLinkerPrediction(
                quantity_id=quantity.id,
                raw_field=raw_field,
                raw_role=raw_role,
                confidence=confidence,
                margin=margin,
                accepted=False,
                final_field=
                    "__UNRESOLVED__",
                final_role=
                    "unspecified",
                link=None,
            )

        # -------------------------------------------------------------
        # Convert FIELD to official ParamName
        # -------------------------------------------------------------

        try:
            param_name = ParamName(
                raw_field
            )

        except ValueError as exc:
            raise RuntimeError(
                "Semantic Linker predicted "
                "unknown Requirement field: "
                f"{raw_field!r}"
            ) from exc

        # -------------------------------------------------------------
        # Convert ROLE to official SemanticRole
        # -------------------------------------------------------------

        try:
            semantic_role = (
                SemanticRole(
                    raw_role
                )
            )

        except ValueError as exc:
            raise RuntimeError(
                "Semantic Linker predicted "
                "unknown semantic role: "
                f"{raw_role!r}"
            ) from exc

        # -------------------------------------------------------------
        # SemanticLink
        # -------------------------------------------------------------

        link = SemanticLink(
            quantity_id=quantity.id,
            field=param_name,
            role=semantic_role,
            evidence=text,
            resolver=
                "semantic_linker_xlmr",
        )

        return SemanticLinkerPrediction(
            quantity_id=quantity.id,
            raw_field=raw_field,
            raw_role=raw_role,
            confidence=confidence,
            margin=margin,
            accepted=True,
            final_field=raw_field,
            final_role=raw_role,
            link=link,
        )

    # =================================================================
    # INFORMATION / DEBUG
    # =================================================================

    def info(self) -> dict:
        return {
            "artifact_dir": str(
                self.artifact_dir
            ),

            "device": str(
                self.device
            ),

            "num_fields": len(
                self.field_labels
            ),

            "num_roles": len(
                self.role_labels
            ),

            "max_length":
                self.max_length,

            "confidence_threshold":
                self.confidence_threshold,

            "margin_threshold":
                self.margin_threshold,

            "unknown_direct_fields":
                sorted(
                    self._UNKNOWN_DIRECT_FIELDS
                ),

            "production_safety_gate":
                True,
        }