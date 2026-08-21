from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

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
# CONSTANTS
# =====================================================================

UNRESOLVED_FIELD = "__UNRESOLVED__"
UNSPECIFIED_ROLE = "unspecified"


# =====================================================================
# RUNTIME RESULT
# =====================================================================


@dataclass(frozen=True)
class SemanticLinkerPrediction:
    """
    Result of one Semantic Linker inference.

    V3.3 semantics
    --------------

    raw_field:
        Best compatible FIELD predicted by the FIELD head.

    raw_role:
        Best compatible ROLE predicted by the ROLE head after FIELD
        selection.

    field_accepted:
        True when:
        - FIELD is not __UNRESOLVED__
        - FIELD calibration accepts the prediction
        - deterministic production safety gate accepts it

    role_abstained:
        True only when FIELD is valid/resolved but ROLE calibration is
        not sufficiently confident.

        In that situation:

            final_field = predicted FIELD
            final_role = "unspecified"

        This is the important V3.3 behavior: uncertain ROLE must NOT
        destroy a reliable FIELD prediction.

    accepted:
        True when a usable SemanticLink can be emitted.

        Therefore a ROLE abstention does NOT imply accepted=False.
        A resolved FIELD + ROLE=unspecified is still a valid link.

    confidence / margin:
        Backward-compatible aliases for the confidence/margin of the
        final semantic decision:

        - normally ROLE confidence/margin
        - FIELD confidence/margin when FIELD is rejected

        New code should prefer the explicit:
            field_confidence
            field_margin
            role_confidence
            role_margin
    """

    quantity_id: str

    raw_field: str
    raw_role: str

    field_confidence: float
    field_margin: float

    role_confidence: float
    role_margin: float

    field_accepted: bool
    role_abstained: bool
    safety_gate_passed: bool

    accepted: bool

    final_field: str
    final_role: str

    # Backward-compatible aggregate values.
    confidence: float
    margin: float

    link: Optional[SemanticLink] = None

    def to_dict(self) -> dict:
        return {
            "quantity_id": self.quantity_id,

            "raw_field": self.raw_field,
            "raw_role": self.raw_role,

            "field_confidence": self.field_confidence,
            "field_margin": self.field_margin,

            "role_confidence": self.role_confidence,
            "role_margin": self.role_margin,

            "field_accepted": self.field_accepted,
            "role_abstained": self.role_abstained,
            "safety_gate_passed": self.safety_gate_passed,

            "accepted": self.accepted,

            "final_field": self.final_field,
            "final_role": self.final_role,

            # Keep these for code written against the old runtime.
            "confidence": self.confidence,
            "margin": self.margin,

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
          /     \\
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Same representation used during training.
        hidden = output.last_hidden_state[:, 0, :]

        hidden = self.dropout(hidden)

        return (
            self.field_head(hidden),
            self.role_head(hidden),
        )


# =====================================================================
# SEMANTIC LINKER V3.3 RUNTIME
# =====================================================================


class SemanticLinkerRuntime:
    """
    Semantic Linker V3.3 inference runtime.

    Runtime pipeline
    ----------------

        original text + Quantity
                ↓
        SemanticLinkerInputBuilder
                ↓
        text with [Q] target [/Q]
                ↓
             XLM-R
                ↓
        FIELD logits + ROLE logits
                ↓
       FIELD compatibility mask
                ↓
          FIELD prediction
                ↓
       FIELD calibration by dimension
                ↓
       deterministic safety gate
                ↓
             FIELD OK?
             /       \\
           no         yes
           ↓           ↓
       UNRESOLVED   ROLE mask
                       ↓
                  ROLE prediction
                       ↓
                 ROLE calibration
                       ↓
                 ROLE reliable?
                  /          \\
                yes          no
                 ↓            ↓
             predicted     unspecified
               ROLE       role_abstained
                  \\          /
                   SemanticLink

    Important
    ---------
    This runtime does NOT call the LLM fallback itself.
    The surrounding cascade remains responsible for that decision.
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

    # Unitless UNKNOWN values may only be accepted directly for
    # count-like concepts with explicit lexical evidence.
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
            "clent",
            "clents",
            "clint",
            "clints",
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
        # Validate V3.3 calibration contract
        # -------------------------------------------------------------

        self._validate_threshold_contract()

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

        # V3.3 mappings + legacy evaluation compatibility.
        # Supports both:
        #   id_to_field[class_id]
        #   id_to_field[field_name]

        self.id_to_field = {}

        for index, label in enumerate(self.field_labels):
            self.id_to_field[index] = label
            self.id_to_field[label] = label

        self.id_to_role = {}

        for index, label in enumerate(self.role_labels):
            self.id_to_role[index] = label
            self.id_to_role[label] = label

        # -------------------------------------------------------------
        # V3.3 hierarchical calibration
        # -------------------------------------------------------------

        self.field_thresholds = dict(
            self.thresholds[
                "field_thresholds_by_dimension"
            ]
        )

        self.role_thresholds = dict(
            self.thresholds[
                "role_thresholds_by_field"
            ]
        )

        self.policy_version = str(
            self.thresholds.get(
                "policy_version",
                "unknown",
            )
        )

        self.calibration_revision = str(
            self.thresholds.get(
                "calibration_revision",
                self.policy_version,
            )
        )

        # Backward compatibility for legacy evaluation scripts.
        # V3.3 uses field/role-specific thresholds internally.
        self.confidence_threshold = 0.48
        self.margin_threshold = 0.05

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

        encoder_hidden_size = int(
            self.model.encoder.config.hidden_size
        )

        if encoder_hidden_size != hidden_size:
            raise RuntimeError(
                "Encoder/head hidden-size mismatch: "
                f"encoder={encoder_hidden_size}, "
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
    # ARTIFACT DISCOVERY / VALIDATION
    # =================================================================

    @staticmethod
    def _discover_artifact_dir() -> Path:
        """
        Automatically discover the Semantic Linker artifact under:

            requirement_extractor_v2/artifacts/

        If several Semantic Linker artifacts exist, artifact_dir must be
        supplied explicitly. This avoids accidentally loading the old V1.
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
                "Artifacts directory not found: "
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
                "Several Semantic Linker artifacts were found. "
                "Pass artifact_dir explicitly to avoid loading "
                "the wrong model.\n"
                f"Candidates: {candidates}"
            )

        return candidates[0]

    def _validate_artifact_directory(
        self,
    ) -> None:

        if not self.artifact_dir.is_dir():
            raise FileNotFoundError(
                "Artifact directory not found: "
                f"{self.artifact_dir}"
            )

        encoder_dir = (
            self.artifact_dir
            / "encoder"
        )

        tokenizer_dir = (
            self.artifact_dir
            / "tokenizer"
        )

        if not encoder_dir.is_dir():
            raise FileNotFoundError(
                "Missing encoder/ directory"
            )

        if not tokenizer_dir.is_dir():
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

    def _validate_threshold_contract(
        self,
    ) -> None:
        """
        Fail immediately if an old V1/V2-style global threshold file is
        accidentally used with the V3.3 runtime.
        """

        required = (
            "field_thresholds_by_dimension",
            "role_thresholds_by_field",
        )

        missing = [
            key
            for key in required
            if key not in self.thresholds
        ]

        if missing:
            raise RuntimeError(
                "Incompatible thresholds.json. "
                "SemanticLinkerRuntime V3.3 requires hierarchical "
                "FIELD/ROLE calibration.\n"
                f"Missing keys: {missing}\n"
                f"Loaded from: "
                f"{self.artifact_dir / 'thresholds.json'}"
            )

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
    # COMPATIBILITY HELPERS
    # =================================================================

    def _allowed_fields(
        self,
        dimension: str,
    ) -> Tuple[str, ...]:

        fields_by_dimension = (
            self.compatibility[
                "allowed_fields_by_dimension"
            ]
        )

        if dimension not in fields_by_dimension:
            raise ValueError(
                "Unsupported quantity dimension: "
                f"{dimension!r}"
            )

        allowed = set(
            fields_by_dimension[
                dimension
            ]
        )

        # Preserve exact exported label order.
        result = tuple(
            field
            for field in self.field_labels
            if field in allowed
        )

        if not result:
            raise RuntimeError(
                "No compatible FIELD label for "
                f"dimension={dimension!r}"
            )

        return result

    def _allowed_roles(
        self,
        field: str,
    ) -> Tuple[str, ...]:

        roles_by_field = (
            self.compatibility[
                "allowed_roles_by_field"
            ]
        )

        if field not in roles_by_field:
            raise RuntimeError(
                "No ROLE compatibility rule for "
                f"FIELD={field!r}"
            )

        allowed = set(
            roles_by_field[field]
        )

        result = tuple(
            role
            for role in self.role_labels
            if role in allowed
        )

        if not result:
            raise RuntimeError(
                "No compatible ROLE labels for "
                f"FIELD={field!r}"
            )

        return result

    # =================================================================
    # GENERIC MASKED DECODER
    # =================================================================

    @staticmethod
    def _decode_subset(
        logits: torch.Tensor,
        label_to_id: dict,
        allowed_labels: Sequence[str],
    ) -> Tuple[str, float, float]:
        """
        Decode one classification head only among allowed labels.

        Probabilities are normalized across the compatibility-constrained
        subset, exactly as required by hierarchical V3.x inference.

        Returns:
            best_label
            confidence
            margin
        """

        if logits.ndim != 2:
            raise ValueError(
                "Expected logits with shape [batch, classes], "
                f"got {tuple(logits.shape)}"
            )

        if logits.shape[0] != 1:
            raise ValueError(
                "SemanticLinkerRuntime currently expects "
                "batch size 1."
            )

        ids = [
            label_to_id[label]
            for label in allowed_labels
        ]

        subset_logits = logits[
            0,
            ids,
        ]

        probabilities = torch.softmax(
            subset_logits,
            dim=0,
        )

        order = torch.argsort(
            probabilities,
            descending=True,
        )

        best_local_index = int(
            order[0].item()
        )

        best_label = (
            allowed_labels[
                best_local_index
            ]
        )

        confidence = float(
            probabilities[
                best_local_index
            ].item()
        )

        if len(order) > 1:

            second_local_index = int(
                order[1].item()
            )

            second_probability = float(
                probabilities[
                    second_local_index
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
            best_label,
            confidence,
            margin,
        )

    # =================================================================
    # HIERARCHICAL FIELD DECODER
    # =================================================================

    def _decode_field(
        self,
        field_logits: torch.Tensor,
        dimension: str,
    ) -> Tuple[str, float, float]:

        allowed_fields = (
            self._allowed_fields(
                dimension
            )
        )

        return self._decode_subset(
            logits=field_logits,
            label_to_id=self.field_to_id,
            allowed_labels=allowed_fields,
        )

    # =================================================================
    # HIERARCHICAL ROLE DECODER
    # =================================================================

    def _decode_role(
        self,
        role_logits: torch.Tensor,
        field: str,
    ) -> Tuple[str, float, float]:

        allowed_roles = (
            self._allowed_roles(
                field
            )
        )

        return self._decode_subset(
            logits=role_logits,
            label_to_id=self.role_to_id,
            allowed_labels=allowed_roles,
        )

    # =================================================================
    # V3.3 CALIBRATION
    # =================================================================

    def _field_threshold_for_dimension(
        self,
        dimension: str,
    ) -> Tuple[float, float]:

        config = (
            self.field_thresholds
            .get(dimension)
        )

        if config is None:
            raise RuntimeError(
                "No FIELD calibration threshold "
                f"for dimension={dimension!r}"
            )

        confidence_threshold = float(
            config.get(
                "confidence_threshold",
                0.0,
            )
        )

        margin_threshold = float(
            config.get(
                "margin_threshold",
                0.0,
            )
        )

        return (
            confidence_threshold,
            margin_threshold,
        )

    def _role_threshold_for_field(
        self,
        field: str,
    ) -> Tuple[float, float]:

        config = (
            self.role_thresholds
            .get(field)
        )

        if config is None:
            raise RuntimeError(
                "No ROLE calibration threshold "
                f"for FIELD={field!r}"
            )

        confidence_threshold = float(
            config.get(
                "confidence_threshold",
                0.48,
            )
        )

        margin_threshold = float(
            config.get(
                "margin_threshold",
                0.05,
            )
        )

        return (
            confidence_threshold,
            margin_threshold,
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
        Return local context around the target quantity.

        Prevents a keyword belonging to another quantity elsewhere in a
        multi-quantity sentence from validating the wrong target.
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

        return text[
            start:end
        ]

    @staticmethod
    def _contains_semantic_anchor(
        text: str,
        anchors: Sequence[str],
    ) -> bool:

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
        Deterministic guard executed after FIELD prediction.

        Known dimensions:
            keep trained compatibility behavior.

        UNKNOWN:
            direct production acceptance is restricted to count-like
            fields with explicit local lexical support.

        This guard may veto a prediction but must never invent a field.
        """

        dimension = (
            quantity.dimension.value
        )

        if dimension != "unknown":
            return True

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
        Predict semantic FIELD + ROLE for exactly one detected Quantity.

        V3.3 rule:
            FIELD and ROLE abstention are independent.

        A rejected ROLE never converts a reliable FIELD into
        __UNRESOLVED__.
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

        field_logits = (
            field_logits
            .float()
            .cpu()
        )

        role_logits = (
            role_logits
            .float()
            .cpu()
        )

        dimension = (
            quantity.dimension.value
        )

        # =============================================================
        # STAGE 1: FIELD
        # =============================================================

        (
            raw_field,
            field_confidence,
            field_margin,
        ) = self._decode_field(
            field_logits=field_logits,
            dimension=dimension,
        )

        (
            field_conf_threshold,
            field_margin_threshold,
        ) = (
            self._field_threshold_for_dimension(
                dimension
            )
        )

        field_calibration_passed = (
            raw_field
            != UNRESOLVED_FIELD
            and
            field_confidence
            >= field_conf_threshold
            and
            field_margin
            >= field_margin_threshold
        )

        safety_gate_passed = False

        if field_calibration_passed:

            safety_gate_passed = (
                self._passes_production_safety_gate(
                    text=text,
                    quantity=quantity,
                    predicted_field=
                        raw_field,
                )
            )

        field_accepted = (
            field_calibration_passed
            and
            safety_gate_passed
        )

        # -------------------------------------------------------------
        # FIELD rejection
        # -------------------------------------------------------------

        if not field_accepted:

            return SemanticLinkerPrediction(
                quantity_id=quantity.id,

                raw_field=raw_field,
                raw_role=UNSPECIFIED_ROLE,

                field_confidence=
                    field_confidence,
                field_margin=
                    field_margin,

                role_confidence=0.0,
                role_margin=0.0,

                field_accepted=False,
                role_abstained=False,
                safety_gate_passed=
                    safety_gate_passed,

                accepted=False,

                final_field=
                    UNRESOLVED_FIELD,
                final_role=
                    UNSPECIFIED_ROLE,

                # Backward compatibility.
                confidence=
                    field_confidence,
                margin=
                    field_margin,

                link=None,
            )

        # =============================================================
        # STAGE 2: ROLE, conditioned on resolved FIELD
        # =============================================================

        (
            raw_role,
            role_confidence,
            role_margin,
        ) = self._decode_role(
            role_logits=role_logits,
            field=raw_field,
        )

        (
            role_conf_threshold,
            role_margin_threshold,
        ) = (
            self._role_threshold_for_field(
                raw_field
            )
        )

        # A raw "unspecified" is a legitimate model prediction,
        # NOT an abstention.
        if raw_role == UNSPECIFIED_ROLE:

            final_role = (
                UNSPECIFIED_ROLE
            )

            role_abstained = False

        else:

            role_calibration_passed = (
                role_confidence
                >= role_conf_threshold
                and
                role_margin
                >= role_margin_threshold
            )

            if role_calibration_passed:

                final_role = raw_role

                role_abstained = False

            else:

                # V3.3 key behavior:
                #
                # keep FIELD
                # abstain only on ROLE
                final_role = (
                    UNSPECIFIED_ROLE
                )

                role_abstained = True

        final_field = raw_field

        # =============================================================
        # Convert FIELD to official ParamName
        # =============================================================

        try:
            param_name = ParamName(
                final_field
            )

        except ValueError as exc:
            raise RuntimeError(
                "Semantic Linker predicted "
                "unknown Requirement field: "
                f"{final_field!r}"
            ) from exc

        # =============================================================
        # Convert ROLE to official SemanticRole
        # =============================================================

        try:
            semantic_role = (
                SemanticRole(
                    final_role
                )
            )

        except ValueError as exc:
            raise RuntimeError(
                "Semantic Linker predicted "
                "unknown semantic role: "
                f"{final_role!r}"
            ) from exc

        # =============================================================
        # Build SemanticLink
        # =============================================================

        link = SemanticLink(
            quantity_id=quantity.id,
            field=param_name,
            role=semantic_role,
            evidence=text,
            resolver=
                "semantic_linker_xlmr_v3_3",
        )

        # -------------------------------------------------------------
        # Backward-compatible confidence
        # -------------------------------------------------------------
        #
        # If ROLE abstains, FIELD is the actual accepted decision.
        # Otherwise ROLE is the last hierarchical decision.

        if role_abstained:

            aggregate_confidence = (
                field_confidence
            )

            aggregate_margin = (
                field_margin
            )

        else:

            aggregate_confidence = (
                role_confidence
            )

            aggregate_margin = (
                role_margin
            )

        return SemanticLinkerPrediction(
            quantity_id=quantity.id,

            raw_field=raw_field,
            raw_role=raw_role,

            field_confidence=
                field_confidence,
            field_margin=
                field_margin,

            role_confidence=
                role_confidence,
            role_margin=
                role_margin,

            field_accepted=True,
            role_abstained=
                role_abstained,
            safety_gate_passed=True,

            # A resolved FIELD produces a usable SemanticLink even if
            # ROLE abstains to unspecified.
            accepted=True,

            final_field=
                final_field,
            final_role=
                final_role,

            confidence=
                aggregate_confidence,
            margin=
                aggregate_margin,

            link=link,
        )


    # =================================================================
    # LEGACY EVALUATION COMPATIBILITY (V3.2 -> V3.3)
    # =================================================================

    def _decode_valid_pair(
        self,
        field_logits: torch.Tensor,
        role_logits: torch.Tensor,
        dimension: str,
    ):
        """
        Compatibility wrapper for old evaluation scripts.

        Older V3.2 evaluation code expected a single pair decoder:
            _decode_valid_pair()

        V3.3 uses hierarchical decoding:
            FIELD -> ROLE

        This wrapper preserves the old API without changing the V3.3
        inference behavior.
        """

        (
            field,
            field_confidence,
            field_margin,
        ) = self._decode_field(
            field_logits=field_logits,
            dimension=dimension,
        )

        (
            role,
            role_confidence,
            role_margin,
        ) = self._decode_role(
            role_logits=role_logits,
            field=field,
        )

        # Legacy evaluation scripts expect exactly four values:
        # field, role, confidence, margin
        #
        # V3.3 internally keeps the hierarchical FIELD -> ROLE logic.
        # The wrapper only exposes the old interface.

        return (
            field,
            role,
            role_confidence,
            role_margin,
        )

    # =================================================================
    # INFORMATION / DEBUG
    # =================================================================

    def info(self) -> dict:

        role_threshold_summary = {}

        for field, config in (
            self.role_thresholds.items()
        ):

            role_threshold_summary[field] = {
                "confidence_threshold":
                    float(
                        config.get(
                            "confidence_threshold",
                            0.0,
                        )
                    ),

                "margin_threshold":
                    float(
                        config.get(
                            "margin_threshold",
                            0.0,
                        )
                    ),
            }

        field_threshold_summary = {}

        for dimension, config in (
            self.field_thresholds.items()
        ):

            field_threshold_summary[
                dimension
            ] = {
                "confidence_threshold":
                    float(
                        config.get(
                            "confidence_threshold",
                            0.0,
                        )
                    ),

                "margin_threshold":
                    float(
                        config.get(
                            "margin_threshold",
                            0.0,
                        )
                    ),
            }

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

            "policy_version":
                self.policy_version,

            "calibration_revision":
                self.calibration_revision,

            "decoder":
                self.thresholds.get(
                    "decoder",
                    "FIELD_then_ROLE",
                ),

            "field_thresholds_by_dimension":
                field_threshold_summary,

            "role_thresholds_by_field":
                role_threshold_summary,

            "unknown_direct_fields":
                sorted(
                    self._UNKNOWN_DIRECT_FIELDS
                ),

            "production_safety_gate":
                True,

            "hierarchical_decoding":
                True,

            "role_abstention_preserves_field":
                True,
        }