from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import (
    Quantity,
    SemanticLink,
)

from .quantity_scanner import QuantityScanner
from .explicit_pattern_resolver import ExplicitPatternResolver

from .semantic_linker.runtime import (
    SemanticLinkerPrediction,
    SemanticLinkerRuntime,
)

from .llm_fallback_extractor import (
    LLMFallbackExtractor,
)


# =====================================================================
# TRACE
# =====================================================================


@dataclass
class QuantityRouteTrace:
    """
    Trace de résolution d'une Quantity à travers la cascade.
    """

    quantity_id: str

    explicit_attempted: bool = False
    explicit_resolved: bool = False

    semantic_attempted: bool = False
    semantic_accepted: bool = False
    semantic_confidence: Optional[float] = None
    semantic_margin: Optional[float] = None
    semantic_raw_field: Optional[str] = None
    semantic_raw_role: Optional[str] = None

    llm_attempted: bool = False
    llm_resolved: bool = False

    final_resolver: Optional[str] = None
    final_status: str = "unresolved"

    def to_dict(self) -> dict:
        return {
            "quantity_id": self.quantity_id,

            "explicit_attempted":
                self.explicit_attempted,
            "explicit_resolved":
                self.explicit_resolved,

            "semantic_attempted":
                self.semantic_attempted,
            "semantic_accepted":
                self.semantic_accepted,
            "semantic_confidence":
                self.semantic_confidence,
            "semantic_margin":
                self.semantic_margin,
            "semantic_raw_field":
                self.semantic_raw_field,
            "semantic_raw_role":
                self.semantic_raw_role,

            "llm_attempted":
                self.llm_attempted,
            "llm_resolved":
                self.llm_resolved,

            "final_resolver":
                self.final_resolver,
            "final_status":
                self.final_status,
        }


# =====================================================================
# CASCADE RESULT
# =====================================================================


@dataclass
class SelectiveCascadeResult:
    """
    Résultat de la cascade sémantique.

    Attention :
    les SemanticLink produits ici sont encore des CANDIDATS.

    Ils devront être envoyés ensuite au DeterministicVerifier.
    """

    text: str

    quantities: List[Quantity] = field(
        default_factory=list
    )

    links: List[SemanticLink] = field(
        default_factory=list
    )

    unresolved_quantity_ids: List[str] = field(
        default_factory=list
    )

    traces: Dict[
        str,
        QuantityRouteTrace,
    ] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "text": self.text,

            "quantities": [
                quantity.to_dict()
                for quantity in self.quantities
            ],

            "links": [
                link.to_dict()
                for link in self.links
            ],

            "unresolved_quantity_ids":
                list(
                    self.unresolved_quantity_ids
                ),

            "traces": {
                quantity_id: trace.to_dict()
                for quantity_id, trace
                in self.traces.items()
            },
        }


# =====================================================================
# SELECTIVE CASCADE
# =====================================================================


class SelectiveCascade:
    """
    Orchestrateur sélectif V2.

    Routing:

        QuantityScanner
              ↓
        ExplicitPatternResolver
              ↓ unresolved only
        SemanticLinkerRuntime
              ↓ rejected only
        LLMFallbackExtractor

    Important:
        cette classe ne réalise PAS encore la validation métier finale.
    """

    def __init__(
        self,
        scanner: Optional[
            QuantityScanner
        ] = None,
        explicit_resolver: Optional[
            ExplicitPatternResolver
        ] = None,
        semantic_linker: Optional[
            SemanticLinkerRuntime
        ] = None,
        llm_fallback: Optional[
            LLMFallbackExtractor
        ] = None,
    ) -> None:

        self.scanner = (
            scanner
            or QuantityScanner()
        )

        self.explicit_resolver = (
            explicit_resolver
            or ExplicitPatternResolver()
        )

        self.semantic_linker = (
            semantic_linker
            or SemanticLinkerRuntime()
        )

        self.llm_fallback = (
            llm_fallback
            or LLMFallbackExtractor()
        )

    # =================================================================
    # PUBLIC API
    # =================================================================

    def resolve(
        self,
        text: str,
        previous_question: Optional[str] = None,
    ) -> SelectiveCascadeResult:

        # -------------------------------------------------------------
        # 1. Quantity Scanner
        # -------------------------------------------------------------

        quantities = self.scanner.scan(
            text
        )

        traces = {
            quantity.id:
                QuantityRouteTrace(
                    quantity_id=quantity.id
                )
            for quantity in quantities
        }

        if not quantities:
            return SelectiveCascadeResult(
                text=text,
                quantities=[],
                links=[],
                unresolved_quantity_ids=[],
                traces={},
            )

        quantity_by_id = {
            quantity.id: quantity
            for quantity in quantities
        }

        # -------------------------------------------------------------
        # 2. Explicit Resolver
        # -------------------------------------------------------------

        explicit_result = (
            self.explicit_resolver.resolve(
                text,
                quantities,
            )
        )

        links: List[
            SemanticLink
        ] = list(
            explicit_result.links
        )

        explicitly_resolved_ids = {
            link.quantity_id
            for link in explicit_result.links
        }

        for quantity in quantities:

            trace = traces[
                quantity.id
            ]

            trace.explicit_attempted = True

            if (
                quantity.id
                in explicitly_resolved_ids
            ):
                trace.explicit_resolved = True
                trace.final_resolver = (
                    "explicit_pattern"
                )
                trace.final_status = (
                    "resolved"
                )

        # -------------------------------------------------------------
        # 3. Process ONLY explicitly unresolved quantities
        # -------------------------------------------------------------

        final_unresolved: List[str] = []

        for quantity_id in (
            explicit_result
            .unresolved_quantity_ids
        ):

            quantity = (
                quantity_by_id[
                    quantity_id
                ]
            )

            trace = traces[
                quantity_id
            ]

            # ---------------------------------------------------------
            # 3A. Semantic Linker XLM-R
            # ---------------------------------------------------------

            trace.semantic_attempted = True

            semantic_prediction = (
                self.semantic_linker.predict(
                    text=text,
                    quantity=quantity,
                    previous_question=
                        previous_question,
                )
            )

            self._update_semantic_trace(
                trace=trace,
                prediction=
                    semantic_prediction,
            )

            if (
                semantic_prediction.accepted
                and
                semantic_prediction.link
                is not None
            ):

                links.append(
                    semantic_prediction.link
                )

                trace.semantic_accepted = True
                trace.final_resolver = (
                    "semantic_linker_xlmr"
                )
                trace.final_status = (
                    "resolved"
                )

                # CRITICAL:
                # accepted Transformer result NEVER reaches LLM.
                continue

            # ---------------------------------------------------------
            # 3B. LLM FALLBACK — only after Semantic abstention
            # ---------------------------------------------------------

            trace.llm_attempted = True

            llm_link = (
                self.llm_fallback
                .resolve_quantity(
                    user_text=text,
                    quantity=quantity,
                    previous_question=
                        previous_question,
                )
            )

            if llm_link is not None:

                links.append(
                    llm_link
                )

                trace.llm_resolved = True
                trace.final_resolver = (
                    "llm_fallback"
                )
                trace.final_status = (
                    "resolved"
                )

                continue

            # ---------------------------------------------------------
            # 3C. Still unresolved
            # ---------------------------------------------------------

            trace.final_status = (
                "unresolved"
            )

            trace.final_resolver = None

            final_unresolved.append(
                quantity_id
            )

        # -------------------------------------------------------------
        # 4. Stable ordering
        # -------------------------------------------------------------

        quantity_order = {
            quantity.id: index
            for index, quantity
            in enumerate(quantities)
        }

        links.sort(
            key=lambda link:
                quantity_order.get(
                    link.quantity_id,
                    10**9,
                )
        )

        # -------------------------------------------------------------
        # 5. Result
        # -------------------------------------------------------------

        return SelectiveCascadeResult(
            text=text,
            quantities=quantities,
            links=links,
            unresolved_quantity_ids=
                final_unresolved,
            traces=traces,
        )

    # =================================================================
    # INTERNAL HELPERS
    # =================================================================

    @staticmethod
    def _update_semantic_trace(
        trace: QuantityRouteTrace,
        prediction: SemanticLinkerPrediction,
    ) -> None:

        trace.semantic_accepted = (
            prediction.accepted
        )

        trace.semantic_confidence = (
            prediction.confidence
        )

        trace.semantic_margin = (
            prediction.margin
        )

        trace.semantic_raw_field = (
            prediction.raw_field
        )

        trace.semantic_raw_role = (
            prediction.raw_role
        )

    # =================================================================
    # DEBUG
    # =================================================================

    def info(self) -> dict:
        return {
            "semantic_linker":
                self.semantic_linker.info(),

            "llm_fallback":
                self.llm_fallback.info(),
        }