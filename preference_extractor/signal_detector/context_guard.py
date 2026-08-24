from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Pattern, Tuple


class PreferenceGuardDecision(str, Enum):
    """
    High-precision deterministic decision used before final Layer-1 output.

    PASS_TO_MODEL:
        No safe deterministic conclusion. Keep the Transformer decision.

    FORCE_SIGNAL:
        Explicit current-user preference / trade-off construction.

    FORCE_NO_SIGNAL:
        Explicit evidence that preference-looking wording does not describe
        the current user's requirement.
    """

    PASS_TO_MODEL = "PASS_TO_MODEL"
    FORCE_SIGNAL = "FORCE_SIGNAL"
    FORCE_NO_SIGNAL = "FORCE_NO_SIGNAL"


@dataclass(frozen=True)
class PreferenceGuardResult:
    decision: PreferenceGuardDecision
    reason: Optional[str] = None
    evidence: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class _GuardRule:
    name: str
    decision: PreferenceGuardDecision
    patterns: Tuple[Pattern[str], ...]


def _compile(*patterns: str) -> Tuple[Pattern[str], ...]:
    return tuple(
        re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
        for pattern in patterns
    )


# ---------------------------------------------------------------------
# NEGATIVE / SOURCE GUARDS
# ---------------------------------------------------------------------
#
# These are intentionally narrow. A keyword such as "not", "vendor", "can",
# or "priority" is NOT sufficient by itself. A complete discourse structure
# must show that the preference-looking phrase is quoted/rejected, merely
# reported as absent, or describes a system capability rather than the
# current user's choice.
# ---------------------------------------------------------------------

_NEGATIVE_RULES: Tuple[_GuardRule, ...] = (
    _GuardRule(
        name="quoted_third_party_explicitly_rejected",
        decision=PreferenceGuardDecision.FORCE_NO_SIGNAL,
        patterns=_compile(
            r"\b(?:vendor|supplier)\b.{0,80}\b(?:brochure|document|proposal)\b"
            r".{0,180}\b(?:says?|states?|claims?|dit|indique|affirme)\b"
            r".{0,260}\b(?:not\s+our\s+requirement|not\s+our\s+request|"
            r"n['’]est\s+pas\s+notre\s+exigence|"
            r"ne\s+correspond\s+pas\s+[àa]\s+notre\s+demande)\b",

            r"\bbrochure\s+du\s+fournisseur\b"
            r".{0,180}\b(?:dit|indique|affirme)\b"
            r".{0,260}\b(?:n['’]est\s+pas\s+notre\s+exigence|"
            r"not\s+our\s+requirement)\b",
        ),
    ),
    _GuardRule(
        name="negated_preference_report",
        decision=PreferenceGuardDecision.FORCE_NO_SIGNAL,
        patterns=_compile(
            r"\b(?:minutes|meeting\s+minutes)\b"
            r".{0,120}\b(?:do\s+not|does\s+not|did\s+not)\s+"
            r"(?:state|say|indicate|record)\b"
            r".{0,180}\b(?:prefer(?:red|ence)?|priorit(?:y|ized|ised))\b",

            r"\b(?:compte\s+rendu|proc[eè]s[- ]verbal)\b"
            r".{0,120}\bne\s+"
            r"(?:dit|disent|pr[eé]cise|pr[eé]cisent|mentionne|mentionnent|"
            r"indique|indiquent)\s+pas\b"
            r".{0,180}\b(?:privil[eé]gi|priorit[eé]|pr[eé]f[eé]r)\w*\b",

            r"\b(?:minutes|meeting\s+minutes)\b"
            r".{0,120}\bne\s+"
            r"(?:disent|dit|pr[eé]cisent|mentionnent|indiquent)\s+pas\b"
            r".{0,180}\b(?:preferred|priorit|privil[eé]gi)\w*\b",
        ),
    ),
    _GuardRule(
        name="rejected_hypothesis_or_draft",
        decision=PreferenceGuardDecision.FORCE_NO_SIGNAL,
        patterns=_compile(
            r"\b(?:a\s+)?(?:rejected|discarded)\s+"
            r"(?:draft|proposal|option|hypothesis)\b"
            r".{0,260}\b(?:does\s+not|doesn['’]t|did\s+not)\s*describe\b"
            r".{0,120}\b(?:current\s+request|current\s+requirement|"
            r"la\s+current\s+request)\b",

            r"\bbrouillon\s+rejet[eé]\b"
            r".{0,260}\b(?:ne\s+d[eé]crit\s+pas|ne\s+correspond\s+pas)\b"
            r".{0,120}\b(?:demande|exigence)\s+actuelle\b",
        ),
    ),
    _GuardRule(
        name="system_capability_not_user_preference",
        decision=PreferenceGuardDecision.FORCE_NO_SIGNAL,
        patterns=_compile(
            r"\b(?:the\s+)?(?:controller|platform|system)\b"
            r".{0,60}\b(?:can|peut)\b"
            r".{0,100}\b(?:tune|adjust|regulate|optimi[sz]e|"
            r"r[eé]gler|ajuster|optimiser|adapter)\b"
            r".{0,120}\b(?:automatically|automatiquement)\b",

            r"\b(?:le\s+|la\s+)?(?:contr[oô]leur|plateforme|syst[eè]me)\b"
            r".{0,60}\bpeut\b"
            r".{0,100}\b(?:r[eé]gler|ajuster|optimiser|adapter)\b"
            r".{0,120}\b(?:automatiquement|automatically)\b",
        ),
    ),
    _GuardRule(
        name="lexical_priority_queue_trap",
        decision=PreferenceGuardDecision.FORCE_NO_SIGNAL,
        patterns=_compile(
            r"\bpriority\s+queue\b",
            r"\bfile\s+de\s+priorit[eé]\b",
        ),
    ),
    _GuardRule(
        name="bare_deployment_fact",
        decision=PreferenceGuardDecision.FORCE_NO_SIGNAL,
        patterns=_compile(
            r"^\s*lustre\s+will\s+be\s+deployed\s+on\s+the\s+"
            r"(?:production|planned|compute)\s+cluster\s*[.!]?\s*$",

            r"^\s*lustre\s+sera\s+d[eé]ploy[eé]\s+sur\s+le\s+cluster\s+"
            r"(?:de\s+)?(?:production|calcul)\s*[.!]?\s*$",
        ),
    ),
)


# ---------------------------------------------------------------------
# POSITIVE / EXPLICIT CHOICE GUARDS
# ---------------------------------------------------------------------
#
# These rules require an actual selection/trade-off structure. Merely naming
# "performance", "cost", "power", or "reliability" is not enough.
# ---------------------------------------------------------------------

_POSITIVE_RULES: Tuple[_GuardRule, ...] = (
    _GuardRule(
        name="conditional_choice_preference",
        decision=PreferenceGuardDecision.FORCE_SIGNAL,
        patterns=_compile(
            r"\bif\b"
            r".{0,140}\b(?:options?|designs?|architectures?)\b"
            r".{0,140}\b(?:select|choose|prefer)\b"
            r".{0,140}\b(?:over|rather\s+than)\b",

            r"\bsi\b"
            r".{0,140}\b(?:options?|conceptions?|architectures?)\b"
            r".{0,140}\b(?:chois(?:ir|issez|issons)|s[eé]lectionn\w*|"
            r"privil[eé]gi\w*)\b"
            r".{0,140}\b(?:plut[oô]t\s+que|par\s+rapport\s+[àa]|sur)\b",
        ),
    ),
    _GuardRule(
        name="question_tradeoff_preference",
        decision=PreferenceGuardDecision.FORCE_SIGNAL,
        patterns=_compile(
            r"\b(?:can|could|should)\s+we\b"
            r".{0,100}\b(?:optimi[sz]e|optimiser|prioriti[sz]e|prioriser|"
            r"favor|favour|favoriser|privil[eé]gier)\b"
            r".{0,180}\b(?:at\s+the\s+expense\s+of|"
            r"au\s+d[eé]triment\s+de)\b",

            r"\b(?:pouvons[- ]nous|peut[- ]on|devrions[- ]nous)\b"
            r".{0,100}\b(?:optimiser|prioriser|favoriser|privil[eé]gier)\b"
            r".{0,180}\b(?:au\s+d[eé]triment\s+de|"
            r"at\s+the\s+expense\s+of)\b",
        ),
    ),
    _GuardRule(
        name="final_choice_preference",
        decision=PreferenceGuardDecision.FORCE_SIGNAL,
        patterns=_compile(
            r"\b(?:for\s+the\s+final\s+choice|final\s+choice)\b"
            r".{0,180}\b(?:favor|favour|prefer|prioriti[sz]e|"
            r"privil[eé]gi|pr[eé]f[eé]r|favoris|prioris)\w*\b",

            r"\b(?:pour\s+le\s+choix\s+final|choix\s+final)\b"
            r".{0,180}\b(?:privil[eé]gi|pr[eé]f[eé]r|favoris|prioris|"
            r"prefer|prioriti[sz])\w*\b",
        ),
    ),
    _GuardRule(
        name="telegraphic_ranked_choice",
        decision=PreferenceGuardDecision.FORCE_SIGNAL,
        patterns=_compile(
            r"\bproduction\s+choice\b"
            r".{0,100}\bfirst\b"
            r".{0,100}\b(?:secondary|second)\b",

            r"\bchoix\s+production\b"
            r".{0,100}\bd['’]abord\b"
            r".{0,100}\b(?:ensuite|secondaire)\b",
        ),
    ),
)


class PreferenceContextGuard:
    """
    High-precision discourse guard for Layer-1 preference-signal detection.

    The Transformer remains the default classifier.

    Resolution order:
        1. Strong negative/source rules.
        2. Strong positive current-user choice/trade-off rules.
        3. PASS_TO_MODEL.

    Negative rules intentionally have precedence because a third-party quote
    may itself contain a perfectly positive preference sentence while the
    surrounding sentence explicitly rejects it as the user's requirement.
    """

    def __init__(self) -> None:
        self.negative_rules = _NEGATIVE_RULES
        self.positive_rules = _POSITIVE_RULES

    @staticmethod
    def _evidence(match: re.Match[str]) -> str:
        evidence = " ".join(match.group(0).split())
        if len(evidence) > 280:
            evidence = evidence[:277] + "..."
        return evidence

    @staticmethod
    def _match_rule(
        text: str,
        rule: _GuardRule,
    ) -> Optional[PreferenceGuardResult]:
        for pattern in rule.patterns:
            match = pattern.search(text)
            if match is None:
                continue

            return PreferenceGuardResult(
                decision=rule.decision,
                reason=rule.name,
                evidence=PreferenceContextGuard._evidence(match),
            )

        return None

    def resolve(self, text: str) -> PreferenceGuardResult:
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")

        for rule in self.negative_rules:
            result = self._match_rule(text, rule)
            if result is not None:
                return result

        for rule in self.positive_rules:
            result = self._match_rule(text, rule)
            if result is not None:
                return result

        return PreferenceGuardResult(
            decision=PreferenceGuardDecision.PASS_TO_MODEL,
            reason=None,
            evidence=None,
        )

    def info(self) -> dict:
        return {
            "name": "PreferenceContextGuard",
            "policy": "high_precision_deterministic_then_transformer",
            "negative_rule_count": len(self.negative_rules),
            "positive_rule_count": len(self.positive_rules),
            "negative_precedence": True,
        }
