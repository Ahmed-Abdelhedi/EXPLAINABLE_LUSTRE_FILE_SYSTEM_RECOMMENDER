from __future__ import annotations

from typing import Iterable, List, Optional

from requirement_extractor_v2.candidate_relation_resolver import CandidateRelationResolver
from requirement_extractor_v2.models import (
    ParamName,
    Quantity,
    QuantityDimension,
    SemanticLink,
    SemanticRole,
    VerificationDecision,
    VerificationStatus,
)
from requirement_extractor_v2.relation_aware_verifier import (
    RelationAwareDeterministicVerifier,
)


class AlwaysVerifiedBaseVerifier:
    """Small deterministic stub used to isolate the relation safety gate."""

    def verify_many(
        self,
        quantities: Iterable[Quantity],
        links: Iterable[SemanticLink],
        source_text: Optional[str] = None,
    ) -> List[VerificationDecision]:
        link_by_id = {link.quantity_id: link for link in links}
        decisions: List[VerificationDecision] = []
        for quantity in quantities:
            link = link_by_id.get(quantity.id)
            if link is None:
                decisions.append(
                    VerificationDecision(
                        status=VerificationStatus.UNRESOLVED,
                        quantity_id=quantity.id,
                        value=quantity.value,
                        unit=quantity.unit,
                        reasons=["no_semantic_link"],
                    )
                )
                continue

            decisions.append(
                VerificationDecision(
                    status=VerificationStatus.VERIFIED,
                    quantity_id=quantity.id,
                    field=link.field,
                    role=link.role,
                    value=quantity.value,
                    unit=quantity.unit,
                    evidence=link.evidence,
                    reasons=["stub_verified"],
                )
            )
        return decisions


def throughput_q(text: str, token: str, qid: str, value: float) -> Quantity:
    start = text.find(token)
    return Quantity(
        id=qid,
        raw=token,
        normalized=token,
        value=value,
        unit="GB/s",
        dimension=QuantityDimension.THROUGHPUT,
        start=start,
        end=start + len(token),
        source_text=text,
    )


def percent_q(text: str, token: str, qid: str, value: float) -> Quantity:
    start = text.find(token)
    return Quantity(
        id=qid,
        raw=token,
        normalized=token,
        value=value,
        unit="%",
        dimension=QuantityDimension.PERCENT,
        start=start,
        end=start + len(token),
        source_text=text,
    )


def target_link(qid: str, field: ParamName) -> SemanticLink:
    return SemanticLink(
        quantity_id=qid,
        field=field,
        role=SemanticRole.TARGET,
        evidence="test",
        resolver="test",
    )


def ratio_link(qid: str) -> SemanticLink:
    return SemanticLink(
        quantity_id=qid,
        field=ParamName.read_write_ratio,
        role=SemanticRole.RATIO_COMPONENT,
        evidence="test",
        resolver="test",
    )


def run() -> None:
    verifier = RelationAwareDeterministicVerifier(
        base_verifier=AlwaysVerifiedBaseVerifier(),
        relation_resolver=CandidateRelationResolver(),
    )

    # Alternative: neither value may survive as VERIFIED.
    text = "Write throughput could be 20 or 30 GB/s."
    q1 = throughput_q(text, "20", "q1", 20)
    q2 = throughput_q(text, "30 GB/s", "q2", 30)
    decisions = verifier.verify_many(
        [q1, q2],
        [
            target_link("q1", ParamName.target_write_gbps),
            target_link("q2", ParamName.target_write_gbps),
        ],
        source_text=text,
    )
    assert all(item.status == VerificationStatus.AMBIGUOUS for item in decisions)

    # Correction: old value is suppressed; new value remains VERIFIED.
    text = "Change read throughput from 20 to 30 GB/s."
    q1 = throughput_q(text, "20", "q1", 20)
    q2 = throughput_q(text, "30 GB/s", "q2", 30)
    decisions = verifier.verify_many(
        [q1, q2],
        [
            target_link("q1", ParamName.target_read_gbps),
            target_link("q2", ParamName.target_read_gbps),
        ],
        source_text=text,
    )
    by_id = {item.quantity_id: item for item in decisions}
    assert by_id["q1"].status == VerificationStatus.UNRESOLVED
    assert by_id["q2"].status == VerificationStatus.VERIFIED

    # Different fields: both values stay VERIFIED.
    text = "Read target 20 GB/s and write target 30 GB/s."
    q1 = throughput_q(text, "20 GB/s", "q1", 20)
    q2 = throughput_q(text, "30 GB/s", "q2", 30)
    decisions = verifier.verify_many(
        [q1, q2],
        [
            target_link("q1", ParamName.target_read_gbps),
            target_link("q2", ParamName.target_write_gbps),
        ],
        source_text=text,
    )
    assert all(item.status == VerificationStatus.VERIFIED for item in decisions)

    # A lone ratio component must NOT update the state automatically.
    text = "The workload is 70% read."
    q1 = percent_q(text, "70%", "q1", 70)
    decisions = verifier.verify_many(
        [q1],
        [ratio_link("q1")],
        source_text=text,
    )
    assert decisions[0].status == VerificationStatus.AMBIGUOUS
    assert "incomplete_read_write_ratio_requires_both_components" in decisions[0].reasons

    # A complete ratio pair remains valid.
    text = "The workload is 70% read and 30% write."
    q1 = percent_q(text, "70%", "q1", 70)
    q2 = percent_q(text, "30%", "q2", 30)
    decisions = verifier.verify_many(
        [q1, q2],
        [ratio_link("q1"), ratio_link("q2")],
        source_text=text,
    )
    assert all(item.status == VerificationStatus.VERIFIED for item in decisions)

    # An inconsistent pair must not be accepted automatically.
    text = "The workload is 70% read and 40% write."
    q1 = percent_q(text, "70%", "q1", 70)
    q2 = percent_q(text, "40%", "q2", 40)
    decisions = verifier.verify_many(
        [q1, q2],
        [ratio_link("q1"), ratio_link("q2")],
        source_text=text,
    )
    assert all(item.status == VerificationStatus.AMBIGUOUS for item in decisions)

    print("RelationAwareDeterministicVerifier regression test: PASS")


if __name__ == "__main__":
    run()