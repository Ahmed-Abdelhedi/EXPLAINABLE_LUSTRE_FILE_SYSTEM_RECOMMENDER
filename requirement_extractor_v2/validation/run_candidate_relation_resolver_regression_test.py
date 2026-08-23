from __future__ import annotations

from typing import List, Tuple

from requirement_extractor_v2.candidate_relation_resolver import (
    CandidateRelationResolver,
    CandidateRelationType,
)
from requirement_extractor_v2.models import (
    ParamName,
    Quantity,
    QuantityDimension,
    SemanticLink,
    SemanticRole,
)


def make_quantity(
    text: str,
    token: str,
    qid: str,
    value: float,
    unit: str | None,
    dimension: QuantityDimension,
    occurrence: int = 0,
) -> Quantity:
    start = -1
    cursor = 0
    for _ in range(occurrence + 1):
        start = text.find(token, cursor)
        if start < 0:
            raise AssertionError(f"token not found: {token!r} in {text!r}")
        cursor = start + len(token)

    return Quantity(
        id=qid,
        raw=token,
        normalized=token,
        value=value,
        unit=unit,
        dimension=dimension,
        start=start,
        end=start + len(token),
        source_text=text,
    )


def link(qid: str, field: ParamName, role: SemanticRole) -> SemanticLink:
    return SemanticLink(
        quantity_id=qid,
        field=field,
        role=role,
        evidence="test",
        resolver="regression_test",
    )


def field_relation(result, field_name: ParamName):
    matches = [
        relation
        for relation in result.relations
        if relation.fields == [field_name]
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one relation for {field_name.value}, got {len(matches)}"
        )
    return matches[0]


def run() -> None:
    resolver = CandidateRelationResolver()
    cases: List[Tuple[str, CandidateRelationType]] = []

    # 1. English alternative.
    text = "Write throughput could be 20 or 30 GB/s."
    q1 = make_quantity(text, "20", "q1", 20, "GB/s", QuantityDimension.THROUGHPUT)
    q2 = make_quantity(text, "30 GB/s", "q2", 30, "GB/s", QuantityDimension.THROUGHPUT)
    result = resolver.resolve(
        text,
        [q1, q2],
        [
            link("q1", ParamName.target_write_gbps, SemanticRole.TARGET),
            link("q2", ParamName.target_write_gbps, SemanticRole.TARGET),
        ],
    )
    relation = field_relation(result, ParamName.target_write_gbps)
    assert relation.relation == CandidateRelationType.ALTERNATIVE
    assert set(relation.blocked_quantity_ids) == {"q1", "q2"}
    cases.append(("alternative_en", relation.relation))

    # 2. French alternative.
    text = "La taille maximale peut être 40 ou 60 GB."
    q1 = make_quantity(text, "40", "q1", 40, "GB", QuantityDimension.FILE_SIZE)
    q2 = make_quantity(text, "60 GB", "q2", 60, "GB", QuantityDimension.FILE_SIZE)
    result = resolver.resolve(
        text,
        [q1, q2],
        [
            link("q1", ParamName.max_file_size_gb, SemanticRole.MAXIMUM_LIMIT),
            link("q2", ParamName.max_file_size_gb, SemanticRole.MAXIMUM_LIMIT),
        ],
    )
    relation = field_relation(result, ParamName.max_file_size_gb)
    assert relation.relation == CandidateRelationType.ALTERNATIVE
    cases.append(("alternative_fr", relation.relation))

    # 3. Range.
    text = "Read throughput should be between 20 and 30 GB/s."
    q1 = make_quantity(text, "20", "q1", 20, "GB/s", QuantityDimension.THROUGHPUT)
    q2 = make_quantity(text, "30 GB/s", "q2", 30, "GB/s", QuantityDimension.THROUGHPUT)
    result = resolver.resolve(
        text,
        [q1, q2],
        [
            link("q1", ParamName.target_read_gbps, SemanticRole.TARGET),
            link("q2", ParamName.target_read_gbps, SemanticRole.TARGET),
        ],
    )
    relation = field_relation(result, ParamName.target_read_gbps)
    assert relation.relation == CandidateRelationType.RANGE
    assert relation.blocks_automatic_acceptance
    cases.append(("range", relation.relation))

    # 4. Explicit correction with old and new values.
    text = "Change read throughput from 20 to 30 GB/s."
    q1 = make_quantity(text, "20", "q1", 20, "GB/s", QuantityDimension.THROUGHPUT)
    q2 = make_quantity(text, "30 GB/s", "q2", 30, "GB/s", QuantityDimension.THROUGHPUT)
    result = resolver.resolve(
        text,
        [q1, q2],
        [
            link("q1", ParamName.target_read_gbps, SemanticRole.TARGET),
            link("q2", ParamName.target_read_gbps, SemanticRole.TARGET),
        ],
    )
    relation = field_relation(result, ParamName.target_read_gbps)
    assert relation.relation == CandidateRelationType.CORRECTION
    assert relation.selected_quantity_ids == ["q2"]
    assert relation.blocked_quantity_ids == ["q1"]
    cases.append(("correction", relation.relation))

    # 5. Two different fields are safe together.
    text = "Read target is 20 GB/s and write target is 30 GB/s."
    q1 = make_quantity(text, "20 GB/s", "q1", 20, "GB/s", QuantityDimension.THROUGHPUT)
    q2 = make_quantity(text, "30 GB/s", "q2", 30, "GB/s", QuantityDimension.THROUGHPUT)
    result = resolver.resolve(
        text,
        [q1, q2],
        [
            link("q1", ParamName.target_read_gbps, SemanticRole.TARGET),
            link("q2", ParamName.target_write_gbps, SemanticRole.TARGET),
        ],
    )
    assert any(
        relation.relation == CandidateRelationType.MULTIPLE_FIELDS
        for relation in result.relations
    )
    assert not result.has_blocking_relation
    cases.append(("multiple_fields", CandidateRelationType.MULTIPLE_FIELDS))

    # 6. Existing ratio representation remains safe.
    text = "Workload is 70% read and 30% write."
    q1 = make_quantity(text, "70%", "q1", 70, "%", QuantityDimension.PERCENT)
    q2 = make_quantity(text, "30%", "q2", 30, "%", QuantityDimension.PERCENT)
    result = resolver.resolve(
        text,
        [q1, q2],
        [
            link("q1", ParamName.read_write_ratio, SemanticRole.RATIO_COMPONENT),
            link("q2", ParamName.read_write_ratio, SemanticRole.RATIO_COMPONENT),
        ],
    )
    relation = field_relation(result, ParamName.read_write_ratio)
    assert relation.relation == CandidateRelationType.SINGLE_VALUE
    assert not relation.blocks_automatic_acceptance
    cases.append(("ratio_composite", relation.relation))

    # 7. Same field + two unrelated different values => conflict.
    text = "Read target 20 GB/s; read target 30 GB/s."
    q1 = make_quantity(text, "20 GB/s", "q1", 20, "GB/s", QuantityDimension.THROUGHPUT)
    q2 = make_quantity(text, "30 GB/s", "q2", 30, "GB/s", QuantityDimension.THROUGHPUT)
    result = resolver.resolve(
        text,
        [q1, q2],
        [
            link("q1", ParamName.target_read_gbps, SemanticRole.TARGET),
            link("q2", ParamName.target_read_gbps, SemanticRole.TARGET),
        ],
    )
    relation = field_relation(result, ParamName.target_read_gbps)
    assert relation.relation == CandidateRelationType.CONFLICT
    assert relation.blocks_automatic_acceptance
    cases.append(("conflict", relation.relation))

    print("CandidateRelationResolver regression test: PASS")
    for name, relation_type in cases:
        print(f"  - {name}: {relation_type.value}")


if __name__ == "__main__":
    run()
