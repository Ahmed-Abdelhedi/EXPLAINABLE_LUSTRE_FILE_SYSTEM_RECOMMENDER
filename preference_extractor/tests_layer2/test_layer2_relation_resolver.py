from preference_extractor.layer2.labels import (
    PreferenceDimension,
)
from preference_extractor.layer2.relation_resolver import (
    ComparativeRelationResolver,
)


def test_english_comparative_is_preserved():
    resolver = ComparativeRelationResolver()

    result = resolver.resolve(
        "Performance is more important than cost."
    )

    assert len(result.relations) == 1
    assert result.relations[0].higher == (
        PreferenceDimension.PERFORMANCE
    )
    assert result.relations[0].lower == (
        PreferenceDimension.COST
    )

    assert (
        PreferenceDimension.PERFORMANCE
        in result.relative_only_dimensions
    )
    assert (
        PreferenceDimension.COST
        in result.relative_only_dimensions
    )


def test_french_comparative_is_preserved():
    resolver = ComparativeRelationResolver()

    result = resolver.resolve(
        "La fiabilité est plus importante que le coût."
    )

    assert len(result.relations) == 1
    assert result.relations[0].higher == (
        PreferenceDimension.RELIABILITY
    )
    assert result.relations[0].lower == (
        PreferenceDimension.COST
    )


def test_absolute_cue_prevents_relative_only_override():
    resolver = ComparativeRelationResolver()

    result = resolver.resolve(
        "Performance is very important and more important than cost."
    )

    assert result.relations
    assert not result.relative_only_dimensions
