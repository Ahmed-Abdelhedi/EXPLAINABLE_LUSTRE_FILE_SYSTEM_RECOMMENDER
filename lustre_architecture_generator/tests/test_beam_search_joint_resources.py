"""Regress the generic phantom completion caused by separate resource minima."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from test_beam_search import beam, search
from full_architecture.feasibility_coverage import analyze_case_feasibility_domain


@pytest.mark.parametrize("width", [1, 32])
def test_joint_resource_conflict_retains_a_real_h10_valid_completion(width):
    fixture = json.loads((Path(__file__).parent / "fixtures/beam_joint_resource_conflict.json").read_text(encoding="utf-8"))
    h, c = fixture["handoff"], fixture["hardware_catalog"]
    h["case_id"] = "RENAMED_JOINT_RESOURCE_REGRESSION"
    witness = analyze_case_feasibility_domain(handoff=h, hardware_catalog=c, max_paths_per_variant=1)
    assert witness["recovered_valid_architecture"] is True
    result = search(h, c, width=width, paths=1)
    assert result["best_validated_architecture"] is not None
    assert result["best_validated_architecture"]["h10"]["decision"] == "VALID"
    assert result["summary"]["complete_architectures_produced"] <= width


@pytest.mark.parametrize("mdt,expected", [([(3., 3.)], False), ([(1., 1.)], True), ([], False)])
def test_independent_minima_do_not_establish_joint_feasibility(mdt, expected):
    envelopes = ({"resource_points": beam._pareto_resources(mdt)},
                 {"resource_points": beam._pareto_resources([(1., 10.), (10., 1.)])})
    assert beam._joint_resources_possible(envelopes, {"max_budget_usd": 12., "max_power_w": 12.}) is expected


def test_pareto_compaction_preserves_every_feasible_resource_choice():
    points = [(1., 9.), (2., 10.), (1., 10.), (5., 5.), (5., 5.), (9., 1.), (10., 1.)]
    frontier = beam._pareto_resources(points)
    assert frontier == ((1., 9.), (5., 5.), (9., 1.))
    for cost, power in points:
        assert any(c <= cost and p <= power for c, p in frontier)


def test_linear_joint_query_matches_independent_cartesian_resource_checks():
    generator = random.Random(1789)
    for _ in range(300):
        mdt = [(float(generator.randrange(30)), float(generator.randrange(30))) for _ in range(9)]
        ost = [(float(generator.randrange(30)), float(generator.randrange(30))) for _ in range(11)]
        budget, power = generator.randrange(60), generator.randrange(60)
        expected = any(m[0]+o[0] <= budget and m[1]+o[1] <= power for m in mdt for o in ost)
        envelopes = ({"resource_points": beam._pareto_resources(mdt)},
                     {"resource_points": beam._pareto_resources(ost)})
        assert beam._joint_resources_possible(envelopes, {"max_budget_usd": budget, "max_power_w": power}) is expected


def test_joint_resource_boundary_is_conservative_at_large_float_scales():
    maximum = 2.**40
    envelopes = ({"resource_points": ((maximum/2, 1.),)},
                 {"resource_points": ((maximum/2+math.ulp(maximum), 1.),)})
    assert beam._joint_resources_possible(envelopes, {"max_budget_usd": maximum, "max_power_w": 2.})
    assert not beam._joint_resources_possible(envelopes, {"max_budget_usd": maximum-1., "max_power_w": 2.})
