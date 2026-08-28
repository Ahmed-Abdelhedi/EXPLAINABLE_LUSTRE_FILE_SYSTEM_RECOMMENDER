# H9 — Architecture scoring

H9 defines how complete H8 architectures are compared by a deterministic soft
score. It does **not** decide final validity.

This follows the project rule:

**hard constraints are never transformed into soft penalties.**

H9 therefore produces two separate outputs:

1. a preference-weighted soft score;
2. a diagnostic snapshot of hard checks that can already be computed
   (performance requirements, total budget lower bound and total power lower
   bound).

The snapshot does not remove an architecture and does not claim final
feasibility. The independent H10 validator remains the only component that can
declare a complete architecture valid or invalid.

## Preference weights

H9 consumes the numeric weights already produced upstream:

- `performance_priority`
- `cost_priority`
- `power_priority`
- `reliability_priority`

It does not convert HIGH/MEDIUM/LOW labels itself. The values are normalized to
sum to one while preserving their relative ratios.

## Performance component

For every active MDT/OST technical requirement:

`headroom_score = max(0, 1 - required / provided)`

Properties:

- exact satisfaction -> 0;
- provided = 2 × required -> 0.5;
- score tends toward 1 as headroom increases;
- requirements equal to zero are excluded from the mean.

No fixed saturation threshold is invented.

## Cost and power components

Lower is better. H9 uses inverse min-max normalization across the complete H8
architecture pool supplied for one case.

These components express user preference only. They do not replace the hard
budget/power checks that H10 will enforce.

## Reliability component

Reliability is explicitly a **reference proxy**, not a predicted failure
probability.

It is the equal mean of:

1. numeric drive reliability evidence;
2. protection fault-tolerance proxy;
3. HA-presence proxy.

Drive evidence uses endurance, MTBF, warranty and workload rating where
applicable. Values are normalized only within the same role and media family.
Missing applicable evidence receives no reliability credit. Attributes absent
from the whole media family are ignored.

## Pre-H10 ranking

H9 can rank the generated architectures by score so the scoring policy can be
tested and later reused by Beam Search.

That rank is explicitly **pre-H10**. A high-scoring architecture that violates
a hard constraint is not a valid recommendation. H10 must filter the ranked
pool before the final winner is selected.

## Boundaries

H9:

- preserves `PENDING_FULL_VALIDATOR`;
- never sets `validation.is_valid = true`;
- does not claim vendor reliability;
- does not apply Beam Search;
- does not decide final feasibility;
- does not replace H10.
