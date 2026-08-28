# H10-B — Feasibility coverage after the 16-architecture validation cap

The global H10 run validated execution on all 1,200 cases, but only 488 cases
had at least one valid architecture among the 16 architectures generated under
the controlled H8 limits.

This does not mean the other 712 cases are globally infeasible.

The baseline generation used:

- Top-K = 10;
- max hardware paths per protection variant = 2;
- max role options = 4 for MDT and 4 for OST;
- therefore 4 × 4 = 16 complete architectures per case.

H10-B first isolates one truncation source: `max_role_options=4`.

## H10-B step A

For every baseline case without a valid architecture:

1. keep Top-K = 10;
2. keep `max_paths_per_variant = 2`;
3. remove the role-option cap;
4. enumerate all MDT and OST role options available in that domain;
5. inspect the full MDT × OST cost/power feasibility domain;
6. when a pair satisfies budget and power, materialize the exact H7 state and
   confirm it with the independent H10 validator.

This is much cheaper than validating every Cartesian pair because cost and
power are additive across the isolated MDT and OST role instances.

## What H10-B can conclude

If a case is recovered, the earlier absence of a valid architecture was caused
by the role-option truncation.

If a case remains unresolved, H10-B records whether:

- even the minimum possible cost exceeds budget;
- even the minimum possible power exceeds the power limit;
- both lower bounds exceed;
- or the individual lower bounds fit but no single pair satisfies both
  simultaneously.

An unresolved case is **not yet declared globally infeasible** because the H6
hardware-path cap is still 2 and Top-K is still 10.

If unresolved cases remain after this stage, the next coverage stage must study
the path cap before the Architecture Layer can be frozen.
