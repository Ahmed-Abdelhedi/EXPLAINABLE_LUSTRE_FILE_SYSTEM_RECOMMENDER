# H10-C — Full hardware-path coverage

After the global H10-B run, 578 of the 712 baseline-unresolved cases were
recovered simply by removing the H8 `max_role_options=4` cap.

The coverage increased from 488/1200 to 1066/1200.

134 cases remain unresolved.

H10-C studies the remaining truncation source in the architecture generation
domain: H6 returned at most two compatible hardware paths per
candidate/protection variant during H10-B.

## Domain examined

H10-C keeps:

- the current Top-K supplied by H2 (K=10);
- the current reference hardware catalog;
- all protection profiles;
- the H5 arithmetic;
- H6 compatibility rules.

It removes:

- `max_role_options`;
- the hardware-path truncation.

For the current catalog, the complete H6 loop domain is bounded by:

`servers × controllers × networks × HA profiles × (DIRECT + enclosures)`

H10-C passes that exact safe upper bound as `max_paths`, so every compatible
path in the current reference catalog can be returned.

## Pareto reduction

The full path domain can contain many equivalent or dominated options.

Because the remaining H10 violations are budget and power, an option can be
discarded safely if another option for the same role uses no more cost and no
more power.

This creates a cost/power Pareto frontier without changing feasibility.

Any recovered pair is still materialized as a full ArchitectureState and
confirmed independently by H10.

## Interpretation

Recovered cases demonstrate that the earlier `max_paths=2` cap hid a valid
solution.

Cases that remain unresolved after H10-C have no feasible cost/power pair
within:

- current Top-K = 10;
- all current protection profiles;
- the full compatible hardware-path domain;
- the current reference hardware catalog.

They are **not** declared globally infeasible because Top-K truncation and
reference-catalog scope still remain external limits.
