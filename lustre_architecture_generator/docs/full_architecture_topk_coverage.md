# H10-D — Top-K feasibility coverage

After H10-C, the architecture layer has confirmed at least one valid
architecture for 1,090 of 1,200 cases.

110 cases remain unresolved inside the following domain:

- Top-K = 10;
- every current protection profile;
- no role-option cap;
- the full compatible hardware-path domain of the current reference catalog;
- final confirmation by the independent H10 validator.

H10-D studies whether the remaining limitation is the drive candidate Top-K.

## Experiment

For each of the 110 unresolved cases:

1. build the normal H2 runtime handoff at K=20;
2. enumerate the complete H10-C domain;
3. confirm every recovered feasible pair with H10;
4. if still unresolved, repeat at K=50.

The first K that recovers a valid architecture is recorded.

## Important distinction

This experiment measures **feasibility coverage**, not recommendation quality
or K optimality.

It does not replace the later K × beam sensitivity evaluation requested for
the search layer.

No Beam Search and no H9 architecture score are used to decide feasibility.

## Interpretation after K=50

A case still unresolved after H10-D means:

> no feasible architecture was found through K=50 using the current
> protection profiles and the complete compatible path domain of the current
> reference hardware catalog.

This is still not a proof of global infeasibility because:

- candidates ranked below K=50 are outside the experiment;
- the hardware catalog is a reference catalog rather than an exhaustive
  representation of all real products.
