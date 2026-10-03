# Audit des absences de solution Beam

Frozen public per-role enumeration and feasibility_coverage.analyze_case_feasibility_domain; scalar cost/power check, at most one COMPLETE/H10 witness, no H9 pool or quality approximation.

Same Top-K, same max_paths_per_variant as the campaign. Domain infeasibility does not imply global infeasibility.

44 couples case×K ; 220 configurations sans solution ; 1 témoins faisables manqués.

| Case | K | Largeurs sans solution | Classification | Options MDT | Options OST | Infaisable dans le domaine |
|---|---|---|---|---|---|---|
| REQ_000220 | 5 | [1, 4, 8, 16, 32] | BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED | 30 | 30 | True |
| REQ_000220 | 10 | [1, 4, 8, 16, 32] | BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED | 60 | 60 | True |
| REQ_000220 | 20 | [1, 4, 8, 16, 32] | BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED | 120 | 120 | True |
| REQ_000220 | 50 | [1, 4, 8, 16, 32] | BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED | 300 | 276 | True |
| REQ_000221 | 5 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 30 | 30 | True |
| REQ_000221 | 10 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 60 | 60 | True |
| REQ_000221 | 20 | [1, 4, 8, 16, 32] | JOINT_BUDGET_POWER_CONFLICT | 120 | 120 | True |
| REQ_000221 | 50 | [1, 4, 8, 16, 32] | JOINT_BUDGET_POWER_CONFLICT | 300 | 300 | True |
| REQ_000267 | 5 | [1, 4, 8, 16, 32] | JOINT_BUDGET_POWER_CONFLICT | 30 | 30 | True |
| REQ_000267 | 10 | [1, 4, 8, 16, 32] | JOINT_BUDGET_POWER_CONFLICT | 60 | 60 | True |
| REQ_000267 | 20 | [1, 4, 8, 16, 32] | JOINT_BUDGET_POWER_CONFLICT | 120 | 120 | True |
| REQ_000267 | 50 | [1, 4, 8, 16, 32] | FEASIBLE_PAIR_EXISTS | 300 | 300 | False |
| REQ_000291 | 5 | [1, 4, 8, 16, 32] | JOINT_BUDGET_POWER_CONFLICT | 30 | 24 | True |
| REQ_000323 | 5 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 30 | 30 | True |
| REQ_000323 | 10 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 60 | 54 | True |
| REQ_000323 | 20 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 120 | 114 | True |
| REQ_000323 | 50 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 300 | 228 | True |
| REQ_000366 | 5 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 30 | 24 | True |
| REQ_000366 | 10 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 60 | 54 | True |
| REQ_000391 | 5 | [1, 4, 8, 16, 32] | BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED | 30 | 30 | True |
| REQ_000391 | 10 | [1, 4, 8, 16, 32] | BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED | 60 | 60 | True |
| REQ_000391 | 20 | [1, 4, 8, 16, 32] | BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED | 120 | 120 | True |
| REQ_000391 | 50 | [1, 4, 8, 16, 32] | BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED | 300 | 300 | True |
| REQ_000637 | 5 | [1, 4, 8, 16, 32] | BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED | 30 | 30 | True |
| REQ_000637 | 10 | [1, 4, 8, 16, 32] | BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED | 60 | 60 | True |
| REQ_000637 | 20 | [1, 4, 8, 16, 32] | BUDGET_LOWER_BOUND_EXCEEDS | 120 | 120 | True |
| REQ_000637 | 50 | [1, 4, 8, 16, 32] | BUDGET_LOWER_BOUND_EXCEEDS | 300 | 300 | True |
| REQ_000657 | 5 | [1, 4, 8, 16, 32] | BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED | 30 | 24 | True |
| REQ_000657 | 10 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 60 | 54 | True |
| REQ_000696 | 5 | [1, 4, 8, 16, 32] | JOINT_BUDGET_POWER_CONFLICT | 30 | 30 | True |
| REQ_000696 | 10 | [1, 4, 8, 16, 32] | JOINT_BUDGET_POWER_CONFLICT | 60 | 60 | True |
| REQ_000696 | 20 | [1, 4, 8, 16, 32] | JOINT_BUDGET_POWER_CONFLICT | 120 | 120 | True |
| REQ_000696 | 50 | [1, 4, 8, 16, 32] | JOINT_BUDGET_POWER_CONFLICT | 300 | 294 | True |
| REQ_000857 | 5 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 30 | 30 | True |
| REQ_000989 | 5 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 30 | 18 | True |
| REQ_000989 | 10 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 60 | 42 | True |
| REQ_001042 | 5 | [1, 4, 8, 16, 32] | BUDGET_LOWER_BOUND_EXCEEDS | 30 | 30 | True |
| REQ_001042 | 10 | [1, 4, 8, 16, 32] | BUDGET_LOWER_BOUND_EXCEEDS | 60 | 30 | True |
| REQ_001042 | 20 | [1, 4, 8, 16, 32] | BUDGET_LOWER_BOUND_EXCEEDS | 120 | 30 | True |
| REQ_001042 | 50 | [1, 4, 8, 16, 32] | BUDGET_LOWER_BOUND_EXCEEDS | 300 | 30 | True |
| REQ_001078 | 5 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 30 | 30 | True |
| REQ_001078 | 10 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 60 | 60 | True |
| REQ_001078 | 20 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 120 | 120 | True |
| REQ_001078 | 50 | [1, 4, 8, 16, 32] | POWER_LOWER_BOUND_EXCEEDS | 300 | 282 | True |
