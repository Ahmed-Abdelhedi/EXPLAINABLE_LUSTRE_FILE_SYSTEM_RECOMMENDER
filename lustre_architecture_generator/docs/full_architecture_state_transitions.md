# H7 — ArchitectureState and deterministic transitions

H7 materializes one already-compatible set of choices into a complete physical
`ArchitectureState`.

It still does **not search** for the best choices and does **not use Beam Search**.

## State schema v1.1

The state contains ranking/handoff provenance, original requirements, selected
MDT/OST drives, selected protection results, compatible hardware paths, final
physical-drive counts, provisional target-group counts, MDS/OSS counts,
controller/enclosure/network-adapter counts, aggregate performance, cost/power,
validation status and an explainability trace.

## Legal transition order

`EMPTY → DRIVES_SELECTED → PROTECTION_SELECTED → SERVERS_SELECTED → ENCLOSURES_SELECTED → NETWORK_SELECTED → COMPLETE`

Every transition returns a deep copy; the input state is not mutated.

## COMPLETE is not yet VALIDATED

At H7 a complete state has:

- `validation.is_complete = true`
- `validation.is_valid = false`
- `validation.status = PENDING_FULL_VALIDATOR`

The future deterministic full-architecture validator is responsible for final
hard validity.

## Target-count convention

H7 uses `mdt_count = MDT protection group_count` and
`ost_count = OST protection group_count` as a provisional target-group
representation. This is explicit and traceable, but it is not claimed as a
final performance-optimal Lustre target-layout policy. It must be validated or
refined before the architecture layer is frozen.

## Runtime validation selection

The H7 validator uses the first ranked candidate / first protection profile /
first compatible path only to prove state-transition coverage on real cases.
It is not an optimization or recommendation policy.
