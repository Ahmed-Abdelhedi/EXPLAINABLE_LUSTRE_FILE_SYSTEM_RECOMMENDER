# H5 — Protection / RAID arithmetic

H5 converts the ranking-layer `raw_minimum_drive_count` lower bound into a
physical whole-group drive count for each candidate/protection-profile pair.

H5 does **not choose** the best RAID profile. It only calculates each legal
profile variant deterministically.

## Capacity

Usable capacity per RAID group is:

`per_drive_capacity_tib * data_drives_per_group`

## Read performance

Read IOPS / read bandwidth per group is modeled as:

`per_drive_read * total_group_drives * read_efficiency`

## Write performance

Write IOPS / write bandwidth per group is modeled as:

`per_drive_write * data_drives_per_group * write_efficiency`

This makes mirror/parity overhead explicit and configurable.

## Required group count

The final number of groups is the maximum of:
- groups required to contain the frozen pre-RAID lower bound;
- groups required for capacity;
- groups required for read performance;
- groups required for write performance;
- for OST, groups required for total throughput.

Then:

`physical_drive_count = group_count * group_size`

So the result is always a complete RAID group count and can never be below the
pre-RAID lower bound.

## Cost / power

Per-drive cost and power are derived from the original deterministic lower-bound
aggregates and scaled by the final physical drive count.

Server/controller/enclosure/network cost and power are **not** included yet.
They belong to later full-architecture composition.

## Scientific boundary

The protection efficiency factors come from the H4
`POLICY_REFERENCE_NOT_VENDOR_BENCHMARKED` catalog. They are policy/reference
parameters, not independently benchmarked vendor performance.
