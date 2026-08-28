# Full Architecture Hardware Scope — Schema v1.0

This document freezes the semantic scope of the physical architecture layer
before any Beam Search implementation.

## Components

### ServerProfile
Represents an MDS, OSS, or dual-role server platform.
Required dimensions include:
- CPU cores and memory;
- PCIe slot/lane budget;
- local drive-bay capacities by form factor;
- native drive protocols;
- network-interface support;
- dual-PSU capability;
- price and power.

### ControllerProfile
Represents an HBA, RAID controller, or NVMe-switch-like adapter.
Required dimensions include:
- supported drive protocols;
- number of ports;
- aggregate storage-side bandwidth in GB/s;
- PCIe generation and lanes;
- multipath capability;
- price and power.

### EnclosureProfile
Represents external JBOD/JBOF storage.
Required dimensions include:
- drive protocols and form factors;
- number of drive bays;
- uplink count;
- uplink bandwidth in GB/s;
- redundant-path capability;
- price and power.

### NetworkProfile
Represents the server-side Lustre data network.
Important: network link speed is expressed in **Gbit/s**, not GB/s.
The later compatibility layer must convert it to usable GB/s using
`link_speed_gbit_s / 8 * usable_efficiency`.

### ProtectionProfile
Represents RAID1/10/5/6 group arithmetic.
The profile contains:
- minimum drives per group;
- data drives;
- parity drives;
- mirror copies;
- capacity/read/write efficiency;
- fault tolerance.

No final physical count is computed in H3.

### HAProfile
Represents NONE / ACTIVE_PASSIVE / ACTIVE_ACTIVE high-availability policy.
It defines minimum node multiplicity and requirements for shared storage and
redundant networking.

## ArchitectureState

`ArchitectureState` is the future search state manipulated by the deterministic
generator and, only later, Beam Search.

It already reserves fields for:
- selected drives;
- selected protection profiles;
- MDS/OSS servers;
- controller;
- enclosure;
- network;
- HA;
- final physical counts;
- aggregate capacity/performance/cost/power;
- validation status and violations;
- explainability trace.

At H3 the state starts at stage `EMPTY`; no hardware decision is yet made.

## Units

- drive/OST throughput: GB/s
- controller/enclosure bandwidth: GB/s
- network line rate: Gbit/s
- capacity: TiB
- metadata performance: IOPS
- cost: USD
- power: W

This explicit separation avoids repeating the historical `_gbps` naming debt in
the frozen sizing/ranking modules.
