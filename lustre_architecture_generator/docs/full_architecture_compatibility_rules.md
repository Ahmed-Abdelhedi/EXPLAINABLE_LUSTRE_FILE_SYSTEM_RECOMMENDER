# H6 — Deterministic Hardware Compatibility Rules

H6 sits after H5 RAID/protection arithmetic and before full architecture
generation.

It does not score or select a final architecture.

For each drive/protection variant it can determine whether a hardware path is
compatible and estimate only the minimum resources needed to make that path
possible.

## Rules

H6 checks:
- MDS/OSS server role compatibility;
- drive protocol -> controller compatibility;
- NVMe PCIe generation compatibility when available;
- drive protocol/form factor -> direct server bays;
- drive protocol/form factor -> enclosure;
- controller -> server PCIe slots/lane budget;
- direct vs external enclosure attachment;
- physical drive count -> available bays/enclosures;
- controller port count;
- controller aggregate bandwidth for OST requirements;
- enclosure uplink bandwidth for OST requirements;
- network Gbit/s -> usable GB/s conversion;
- network adapter multiplicity;
- HA -> minimum server multiplicity;
- HA -> dual PSU, multipath, redundant network and external shared storage.

## Important boundary

The returned server/controller/enclosure/network counts are *minimum resource
requirements for a compatibility path*. They are not yet the final
ArchitectureState/BOM.

No architecture score is calculated in H6.

## 2.5-inch form-factor normalization

`FF_2_5`, `FF_U2` and `FF_U3` are treated as the same mechanical 2.5-inch
family only after protocol compatibility is checked separately. This is a
catalog normalization rule, not permission to connect SATA/SAS/NVMe protocols
interchangeably.

## Network units

Network catalog values remain Gbit/s.

Usable adapter bandwidth is:

`link_speed_gbit_s / 8 * ports_per_adapter * usable_efficiency`

The result is GB/s.
