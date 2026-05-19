from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import IntEnum

from .constants import VIDEO_CRACK_START, VIDEO_MASK_START

PlainVm1Constraints = dict[int, frozenset[int]]
Vm1Constraints = dict[int, "Vm1ConstraintEntry"]


class ConstraintTrust(IntEnum):
    # Engineering-trusted: both superframe markers are independently keyless-readable.
    BOTH_MARKERS = 3
    # High-trust structural evidence: one observed marker plus exact observed sizes.
    SINGLE_MARKER_EXACT_SIZE = 2
    # Empirical C9 template for tiny files; useful but not a VP9 spec rule.
    C9_TEMPLATE = 1


@dataclass(frozen=True)
class Vm1ConstraintEntry:
    values: frozenset[int]
    trust: ConstraintTrust
    reason: str
    support: int = 1


@dataclass(frozen=True)
class Vm1Evidence:
    column: int
    values: frozenset[int]
    trust: ConstraintTrust
    reason: str


@dataclass(frozen=True)
class PlaintextConstraint:
    payload_offset: int
    allowed_values: frozenset[int]
    reason: str


@dataclass
class Vp9ConstraintStats:
    attempted_frames: int = 0
    matched_frames: int = 0
    plaintext_constraints: int = 0
    vm1_columns: set[int] = field(default_factory=set)
    reason_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    extractor_counts: Counter[str] = field(default_factory=Counter)
    same_trust_conflicts: Counter[str] = field(default_factory=Counter)
    cross_trust_conflicts: Counter[str] = field(default_factory=Counter)
    # Evidence with trust <= this value is kept for reporting but not used by the solver.
    disabled_trust_threshold: int = 0
    evidences: list[Vm1Evidence] = field(default_factory=list, repr=False)
    _conflict_fingerprints: set[tuple] = field(default_factory=set, repr=False)

    def add_evidence(self, evidence: Vm1Evidence, *, count_reason: bool = True) -> None:
        # Compare with all historical evidence, even if its trust level is disabled,
        # so conflicts remain visible in the report.
        for old in self.evidences:
            if old.column != evidence.column:
                continue
            if old.values & evidence.values:
                continue

            if old.trust == evidence.trust:
                conflict_key = old.trust.name
                counter_key = ("same", conflict_key)
                disabled_level = int(old.trust)
            else:
                low = min(old.trust, evidence.trust)
                high = max(old.trust, evidence.trust)
                conflict_key = f"{low.name}_vs_{high.name}"
                counter_key = ("cross", conflict_key)
                disabled_level = int(low)

            value_key = tuple(sorted(evidence.values))
            fingerprint = (evidence.column, counter_key, value_key)
            if fingerprint not in self._conflict_fingerprints:
                self._conflict_fingerprints.add(fingerprint)
                if counter_key[0] == "same":
                    self.same_trust_conflicts[conflict_key] += 1
                else:
                    self.cross_trust_conflicts[conflict_key] += 1
            self.disabled_trust_threshold = max(
                self.disabled_trust_threshold,
                disabled_level,
            )

        self.evidences.append(evidence)
        self.vm1_columns.add(evidence.column)
        if count_reason:
            self.plaintext_constraints += 1
            self.reason_counts[(evidence.reason, evidence.trust.name)] += 1

    @property
    def conflict_total(self) -> int:
        return sum(self.same_trust_conflicts.values()) + sum(
            self.cross_trust_conflicts.values()
        )

    def merge(self, other: "Vp9ConstraintStats") -> None:
        self.attempted_frames += other.attempted_frames
        self.matched_frames += other.matched_frames
        self.extractor_counts.update(other.extractor_counts)
        for evidence in other.evidences:
            self.add_evidence(evidence)

    def as_report(self) -> dict:
        disabled_names = [
            trust.name
            for trust in ConstraintTrust
            if int(trust) <= self.disabled_trust_threshold
        ]
        return {
            "attempted_frames": self.attempted_frames,
            "matched_frames": self.matched_frames,
            "plaintext_constraints": self.plaintext_constraints,
            "reason_counts": _nested_reason_counts(self.reason_counts),
            "extractor_counts": dict(sorted(self.extractor_counts.items())),
            "conflict_counts": {
                "same_trust": dict(sorted(self.same_trust_conflicts.items())),
                "cross_trust": dict(sorted(self.cross_trust_conflicts.items())),
            },
            "disabled_trust_threshold": self.disabled_trust_threshold,
            "disabled_trust_names": disabled_names,
        }


def _nested_reason_counts(
    counts: Counter[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = {}
    for (reason, trust), count in sorted(counts.items()):
        nested.setdefault(reason, {})[trust] = count
    return nested


@dataclass(frozen=True)
class SuperframeIndex:
    marker: int
    bytes_per_size: int
    frame_count: int
    index_start: int
    index_size: int
    frame_size: int


@dataclass(frozen=True)
class _MarkerCandidate:
    index: SuperframeIndex
    source: str
    trust: ConstraintTrust


# Empirical mapping for common C9 superframes:
# hidden inter-frame prefix -> shown inter-frame prefix.
_C9_SECOND_FRAME_4BYTE_PREFIXES = {
    bytes.fromhex("84008049"): bytes.fromhex("86004096"),
    bytes.fromhex("84004085"): bytes.fromhex("8600410e"),
}


_MAX_EXACT_SIZE_UNKNOWN_BYTES = 2
_C9_TEMPLATE_REASON = "vp9_c9_second_frame_4byte_header"


def payload_starts_vp9_stream(payload: bytes) -> bool:
    return len(payload) >= 12 and payload[:4] == b"DKIF" and payload[8:12] in (
        b"VP90",
        b"vp90",
    )


def extract_vp9_superframe_constraints(
    payload: bytes,
    *,
    enable_c9_template: bool = True,
) -> Vp9ConstraintStats:
    stats = Vp9ConstraintStats()

    for frame_start, frame_end in _iter_vp9_frame_ranges(payload):
        stats.attempted_frames += 1
        candidate = _detect_superframe_index(
            payload,
            frame_start,
            frame_end,
            enable_c9_template=enable_c9_template,
        )
        if candidate is None:
            continue

        stats.matched_frames += 1
        stats.extractor_counts[candidate.source] += 1
        index = candidate.index

        _record_plaintext_constraints(
            payload,
            _verified_superframe_constraints(payload, index),
            candidate.trust,
            stats,
        )

        if enable_c9_template:
            # Empirical C9 template; keep isolated from structural evidence.
            _record_plaintext_constraints(
                payload,
                _c9_second_frame_4byte_header_constraints(payload, frame_start, index),
                ConstraintTrust.C9_TEMPLATE,
                stats,
            )

    return stats


def merge_vm1_constraints(
    base: Vm1Constraints,
    extra: Vm1Constraints,
    stats: Vp9ConstraintStats | None = None,
) -> Vm1Constraints:
    if stats is None:
        return _merge_final_constraints(base, extra)
    for column, entry in extra.items():
        stats.add_evidence(Vm1Evidence(column, entry.values, entry.trust, entry.reason))
    return build_vm1_constraints(stats)


def build_vm1_constraints(stats: Vp9ConstraintStats) -> Vm1Constraints:
    constraints: Vm1Constraints = {}
    for evidence in stats.evidences:
        if int(evidence.trust) <= stats.disabled_trust_threshold:
            continue
        entry = Vm1ConstraintEntry(evidence.values, evidence.trust, evidence.reason)
        current = constraints.get(evidence.column)
        if current is None:
            constraints[evidence.column] = entry
            continue

        intersection = current.values & entry.values
        if not intersection:
            return {}
        constraints[evidence.column] = _merge_compatible_entries(
            current,
            entry,
            frozenset(intersection),
        )
    return constraints


def _merge_final_constraints(base: Vm1Constraints, extra: Vm1Constraints) -> Vm1Constraints:
    merged = dict(base)
    for column, entry in extra.items():
        current = merged.get(column)
        if current is None:
            merged[column] = entry
            continue
        intersection = current.values & entry.values
        if not intersection:
            continue
        merged[column] = _merge_compatible_entries(
            current,
            entry,
            frozenset(intersection),
        )
    return merged


def _merge_compatible_entries(
    current: Vm1ConstraintEntry,
    entry: Vm1ConstraintEntry,
    values: frozenset[int],
) -> Vm1ConstraintEntry:
    if current.trust > entry.trust:
        trust = current.trust
        reason = current.reason
    else:
        trust = entry.trust
        reason = entry.reason
    support = current.support + entry.support if current.reason == entry.reason else 1
    return Vm1ConstraintEntry(values, trust, reason, support)


def plain_vm1_constraints(constraints: Vm1Constraints) -> PlainVm1Constraints:
    return {column: entry.values for column, entry in constraints.items()}


def format_vm1_constraints(constraints: Vm1Constraints) -> dict[str, list[str]]:
    return {
        str(column): [f"{value:02X}" for value in sorted(entry.values)]
        for column, entry in sorted(constraints.items())
    }


def _record_plaintext_constraints(
    payload: bytes,
    plaintext_constraints: list[PlaintextConstraint],
    trust: ConstraintTrust,
    stats: Vp9ConstraintStats,
) -> None:
    # Plaintext facts only become evidence when the offset has a single-vm1 relation.
    for constraint in plaintext_constraints:
        mapped = _vm1_values_from_plaintext_constraint(
            payload,
            constraint.payload_offset,
            constraint.allowed_values,
        )
        if mapped is None:
            continue
        column, allowed = mapped
        stats.add_evidence(Vm1Evidence(column, allowed, trust, constraint.reason))


def _detect_superframe_index(
    payload: bytes,
    frame_start: int,
    frame_end: int,
    *,
    enable_c9_template: bool = True,
) -> _MarkerCandidate | None:
    # Prefer spec-backed evidence first; the C9 template is only a fallback.
    both_markers: list[_MarkerCandidate] = []
    exact_size: list[_MarkerCandidate] = []
    template: list[_MarkerCandidate] = []

    for marker in range(0xC0, 0xE0):
        index = _try_parse_superframe_index(payload, frame_start, frame_end, marker)
        if index is None:
            continue

        if _both_markers_observed(payload, index):
            both_markers.append(
                _MarkerCandidate(
                    index,
                    _extractor_name("both_marker"),
                    ConstraintTrust.BOTH_MARKERS,
                )
            )
        elif _single_marker_with_exact_size(payload, index):
            exact_size.append(
                _MarkerCandidate(
                    index,
                    _extractor_name("exact_size"),
                    ConstraintTrust.SINGLE_MARKER_EXACT_SIZE,
                )
            )
        # C9 template fallback. Use only when no stronger marker candidate exists.
        elif enable_c9_template and _is_c9_template_supported(payload, frame_start, index):
            template.append(
                _MarkerCandidate(
                    index,
                    "c9_template_superframe",
                    ConstraintTrust.C9_TEMPLATE,
                )
            )

    if len(both_markers) == 1:
        return both_markers[0]
    if both_markers:
        return None
    if len(exact_size) == 1:
        return exact_size[0]
    if exact_size:
        return None
    if len(template) == 1:
        return template[0]
    return None


def _try_parse_superframe_index(
    payload: bytes,
    frame_start: int,
    frame_end: int,
    marker: int,
) -> SuperframeIndex | None:
    if not _is_vp9_superframe_marker(marker):
        return None

    bytes_per_size, frame_count, index_size = _superframe_meta(marker)
    index_start = frame_end - index_size
    if index_start < frame_start or index_start < 0 or frame_end > len(payload):
        return None

    start_check = _plain_byte_can_be(payload, index_start, marker)
    end_check = _plain_byte_can_be(payload, frame_end - 1, marker)
    if start_check is None or end_check is None:
        return None
    if not start_check[0] or not end_check[0]:
        return None

    total_subframe_size = frame_end - frame_start - index_size
    if not _superframe_sizes_feasible(
        payload,
        index_start,
        total_subframe_size,
        bytes_per_size,
        frame_count,
    ):
        return None

    return SuperframeIndex(
        marker=marker,
        bytes_per_size=bytes_per_size,
        frame_count=frame_count,
        index_start=index_start,
        index_size=index_size,
        frame_size=frame_end - frame_start,
    )


def _extractor_name(source: str) -> str:
    return f"{source}_superframe"


def _both_markers_observed(payload: bytes, index: SuperframeIndex) -> bool:
    return (
        _keyless_plain_byte(payload, index.index_start) == index.marker
        and _keyless_plain_byte(
            payload,
            index.index_start + index.index_size - 1,
        )
        == index.marker
    )


def _single_marker_with_exact_size(payload: bytes, index: SuperframeIndex) -> bool:
    start_observed = _keyless_plain_byte(payload, index.index_start) == index.marker
    end_observed = (
        _keyless_plain_byte(payload, index.index_start + index.index_size - 1)
        == index.marker
    )
    if not (start_observed or end_observed):
        return False
    if start_observed and end_observed:
        return False
    # One observed marker plus exact observed sizes is strong, but not absolute.
    return _superframe_sizes_known_and_exact(payload, index)


def _is_c9_template_supported(
    payload: bytes,
    frame_start: int,
    index: SuperframeIndex,
) -> bool:
    if index.marker != 0xC9:
        return False
    return _c9_second_frame_template_is_verified(payload, frame_start, index)


def _verified_superframe_constraints(
    payload: bytes,
    index: SuperframeIndex,
) -> list[PlaintextConstraint]:
    constraints = _superframe_marker_constraints(payload, index)
    constraints.extend(_exact_superframe_size_constraints(payload, index))
    return constraints


def _superframe_marker_constraints(
    payload: bytes,
    index: SuperframeIndex,
) -> list[PlaintextConstraint]:
    constraints: list[PlaintextConstraint] = []
    marker_offsets = (
        (index.index_start, "vp9_superframe_start_marker"),
        (index.index_start + index.index_size - 1, "vp9_superframe_tail_marker"),
    )
    for offset, reason in marker_offsets:
        if _single_vm1_relation(payload, offset) is None:
            continue
        constraints.append(
            PlaintextConstraint(offset, frozenset({index.marker}), reason)
        )
    return constraints


def _exact_superframe_size_constraints(
    payload: bytes,
    index: SuperframeIndex,
) -> list[PlaintextConstraint]:
    sizes = _solve_exact_superframe_sizes(payload, index)
    if sizes is None:
        return []

    constraints: list[PlaintextConstraint] = []
    offset = index.index_start + 1
    for size in sizes:
        for byte_index in range(index.bytes_per_size):
            value = (size >> (8 * byte_index)) & 0xFF
            constraints.append(
                PlaintextConstraint(
                    offset,
                    frozenset({value}),
                    "vp9_superframe_exact_size",
                )
            )
            offset += 1
    return constraints


def _c9_second_frame_4byte_header_constraints(
    payload: bytes,
    frame_start: int,
    index: SuperframeIndex,
) -> list[PlaintextConstraint]:
    if index.marker != 0xC9 or index.bytes_per_size != 2 or index.frame_count != 2:
        return []

    sizes = _solve_exact_superframe_sizes(payload, index)
    if sizes is None:
        return []

    first_size = sizes[0]
    second_frame_offset = frame_start + first_size
    if second_frame_offset <= frame_start or second_frame_offset + 4 > index.index_start:
        return []

    header_values = [_keyless_plain_byte(payload, frame_start + i) for i in range(4)]
    if any(value is None for value in header_values):
        return []

    first_header = bytes(int(value) for value in header_values)
    expected = _C9_SECOND_FRAME_4BYTE_PREFIXES.get(first_header)
    if expected is None:
        return []

    # Empirical C9-only header template; keep isolated and lower-trust.
    return [
        PlaintextConstraint(
            second_frame_offset + i,
            frozenset({value}),
            _C9_TEMPLATE_REASON,
        )
        for i, value in enumerate(expected)
    ]


def _c9_second_frame_template_is_verified(
    payload: bytes,
    frame_start: int,
    index: SuperframeIndex,
) -> bool:
    if index.marker != 0xC9 or index.bytes_per_size != 2 or index.frame_count != 2:
        return False

    sizes = _solve_exact_superframe_sizes(payload, index)
    if sizes is None:
        return False

    first_size = sizes[0]
    second_frame_offset = frame_start + first_size
    if second_frame_offset <= frame_start or second_frame_offset + 4 > index.index_start:
        return False

    header_values = [_keyless_plain_byte(payload, frame_start + i) for i in range(4)]
    if any(value is None for value in header_values):
        return False

    first_header = bytes(int(value) for value in header_values)
    return _C9_SECOND_FRAME_4BYTE_PREFIXES.get(first_header) is not None


def _solve_exact_superframe_sizes(
    payload: bytes,
    index: SuperframeIndex,
) -> list[int] | None:
    # Exact size evidence is emitted only when all unknown bytes have one solution.
    total_subframe_size = index.frame_size - index.index_size
    if total_subframe_size < index.frame_count:
        return None

    known: list[int | None] = []
    offset = index.index_start + 1
    for _ in range(index.frame_count * index.bytes_per_size):
        known.append(_keyless_plain_byte(payload, offset))
        offset += 1

    unknown_count = sum(value is None for value in known)
    if unknown_count > _MAX_EXACT_SIZE_UNKNOWN_BYTES:
        return None

    possible: list[list[int]] = []
    candidate_values: list[int] = [0] * len(known)

    def search(byte_pos: int) -> bool:
        if byte_pos == len(known):
            sizes = _size_bytes_to_values(candidate_values, index.bytes_per_size)
            if any(size <= 0 for size in sizes):
                return True
            if sum(sizes) == total_subframe_size:
                possible.append(sizes)
            return len(possible) <= 1

        value = known[byte_pos]
        if value is not None:
            candidate_values[byte_pos] = value
            return search(byte_pos + 1)

        for guessed in range(256):
            candidate_values[byte_pos] = guessed
            if not search(byte_pos + 1):
                return False
        return True

    search(0)
    return possible[0] if len(possible) == 1 else None


def _size_bytes_to_values(values: list[int], bytes_per_size: int) -> list[int]:
    sizes: list[int] = []
    for start in range(0, len(values), bytes_per_size):
        size = 0
        for byte_index in range(bytes_per_size):
            size |= values[start + byte_index] << (8 * byte_index)
        sizes.append(size)
    return sizes


def _superframe_sizes_known_and_exact(
    payload: bytes,
    index: SuperframeIndex,
) -> bool:
    sizes = _solve_observed_superframe_sizes(payload, index)
    if sizes is None:
        return False
    return sum(sizes) == index.frame_size - index.index_size


def _solve_observed_superframe_sizes(
    payload: bytes,
    index: SuperframeIndex,
) -> list[int] | None:
    values: list[int] = []
    offset = index.index_start + 1
    for _ in range(index.frame_count * index.bytes_per_size):
        value = _keyless_plain_byte(payload, offset)
        if value is None:
            return None
        values.append(value)
        offset += 1
    sizes = _size_bytes_to_values(values, index.bytes_per_size)
    if any(size <= 0 for size in sizes):
        return None
    return sizes


def _superframe_sizes_feasible(
    payload: bytes,
    index_start: int,
    total_subframe_size: int,
    bytes_per_size: int,
    frame_count: int,
) -> bool:
    if total_subframe_size < frame_count:
        return False

    min_sum = 0
    max_sum = 0
    pos = index_start + 1
    all_known = True
    for _ in range(frame_count):
        min_value = 0
        max_value = 0
        for byte_index in range(bytes_per_size):
            coefficient = 1 << (8 * byte_index)
            known = _keyless_plain_byte(payload, pos + byte_index)
            if known is None:
                all_known = False
                max_value += 0xFF * coefficient
            else:
                min_value += known * coefficient
                max_value += known * coefficient
        min_sum += min_value
        max_sum += max_value
        pos += bytes_per_size

    if all_known:
        return min_sum == total_subframe_size
    return min_sum <= total_subframe_size <= max_sum


def _plain_byte_can_be(payload: bytes, offset: int, value: int) -> tuple[bool, bool] | None:
    known = _keyless_plain_byte(payload, offset)
    if known is not None:
        return known == value, True
    if _is_direct_vm1_offset(offset):
        return True, False
    return None


def _vm1_values_from_plaintext_constraint(
    payload: bytes,
    payload_offset: int,
    allowed_plaintext: frozenset[int],
) -> tuple[int, frozenset[int]] | None:
    mapped = _single_vm1_relation(payload, payload_offset)
    if mapped is None:
        return None

    column, known_xor, xor_ff = mapped
    ff = 0xFF if xor_ff else 0x00
    return column, frozenset(known_xor ^ value ^ ff for value in allowed_plaintext)


def _keyless_plain_byte(payload: bytes, payload_offset: int) -> int | None:
    if payload_offset < 0 or payload_offset >= len(payload):
        return None
    if payload_offset < VIDEO_MASK_START:
        return payload[payload_offset]

    if payload_offset < VIDEO_CRACK_START:
        head_block = (payload_offset - VIDEO_MASK_START) // 32
        column = (payload_offset - VIDEO_MASK_START) & 0x1F
        if head_block % 2 != 0:
            return None
        current_s = _encrypted_prefix_xor(payload, head_block, column)
        if current_s is None:
            return None
        return payload[payload_offset] ^ current_s ^ 0xFF

    relative = payload_offset - VIDEO_CRACK_START
    block_index = relative // 32
    column = relative & 0x1F
    if block_index % 2 == 0:
        return None
    return _encrypted_prefix_xor(payload, block_index, column)


def _single_vm1_relation(payload: bytes, payload_offset: int) -> tuple[int, int, bool] | None:
    """Return (vm1_column, known_xor, xor_ff).

    The plaintext relation is:
      plaintext = known_xor ^ vm1[column] ^ (0xFF if xor_ff else 0)

    This covers the direct single-byte vm1 positions in both encrypted regions:
      - 0x60..0x7F, 0xA0..0xBF, 0xE0..0xFF, 0x120..0x13F
      - even 32-byte blocks at 0x140 and later
    """
    if payload_offset < 0 or payload_offset >= len(payload):
        return None

    if VIDEO_MASK_START <= payload_offset < VIDEO_CRACK_START:
        head_block = (payload_offset - VIDEO_MASK_START) // 32
        column = (payload_offset - VIDEO_MASK_START) & 0x1F
        if head_block % 2 == 0:
            return None
        current_s = _encrypted_prefix_xor(payload, head_block, column)
        if current_s is None:
            return None
        return column, payload[payload_offset] ^ current_s, False

    if payload_offset >= VIDEO_CRACK_START:
        relative = payload_offset - VIDEO_CRACK_START
        block_index = relative // 32
        column = relative & 0x1F
        if block_index % 2 != 0:
            return None
        current_s = _encrypted_prefix_xor(payload, block_index, column)
        if current_s is None:
            return None
        return column, current_s, True

    return None


def _encrypted_prefix_xor(payload: bytes, block_index: int, column: int) -> int | None:
    value = 0
    for block in range(block_index + 1):
        pos = VIDEO_CRACK_START + block * 32 + column
        if pos >= len(payload):
            return None
        value ^= payload[pos]
    return value


def _is_direct_vm1_offset(payload_offset: int) -> bool:
    if VIDEO_MASK_START <= payload_offset < VIDEO_CRACK_START:
        return ((payload_offset - VIDEO_MASK_START) // 32) % 2 != 0
    if payload_offset < VIDEO_CRACK_START:
        return False
    relative = payload_offset - VIDEO_CRACK_START
    return (relative // 32) % 2 == 0


def _iter_vp9_frame_ranges(payload: bytes) -> Iterable[tuple[int, int]]:
    if len(payload) >= 44 and payload[:4] == b"DKIF":
        if payload[8:12] not in (b"VP90", b"vp90"):
            return
        frame_size = int.from_bytes(payload[32:36], "little")
        frame_start = 44
        frame_end = frame_start + frame_size
        if 0 < frame_size and frame_end <= len(payload):
            yield frame_start, frame_end
        return

    if len(payload) >= 12:
        frame_size = int.from_bytes(payload[:4], "little")
        frame_start = 12
        frame_end = frame_start + frame_size
        if 0 < frame_size and frame_end <= len(payload):
            yield frame_start, frame_end


def _is_vp9_superframe_marker(marker: int) -> bool:
    return (marker & 0xE0) == 0xC0


def _superframe_meta(marker: int) -> tuple[int, int, int]:
    bytes_per_size = ((marker >> 3) & 0x03) + 1
    frame_count = (marker & 0x07) + 1
    index_size = 2 + bytes_per_size * frame_count
    return bytes_per_size, frame_count, index_size
