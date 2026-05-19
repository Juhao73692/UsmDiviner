from __future__ import annotations

import heapq
import mmap
import struct
from array import array
from collections.abc import Iterable
from pathlib import Path

from .constants import (
    SOLVER_BEAM,
    SOLVER_L1_BEAM,
    BIGRAM_ADAPT_MIN_HITS,
    BIGRAM_LOW_CONF_FF_WEIGHT,
    BIGRAM_LOW_CONF_ZERO_WEIGHT,
    BIGRAM_RATIO_MAX,
    BIGRAM_RATIO_MIN,
    BIGRAM_WEIGHT_TOTAL,
    SIG_SFV,
    VIDEO_CRACK_START,
    VIDEO_MASK_START,
)
from .exceptions import UsmFormatError
from .vp9_superframe_constraints import (
    Vm1Constraints,
    Vp9ConstraintStats,
    extract_vp9_superframe_constraints,
    format_vm1_constraints,
    build_vm1_constraints,
    payload_starts_vp9_stream,
    plain_vm1_constraints,
)

ScoreMatrix = list[array]
Candidate = tuple[int, list[int]]
BigramWeights = tuple[int, int]


def crack_keys_from_usm(
    usm_path: str | Path,
    max_video_bytes: int | None = None,
    beam_size: int = SOLVER_BEAM,
    l1_beam_size: int = SOLVER_L1_BEAM,
) -> tuple[bytes | None, bytes | None, dict]:
    path = Path(usm_path)
    if path.stat().st_size == 0:
        raise UsmFormatError("empty USM file")

    with path.open("rb") as fp, mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as data:
        return _crack_from_buffer(data, max_video_bytes, beam_size, l1_beam_size)


def _crack_from_buffer(
    data,
    max_video_bytes: int | None,
    beam_size: int,
    l1_beam_size: int,
) -> tuple[bytes | None, bytes | None, dict]:
    offset = 0
    data_len = len(data)
    unigram, bigram = _make_score_matrices()
    video_blocks_found = 0
    chunks_seen = 0
    video_crack_bytes_used = 0
    sample_rows = 0
    odd_bigram_zero = 0
    odd_bigram_ff = 0
    vp9_constraints: Vm1Constraints = {}
    vp9_constraint_stats = Vp9ConstraintStats()
    vp9_stream_detected = False
    crack_limited = max_video_bytes is not None
    enable_c9_template = data_len < 2_000_000

    while offset + 32 <= data_len:
        header = data[offset:offset + 32]
        try:
            signature, data_size = struct.unpack(">II", header[:8])
            data_offset = header[9]
            padding_size = struct.unpack(">H", header[10:12])[0]
            data_type = header[15]
        except (IndexError, struct.error) as exc:
            raise UsmFormatError(f"bad USM header at 0x{offset:X}") from exc

        next_offset = offset + data_size + 8
        if data_size < 0x18 or next_offset > data_len:
            raise UsmFormatError(f"bad USM chunk at 0x{offset:X}: data_size={data_size}")

        payload_start = offset + 8 + data_offset
        payload_size = data_size - data_offset - padding_size
        payload_end = payload_start + payload_size
        if payload_size < 0 or payload_end > data_len:
            raise UsmFormatError(f"bad USM payload at 0x{offset:X}")

        chunks_seen += 1
        if signature == SIG_SFV and data_type == 0:
            if not vp9_stream_detected:
                header_probe_size = min(payload_size, 44)
                header_probe = data[payload_start:payload_start + header_probe_size]
                if payload_starts_vp9_stream(header_probe):
                    vp9_stream_detected = True

            if payload_size - VIDEO_MASK_START >= 0x200:
                encrypted_start = payload_start + VIDEO_CRACK_START
                encrypted_size = payload_size - VIDEO_CRACK_START
                constraint_encrypted_size = encrypted_size
                if max_video_bytes is not None:
                    remaining = max_video_bytes - video_crack_bytes_used
                    if remaining < 32:
                        break
                    constraint_encrypted_size = min(encrypted_size, remaining)
                    encrypted_size = constraint_encrypted_size

                encrypted_size -= encrypted_size % 32
                if encrypted_size >= 32:
                    constraint_payload_size = min(
                        payload_size,
                        VIDEO_CRACK_START + constraint_encrypted_size,
                    )
                    payload = data[payload_start:payload_start + constraint_payload_size]
                    if vp9_stream_detected:
                        new_stats = extract_vp9_superframe_constraints(
                            payload,
                            enable_c9_template=enable_c9_template,
                        )
                        vp9_constraint_stats.merge(new_stats)
                        vp9_constraints = build_vm1_constraints(vp9_constraint_stats)

                    video_blocks_found += 1
                    video_crack_bytes_used += encrypted_size
                    rows, odd_zero, odd_ff = _accumulate_score_matrices(
                        data,
                        encrypted_start,
                        encrypted_size,
                        unigram,
                        bigram,
                    )
                    sample_rows += rows
                    odd_bigram_zero += odd_zero
                    odd_bigram_ff += odd_ff
                    if max_video_bytes is not None and video_crack_bytes_used >= max_video_bytes:
                        break

        offset = next_offset

    bigram_zero_weight, bigram_ff_weight, odd_ratio = (
        _estimate_bigram_weights(
            odd_bigram_zero,
            odd_bigram_ff,
        )
    )
    bigram_weights = (bigram_zero_weight, bigram_ff_weight)

    if video_blocks_found == 0 or sample_rows == 0:
        return None, None, {
            "video_blocks_found": video_blocks_found,
            "chunks_seen": chunks_seen,
            "reason": "no encrypted/long-enough @SFV video blocks found",
            "solver": "bigram",
            "fast_enabled": crack_limited,
            "video_crack_bytes_limit": max_video_bytes,
            "video_crack_bytes_used": video_crack_bytes_used,
            "bigram_zero_weight": bigram_zero_weight,
            "bigram_ff_weight": bigram_ff_weight,
            "odd_bigram_zero": odd_bigram_zero,
            "odd_bigram_ff": odd_bigram_ff,
            "odd_bigram_ratio": odd_ratio,
            "beam_size": beam_size,
            "l1_beam_size": l1_beam_size,
            **_vp9_report(vp9_stream_detected, vp9_constraints, vp9_constraint_stats),
        }

    best_score, best_vm1 = _solve_vm1_bigram(
        unigram,
        bigram,
        beam_size,
        l1_beam_size,
        bigram_weights,
        plain_vm1_constraints(vp9_constraints),
    )

    if best_vm1 is None:
        return None, None, {
            "video_blocks_found": video_blocks_found,
            "reason": "solver produced no candidate",
            "solver": "bigram",
            "fast_enabled": crack_limited,
            "video_crack_bytes_limit": max_video_bytes,
            "video_crack_bytes_used": video_crack_bytes_used,
            "bigram_zero_weight": bigram_zero_weight,
            "bigram_ff_weight": bigram_ff_weight,
            "odd_bigram_zero": odd_bigram_zero,
            "odd_bigram_ff": odd_bigram_ff,
            "odd_bigram_ratio": odd_ratio,
            "beam_size": beam_size,
            "l1_beam_size": l1_beam_size,
            **_vp9_report(vp9_stream_detected, vp9_constraints, vp9_constraint_stats),
        }

    key1 = bytes([
        best_vm1[0],
        best_vm1[1],
        best_vm1[2],
        (best_vm1[3] + 0x34) & 0xFF,
    ])
    key2 = bytes([
        (best_vm1[4] - 0xF9) & 0xFF,
        best_vm1[5] ^ 0x13,
        (best_vm1[6] - 0x61) & 0xFF,
        0x00,
    ])
    return key1, key2, {
        "video_blocks_found": video_blocks_found,
        "chunks_seen": chunks_seen,
        "solver": "bigram",
        "solver_score": best_score,
        "samples": sample_rows,
        "fast_enabled": crack_limited,
        "video_crack_bytes_limit": max_video_bytes,
        "video_crack_bytes_used": video_crack_bytes_used,
        "bigram_zero_weight": bigram_zero_weight,
        "bigram_ff_weight": bigram_ff_weight,
        "odd_bigram_zero": odd_bigram_zero,
        "odd_bigram_ff": odd_bigram_ff,
        "odd_bigram_ratio": odd_ratio,
        "beam_size": beam_size,
        "l1_beam_size": l1_beam_size,
        **_vp9_report(vp9_stream_detected, vp9_constraints, vp9_constraint_stats),
    }


def _vp9_report(
    vp9_stream_detected: bool,
    constraints: Vm1Constraints,
    stats: Vp9ConstraintStats,
) -> dict:
    if not vp9_stream_detected:
        return {}

    return {
        "vp9": {
            "constraints": format_vm1_constraints(constraints),
            "constraint_conflicts": stats.conflict_total,
            "constraint_stats": stats.as_report(),
        }
    }


def _make_score_matrices() -> tuple[ScoreMatrix, ScoreMatrix]:
    unigram = [array("I", [0]) * 256 for _ in range(32)]
    bigram = [array("I", [0]) * 65536 for _ in range(31)]
    return unigram, bigram


def _accumulate_score_matrices(
    data,
    start: int,
    size: int,
    unigram: ScoreMatrix,
    bigram: ScoreMatrix,
) -> tuple[int, int, int]:
    num_blocks = size // 32
    current_s = bytearray(32)
    rows = 0
    odd_bigram_zero = 0
    odd_bigram_ff = 0

    for i in range(num_blocks):
        block_start = start + i * 32
        block = data[block_start:block_start + 32]
        for j, value in enumerate(block):
            current_s[j] ^= value

        if i % 2 == 0:
            rows += 1
            for j, value in enumerate(current_s):
                unigram[j][value] += 1
                unigram[j][value ^ 0xFF] += 1

            for j in range(31):
                left = current_s[j]
                right = current_s[j + 1]
                bigram[j][(left << 8) | right] += 1
        else:
            for j in range(31):
                left = current_s[j]
                right = current_s[j + 1]
                if left == 0x00 and right == 0x00:
                    odd_bigram_zero += 1
                elif left == 0xFF and right == 0xFF:
                    odd_bigram_ff += 1

    return rows, odd_bigram_zero, odd_bigram_ff


def _estimate_bigram_weights(
    odd_zero: int,
    odd_ff: int,
) -> tuple[int, int, float | None]:
    total_hits = odd_zero + odd_ff
    if total_hits < BIGRAM_ADAPT_MIN_HITS:
        raw_ratio = (odd_zero / odd_ff) if odd_ff else BIGRAM_RATIO_MAX
        return BIGRAM_LOW_CONF_ZERO_WEIGHT, BIGRAM_LOW_CONF_FF_WEIGHT, raw_ratio

    raw_ratio = (odd_zero / odd_ff) if odd_ff else BIGRAM_RATIO_MAX
    adjusted_ratio = max(BIGRAM_RATIO_MIN, min(BIGRAM_RATIO_MAX, raw_ratio))
    zero_weight = round(BIGRAM_WEIGHT_TOTAL * adjusted_ratio / (1.0 + adjusted_ratio))
    ff_weight = BIGRAM_WEIGHT_TOTAL - zero_weight
    return zero_weight, ff_weight, raw_ratio


def _solve_vm1_bigram(
    unigram: ScoreMatrix,
    bigram: ScoreMatrix,
    beam_size: int,
    l1_beam_size: int,
    bigram_weights: BigramWeights,
    known_vm1: Vm1Constraints | None = None,
) -> tuple[int, list[int] | None]:
    beam = max(1, beam_size)
    l1_beam = max(1, l1_beam_size)
    known = known_vm1 or {}

    level1 = _top(_iter_level1(unigram, bigram, bigram_weights, known), l1_beam)
    level2 = _extend_level0(level1, unigram, bigram, beam, bigram_weights, known)
    level3 = _extend_level3(level2, unigram, bigram, beam, bigram_weights, known)
    level4 = _extend_level4(level3, unigram, bigram, beam, bigram_weights, known)
    level5 = _extend_level6(level4, unigram, bigram, beam, bigram_weights, known)
    level6 = _extend_level5(level5, unigram, bigram, beam, bigram_weights, known)

    if not level6:
        return -1, None

    best_score, best_vm1 = level6[0]
    return best_score, best_vm1


def _iter_level1(
    unigram: ScoreMatrix,
    bigram: ScoreMatrix,
    bigram_weights: BigramWeights,
    known_vm1: Vm1Constraints,
) -> Iterable[Candidate]:
    for v1 in range(256):
        for v2 in range(256):
            v = [0] * 32
            v[1] = v1
            v[2] = v2
            v[8] = (v[2] + v[1]) & 0xFF
            v[10] = v[2] ^ 0xFF
            v[11] = v[1] ^ 0xFF
            v[15] = (v[10] - v[11]) & 0xFF
            v[16] = (v[8] - v[15]) & 0xFF
            v[18] = v[15] ^ 0xFF
            if not _matches_known(v, known_vm1, (1, 2, 8, 10, 11, 15, 16, 18)):
                continue
            score = (
                unigram[1][v[1]]
                + unigram[2][v[2]]
                + unigram[8][v[8]]
                + unigram[10][v[10]]
                + unigram[11][v[11]]
                + unigram[15][v[15]]
                + unigram[16][v[16]]
                + unigram[18][v[18]]
                + (
                    _bg(bigram, 1, v[1], v[2], bigram_weights)
                    + _bg(bigram, 10, v[10], v[11], bigram_weights)
                    + _bg(bigram, 15, v[15], v[16], bigram_weights)
                )
            )
            yield score, v


def _extend_level0(
    candidates: list[Candidate],
    unigram: ScoreMatrix,
    bigram: ScoreMatrix,
    beam: int,
    bigram_weights: BigramWeights,
    known_vm1: Vm1Constraints,
) -> list[Candidate]:
    out: list[Candidate] = []
    for prev_score, prev_v in candidates:
        for v0 in range(256):
            v = prev_v.copy()
            v[0] = v0
            v[7] = v[0] ^ 0xFF
            v[9] = (v[1] - v[7]) & 0xFF
            v[12] = (v[11] + v[9]) & 0xFF
            v[17] = v[16] ^ v[7]
            if not _matches_known(v, known_vm1, (0, 7, 9, 12, 17)):
                continue
            score = prev_score + (
                unigram[0][v[0]]
                + unigram[7][v[7]]
                + unigram[9][v[9]]
                + unigram[12][v[12]]
                + unigram[17][v[17]]
                + (
                    _bg(bigram, 0, v[0], v[1], bigram_weights)
                    + _bg(bigram, 7, v[7], v[8], bigram_weights)
                    + _bg(bigram, 8, v[8], v[9], bigram_weights)
                    + _bg(bigram, 9, v[9], v[10], bigram_weights)
                    + _bg(bigram, 11, v[11], v[12], bigram_weights)
                    + _bg(bigram, 16, v[16], v[17], bigram_weights)
                    + _bg(bigram, 17, v[17], v[18], bigram_weights)
                )
            )
            out.append((score, v))
    return _top(out, beam)


def _extend_level3(
    candidates: list[Candidate],
    unigram: ScoreMatrix,
    bigram: ScoreMatrix,
    beam: int,
    bigram_weights: BigramWeights,
    known_vm1: Vm1Constraints,
) -> list[Candidate]:
    out: list[Candidate] = []
    for prev_score, prev_v in candidates:
        for v3 in range(256):
            v = prev_v.copy()
            v[3] = v3
            v[13] = (v[8] - v[3]) & 0xFF
            v[14] = v[13] ^ 0xFF
            v[19] = v[3] ^ 0x10
            v[23] = (v[19] - v[15]) & 0xFF
            v[25] = (0x21 - v[19]) & 0xFF
            v[28] = (v[23] + 0x44) & 0xFF
            if not _matches_known(v, known_vm1, (3, 13, 14, 19, 23, 25, 28)):
                continue
            score = prev_score + (
                unigram[3][v[3]]
                + unigram[13][v[13]]
                + unigram[14][v[14]]
                + unigram[19][v[19]]
                + unigram[23][v[23]]
                + unigram[25][v[25]]
                + unigram[28][v[28]]
                + (
                    _bg(bigram, 2, v[2], v[3], bigram_weights)
                    + _bg(bigram, 12, v[12], v[13], bigram_weights)
                    + _bg(bigram, 13, v[13], v[14], bigram_weights)
                    + _bg(bigram, 14, v[14], v[15], bigram_weights)
                    + _bg(bigram, 18, v[18], v[19], bigram_weights)
                )
            )
            out.append((score, v))
    return _top(out, beam)


def _extend_level4(
    candidates: list[Candidate],
    unigram: ScoreMatrix,
    bigram: ScoreMatrix,
    beam: int,
    bigram_weights: BigramWeights,
    known_vm1: Vm1Constraints,
) -> list[Candidate]:
    out: list[Candidate] = []
    for prev_score, prev_v in candidates:
        for v4 in range(256):
            v = prev_v.copy()
            v[4] = v4
            v[20] = (v[4] - 0x32) & 0xFF
            v[26] = v[20] ^ v[23]
            v[29] = (v[3] + v[4]) & 0xFF
            v[31] = v[29] ^ v[19]
            if not _matches_known(v, known_vm1, (4, 20, 26, 29, 31)):
                continue
            score = prev_score + (
                unigram[4][v[4]]
                + unigram[20][v[20]]
                + unigram[26][v[26]]
                + unigram[29][v[29]]
                + unigram[31][v[31]]
                + (
                    _bg(bigram, 3, v[3], v[4], bigram_weights)
                    + _bg(bigram, 19, v[19], v[20], bigram_weights)
                    + _bg(bigram, 25, v[25], v[26], bigram_weights)
                    + _bg(bigram, 28, v[28], v[29], bigram_weights)
                )
            )
            out.append((score, v))
    return _top(out, beam)


def _extend_level6(
    candidates: list[Candidate],
    unigram: ScoreMatrix,
    bigram: ScoreMatrix,
    beam: int,
    bigram_weights: BigramWeights,
    known_vm1: Vm1Constraints,
) -> list[Candidate]:
    out: list[Candidate] = []
    for prev_score, prev_v in candidates:
        for v6 in range(256):
            v = prev_v.copy()
            v[6] = v6
            v[22] = v[6] ^ 0xF3
            v[27] = (v[22] + v[22]) & 0xFF
            if not _matches_known(v, known_vm1, (6, 22, 27)):
                continue
            score = prev_score + (
                unigram[6][v[6]]
                + unigram[22][v[22]]
                + unigram[27][v[27]]
                + (
                    _bg(bigram, 6, v[6], v[7], bigram_weights)
                    + _bg(bigram, 26, v[26], v[27], bigram_weights)
                    + _bg(bigram, 27, v[27], v[28], bigram_weights)
                )
            )
            out.append((score, v))
    return _top(out, beam)


def _extend_level5(
    candidates: list[Candidate],
    unigram: ScoreMatrix,
    bigram: ScoreMatrix,
    beam: int,
    bigram_weights: BigramWeights,
    known_vm1: Vm1Constraints,
) -> list[Candidate]:
    out: list[Candidate] = []
    for prev_score, prev_v in candidates:
        for v5 in range(256):
            v = prev_v.copy()
            v[5] = v5
            v[21] = (v[5] + 0xED) & 0xFF
            v[24] = (v[21] + v[7]) & 0xFF
            v[30] = (v[5] - v[22]) & 0xFF
            if not _matches_known(v, known_vm1, (5, 21, 24, 30)):
                continue
            score = prev_score + (
                unigram[5][v[5]]
                + unigram[21][v[21]]
                + unigram[24][v[24]]
                + unigram[30][v[30]]
                + (
                    _bg(bigram, 4, v[4], v[5], bigram_weights)
                    + _bg(bigram, 5, v[5], v[6], bigram_weights)
                    + _bg(bigram, 20, v[20], v[21], bigram_weights)
                    + _bg(bigram, 21, v[21], v[22], bigram_weights)
                    + _bg(bigram, 22, v[22], v[23], bigram_weights)
                    + _bg(bigram, 23, v[23], v[24], bigram_weights)
                    + _bg(bigram, 24, v[24], v[25], bigram_weights)
                    + _bg(bigram, 29, v[29], v[30], bigram_weights)
                    + _bg(bigram, 30, v[30], v[31], bigram_weights)
                )
            )
            out.append((score, v))
    return _top(out, beam)


def _matches_known(
    vm1: list[int],
    known_vm1: Vm1Constraints,
    indices: tuple[int, ...],
) -> bool:
    for index in indices:
        allowed = known_vm1.get(index)
        if allowed is not None and vm1[index] not in allowed:
            return False
    return True


def _bg(
    bigram: ScoreMatrix,
    index: int,
    left: int,
    right: int,
    bigram_weights: BigramWeights,
) -> int:
    bigram_zero_weight, bigram_ff_weight = bigram_weights
    pair_ff = (left << 8) | right
    pair_zero = ((left ^ 0xFF) << 8) | (right ^ 0xFF)
    return (
        bigram_ff_weight * bigram[index][pair_ff]
        + bigram_zero_weight * bigram[index][pair_zero]
    )


def _top(candidates: Iterable[Candidate], beam: int) -> list[Candidate]:
    return heapq.nlargest(beam, candidates, key=lambda x: x[0])
