from __future__ import annotations

import mmap
import struct
from collections import Counter
from pathlib import Path

from .constants import SIG_SFV, VIDEO_CRACK_START, VIDEO_MASK_START
from .exceptions import UsmFormatError


def crack_keys_from_usm(
    usm_path: str | Path,
    max_video_bytes: int | None = None,
) -> tuple[bytes | None, bytes | None, dict]:
    path = Path(usm_path)
    if path.stat().st_size == 0:
        raise UsmFormatError("empty USM file")

    with path.open("rb") as fp, mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as data:
        return _crack_from_buffer(data, max_video_bytes)


def _crack_from_buffer(
    data,
    max_video_bytes: int | None,
) -> tuple[bytes | None, bytes | None, dict]:
    offset = 0
    data_len = len(data)
    counters: list[Counter[int]] = [Counter() for _ in range(32)]
    video_blocks_found = 0
    chunks_seen = 0
    video_crack_bytes_used = 0
    crack_limited = max_video_bytes is not None

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
            if payload_size - VIDEO_MASK_START >= 0x200:
                encrypted_start = payload_start + VIDEO_CRACK_START
                encrypted_size = payload_size - VIDEO_CRACK_START
                if max_video_bytes is not None:
                    remaining = max_video_bytes - video_crack_bytes_used
                    if remaining < 32:
                        break
                    encrypted_size = min(encrypted_size, remaining)
                if encrypted_size >= 32:
                    video_blocks_found += 1
                    video_crack_bytes_used += encrypted_size
                    _accumulate_solver_counters(data, encrypted_start, encrypted_size, counters)
                    if max_video_bytes is not None and video_crack_bytes_used >= max_video_bytes:
                        break

        offset = next_offset

    sample_count = _counter_sample_count(counters)
    if video_blocks_found == 0 or sample_count == 0:
        return None, None, {
            "video_blocks_found": video_blocks_found,
            "chunks_seen": chunks_seen,
            "reason": "no encrypted/long-enough @SFV video blocks found",
            "fast_enabled": crack_limited,
            "video_crack_bytes_limit": max_video_bytes,
            "video_crack_bytes_used": video_crack_bytes_used,
        }

    joint_scores = _build_joint_scores(counters)
    best_score, best_vm1 = _solve_vm1(joint_scores)

    if best_vm1 is None:
        return None, None, {
            "video_blocks_found": video_blocks_found,
            "reason": "solver produced no candidate",
            "fast_enabled": crack_limited,
            "video_crack_bytes_limit": max_video_bytes,
            "video_crack_bytes_used": video_crack_bytes_used,
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
        "solver_score": best_score,
        "samples": sample_count,
        "fast_enabled": crack_limited,
        "video_crack_bytes_limit": max_video_bytes,
        "video_crack_bytes_used": video_crack_bytes_used,
    }


def _accumulate_solver_counters(data, start: int, size: int, counters: list[Counter[int]]) -> None:
    num_blocks = size // 32
    current_s = bytearray(32)
    for i in range(num_blocks):
        block_start = start + i * 32
        block = data[block_start:block_start + 32]
        for j, value in enumerate(block):
            current_s[j] ^= value
        if i % 2 == 0:
            for j, value in enumerate(current_s):
                counters[j][value] += 1


def _counter_sample_count(counters: list[Counter[int]]) -> int:
    return sum(sum(counter.values()) for counter in counters)


def _build_joint_scores(counters: list[Counter[int]]) -> list[list[int]]:
    joint_scores: list[list[int]] = []
    for i in range(32):
        scores = [0] * 256
        counter = counters[i]
        for val in range(256):
            scores[val] = counter[val] + counter[val ^ 0xFF]
        joint_scores.append(scores)
    return joint_scores


def _solve_vm1(joint_scores: list[list[int]]) -> tuple[int, list[int] | None]:
    best_score = -1
    best_vm1: list[int] | None = None

    v1v2_scores = []
    for v1 in range(256):
        for v2 in range(256):
            v8 = (v2 + v1) & 0xFF
            v10 = v2 ^ 0xFF
            v11 = v1 ^ 0xFF
            v15 = (v10 - v11) & 0xFF
            v16 = (v8 - v15) & 0xFF
            v18 = v15 ^ 0xFF
            score = (
                joint_scores[1][v1] + joint_scores[2][v2] + joint_scores[8][v8]
                + joint_scores[10][v10] + joint_scores[11][v11] + joint_scores[15][v15]
                + joint_scores[16][v16] + joint_scores[18][v18]
            )
            v1v2_scores.append((score, v1, v2, v8, v10, v11, v15, v16, v18))

    v6_candidates = _v6_candidates(joint_scores)

    top_v1v2 = sorted(v1v2_scores, key=lambda x: x[0], reverse=True)[:16]
    for s_12, v1, v2, v8, v10, v11, v15, v16, v18 in top_v1v2:
        v3_branches = []
        for v3_candidate in _v3_candidates(joint_scores, v8, v15):
            _, v3, _, _, v19, v23, _, _ = v3_candidate
            v3_branches.append((v3_candidate, _v4_candidates(joint_scores, v3, v19, v23)))

        for s_0, v0, v7, v9, v12, v17 in _v0_candidates(joint_scores, v1, v11, v16):
            v5_cache = {
                v22: _v5_candidates(joint_scores, v7, v22)
                for _, _, v22, _ in v6_candidates
            }
            for v3_candidate, v4_candidates in v3_branches:
                s_3, v3, v13, v14, v19, v23, v25, v28 = v3_candidate
                for s_4, v4, v20, v26, v29, v31 in v4_candidates:
                    for s_6, v6, v22, v27 in v6_candidates:
                        for s_5, v5, v21, v24, v30 in v5_cache[v22]:
                            total_score = s_12 + s_0 + s_3 + s_4 + s_6 + s_5
                            if total_score > best_score:
                                best_score = total_score
                                best_vm1 = [
                                    v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                    v10, v11, v12, v13, v14, v15, v16, v17,
                                    v18, v19, v20, v21, v22, v23, v24, v25,
                                    v26, v27, v28, v29, v30, v31,
                                ]
    return best_score, best_vm1


def _v0_candidates(
    joint_scores: list[list[int]],
    v1: int,
    v11: int,
    v16: int,
) -> list[tuple[int, ...]]:
    out = []
    for v0 in range(256):
        v7 = v0 ^ 0xFF
        v9 = (v1 - v7) & 0xFF
        v12 = (v11 + v9) & 0xFF
        v17 = v16 ^ v7
        score = (
            joint_scores[0][v0]
            + joint_scores[7][v7]
            + joint_scores[9][v9]
            + joint_scores[12][v12]
            + joint_scores[17][v17]
        )
        out.append((score, v0, v7, v9, v12, v17))
    return sorted(out, key=lambda x: x[0], reverse=True)[:5]


def _v3_candidates(joint_scores: list[list[int]], v8: int, v15: int) -> list[tuple[int, ...]]:
    out = []
    for v3 in range(256):
        v13 = (v8 - v3) & 0xFF
        v14 = v13 ^ 0xFF
        v19 = v3 ^ 0x10
        v23 = (v19 - v15) & 0xFF
        v25 = (0x21 - v19) & 0xFF
        v28 = (v23 + 0x44) & 0xFF
        score = (
            joint_scores[3][v3] + joint_scores[13][v13] + joint_scores[14][v14]
            + joint_scores[19][v19] + joint_scores[23][v23] + joint_scores[25][v25]
            + joint_scores[28][v28]
        )
        out.append((score, v3, v13, v14, v19, v23, v25, v28))
    return sorted(out, key=lambda x: x[0], reverse=True)[:5]


def _v4_candidates(
    joint_scores: list[list[int]],
    v3: int,
    v19: int,
    v23: int,
) -> list[tuple[int, ...]]:
    out = []
    for v4 in range(256):
        v20 = (v4 - 0x32) & 0xFF
        v26 = v20 ^ v23
        v29 = (v3 + v4) & 0xFF
        v31 = v29 ^ v19
        score = (
            joint_scores[4][v4]
            + joint_scores[20][v20]
            + joint_scores[26][v26]
            + joint_scores[29][v29]
            + joint_scores[31][v31]
        )
        out.append((score, v4, v20, v26, v29, v31))
    return sorted(out, key=lambda x: x[0], reverse=True)[:5]


def _v6_candidates(joint_scores: list[list[int]]) -> list[tuple[int, ...]]:
    out = []
    for v6 in range(256):
        v22 = v6 ^ 0xF3
        v27 = (v22 + v22) & 0xFF
        score = joint_scores[6][v6] + joint_scores[22][v22] + joint_scores[27][v27]
        out.append((score, v6, v22, v27))
    return sorted(out, key=lambda x: x[0], reverse=True)[:8]


def _v5_candidates(joint_scores: list[list[int]], v7: int, v22: int) -> list[tuple[int, ...]]:
    out = []
    for v5 in range(256):
        v21 = (v5 + 0xED) & 0xFF
        v24 = (v21 + v7) & 0xFF
        v30 = (v5 - v22) & 0xFF
        score = (
            joint_scores[5][v5]
            + joint_scores[21][v21]
            + joint_scores[24][v24]
            + joint_scores[30][v30]
        )
        out.append((score, v5, v21, v24, v30))
    return sorted(out, key=lambda x: x[0], reverse=True)[:5]
