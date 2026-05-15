from __future__ import annotations

import dataclasses

from .models import HcaInfo


def _tag(buf: bytes, off: int, header_masked: bool) -> bytes:
    t = buf[off:off + 4]
    if len(t) < 4:
        return b""
    return bytes((b & 0x7F) for b in t) if header_masked else t


def _be16(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off:off + 2], "big")


def _be32(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off:off + 4], "big")


def parse_hca_info(buf: bytes) -> HcaInfo | None:
    if len(buf) < 0x20:
        return None
    magic_le = int.from_bytes(buf[:4], "little")
    if magic_le == 0x00414348:
        header_masked = False
    elif (magic_le & 0x7F7F7F7F) == 0x00414348:
        header_masked = True
    else:
        return None

    version = _be16(buf, 4)
    data_offset = _be16(buf, 6)
    if data_offset < 8 or data_offset > len(buf):
        return None

    off = 8
    if _tag(buf, off, header_masked) != b"fmt\x00" or off + 16 > len(buf):
        return None
    channel_count = buf[off + 4]
    sample_rate = (buf[off + 5] << 16) | (buf[off + 6] << 8) | buf[off + 7]
    block_count = _be32(buf, off + 8)
    off += 16

    comp_tag = _tag(buf, off, header_masked)
    if comp_tag == b"comp" and off + 16 <= len(buf):
        block_size = _be16(buf, off + 4)
        off += 16
    elif comp_tag == b"dec\x00" and off + 12 <= len(buf):
        block_size = _be16(buf, off + 4)
        off += 12
    else:
        return None

    if _tag(buf, off, header_masked) == b"vbr\x00" and off + 8 <= len(buf):
        off += 8
    if _tag(buf, off, header_masked) == b"ath\x00" and off + 6 <= len(buf):
        off += 6
    if _tag(buf, off, header_masked) == b"loop" and off + 16 <= len(buf):
        off += 16

    ciph_type = 0
    if _tag(buf, off, header_masked) == b"ciph" and off + 6 <= len(buf):
        ciph_type = _be16(buf, off + 4)

    if block_size <= 0 or block_size > 0xFFFF or channel_count <= 0 or channel_count > 16:
        return None
    return HcaInfo(
        header_masked,
        version,
        data_offset,
        channel_count,
        sample_rate,
        block_count,
        block_size,
        ciph_type,
    )


def _make_crc16_table() -> list[int]:
    table = []
    poly = 0x8005
    for i in range(256):
        r = i << 8
        for _ in range(8):
            r = ((r << 1) ^ poly) & 0xFFFF if r & 0x8000 else (r << 1) & 0xFFFF
        table.append(r)
    return table


_CRC16_TABLE = _make_crc16_table()


def hca_crc16(data: bytes) -> int:
    s = 0
    for b in data:
        s = ((s << 8) ^ _CRC16_TABLE[((s >> 8) ^ b) & 0xFF]) & 0xFFFF
    return s


def score_hca_crc(buf: bytes, info: HcaInfo, max_blocks: int = 12) -> tuple[int, int]:
    valid = 0
    checked = 0
    off = info.data_offset
    for _ in range(min(max_blocks, info.block_count if info.block_count else max_blocks)):
        if off + info.block_size > len(buf):
            break
        block = buf[off:off + info.block_size]
        checked += 1
        if hca_crc16(block) == 0:
            valid += 1
        off += info.block_size
    return valid, checked


def is_adx(buf: bytes) -> bool:
    if len(buf) < 8 or buf[0:2] != b"\x80\x00":
        return False
    header_end = int.from_bytes(buf[2:4], "big") + 4
    return 4 <= header_end < min(len(buf), 0x4000)


def classify_audio(buf: bytes) -> dict:
    hca = parse_hca_info(buf)
    if hca:
        valid, checked = score_hca_crc(buf, hca)
        hca.valid_crc_blocks = valid
        hca.checked_crc_blocks = checked
        return {
            "format": "hca",
            "hca": dataclasses.asdict(hca),
            "score": 1000 + valid * 10 + checked,
        }
    if is_adx(buf):
        return {"format": "adx", "score": 500}
    return {"format": "unknown", "score": 0}


def _find_mpeg_start_codes(buf: bytes, limit: int = 8192) -> list[tuple[int, int]]:
    codes: list[tuple[int, int]] = []
    end = min(len(buf), limit)
    i = 0
    while i + 3 < end:
        if buf[i:i + 3] == b"\x00\x00\x01":
            codes.append((i, buf[i + 3]))
            i += 4
            continue
        i += 1
    return codes


def _valid_mpeg_sequence_header(buf: bytes, pos: int) -> bool:
    if pos + 8 > len(buf) or buf[pos:pos + 4] != b"\x00\x00\x01\xB3":
        return False
    b0, b1, b2, b3 = buf[pos + 4:pos + 8]
    width = (b0 << 4) | (b1 >> 4)
    height = ((b1 & 0x0F) << 8) | b2
    frame_rate_code = b3 & 0x0F
    return width > 0 and height > 0 and 1 <= frame_rate_code <= 8


def _detect_mpeg_video(buf: bytes) -> dict | None:
    codes = _find_mpeg_start_codes(buf)
    sequence_positions = [pos for pos, code in codes if code == 0xB3]
    if not sequence_positions:
        return None
    if not any(_valid_mpeg_sequence_header(buf, pos) for pos in sequence_positions):
        return None

    for pos, code in codes:
        if code == 0xB5 and pos + 4 < len(buf) and (buf[pos + 4] >> 4) == 1:
            return {
                "format": "mpeg2",
                "extension": "m2v",
                "codec": "mpeg2video",
                "magic": buf[:16].hex(),
            }
    return {"format": "mpeg1", "extension": "m1v", "codec": "mpeg1video", "magic": buf[:16].hex()}


def _find_annexb_start_codes(buf: bytes, limit: int = 4096) -> list[int]:
    positions: list[int] = []
    end = min(len(buf), limit)
    i = 0
    while i + 3 < end:
        if buf[i:i + 3] == b"\x00\x00\x01":
            positions.append(i + 3)
            i += 3
            continue
        if i + 4 < end and buf[i:i + 4] == b"\x00\x00\x00\x01":
            positions.append(i + 4)
            i += 4
            continue
        i += 1
    return positions


def _is_h264_annexb(buf: bytes) -> bool:
    nal_types: list[int] = []
    for pos in _find_annexb_start_codes(buf):
        if pos >= len(buf):
            continue
        nal_type = buf[pos] & 0x1F
        if 1 <= nal_type <= 12:
            nal_types.append(nal_type)
    if not nal_types:
        return False
    return (
        7 in nal_types
        or 8 in nal_types
        or 9 in nal_types
        or any(1 <= t <= 5 for t in nal_types)
    )


def detect_video_stream(buf: bytes) -> dict:
    if buf.startswith(b"DKIF"):
        fourcc = ""
        if len(buf) >= 12:
            try:
                fourcc = buf[8:12].decode("ascii", errors="replace").strip("\x00")
            except UnicodeDecodeError:
                fourcc = ""
        return {
            "format": "ivf",
            "extension": "ivf",
            "codec": fourcc or "ivf",
            "magic": buf[:16].hex(),
        }

    mpeg = _detect_mpeg_video(buf)
    if mpeg:
        return mpeg

    if _is_h264_annexb(buf):
        return {"format": "h264", "extension": "264", "codec": "h264", "magic": buf[:16].hex()}

    return {"format": "unknown", "extension": "bin", "codec": "unknown", "magic": buf[:16].hex()}
