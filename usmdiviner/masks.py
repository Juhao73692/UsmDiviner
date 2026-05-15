from __future__ import annotations

from .constants import AUDIO_MASK_START, VIDEO_MASK_START


def make_masks(key1: bytes, key2: bytes) -> tuple[bytes, bytes, bytes]:
    if len(key1) != 4 or len(key2) != 4:
        raise ValueError("key1/key2 must be 4 bytes each")

    v = [0] * 0x20
    v[0x00] = key1[0]
    v[0x01] = key1[1]
    v[0x02] = key1[2]
    v[0x03] = (key1[3] - 0x34) & 0xFF
    v[0x04] = (key2[0] + 0xF9) & 0xFF
    v[0x05] = key2[1] ^ 0x13
    v[0x06] = (key2[2] + 0x61) & 0xFF
    v[0x07] = v[0x00] ^ 0xFF
    v[0x08] = (v[0x02] + v[0x01]) & 0xFF
    v[0x09] = (v[0x01] - v[0x07]) & 0xFF
    v[0x0A] = v[0x02] ^ 0xFF
    v[0x0B] = v[0x01] ^ 0xFF
    v[0x0C] = (v[0x0B] + v[0x09]) & 0xFF
    v[0x0D] = (v[0x08] - v[0x03]) & 0xFF
    v[0x0E] = v[0x0D] ^ 0xFF
    v[0x0F] = (v[0x0A] - v[0x0B]) & 0xFF
    v[0x10] = (v[0x08] - v[0x0F]) & 0xFF
    v[0x11] = v[0x10] ^ v[0x07]
    v[0x12] = v[0x0F] ^ 0xFF
    v[0x13] = v[0x03] ^ 0x10
    v[0x14] = (v[0x04] - 0x32) & 0xFF
    v[0x15] = (v[0x05] + 0xED) & 0xFF
    v[0x16] = v[0x06] ^ 0xF3
    v[0x17] = (v[0x13] - v[0x0F]) & 0xFF
    v[0x18] = (v[0x15] + v[0x07]) & 0xFF
    v[0x19] = (0x21 - v[0x13]) & 0xFF
    v[0x1A] = v[0x14] ^ v[0x17]
    v[0x1B] = (v[0x16] + v[0x16]) & 0xFF
    v[0x1C] = (v[0x17] + 0x44) & 0xFF
    v[0x1D] = (v[0x03] + v[0x04]) & 0xFF
    v[0x1E] = (v[0x05] - v[0x16]) & 0xFF
    v[0x1F] = v[0x1D] ^ v[0x13]

    video_mask1 = bytes(v)
    video_mask2 = bytes(x ^ 0xFF for x in video_mask1)
    table2 = b"URUC"
    audio_mask = bytes(
        table2[(i >> 1) & 3] if (i & 1) else (video_mask1[i] ^ 0xFF)
        for i in range(0x20)
    )
    return video_mask1, video_mask2, audio_mask


def _xor_int_block(data: bytearray, start: int, size: int, mask: bytes) -> None:
    if size <= 0:
        return
    value = int.from_bytes(data[start:start + size], "little") ^ int.from_bytes(
        mask[:size],
        "little",
    )
    data[start:start + size] = value.to_bytes(size, "little")


def unmask_video_payload(payload: bytes, video_mask1: bytes, video_mask2: bytes) -> bytes:
    data = bytearray(payload)
    size = len(data) - VIDEO_MASK_START
    if size < 0x200:
        return bytes(data)

    vm1_int = int.from_bytes(video_mask1, "little")
    vm2_int = int.from_bytes(video_mask2, "little")
    mask_int = vm2_int

    first_start = VIDEO_MASK_START + 0x100
    first_size = size - 0x100
    full_blocks = first_size // 0x20
    tail_size = first_size & 0x1F

    pos = first_start
    for _ in range(full_blocks):
        block_int = int.from_bytes(data[pos:pos + 0x20], "little")
        plain_int = block_int ^ mask_int
        data[pos:pos + 0x20] = plain_int.to_bytes(0x20, "little")
        mask_int = plain_int ^ vm2_int
        pos += 0x20

    if tail_size:
        mask_tail = mask_int.to_bytes(0x20, "little")
        for i in range(tail_size):
            idx = i & 0x1F
            data[pos + i] ^= mask_tail[idx]

    mask_int = vm1_int
    for block in range(0x100 // 0x20):
        pos = VIDEO_MASK_START + block * 0x20
        pos2 = pos + 0x100
        next_mask = mask_int ^ int.from_bytes(data[pos2:pos2 + 0x20], "little")
        plain_int = int.from_bytes(data[pos:pos + 0x20], "little") ^ next_mask
        data[pos:pos + 0x20] = plain_int.to_bytes(0x20, "little")
        mask_int = next_mask

    return bytes(data)


def unmask_audio_payload(payload: bytes, audio_mask: bytes) -> bytes:
    data = bytearray(payload)
    if len(data) <= AUDIO_MASK_START:
        return bytes(data)

    mask_block = audio_mask * (64 * 1024 // len(audio_mask))
    start = AUDIO_MASK_START
    end = len(data)
    while start < end:
        size = min(64 * 1024, end - start)
        _xor_int_block(data, start, size, mask_block)
        start += size
    return bytes(data)
