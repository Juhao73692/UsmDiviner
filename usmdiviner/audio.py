from __future__ import annotations

import tempfile
from pathlib import Path

from .exceptions import ExternalToolError
from .formats import classify_audio
from .masks import unmask_audio_payload
from .models import AudioDecision
from .tools import decode_with_vgmstream, write_hcakey_file


def probe_with_vgmstream(
    vgmstream: str | None,
    ext: str,
    content: bytes,
    key1: bytes,
    key2: bytes,
) -> bool:
    if not vgmstream:
        return False
    with tempfile.TemporaryDirectory() as td:
        probe_path = Path(td) / f"probe.{ext}"
        probe_path.write_bytes(content)
        if ext.lower() == "hca":
            write_hcakey_file(probe_path, key1, key2)
        out = Path(td) / "probe.wav"
        try:
            ok, _ = decode_with_vgmstream(vgmstream, probe_path, out, timeout=30)
        except ExternalToolError:
            return False
        return ok


def decide_audio_for_channel(
    channel: int,
    payloads: list[bytes],
    audio_mask: bytes,
    key1: bytes,
    key2: bytes,
    vgmstream: str | None,
    adx_audio_mask: bool,
) -> tuple[AudioDecision, bytes]:
    raw_joined = b"".join(payloads)
    masked_joined = b"".join(unmask_audio_payload(p, audio_mask) for p in payloads)

    raw_cls = classify_audio(raw_joined)
    masked_cls = classify_audio(masked_joined)

    if raw_cls["format"] == "hca" or masked_cls["format"] == "hca":
        raw_hca = raw_cls.get("hca") or {}
        masked_hca = masked_cls.get("hca") or {}
        raw_valid = int(raw_hca.get("valid_crc_blocks", 0))
        masked_valid = int(masked_hca.get("valid_crc_blocks", 0))
        raw_checked = int(raw_hca.get("checked_crc_blocks", 0))
        masked_checked = int(masked_hca.get("checked_crc_blocks", 0))
        if masked_valid > raw_valid:
            return AudioDecision(
                channel,
                "hca",
                True,
                "high" if masked_valid else "medium",
                (
                    "HCA CRC improves after AudioMask "
                    f"({raw_valid}/{raw_checked} -> {masked_valid}/{masked_checked})"
                ),
                masked_cls.get("hca"),
            ), masked_joined
        if raw_valid >= masked_valid and raw_cls["format"] == "hca":
            return AudioDecision(
                channel,
                "hca",
                False,
                "high" if raw_valid else "medium",
                (
                    "HCA CRC is not improved by AudioMask "
                    f"({raw_valid}/{raw_checked} vs {masked_valid}/{masked_checked})"
                ),
                raw_cls.get("hca"),
            ), raw_joined

    if raw_cls["format"] == "adx" or masked_cls["format"] == "adx":
        if adx_audio_mask:
            return (
                AudioDecision(
                    channel,
                    "adx",
                    True,
                    "medium",
                    "ADX detected; AudioMask enabled by default",
                    None,
                ),
                masked_joined,
            )
        return (
            AudioDecision(
                channel,
                "adx",
                False,
                "medium",
                "ADX detected; AudioMask disabled",
                None,
            ),
            raw_joined,
        )

    if vgmstream:
        for ext in ("hca", "adx"):
            raw_ok = probe_with_vgmstream(vgmstream, ext, raw_joined, key1, key2)
            masked_ok = probe_with_vgmstream(vgmstream, ext, masked_joined, key1, key2)
            if masked_ok and not raw_ok:
                return (
                    AudioDecision(
                        channel,
                        ext,
                        True,
                        "medium",
                        f"vgmstream probe suggests {ext.upper()} after AudioMask",
                        None,
                    ),
                    masked_joined,
                )
            if raw_ok and not masked_ok:
                return (
                    AudioDecision(
                        channel,
                        ext,
                        False,
                        "medium",
                        f"vgmstream probe suggests {ext.upper()} without AudioMask",
                        None,
                    ),
                    raw_joined,
                )

    return (
        AudioDecision(
            channel,
            "unknown",
            False,
            "low",
            "unable to identify audio format or AudioMask state",
            None,
        ),
        raw_joined,
    )
