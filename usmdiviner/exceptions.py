class UsmDivinerError(Exception):
    """Base exception for this project."""


class UsmFormatError(UsmDivinerError):
    """Raised when a USM file cannot be parsed safely."""


class KeyCrackError(UsmDivinerError):
    """Raised when no usable key can be recovered."""


class ExternalToolError(UsmDivinerError):
    """Raised when an external decoder/muxer fails unexpectedly."""
