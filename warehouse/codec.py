"""Binary key / projection codecs.

A key is a 3-tuple whose columns are ``None`` or non-bool int64 values.  To
make equality including ``None`` a plain indexed BLOB comparison we encode a
key into a fixed 25-byte blob: one mask byte (bit ``c`` set when column ``c``
is non-null) followed by three big-endian signed 64-bit slots (zero for a null
column).

Projections are used by the exclusion index.  A projection onto a set of
columns drops the other columns entirely, which gives the "wildcard" behaviour
needed for null-tolerant non-overlap: a right key projects onto the left row's
non-null columns, and a null right value is kept distinct from an integer so
that "both non-null and unequal" can be checked exactly.
"""

from __future__ import annotations

import struct
from typing import Iterable, Optional, Sequence, Tuple

KEY_ARITY = 3
_SLOT = struct.Struct(">q")
_KEY_SIZE = 1 + KEY_ARITY * 8
_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1

ALL_MASKS: Tuple[int, ...] = tuple(range(1 << KEY_ARITY))


def key_mask(key: Sequence[Optional[int]]) -> int:
    """Bit ``c`` is set when column ``c`` is non-null."""

    mask = 0
    for column, value in enumerate(key):
        if value is not None:
            mask |= 1 << column
    return mask


def encode_key(key: Sequence[Optional[int]]) -> bytes:
    """Fixed-width 25-byte encoding of a validated key."""

    mask = key_mask(key)
    out = bytearray(_KEY_SIZE)
    out[0] = mask
    offset = 1
    for value in key:
        if value is None:
            out[offset:offset + 8] = b"\x00" * 8
        else:
            out[offset:offset + 8] = _SLOT.pack(value)
        offset += 8
    return bytes(out)


def decode_key(blob: bytes, mask: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Inverse of :func:`encode_key` for callers that only kept the blob."""

    values = []
    for column in range(KEY_ARITY):
        if mask & (1 << column):
            values.append(_SLOT.unpack(blob[1 + column * 8: 9 + column * 8])[0])
        else:
            values.append(None)
    return tuple(values)  # type: ignore[return-value]


def project_key(key: Sequence[Optional[int]], mask: int) -> bytes:
    """Encode the projection of ``key`` onto the columns selected by ``mask``.

    Layout: the mask byte, then for every selected column one flag byte
    (``0`` = null, ``1`` = integer) followed by eight big-endian bytes.
    """

    out = bytearray()
    out.append(mask)
    for column in range(KEY_ARITY):
        if mask & (1 << column):
            value = key[column]
            if value is None:
                out.append(0)
                out.extend(b"\x00" * 8)
            else:
                out.append(1)
                out.extend(_SLOT.pack(value))
    return bytes(out)


def assignment_projections(
    key: Sequence[Optional[int]], mask: int
) -> Iterable[bytes]:
    """Yield the projection blobs a compatible right key could have.

    For every non-null column of the left ``key`` a compatible right key may
    either be null or carry exactly the same integer, so there are at most
    ``2 ** popcount(mask)`` projections (eight for three columns).  Null
    columns of the left key are free and are dropped by the projection.
    """

    columns = [column for column in range(KEY_ARITY) if mask & (1 << column)]
    total = 1 << len(columns)
    for choice in range(total):
        synthetic = [None, None, None]
        for index, column in enumerate(columns):
            # bit index == 1 -> keep the integer, else force null
            if (choice >> index) & 1:
                synthetic[column] = key[column]
            else:
                synthetic[column] = None
        yield project_key(synthetic, mask)
