"""Row / key model and strict value validation."""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

#: A key is a fixed three-tuple of int64-or-None column values.
KEY_ARITY = 3

#: Maximum UTF-8 byte lengths for the identifier and the payload.
MAX_ID_BYTES = 128
MAX_PAYLOAD_BYTES = 4096

INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1


class Row(NamedTuple):
    """A sparse row: an id, a three-column key and a payload string."""

    row_id: str
    key: Tuple[Optional[int], Optional[int], Optional[int]]
    payload: str


def _check_text(value: object, limit: int, what: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a str, got {type(value).__name__}")
    length = len(value.encode("utf-8"))
    if length > limit:
        raise ValueError(f"{what} is {length} UTF-8 bytes, limit is {limit}")
    return value


def check_key(key: object) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Validate a key triple and return it as a plain tuple.

    Every column must be ``None`` or a non-bool integer inside int64 range.
    """

    if isinstance(key, (str, bytes)) or not isinstance(key, Sequence):
        raise ValueError("key must be a 3-tuple")
    if len(key) != KEY_ARITY:
        raise ValueError(f"key must have {KEY_ARITY} columns, got {len(key)}")
    columns = []
    for index, value in enumerate(key):
        if value is None:
            columns.append(None)
            continue
        # bool is an int subclass but is explicitly forbidden.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"key column {index} must be None or a non-bool int, "
                f"got {type(value).__name__}"
            )
        if value < INT64_MIN or value > INT64_MAX:
            raise ValueError(f"key column {index} is outside int64 range")
        columns.append(int(value))
    return tuple(columns)  # type: ignore[return-value]


def check_row(row: object) -> Row:
    """Validate one left record and return it as a :class:`Row`.

    A malformed record raises :class:`ValueError`.
    """

    if isinstance(row, Row):
        row_id, key, payload = row.row_id, row.key, row.payload
    else:
        try:
            row_id, key, payload = row  # type: ignore[misc]
        except (TypeError, ValueError) as exc:
            raise ValueError("record must be a (row_id, key, payload) triple") from exc

    _check_text(row_id, MAX_ID_BYTES, "row_id")
    if not row_id:
        raise ValueError("row_id must be non-empty")
    columns = check_key(key)
    _check_text(payload, MAX_PAYLOAD_BYTES, "payload")
    return Row(row_id, columns, payload)


def key_has_null(key: Sequence[Optional[int]]) -> bool:
    """True when any key column is ``None``."""

    return any(value is None for value in key)
