"""Batch staging, scan modes and the on-disk left index.

Lifecycle of :func:`open_batch`:

1. Arguments are validated.  Invalid ranges raise :class:`ValueError`.
2. The left source is consumed exactly once, left-to-right.  Every record is
   validated (bad record -> :class:`ValueError`); payloads are appended to a
   fixed number of partition files and row metadata is written into a SQLite
   index that lives in this batch's private temporary directory.  Duplicate
   keys are kept: nothing is deduplicated on the left.
3. The right source is consumed exactly once, left-to-right, and every key is
   validated.  Right keys are spooled to disk; the distinct right keys also
   feed a null-tolerant projection index used by exclusion.  The left Bloom
   filter is sized from the multiset of fully non-null left keys.
4. Sources are handed back to the caller: this module never closes or rewinds
   them, and exceptions raised by a source propagate unchanged.

:meth:`Batch.scan` then streams staged data from disk.  ``"join"`` yields
``(left Row, right key)`` pairs for every fully non-null, Bloom-permitted equal
key; ``"exclude"`` yields the left rows that carry a both-non-null unequal
column against *every* right key, resolving each left row with a small, fixed
number of indexed point lookups.  Each mode may be run at most once, and the
two scans are serial.

Memory is bounded by ``row_budget``: a paused scan never holds more decoded row
keys than the budget (see :attr:`Batch.peak_held`).  Closing the batch, or a
failure while opening it, removes only this batch's staging directory.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from .bloom import BloomFilter
from .codec import ALL_MASKS, assignment_projections, encode_key, key_mask, project_key
from .model import Row, check_key, check_row

MIN_ROW_BUDGET = 4
MAX_ROW_BUDGET = 4096
MIN_PARTITIONS = 1
MAX_PARTITIONS = 32
MIN_FILTER_BYTES = 64
MAX_FILTER_BYTES = 65536

#: Canonical scan mode names and the aliases accepted by :meth:`Batch.scan`.
SCAN_MODES = ("join", "exclude")
_MODE_ALIASES = {
    "join": "join",
    "inner": "join",
    "inner_join": "join",
    "exclude": "exclude",
    "exclusion": "exclude",
}

_SCHEMA = """
CREATE TABLE left_rows (
    rid    INTEGER PRIMARY KEY,
    row_id TEXT    NOT NULL,
    k0     INTEGER,
    k1     INTEGER,
    k2     INTEGER,
    mask   INTEGER NOT NULL,
    keyblob BLOB   NOT NULL,
    part   INTEGER NOT NULL,
    off    INTEGER NOT NULL,
    plen   INTEGER NOT NULL
);
CREATE TABLE left_meta (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE right_keys (
    seq    INTEGER PRIMARY KEY,
    k0     INTEGER,
    k1     INTEGER,
    k2     INTEGER,
    mask   INTEGER NOT NULL,
    keyblob BLOB   NOT NULL
);
CREATE TABLE right_proj (
    mask INTEGER NOT NULL,
    proj BLOB    NOT NULL
);
"""


def _validate_param(name: str, value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int in [{low}, {high}]")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
    return value


def open_batch(
    left: Iterable,
    right: Iterable,
    row_budget: int,
    partitions: int,
    filter_bytes: int,
    temp_parent: Optional[str] = None,
) -> "Batch":
    """Stage a left/right pair into a fresh, private on-disk batch."""

    row_budget = _validate_param("row_budget", row_budget, MIN_ROW_BUDGET, MAX_ROW_BUDGET)
    partitions = _validate_param("partitions", partitions, MIN_PARTITIONS, MAX_PARTITIONS)
    filter_bytes = _validate_param(
        "filter_bytes", filter_bytes, MIN_FILTER_BYTES, MAX_FILTER_BYTES
    )
    if temp_parent is not None and not isinstance(temp_parent, str):
        raise ValueError("temp_parent must be a str or None")

    temp_dir = tempfile.mkdtemp(prefix="warehouse-batch-", dir=temp_parent)
    batch: Optional["Batch"] = None
    try:
        batch = Batch(temp_dir, row_budget, partitions, filter_bytes)
        batch._load_left(left)      # left Rows first, one validating pass
        batch._load_right(right)    # then right keys, one validating pass
        batch._finalize()
        return batch
    except BaseException:
        # Clean only this batch's staging directory, then let the error through.
        if batch is not None:
            batch.close()
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


class Batch:
    """A staged batch.  Obtain one through :func:`open_batch`."""

    def __init__(self, temp_dir: str, row_budget: int, partitions: int,
                 filter_bytes: int) -> None:
        self.temp_dir = temp_dir
        self.row_budget = row_budget
        self.partitions = partitions
        self.filter_bytes = filter_bytes

        self._db_path = os.path.join(temp_dir, "index.sqlite3")
        self._conn = sqlite3.connect(self._db_path, isolation_level=None)
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.execute("PRAGMA journal_mode=MEMORY")
        self._conn.execute("PRAGMA cache_size=-8192")
        self._conn.execute("PRAGMA temp_store=FILE")
        self._conn.execute("BEGIN")

        self._writers: dict = {}
        self._readers: dict = {}
        self._offsets = {part: 0 for part in range(partitions)}

        self._left_count = 0
        self._right_count = 0
        self._full_left_count = 0
        self._bloom: Optional[BloomFilter] = None

        self._closed = False
        self._used = set()
        self._active: Optional[str] = None
        self._held = 0
        self.peak_held = 0
        self.point_queries = 0

    # ------------------------------------------------------------------ staging

    def _partition_for(self, keyblob: bytes) -> int:
        # FNV-1a over the key blob; stable and cheap, no dependency on hash().
        value = 0xCBF29CE484222325
        for byte in keyblob:
            value ^= byte
            value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return value % self.partitions

    def _writer(self, part: int):
        handle = self._writers.get(part)
        if handle is None:
            path = os.path.join(self.temp_dir, f"payload-{part:03d}.bin")
            handle = open(path, "wb")
            self._writers[part] = handle
        return handle

    def _load_left(self, left: Iterable) -> None:
        pending: List[tuple] = []
        sql = (
            "INSERT INTO left_rows"
            " (rid,row_id,k0,k1,k2,mask,keyblob,part,off,plen)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)"
        )
        for record in left:                      # caller's iterator, one pass
            row = check_row(record)              # ValueError on a bad record
            keyblob = encode_key(row.key)
            mask = key_mask(row.key)
            payload = row.payload.encode("utf-8")
            part = self._partition_for(keyblob)
            off = self._offsets[part]
            self._writer(part).write(payload)
            self._offsets[part] = off + len(payload)
            self._left_count += 1
            if mask == 7:
                self._full_left_count += 1
            pending.append(
                (self._left_count, row.row_id, row.key[0], row.key[1], row.key[2],
                 mask, keyblob, part, off, len(payload))
            )
            if len(pending) >= self.row_budget:
                self._conn.executemany(sql, pending)
                pending.clear()
        if pending:
            self._conn.executemany(sql, pending)
        for handle in self._writers.values():
            handle.flush()
        self._conn.execute(
            "INSERT INTO left_meta(name,value) VALUES ('left_count',?)",
            (self._left_count,),
        )
        self._conn.execute(
            "CREATE INDEX left_keyblob_idx ON left_rows(keyblob)"
        )
        self._conn.execute("COMMIT")
        self._conn.execute("BEGIN")

    def _load_right(self, right: Iterable) -> None:
        pending: List[tuple] = []
        sql = (
            "INSERT INTO right_keys(seq,k0,k1,k2,mask,keyblob) VALUES (?,?,?,?,?,?)"
        )
        for record in right:                     # caller's iterator, one pass
            key = check_key(record)              # ValueError on a bad record
            keyblob = encode_key(key)
            mask = key_mask(key)
            self._right_count += 1
            pending.append(
                (self._right_count, key[0], key[1], key[2], mask, keyblob)
            )
            if len(pending) >= self.row_budget:
                self._conn.executemany(sql, pending)
                pending.clear()
        if pending:
            self._conn.executemany(sql, pending)
        self._conn.execute(
            "INSERT INTO left_meta(name,value) VALUES ('right_count',?)",
            (self._right_count,),
        )
        self._conn.execute("COMMIT")
        self._conn.execute("BEGIN")

    def _finalize(self) -> None:
        # Bloom filter over the multiset of fully non-null left full keys.
        self._bloom = BloomFilter(self.filter_bytes, self._full_left_count)
        cursor = self._conn.execute(
            "SELECT keyblob FROM left_rows WHERE mask=7"
        )
        while True:
            rows = cursor.fetchmany(self.row_budget)
            if not rows:
                break
            for (keyblob,) in rows:
                self._bloom.add(keyblob)
        cursor.close()

        # Null-tolerant projection index over distinct right keys.  Distinct is
        # enough here because exclusion only asks whether a compatible key
        # exists, and SQLite DISTINCT treats NULLs as equal.
        distinct = self._conn.execute(
            "SELECT DISTINCT k0,k1,k2 FROM right_keys"
        )
        proj_sql = "INSERT INTO right_proj(mask,proj) VALUES (?,?)"
        pend: List[tuple] = []
        while True:
            rows = distinct.fetchmany(self.row_budget)
            if not rows:
                break
            for k0, k1, k2 in rows:
                key = (k0, k1, k2)
                for mask in ALL_MASKS:
                    pend.append((mask, project_key(key, mask)))
            if len(pend) >= self.row_budget * len(ALL_MASKS):
                self._conn.executemany(proj_sql, pend)
                pend.clear()
        distinct.close()
        if pend:
            self._conn.executemany(proj_sql, pend)
        self._conn.execute("CREATE INDEX right_proj_idx ON right_proj(mask,proj)")
        self._conn.execute("COMMIT")
        for handle in self._writers.values():
            handle.close()
        self._writers.clear()

    # ------------------------------------------------------------------ reading

    def _reader(self, part: int):
        handle = self._readers.get(part)
        if handle is None:
            path = os.path.join(self.temp_dir, f"payload-{part:03d}.bin")
            handle = open(path, "rb")
            self._readers[part] = handle
        return handle

    def _read_payload(self, part: int, off: int, length: int) -> str:
        handle = self._reader(part)
        handle.seek(off)
        return handle.read(length).decode("utf-8")

    def _note_held(self, count: int) -> None:
        self._held = count
        if count > self.peak_held:
            self.peak_held = count

    @property
    def _chunk_size(self) -> int:
        # Reserve one slot of the budget for the row currently being processed
        # (and yielded to the caller) so that a paused scan holds at most
        # ``row_budget`` decoded row keys in total.
        return max(1, self.row_budget - 1)

    @staticmethod
    def _close_cursor(cursor) -> None:
        # An abandoned generator may be finalized after the batch closed.
        try:
            cursor.close()
        except sqlite3.Error:
            pass

    def _right_chunks(self) -> Iterator[List[tuple]]:
        cursor = self._conn.execute(
            "SELECT k0,k1,k2,mask,keyblob FROM right_keys ORDER BY seq"
        )
        try:
            while True:
                rows = cursor.fetchmany(self._chunk_size)
                if not rows:
                    break
                self._note_held(len(rows) + 1)
                yield rows
        finally:
            self._close_cursor(cursor)
            self._held = 0

    def _left_chunks(self) -> Iterator[List[tuple]]:
        cursor = self._conn.execute(
            "SELECT row_id,k0,k1,k2,part,off,plen FROM left_rows ORDER BY rid"
        )
        try:
            while True:
                rows = cursor.fetchmany(self._chunk_size)
                if not rows:
                    break
                self._note_held(len(rows) + 1)
                yield rows
        finally:
            self._close_cursor(cursor)
            self._held = 0

    def _left_matches(self, keyblob: bytes) -> Iterator[Row]:
        cursor = self._conn.execute(
            "SELECT row_id,k0,k1,k2,part,off,plen"
            " FROM left_rows WHERE keyblob=? ORDER BY rid",
            (keyblob,),
        )
        try:
            for row_id, k0, k1, k2, part, off, plen in cursor:
                yield Row(row_id, (k0, k1, k2), self._read_payload(part, off, plen))
        finally:
            self._close_cursor(cursor)

    def _has_compatible_right(self, key: Sequence[Optional[int]]) -> bool:
        """True when some right key has no both-non-null unequal column.

        A compatible right key may be NULL at any of the left row's non-null
        columns or equal to it there, so at most ``2 ** popcount(mask) <= 8``
        projection blobs are probed with one indexed lookup.
        """
        mask = key_mask(key)
        projections = list(assignment_projections(key, mask))
        placeholders = ",".join("?" * len(projections))
        self.point_queries += 1
        row = self._conn.execute(
            "SELECT 1 FROM right_proj WHERE mask=? AND proj IN ("
            + placeholders
            + ") LIMIT 1",
            (mask, *projections),
        ).fetchone()
        return row is not None

    # -------------------------------------------------------------------- scan

    def scan(self, mode: str) -> Iterator:
        """Stream results for ``mode`` (``"join"`` or ``"exclude"``)."""

        canonical = _MODE_ALIASES.get(mode) if isinstance(mode, str) else None
        if canonical is None:
            raise ValueError(
                f"unknown scan mode {mode!r}; expected one of {SCAN_MODES}"
            )
        self._check_open()
        if self._active is not None:
            raise RuntimeError(
                f"scan({self._active!r}) is still in progress; scans are serial"
            )
        if canonical in self._used:
            raise RuntimeError(f"scan mode {canonical!r} has already been consumed")

        self._used.add(canonical)
        inner = self._join_stream() if canonical == "join" else self._exclude_stream()
        return self._serialized(inner, canonical)

    def _serialized(self, inner: Iterator, mode: str) -> Iterator:
        # The active marker is only taken on the first iteration, so a scan
        # object that is created but never started cannot pin the batch.
        if self._active is not None:
            raise RuntimeError(
                f"scan({self._active!r}) is still in progress; scans are serial"
            )
        self._active = mode
        try:
            yield from inner
        finally:
            if self._active == mode:
                self._active = None
            self._held = 0

    def inner_join(self) -> Iterator[Tuple[Row, tuple]]:
        return self.scan("join")

    def exclude(self) -> Iterator[Row]:
        return self.scan("exclude")

    def _join_stream(self) -> Iterator[Tuple[Row, tuple]]:
        bloom = self._bloom
        for chunk in self._right_chunks():
            for k0, k1, k2, mask, keyblob in chunk:
                # Right keys with a NULL column, or fully non-null keys the
                # Bloom filter rules out, skip association detail entirely.
                if mask != 7:
                    continue
                if keyblob not in bloom:
                    continue
                right_key = (k0, k1, k2)
                for left_row in self._left_matches(keyblob):
                    yield left_row, right_key

    def _exclude_stream(self) -> Iterator[Row]:
        for chunk in self._left_chunks():
            for row_id, k0, k1, k2, part, off, plen in chunk:
                key = (k0, k1, k2)
                if self._has_compatible_right(key):
                    continue
                yield Row(row_id, key, self._read_payload(part, off, plen))

    # ------------------------------------------------------------------- close

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("batch is closed")

    @property
    def left_count(self) -> int:
        return self._left_count

    @property
    def right_count(self) -> int:
        return self._right_count

    def close(self) -> None:
        """Close this batch and remove only its own staging directory."""

        if getattr(self, "_closed", True):
            return
        self._closed = True
        for handle in list(getattr(self, "_readers", {}).values()):
            try:
                handle.close()
            except OSError:
                pass
        for handle in list(getattr(self, "_writers", {}).values()):
            try:
                handle.close()
            except OSError:
                pass
        readers = getattr(self, "_readers", None)
        writers = getattr(self, "_writers", None)
        if readers is not None:
            readers.clear()
        if writers is not None:
            writers.clear()
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        shutil.rmtree(getattr(self, "temp_dir", ""), ignore_errors=True)

    def __enter__(self) -> "Batch":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
