"""A fixed-size counting-agnostic Bloom filter over left full keys.

Only false positives are possible, so a negative answer is a proof that no
equal full left key exists and the right key can skip association detail.  The
filter is always exactly ``size_bytes`` bytes and is sized from the multiset of
left full-non-null rows (duplicates are inserted once per occurrence).
"""

from __future__ import annotations

import hashlib
import math

_LN2 = math.log(2.0)


class BloomFilter:
    __slots__ = ("_bits", "_m", "_k")

    def __init__(self, size_bytes: int, item_count: int) -> None:
        if size_bytes < 1:
            raise ValueError("bloom size must be positive")
        self._bits = bytearray(size_bytes)
        self._m = size_bytes * 8
        if item_count > 0:
            ideal = (self._m / item_count) * _LN2
            self._k = max(1, min(64, int(round(ideal))))
        else:
            self._k = 1

    def _positions(self, item: bytes):
        digest = hashlib.blake2b(item, digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:], "big") | 1  # odd step
        m = self._m
        for index in range(self._k):
            yield (h1 + index * h2) % m

    def add(self, item: bytes) -> None:
        for position in self._positions(item):
            self._bits[position >> 3] |= 1 << (position & 7)

    def __contains__(self, item: bytes) -> bool:
        for position in self._positions(item):
            if not (self._bits[position >> 3] >> (position & 7)) & 1:
                return False
        return True

    @property
    def size_bytes(self) -> int:
        return len(self._bits)

    @property
    def hash_count(self) -> int:
        return self._k
