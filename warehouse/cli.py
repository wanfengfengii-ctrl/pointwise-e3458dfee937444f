"""Command line entry point: ``python3 -B -m warehouse.cli {verify,demo}``.

``verify`` runs the acceptance example, the over-budget duplicate-key check and
then the full unittest suite in-process, propagating any failure as a non-zero
exit status.  ``demo`` prints the real business results of the same example.
"""

from __future__ import annotations

import sys
import unittest

from . import Row, open_batch

#: Left rows from the brief: A=(1,2,3) twice, then (2,2,3), then (9,9,9).
LEFT = [
    Row("A1", (1, 2, 3), "left-a"),
    Row("A2", (1, 2, 3), "left-b"),
    Row("A3", (2, 2, 3), "left-c"),
    Row("A4", (9, 9, 9), "left-d"),
]

#: Right keys: A twice, then (2,None,3).
RIGHT = [(1, 2, 3), (1, 2, 3), (2, None, 3)]


def _example_snapshot(budget: int, partitions: int, filter_bytes: int):
    batch = open_batch(LEFT, RIGHT, row_budget=budget, partitions=partitions,
                       filter_bytes=filter_bytes)
    try:
        pairs = [(row.row_id, row.payload, key) for row, key in batch.inner_join()]
        released = [(row.row_id, row.key) for row in batch.exclude()]
        return pairs, released, batch.peak_held
    finally:
        batch.close()


def run_checks() -> None:
    """Assert the acceptance example and the over-budget rekey behaviour."""

    pairs, released, _ = _example_snapshot(4, 2, 64)
    # Two left rows (A1, A2) carry the key A=(1,2,3); the right side sends A
    # twice, so the association is the full 2x2 cartesian product.
    expected_pairs = {
        ("A1", "left-a", (1, 2, 3)),
        ("A2", "left-b", (1, 2, 3)),
    }
    assert {(row_id, payload, key) for row_id, payload, key in pairs} == expected_pairs, pairs
    assert len(pairs) == 4, pairs                    # four association pairs
    assert released == [("A4", (9, 9, 9))], released  # only the last key passes

    # Over-budget duplicate keys: a row budget far smaller than the number of
    # duplicate A keys still yields the full cartesian association while never
    # holding more decoded row keys than the budget at once.
    many_left = [Row(f"A{i}", (1, 2, 3), f"p{i}") for i in range(12)]
    many_right = [(1, 2, 3)] * 9
    batch = open_batch(many_left, many_right, row_budget=4, partitions=1,
                       filter_bytes=64)
    try:
        got = list(batch.inner_join())
        assert len(got) == 12 * 9, len(got)
        assert batch.peak_held <= 4, batch.peak_held
        assert list(batch.exclude()) == []
    finally:
        batch.close()


def cmd_verify(argv) -> int:
    if argv:
        print("verify takes no arguments", file=sys.stderr)
        return 2
    run_checks()
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


def cmd_demo(argv) -> int:
    if argv:
        print("demo takes no arguments", file=sys.stderr)
        return 2
    pairs, released, peak = _example_snapshot(4, 2, 64)
    print(f"association pairs: {len(pairs)}")
    for row_id, payload, key in pairs:
        print(f"  left {row_id} ({payload!r}) <-> right {key}")
    print(f"excluded survivors: {len(released)}")
    for row_id, key in released:
        print(f"  left {row_id} key={key}")
    print(f"peak held decode rows: {peak}")
    return 0


COMMANDS = {"verify": cmd_verify, "demo": cmd_demo}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print("usage: python3 -B -m warehouse.cli {verify|demo}", file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
