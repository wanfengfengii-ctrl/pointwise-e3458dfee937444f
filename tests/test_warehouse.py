"""Behavioural tests for the warehouse batch engine."""

import os
import shutil
import tempfile
import unittest

from warehouse import Row, open_batch
from warehouse.model import check_row, INT64_MAX, INT64_MIN


def rows(*triples):
    return [Row(rid, key, payload) for rid, key, payload in triples]


class ExampleTest(unittest.TestCase):
    """The acceptance example from the brief."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wh-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _batch(self, left, right, **kwargs):
        return open_batch(left, right, temp_parent=self.tmp, **kwargs)

    def test_example_join_and_exclude(self):
        left = rows(
            ("L1", (1, 2, 3), "a"),
            ("L2", (1, 2, 3), "b"),
            ("L3", (2, 2, 3), "c"),
            ("L4", (9, 9, 9), "d"),
        )
        right = [(1, 2, 3), (1, 2, 3), (2, None, 3)]
        batch = self._batch(left, right, row_budget=4, partitions=2,
                            filter_bytes=64)
        pairs = list(batch.inner_join())
        self.assertEqual(4, len(pairs))
        for left_row, right_key in pairs:
            self.assertEqual((1, 2, 3), left_row.key)
            self.assertEqual((1, 2, 3), right_key)
        self.assertEqual({("L1", "a"), ("L2", "b")},
                         {(r.row_id, r.payload) for r, _ in pairs})

        survivors = list(batch.exclude())
        self.assertEqual(1, len(survivors))
        self.assertEqual((9, 9, 9), survivors[0].key)
        self.assertEqual("L4", survivors[0].row_id)
        batch.close()

    def test_scans_are_serial_and_single_use(self):
        batch = self._batch(rows(("L1", (1, 2, 3), "a")), [(1, 2, 3)],
                            row_budget=4, partitions=1, filter_bytes=64)
        list(batch.scan("join"))
        with self.assertRaises(RuntimeError):
            list(batch.scan("join"))
        list(batch.scan("exclude"))
        with self.assertRaises(RuntimeError):
            list(batch.scan("exclude"))
        batch.close()

    def test_inner_scan_must_finish_before_other(self):
        batch = self._batch(rows(("L1", (1, 2, 3), "a")), [(1, 2, 3)],
                            row_budget=4, partitions=1, filter_bytes=64)
        stream = batch.scan("join")
        next(iter(stream))
        with self.assertRaises(RuntimeError):
            batch.scan("exclude")
        batch.close()


class ScanSemanticsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wh-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, left, right, **kwargs):
        kwargs.setdefault("row_budget", 16)
        kwargs.setdefault("partitions", 3)
        kwargs.setdefault("filter_bytes", 128)
        return open_batch(left, right, temp_parent=self.tmp, **kwargs)

    def test_join_requires_both_keys_fully_non_null(self):
        left = rows(("L1", (1, None, 3), "x"), ("L2", (4, 5, 6), "y"))
        right = [(1, None, 3), (4, 5, 6), (4, 5, None)]
        batch = self.build(left, right)
        pairs = list(batch.inner_join())
        self.assertEqual([(("L2", "y"), (4, 5, 6))],
                         [((r.row_id, r.payload), k) for r, k in pairs])
        batch.close()

    def test_bloom_negative_right_key_still_releases_left(self):
        # The right key is fully non-null and absent from the left, so the
        # Bloom filter reports a negative for it.  It must still count for
        # exclusion: every column differs, so the left row is released even
        # though the association scan saw nothing.
        left = rows(("L1", (7, 8, 9), "x"))
        right = [(1, 2, 3)]
        batch = self.build(left, right)
        self.assertEqual([], list(batch.inner_join()))
        survivors = list(batch.exclude())
        self.assertEqual(["L1"], [r.row_id for r in survivors])
        batch.close()

    def test_null_bearing_right_key_blocks_exclusion(self):
        left = rows(("L1", (5, 6, 7), "x"), ("L2", (5, 6, 7), "y"),
                    ("L3", (5, 6, 0), "z"))
        right = [(5, None, 7)]
        batch = self.build(left, right)
        # L1, L2 have no both-non-null unequal column against (5,None,7);
        # L3 differs at the last column.
        survivors = sorted(r.row_id for r in batch.exclude())
        self.assertEqual(["L3"], survivors)
        batch.close()

    def test_left_null_columns_are_free(self):
        left = rows(("L1", (None, 5, None), "x"))
        right = [(99, 5, 99), (99, None, 99)]
        batch = self.build(left, right)
        # First right key agrees at column 1; second is null there.  No
        # compatible right key lacks a both-non-null unequal column because
        # only column 1 is constrained and both agree/null there.
        self.assertEqual([], list(batch.exclude()))
        batch.close()

    def test_exclusion_is_wildcard_over_left_nulls(self):
        left = rows(("L1", (None, 5, None), "x"))
        right = [(99, 4, 99)]
        batch = self.build(left, right)
        survivors = list(batch.exclude())
        self.assertEqual(1, len(survivors))
        self.assertEqual("L1", survivors[0].row_id)
        batch.close()

    def test_duplicate_pairs_are_cartesian(self):
        left = rows(("L", (1, 1, 1), "p"), ("L", (1, 1, 1), "q"))
        right = [(1, 1, 1), (1, 1, 1), (1, 1, 1)]
        batch = self.build(left, right)
        self.assertEqual(6, len(list(batch.inner_join())))
        batch.close()

    def test_empty_left_still_validates_right(self):
        batch = self.build([], [(1, 2, 3)], filter_bytes=64)
        self.assertEqual(0, batch.left_count)
        self.assertEqual(1, batch.right_count)
        self.assertEqual([], list(batch.inner_join()))
        self.assertEqual([], list(batch.exclude()))
        batch.close()

    def test_empty_right_releases_every_left_row(self):
        left = rows(("L1", (1, 2, 3), "x"), ("L2", (None, 2, 3), "y"))
        batch = self.build(left, [])
        self.assertEqual([], list(batch.inner_join()))
        self.assertEqual({"L1", "L2"}, {r.row_id for r in batch.exclude()})
        batch.close()

    def test_empty_left_and_right(self):
        batch = self.build([], [])
        self.assertEqual([], list(batch.inner_join()))
        self.assertEqual([], list(batch.exclude()))
        batch.close()

    def test_negative_and_extreme_int64_values(self):
        left = rows(("L1", (INT64_MIN, 0, INT64_MAX), "x"))
        right = [(INT64_MIN, 0, INT64_MAX)]
        batch = self.build(left, right)
        pairs = list(batch.inner_join())
        self.assertEqual(1, len(pairs))
        self.assertEqual(0, len(list(batch.exclude())))  # equal -> not excluded
        batch.close()


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wh-test-")
        self.children_before = set(os.listdir(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def assertBadRecord(self, left, right):
        with self.assertRaises(ValueError):
            open_batch(left, right, row_budget=4, partitions=1,
                       filter_bytes=64, temp_parent=self.tmp)
        # A failure cleans only this batch's staging directory.
        self.assertEqual(self.children_before, set(os.listdir(self.tmp)))

    def test_short_key(self):
        self.assertBadRecord(rows(("L", (1, 2), "x")), [])

    def test_long_key(self):
        self.assertBadRecord(rows(("L", (1, 2, 3, 4), "x")), [])

    def test_bool_key(self):
        self.assertBadRecord(rows(("L", (True, 2, 3), "x")), [])

    def test_string_key(self):
        self.assertBadRecord(rows(("L", ("1", 2, 3), "x")), [])

    def test_float_key(self):
        self.assertBadRecord(rows(("L", (1.0, 2, 3), "x")), [])

    def test_out_of_range_int(self):
        self.assertBadRecord(rows(("L", (INT64_MAX + 1, 2, 3), "x")), [])

    def test_empty_id(self):
        self.assertBadRecord(rows(("", (1, 2, 3), "x")), [])

    def test_non_string_id(self):
        self.assertBadRecord(rows((7, (1, 2, 3), "x")), [])

    def test_id_byte_limit(self):
        # 43 three-byte characters == 129 UTF-8 bytes.
        self.assertBadRecord(rows(("中" * 43, (1, 2, 3), "x")), [])

    def test_payload_byte_limit(self):
        self.assertBadRecord(rows(("L", (1, 2, 3), "x" * 4097)), [])

    def test_bad_right_key(self):
        self.assertBadRecord(rows(("L", (1, 2, 3), "x")), [(1, 2)])

    def test_bad_right_key_after_good(self):
        self.assertBadRecord(rows(("L", (1, 2, 3), "x")),
                             [(1, 2, 3), (1, True, 3)])

    def test_unicode_payload_at_limit_is_accepted(self):
        # 1366 three-byte characters == 4098 bytes is over; 1365 == 4095 is ok.
        payload = "中" * 1365
        batch = open_batch(rows(("L", (1, 2, 3), payload)), [(1, 2, 3)],
                           row_budget=4, partitions=1, filter_bytes=64,
                           temp_parent=self.tmp)
        pairs = list(batch.inner_join())
        self.assertEqual(payload, pairs[0][0].payload)
        batch.close()


class ArgumentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wh-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_row_budget_bounds(self):
        for bad in (3, 4097, 0, -1):
            with self.assertRaises(ValueError):
                open_batch([], [], row_budget=bad, partitions=1,
                           filter_bytes=64, temp_parent=self.tmp)
        for good in (4, 4096):
            batch = open_batch([], [], row_budget=good, partitions=1,
                               filter_bytes=64, temp_parent=self.tmp)
            batch.close()

    def test_partition_bounds(self):
        for bad in (0, 33):
            with self.assertRaises(ValueError):
                open_batch([], [], row_budget=4, partitions=bad,
                           filter_bytes=64, temp_parent=self.tmp)
        for good in (1, 32):
            batch = open_batch([], [], row_budget=4, partitions=good,
                               filter_bytes=64, temp_parent=self.tmp)
            batch.close()

    def test_filter_bounds(self):
        for bad in (63, 65537):
            with self.assertRaises(ValueError):
                open_batch([], [], row_budget=4, partitions=1,
                           filter_bytes=bad, temp_parent=self.tmp)
        for good in (64, 65536):
            batch = open_batch([], [], row_budget=4, partitions=1,
                               filter_bytes=good, temp_parent=self.tmp)
            self.assertEqual(good, batch._bloom.size_bytes)
            batch.close()

    def test_bool_arguments_rejected(self):
        with self.assertRaises(ValueError):
            open_batch([], [], row_budget=True, partitions=1,
                       filter_bytes=64, temp_parent=self.tmp)

    def test_unknown_scan_mode(self):
        batch = open_batch([], [], row_budget=4, partitions=1,
                           filter_bytes=64, temp_parent=self.tmp)
        with self.assertRaises(ValueError):
            batch.scan("outer")
        batch.close()


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wh-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_close_removes_only_own_staging(self):
        sibling = tempfile.mkdtemp(prefix="wh-sibling-", dir=self.tmp)
        batch = open_batch(rows(("L", (1, 2, 3), "x")), [(1, 2, 3)],
                           row_budget=4, partitions=2, filter_bytes=64,
                           temp_parent=self.tmp)
        staging = batch.temp_dir
        self.assertTrue(os.path.isdir(staging))
        batch.close()
        self.assertFalse(os.path.exists(staging))
        self.assertTrue(os.path.isdir(sibling))

    def test_source_exception_propagates_and_cleans(self):
        class Boom(Exception):
            pass

        def left_source():
            yield ("L1", (1, 2, 3), "x")
            raise Boom("left failed")

        with self.assertRaises(Boom):
            open_batch(left_source(), [], row_budget=4, partitions=1,
                       filter_bytes=64, temp_parent=self.tmp)
        self.assertEqual([], os.listdir(self.tmp))

    def test_right_source_exception_propagates_and_cleans(self):
        class Boom(Exception):
            pass

        def right_source():
            yield (1, 2, 3)
            raise Boom("right failed")

        with self.assertRaises(Boom):
            open_batch(rows(("L1", (1, 2, 3), "x")), right_source(),
                       row_budget=4, partitions=1, filter_bytes=64,
                       temp_parent=self.tmp)
        self.assertEqual([], os.listdir(self.tmp))

    def test_paused_scan_respects_row_budget(self):
        left = rows(*[(f"L{i}", (i, i, i), "p") for i in range(200)])
        right = [(i, i, i) for i in range(200)]
        batch = open_batch(left, right, row_budget=7, partitions=4,
                           filter_bytes=256, temp_parent=self.tmp)
        stream = batch.scan("join")
        next(stream)
        self.assertLessEqual(batch.peak_held, 7)
        for _ in stream:
            self.assertLessEqual(batch.peak_held, 7)
        batch.close()

    def test_point_queries_are_bounded(self):
        left = rows(*[(f"L{i}", (i, 0, 0), "p") for i in range(50)])
        right = [(0, 0, 0)]
        batch = open_batch(left, right, row_budget=8, partitions=2,
                           filter_bytes=64, temp_parent=self.tmp)
        list(batch.exclude())
        # Exactly one point lookup per left row: no scan of right keys.
        self.assertEqual(50, batch.point_queries)
        batch.close()


class DifferentialTest(unittest.TestCase):
    """Cross-check streaming results against a brute-force reference."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wh-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _reference_join(left, right):
        out = []
        for lrow in left:
            if any(v is None for v in lrow.key):
                continue
            for rkey in right:
                if any(v is None for v in rkey):
                    continue
                if tuple(lrow.key) == tuple(rkey):
                    out.append((lrow.row_id, lrow.payload, tuple(rkey)))
        return sorted(out)

    @staticmethod
    def _reference_exclude(left, right):
        kept = []
        for lrow in left:
            ok = True
            for rkey in right:
                disagree = any(
                    lv is not None and rv is not None and lv != rv
                    for lv, rv in zip(lrow.key, rkey)
                )
                if not disagree:
                    ok = False
                    break
            if ok:
                kept.append(lrow.row_id)
        return sorted(kept)

    def test_randomized_against_reference(self):
        import random

        rng = random.Random(20261121)
        pool = [None, -2, -1, 0, 1, 2]
        for trial in range(60):
            n_left = rng.randint(0, 25)
            n_right = rng.randint(0, 25)
            left = rows(*[
                (f"L{i}", tuple(rng.choice(pool) for _ in range(3)), f"p{i}")
                for i in range(n_left)
            ])
            key_pool = [tuple(rng.choice(pool) for _ in range(3))
                        for _ in range(rng.randint(0, 8))]
            right = [rng.choice(key_pool) for _ in range(n_right)] if key_pool else []
            budget = rng.choice([4, 5, 8, 16])
            parts = rng.choice([1, 2, 3, 32])
            fbytes = rng.choice([64, 128, 512, 65536])
            batch = open_batch(left, right, row_budget=budget, partitions=parts,
                               filter_bytes=fbytes, temp_parent=self.tmp)
            try:
                got_join = sorted(
                    (r.row_id, r.payload, tuple(k)) for r, k in batch.inner_join()
                )
                got_exclude = sorted(r.row_id for r in batch.exclude())
            finally:
                batch.close()
            self.assertEqual(self._reference_join(left, right), got_join,
                             f"join mismatch trial {trial}")
            self.assertEqual(self._reference_exclude(left, right), got_exclude,
                             f"exclude mismatch trial {trial}")
        self.assertEqual([], [n for n in os.listdir(self.tmp)])


class PartitioningTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wh-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_payloads_survive_partition_round_trip(self):
        left = rows(*[(f"L{i}", (i, i * 2, i * 3), f"payload-{i}")
                      for i in range(1, 33)])
        batch = open_batch(left, [], row_budget=5, partitions=5,
                           filter_bytes=128, temp_parent=self.tmp)
        survivors = {r.row_id: r.payload for r in batch.exclude()}
        self.assertEqual({f"L{i}": f"payload-{i}" for i in range(1, 33)},
                         survivors)
        part_files = [n for n in os.listdir(batch.temp_dir)
                      if n.startswith("payload-")]
        self.assertTrue(part_files)
        batch.close()


class RowModelTest(unittest.TestCase):
    def test_check_row_accepts_row_and_tuple(self):
        self.assertEqual(("a", (1, None, 3), "p"),
                         tuple(check_row(Row("a", (1, None, 3), "p"))))
        self.assertEqual(("a", (1, None, 3), "p"),
                         tuple(check_row(("a", [1, None, 3], "p"))))

    def test_check_row_rejects_wrong_arity(self):
        with self.assertRaises(ValueError):
            check_row(("a", (1, 2, 3)))


if __name__ == "__main__":
    unittest.main()
