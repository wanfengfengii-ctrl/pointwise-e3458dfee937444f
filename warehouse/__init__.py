"""Streaming warehouse join / exclusion engine.

Public API:

    Row(row_id, key, payload)
    open_batch(left, right, row_budget, partitions, filter_bytes,
               temp_parent=None) -> Batch

The module deliberately keeps only bounded state: left Rows are spooled to a
per-batch temporary SQLite index on disk, while the right source is streamed
exactly once per scan mode.  Nothing is shared between batches.
"""

from .batch import Batch, open_batch
from .model import Row

__all__ = ["Row", "Batch", "open_batch"]
