from __future__ import annotations

from typing import Iterator


class Cache:
    def __init__(self, policy, max_items=None, max_bytes=None, size_estimator=None):
        """Create a cache constrained by independent item and byte budgets.

        ``max_items`` and ``max_bytes`` are independent budgets; eviction
        happens while either budget is exceeded. ``max_bytes=None`` means
        item-count only. ``size_estimator`` is approximate by design.
        """
        raise NotImplementedError("see issue #1.2")

    def has(self, key) -> bool:
        """Return whether ``key`` is present in the cache."""
        raise NotImplementedError("see issue #1.2")

    def get(self, key):
        """Return ``key`` and record access, or raise ``KeyError`` if absent."""
        raise NotImplementedError("see issue #1.2")

    def peek(self, key):
        """Return ``key`` without recording access, or raise ``KeyError``."""
        raise NotImplementedError("see issue #1.2")

    def put(self, key, value, dirty: bool = True) -> None:
        """Store ``value`` under ``key`` and record dirty state."""
        raise NotImplementedError("see issue #1.2")

    def discard(self, key) -> None:
        """Remove ``key`` if present without raising when absent."""
        raise NotImplementedError("see issue #1.2")

    def clear(self) -> None:
        """Remove all cached entries and policy state."""
        raise NotImplementedError("see issue #1.2")

    def __len__(self) -> int:
        """Return the number of cached items."""
        raise NotImplementedError("see issue #1.2")

    def nbytes(self) -> int:
        """Return the approximate cached byte size."""
        raise NotImplementedError("see issue #1.4")

    def is_dirty(self, key) -> bool:
        """Return whether ``key`` has unwritten cache changes."""
        raise NotImplementedError("see issue #1.3")

    def mark_clean(self, key) -> None:
        """Mark ``key`` as having no unwritten cache changes."""
        raise NotImplementedError("see issue #1.3")

    def dirty_items(self) -> Iterator:
        """Iterate over cached items with unwritten changes."""
        raise NotImplementedError("see issue #1.3")

    def evict_candidates(self) -> Iterator:
        """Iterate over keys to shed until cache budgets are met."""
        raise NotImplementedError("see issue #1.2")
