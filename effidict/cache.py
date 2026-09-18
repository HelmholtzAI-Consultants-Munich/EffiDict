from __future__ import annotations

from typing import Iterator


class Cache:
    def __init__(self, policy, max_items=None, max_bytes=None, size_estimator=None):
        """Create a cache constrained by independent item and byte budgets.

        ``max_items`` and ``max_bytes`` are independent budgets: either being
        exceeded means entries have to go. ``max_bytes=None`` means item-count
        only. ``size_estimator`` maps a value to its approximate byte size and
        defaults to something like ``len``; it is approximate by design, and a
        ``Cache`` must actually consult it rather than assume ``len``.

        **The cache does not evict on its own.** ``put`` may take the cache over
        budget; ``evict_candidates`` then names what has to be shed, and the
        owner -- ``Store`` -- peeks each candidate, writes it out if it is dirty,
        and calls ``discard``. That split is forced by I5: a dirty victim has to
        reach disk *before* it leaves memory, and a ``Cache`` holds no backend to
        write it to, so a cache that dropped its own victims would make
        write-before-drop impossible to implement. It is also why the surface
        includes ``peek``, ``is_dirty``, ``dirty_items`` and ``discard`` at all.
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
        """Iterate over keys that must be shed to get back inside budget.

        In the policy's victim order, most-evictable first, and empty when both
        budgets are satisfied. Yielding a key does not remove it: the cache is
        unchanged until the caller calls ``discard``, which is what lets the
        caller write a dirty value out first.
        """
        raise NotImplementedError("see issue #1.2")
