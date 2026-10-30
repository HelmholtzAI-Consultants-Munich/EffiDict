from __future__ import annotations

from itertools import islice
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

        ``max_bytes`` and ``size_estimator`` are recorded but not yet consulted:
        the byte budget needs ``nbytes``, which is issue #1.4.
        """
        self.policy = policy
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.size_estimator = size_estimator
        self._values = {}

    def __iter__(self) -> Iterator:
        """Iterate the cached keys, so ``set(cache)`` is the resident key set."""
        return iter(self._values)

    def has(self, key) -> bool:
        """Return whether ``key`` is present in the cache."""
        return key in self._values

    def get(self, key):
        """Return ``key`` and record access, or raise ``KeyError`` if absent."""
        value = self._values[key]
        self.policy.on_access(key)
        return value

    def peek(self, key):
        """Return ``key`` without recording access, or raise ``KeyError``."""
        return self._values[key]

    def put(self, key, value, dirty: bool = True) -> None:
        """Store ``value`` under ``key`` and record dirty state.

        A rewrite is reported to the policy as an access, not an insertion.
        ``on_insert`` restarts a key's bookkeeping -- frequency back to one --
        so routing an update through it would discard the usage history of a key
        that never left the cache. That is the defect
        ``test_mfu_replacement_keeps_frequency_across_a_value_update`` pins
        against the strategies this class replaces.

        ``dirty`` is recorded by issue #1.3, which adds the clean/dirty tag.
        Until then ``Store`` writes every victim out, not only the dirty ones.
        """
        if key in self._values:
            self.policy.on_access(key)
        else:
            self.policy.on_insert(key)
        self._values[key] = value

    def discard(self, key) -> None:
        """Remove ``key`` if present without raising when absent."""
        if key in self._values:
            del self._values[key]
            self.policy.on_remove(key)

    def clear(self) -> None:
        """Remove all cached entries and policy state."""
        self._values.clear()
        self.policy.clear()

    def __len__(self) -> int:
        """Return the number of cached items."""
        return len(self._values)

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

        Built on ``EvictionPolicy.victims()`` rather than repeated ``victim()``
        calls -- ``victim()`` only advances when a key is removed, so it cannot
        enumerate without mutating.

        Exactly the surplus, never more: the promise is "the keys that must be
        shed to get back inside budget", so shedding further to a low-water mark
        would be a different promise. Returned over a snapshot, so the caller
        may discard what it is handed while walking it.

        Only the item budget is consulted. The byte budget needs ``nbytes``,
        which is issue #1.4.
        """
        if self.max_items is None:
            return iter(())

        surplus = len(self._values) - self.max_items
        if surplus <= 0:
            return iter(())

        if surplus == 1:
            # One write past the bound sheds one key, which is the steady state
            # and nearly every call. ``victims()`` has to build the whole
            # ordering before it yields anything, so asking it for one key costs
            # the size of the cache and a fill becomes quadratic; ``victim()``
            # reads the ordering's endpoint instead. The two name the same key,
            # which ``test_victims_enumerates_in_eviction_order`` pins.
            return iter((self.policy.victim(),))

        # Taken eagerly, not lazily: ``islice`` straight back would interleave
        # the policy's walk with the caller's ``discard`` calls, and a policy
        # that enumerates its live ordering rather than a snapshot would raise
        # mid-eviction.
        candidates = list(islice(self.policy.victims(), surplus))
        return iter(candidates)
