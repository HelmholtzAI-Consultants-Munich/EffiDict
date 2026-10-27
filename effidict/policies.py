from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Iterator, List


class EvictionPolicy(ABC):
    """Decides which key to evict next.

    A policy stores ordering and frequency metadata only; values live in
    ``Cache``. A policy that can reach a backend is a bug.
    """

    @abstractmethod
    def on_insert(self, key: str) -> None:
        """Record that ``key`` was inserted."""
        raise NotImplementedError("see issue #1.1")

    @abstractmethod
    def on_access(self, key: str) -> None:
        """Record that ``key`` was accessed."""
        raise NotImplementedError("see issue #1.1")

    @abstractmethod
    def on_remove(self, key: str) -> None:
        """Forget any bookkeeping for ``key``."""
        raise NotImplementedError("see issue #1.1")

    @abstractmethod
    def victim(self) -> str:
        """Return the next key to evict, or raise ``KeyError`` if empty.

        Equivalent to the first item of :meth:`victims`.
        """
        raise NotImplementedError("see issue #1.1")

    @abstractmethod
    def victims(self) -> Iterator:
        """Iterate every tracked key in eviction order, most-evictable first.

        Records nothing: walking it is not an access and not a removal, and the
        policy is unchanged afterwards.

        ``victim()`` alone cannot support ``Cache.evict_candidates``, which
        promises several keys in order *without* mutating the cache. Asking
        ``victim()`` twice returns the same key, because the only way to advance
        it is ``on_remove`` -- and removing before the caller has written a dirty
        value out is exactly what I5 forbids. Without this method a conforming
        ``Cache`` would have to keep its own copy of the ordering, which is the
        duplication the policy exists to prevent.
        """
        raise NotImplementedError("see issue #1.1")

    @abstractmethod
    def clear(self) -> None:
        """Remove all policy bookkeeping."""
        raise NotImplementedError("see issue #1.1")


class _BookkeepingPolicy(EvictionPolicy):
    """Shared bookkeeping for the seven shipped policies.

    Each subclass answers one question -- in what order would you give these
    keys up? -- by overriding :meth:`_eviction_order` and :meth:`_next_victim`.
    Everything else is identical across the seven, which is what the old
    ``*Replacement`` classes duplicated.

    Three containers, because between them the policies need insertion order,
    last-use order and use counts. Each one is a few empty objects for the
    policies that do not consult it, which is cheaper than seven classes each
    maintaining its own subset.
    """

    def __init__(self) -> None:
        self._inserted: OrderedDict = OrderedDict()
        self._used: OrderedDict = OrderedDict()
        self._uses: dict = {}

    @abstractmethod
    def _eviction_order(self) -> List[str]:
        """The tracked keys, most-evictable first."""

    @abstractmethod
    def _next_victim(self) -> str:
        """Only the most evictable key, with at least one key tracked.

        Separate from :meth:`_eviction_order` because shedding one key is the
        common case, and four of the seven answer it by reading one end of an
        ordering they already maintain. Routing it through the full order would
        make a single eviction cost the size of the cache, and a drain
        quadratic. ``test_victims_enumerates_in_eviction_order`` holds the two
        methods to the same answer, so they cannot drift apart.
        """

    def on_insert(self, key: str) -> None:
        self._inserted[key] = None
        self._used[key] = None
        self._used.move_to_end(key)
        self._uses[key] = 1

    def on_access(self, key: str) -> None:
        # ``move_to_end`` raises ``KeyError`` for an untracked key before
        # anything is mutated, so a stray access cannot leave half-updated
        # bookkeeping behind.
        self._used.move_to_end(key)
        self._uses[key] += 1

    def on_remove(self, key: str) -> None:
        # All three, every time. Dropping a key from the ordering while leaving
        # its count behind is the LFU/MFU defect this split exists to remove:
        # the phantom count can later name a key the cache no longer holds.
        self._inserted.pop(key, None)
        self._used.pop(key, None)
        self._uses.pop(key, None)

    def victim(self) -> str:
        if not self._inserted:
            raise KeyError("no keys are being tracked")
        return self._next_victim()

    def victims(self) -> Iterator:
        # Over a snapshot rather than the live ordering: the caller enumerating
        # victims is on its way to removing some of them, and iterating an
        # ``OrderedDict`` that is being mutated raises ``RuntimeError``.
        return iter(self._eviction_order())

    def clear(self) -> None:
        self._inserted.clear()
        self._used.clear()
        self._uses.clear()


class RandomPolicy(_BookkeepingPolicy):
    """Gives up an arbitrary tracked key.

    Keeps one shuffled ordering of the tracked keys instead of deciding per
    call. A new key goes in at a uniformly chosen position and a removed key
    leaves the rest where they are, which keeps the ordering a uniform
    permutation of whatever is tracked right now: the head is equally likely to
    be any of them, and it stays the same key until that key is removed.

    A draw stored per key, smallest one evicted, satisfies both of those
    properties read individually and is still not random replacement. Eviction
    then selects on the draws themselves -- the keys that survive are the ones
    that drew high -- so the survivors' draws are no longer uniform and an
    incoming key with a fresh draw is almost always the smallest. Measured on a
    cache of ten: that version evicted the incoming key again immediately on
    ~950 of 1000 insertions, where uniform replacement evicts about 91.
    ``test_random_stays_uniform_as_keys_are_replaced`` pins it.

    Inserting into a list is linear in the number of tracked keys, where the
    other six are constant. That is the cost of an ordering that can be read
    without being rebuilt; LFU and MFU pay the same linearity per peek.
    """

    def __init__(self) -> None:
        super().__init__()
        self._shuffled: list = []

    def on_insert(self, key: str) -> None:
        super().on_insert(key)
        # Already tracked: it keeps the position it has, matching the base,
        # where re-inserting a key leaves it where it is in insertion order.
        if key not in self._shuffled:
            self._shuffled.insert(random.randint(0, len(self._shuffled)), key)

    def on_remove(self, key: str) -> None:
        super().on_remove(key)
        if key in self._shuffled:
            self._shuffled.remove(key)

    def clear(self) -> None:
        super().clear()
        self._shuffled.clear()

    def _eviction_order(self) -> List[str]:
        return list(self._shuffled)

    def _next_victim(self) -> str:
        return self._shuffled[0]


class FIFOPolicy(_BookkeepingPolicy):
    """Gives up the key inserted longest ago."""

    def _eviction_order(self) -> List[str]:
        return list(self._inserted)

    def _next_victim(self) -> str:
        return next(iter(self._inserted))


class LIFOPolicy(_BookkeepingPolicy):
    """Gives up the key inserted most recently."""

    def _eviction_order(self) -> List[str]:
        return list(reversed(self._inserted))

    def _next_victim(self) -> str:
        return next(reversed(self._inserted))


class LRUPolicy(_BookkeepingPolicy):
    """Gives up the key used longest ago."""

    def _eviction_order(self) -> List[str]:
        return list(self._used)

    def _next_victim(self) -> str:
        return next(iter(self._used))


class MRUPolicy(_BookkeepingPolicy):
    """Gives up the key used most recently."""

    def _eviction_order(self) -> List[str]:
        return list(reversed(self._used))

    def _next_victim(self) -> str:
        return next(reversed(self._used))


class LFUPolicy(_BookkeepingPolicy):
    """Gives up the least-used key; ties go to whichever arrived first."""

    def _eviction_order(self) -> List[str]:
        # ``sorted`` is stable and ``_inserted`` is already in insertion order,
        # so equal counts come out oldest-first without a second sort key.
        return sorted(self._inserted, key=lambda key: self._uses[key])

    def _next_victim(self) -> str:
        # ``min`` returns the first of several equal keys, which is the same
        # oldest-first tie-break the stable sort above gives.
        return min(self._inserted, key=lambda key: self._uses[key])


class MFUPolicy(_BookkeepingPolicy):
    """Gives up the most-used key; ties go to whichever arrived first."""

    def _eviction_order(self) -> List[str]:
        return sorted(self._inserted, key=lambda key: -self._uses[key])

    def _next_victim(self) -> str:
        # ``max`` also returns the first of several equal keys, so ties go to
        # the oldest here too rather than to the newest.
        return max(self._inserted, key=lambda key: self._uses[key])
