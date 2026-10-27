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
    keys up? -- by overriding :meth:`_eviction_order`. Everything else is
    identical across the seven, which is what the old ``*Replacement`` classes
    duplicated.

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
        for key in self.victims():
            return key
        raise KeyError("no keys are being tracked")

    def victims(self) -> Iterator:
        return iter(self._eviction_order())

    def clear(self) -> None:
        self._inserted.clear()
        self._used.clear()
        self._uses.clear()


class RandomPolicy(_BookkeepingPolicy):
    """Gives up an arbitrary tracked key."""

    def _eviction_order(self) -> List[str]:
        # The module-level generator, never an instance of ``random.Random``:
        # drawing from one stored on the policy would mutate it, and listing
        # victims has to leave the policy untouched.
        return random.sample(list(self._inserted), len(self._inserted))


class FIFOPolicy(_BookkeepingPolicy):
    """Gives up the key inserted longest ago."""

    def _eviction_order(self) -> List[str]:
        return list(self._inserted)


class LIFOPolicy(_BookkeepingPolicy):
    """Gives up the key inserted most recently."""

    def _eviction_order(self) -> List[str]:
        return list(reversed(self._inserted))


class LRUPolicy(_BookkeepingPolicy):
    """Gives up the key used longest ago."""

    def _eviction_order(self) -> List[str]:
        return list(self._used)


class MRUPolicy(_BookkeepingPolicy):
    """Gives up the key used most recently."""

    def _eviction_order(self) -> List[str]:
        return list(reversed(self._used))


class LFUPolicy(_BookkeepingPolicy):
    """Gives up the least-used key; ties go to whichever arrived first."""

    def _eviction_order(self) -> List[str]:
        # ``sorted`` is stable and ``_inserted`` is already in insertion order,
        # so equal counts come out oldest-first without a second sort key.
        return sorted(self._inserted, key=lambda key: self._uses[key])


class MFUPolicy(_BookkeepingPolicy):
    """Gives up the most-used key; ties go to whichever arrived first."""

    def _eviction_order(self) -> List[str]:
        return sorted(self._inserted, key=lambda key: -self._uses[key])
