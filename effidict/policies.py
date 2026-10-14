from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


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
