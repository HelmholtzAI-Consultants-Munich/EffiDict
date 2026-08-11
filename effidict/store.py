from __future__ import annotations

from typing import Iterator


class Store:
    """The single owner of where a key lives.

    Holds the lock. Enforces the tier invariants:

    I1  Disk holds the authoritative copy of every flushed key.
    I2  Cached entries are tagged ``clean`` (matches disk) or ``dirty``
        (newer than disk).
    I3  ``key in store`` is ``cache.has(key) or backend.has(key)`` --
        never a full scan of the keyspace.
    I4  Delete removes from cache *and* backend under one lock; no partial
        state is observable.
    I5  Evicting a **clean** entry is a pure memory drop with zero I/O.
        Evicting a **dirty** entry writes *then* drops, so a key is never
        absent from both tiers.
    I6  A read never marks an entry dirty and never writes the key being
        read.
    I7  After ``flush()`` or ``close()``, every dirty entry is on disk.
    I8  The cache honours a **byte** budget, not just an item count.

    Note that keys legitimately reside in both tiers at once; I1 makes disk
    authoritative rather than exclusive. I5 depends on that overlap.
    """

    def __init__(self, backend, cache, lock=None, owns_storage: bool = True):
        raise NotImplementedError("see issue #1.2")

    @property
    def backend(self):
        """Return the backing persistent store."""
        raise NotImplementedError("see issue #1.2")

    @property
    def cache(self):
        """Return the in-memory cache."""
        raise NotImplementedError("see issue #1.2")

    @property
    def owns_storage(self) -> bool:
        """Return whether this store owns the backend storage lifecycle."""
        raise NotImplementedError("see issue #3.1")

    def get(self, key):
        """Return ``key`` from the store."""
        raise NotImplementedError("see issue #1.2")

    def set(self, key, value) -> None:
        """Store ``value`` under ``key``."""
        raise NotImplementedError("see issue #1.2")

    def delete(self, key) -> None:
        """Remove ``key`` from all tiers."""
        raise NotImplementedError("see issue #1.2")

    def pop(self, key, default=None):
        """Remove ``key`` and return its value, or ``default`` if absent."""
        raise NotImplementedError("see issue #1.2")

    def clear(self) -> None:
        """Remove all entries from all tiers."""
        raise NotImplementedError("see issue #1.2")

    def update(self, items) -> None:
        """Store multiple items."""
        raise NotImplementedError("see issue #4.1")

    def contains(self, key) -> bool:
        """Return whether ``key`` is present in any tier."""
        raise NotImplementedError("see issue #1.2")

    def count(self) -> int:
        """Return the number of distinct keys in the store."""
        raise NotImplementedError("see issue #1.2")

    def iter_keys(self) -> Iterator:
        """Iterate over all distinct keys in the store."""
        raise NotImplementedError("see issue #4.2")

    def in_cache(self, key) -> bool:
        """White-box hook for tier-invariant specs to assert preconditions."""
        raise NotImplementedError("see issue #1.2")

    def flush(self) -> None:
        """Write dirty cached items to the backend."""
        raise NotImplementedError("see issue #1.4")

    def close(self) -> None:
        """Close store resources without destroying storage."""
        raise NotImplementedError("see issue #1.2")

    def destroy(self) -> None:
        """Destroy owned storage resources."""
        raise NotImplementedError("see issue #3.1")

    def clone(self, new_path):
        """Clone the store to ``new_path``."""
        raise NotImplementedError("see issue #3.2")
