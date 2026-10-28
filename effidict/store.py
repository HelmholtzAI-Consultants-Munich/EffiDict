from __future__ import annotations

import threading
from typing import Iterator


class _Missing:
    """Sentinel for "no default was given".

    ``pop(key, default=None)`` cannot tell ``pop(key)`` from ``pop(key, None)``,
    so a ``Store`` written against that signature would have to guess -- and
    guessing "return None" is precisely the defect
    ``test_pop_without_default_raises_keyerror`` pins at the facade (issue 1.2).
    The contract has to be able to express the distinction that ``dict`` makes.
    """

    def __repr__(self):  # pragma: no cover - debugging aid
        return "<no default>"


MISSING = _Missing()


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

    I2 is not implemented here. Without the clean/dirty tag every eviction has
    to assume the cached copy is newer, so :meth:`_spill` writes each victim out
    before dropping it. That satisfies the second half of I5 -- a key is never in
    neither tier -- at the cost of the first half, which issue #1.3 recovers once
    entries know whether disk already has them.

    **No stored value is released while the lock is held.** Dropping the last
    reference to a value runs its ``__del__`` then and there, on this thread,
    inside the critical section -- so overwriting a cached entry, evicting one
    or deleting one hands control to arbitrary user code in the middle of a
    tier operation. A finalizer that queries the store would then read the
    half-finished state I4 forbids: during :meth:`delete`, after the cache has
    dropped the key and before the backend has, ``get`` still answers from
    disk. The mutating methods therefore keep every displaced value alive in a
    local ``released`` list and let it die after the ``with`` block, which runs
    those finalizers against a store that is whole again.

    This makes re-entry impossible rather than reentrant, which is why the
    default lock is a plain :class:`threading.Lock`. It does not cover a value
    whose *serialization* hooks re-enter -- ``__reduce__`` runs under the lock
    by necessity, since I4 forbids releasing it around the write -- and such a
    value deadlocks by design.
    """

    def __init__(self, backend, cache, lock=None, owns_storage: bool = True):
        self._backend = backend
        self._cache = cache
        # A plain ``Lock``, because nothing here re-acquires. The tier
        # operations are written in terms of each other -- ``pop`` is ``get``
        # then ``delete``, ``delete`` asks ``contains`` -- and I4 requires the
        # whole sequence to hold one lock rather than releasing it between the
        # two tiers, so each public method acquires once and calls the private
        # held-lock form of the others. That is what lets an injected lock be
        # any lock: a caller sharing a plain ``Lock`` across stores, or the
        # ``Lock`` protocol of PLAN.md 3.6, which need not be reentrant.
        #
        # Reentrant by default would not make the one caller that re-enters on
        # its own -- a finalizer, run the instant a value's last reference
        # goes -- correct; it would only let it observe a half-applied
        # operation. The ``released`` lists in the mutating methods keep that
        # from happening at all: see the class docstring.
        self._lock = threading.Lock() if lock is None else lock
        self._owns_storage = owns_storage

    @property
    def backend(self):
        """Return the backing persistent store."""
        return self._backend

    @property
    def cache(self):
        """Return the in-memory cache."""
        return self._cache

    @property
    def owns_storage(self) -> bool:
        """Return whether this store owns the backend storage lifecycle."""
        return self._owns_storage

    def get(self, key):
        """Return ``key`` from the store.

        Cache first, then disk; the backend raises ``KeyError`` for a key it
        does not hold, which is the same answer the cache gives, so a miss in
        both tiers reaches the caller unchanged.

        A disk hit is *not* promoted into the cache. Promoting it would evict
        something, and without the clean/dirty tag (issue #1.3) that eviction
        writes -- turning a read-only pass into one disk write per key, which is
        the write amplification ``test_clean_eviction_performs_no_io`` measures.
        Promotion belongs with the tag, not before it.
        """
        with self._lock:
            return self._get(key)

    def _get(self, key):
        """:meth:`get` with the lock already held."""
        try:
            return self._cache.get(key)
        except KeyError:
            return self._backend.deserialize(key)

    def set(self, key, value) -> None:
        """Store ``value`` under ``key``.

        Overwriting a cached key displaces the value that was there, and that
        value must outlive the lock: see the class docstring.
        """
        released = []
        with self._lock:
            if self._cache.has(key):
                released.append(self._cache.peek(key))
            self._cache.put(key, value)
            self._spill(released)
        del released

    def _spill(self, released: list) -> None:
        """Write out and drop whatever the cache says is over budget.

        Write *then* drop, per I5: between the two the key is in both tiers, so
        a reader looking at either one still finds it. Dropping first would open
        a window in which the key is in neither, which is the gap
        ``test_no_key_is_ever_absent_from_both_tiers`` probes.

        Each evicted value goes to ``released``, the caller's list, so that the
        discard here cannot be the drop that runs its ``__del__``.
        """
        for key in self._cache.evict_candidates():
            value = self._cache.peek(key)
            released.append(value)
            self._backend.serialize(key, value)
            self._cache.discard(key)

    def delete(self, key) -> None:
        """Remove ``key`` from all tiers.

        I4: both tiers under one lock. ``del_item`` is tolerant of a key the
        backend never held, so a cache-only key needs no special case here --
        the check in :meth:`_delete` is what turns a missing key into ``KeyError``.
        """
        released = []
        with self._lock:
            self._delete(key, released)
        del released

    def _delete(self, key, released: list) -> None:
        """:meth:`delete` with the lock already held.

        ``released`` takes the cached value, so the window between the two
        tiers -- the one I4 is about -- cannot be handed to a finalizer.
        """
        if not self._contains(key):
            raise KeyError(key)
        if self._cache.has(key):
            released.append(self._cache.peek(key))
        self._cache.discard(key)
        self._backend.del_item(key)

    def pop(self, key, default=MISSING):
        """Remove ``key`` and return its value.

        Returns ``default`` if the key is absent and a default was given; raises
        ``KeyError`` if it is absent and none was. ``dict`` semantics, including
        ``pop(key, None)`` returning ``None`` rather than raising.

        Read then delete, on the same path as :meth:`delete`. The bug this
        replaces popped the in-memory copy and returned, leaving the disk copy
        behind: a second ``pop`` of the same key then succeeded.
        """
        released = []
        with self._lock:
            try:
                value = self._get(key)
            except KeyError:
                if default is MISSING:
                    raise
                return default
            self._delete(key, released)
        del released
        return value

    def clear(self) -> None:
        """Remove all entries from all tiers.

        Every cached value is displaced at once, so ``released`` holds the lot
        until the lock is gone: see the class docstring.
        """
        released = []
        with self._lock:
            released.extend(self._cache.peek(key) for key in self._cache)
            for key in list(self._backend.keys()):
                self._backend.del_item(key)
            self._cache.clear()
        del released

    def update(self, items) -> None:
        """Store multiple items."""
        raise NotImplementedError("see issue #4.1")

    def contains(self, key) -> bool:
        """Return whether ``key`` is present in any tier.

        I3, as the invariant states it: ``cache.has(key) or backend.has(key)``.
        Making ``backend.has`` cheaper than a key listing is issue #2.2; the
        shape of the answer is settled here.
        """
        with self._lock:
            return self._contains(key)

    def _contains(self, key) -> bool:
        """:meth:`contains` with the lock already held."""
        return self._cache.has(key) or self._backend.has(key)

    def count(self) -> int:
        """Return the number of distinct keys in the store."""
        with self._lock:
            return len(self._key_set())

    def iter_keys(self) -> Iterator:
        """Iterate over all distinct keys in the store.

        Materialised: the union is built as a set before anything is yielded.
        Issue #4.2 makes it lazy, which is what
        ``test_keys_does_not_materialise_the_whole_keyspace`` is about. The union
        itself lives here because this is the one object that knows both tiers.
        """
        with self._lock:
            return iter(self._key_set())

    def _key_set(self):
        """Every distinct key across both tiers, with the lock already held."""
        return set(self._cache) | set(self._backend.keys())

    def in_cache(self, key) -> bool:
        """White-box hook for tier-invariant specs to assert preconditions.

        Must **not** acquire the store lock. The I5 specs use this to observe
        tier residency while an eviction is deliberately held open; if this
        blocked behind that eviction it could only ever report state from after
        the write completed, at which point a drop-then-write implementation
        looks identical to a compliant one.
        """
        return self._cache.has(key)

    def load_from_dict(self, dictionary) -> None:
        """Bulk-load ``dictionary`` straight into the backend.

        A pass-through to the backend's own bulk loader, kept so the facade does
        not have to reach past the store to call it. That it bypasses the cache
        is a real defect -- ``test_batch_and_single_writes_are_equivalent`` pins
        it -- and it is issue #4.1's to fix along with :meth:`update`.
        """
        with self._lock:
            self._backend.load_from_dict(dictionary)

    def flush(self) -> None:
        """Write dirty cached items to the backend."""
        raise NotImplementedError("see issue #1.4")

    def close(self) -> None:
        """Close store resources without destroying storage.

        Destroys today, which is what ``EffiDict.close`` has always done. Issue
        #3.1 separates the two; doing it here instead would be a lifecycle change
        in a key-placement PR, and three lifecycle specs are marked as expected
        failures against today's behaviour.
        """
        with self._lock:
            self._backend.destroy()

    def destroy(self) -> None:
        """Destroy owned storage resources."""
        raise NotImplementedError("see issue #3.1")

    def clone(self, new_path):
        """Clone the store to ``new_path``."""
        raise NotImplementedError("see issue #3.2")
