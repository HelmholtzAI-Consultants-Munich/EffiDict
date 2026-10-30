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


#: Which locks this thread is inside, by ``id``. Thread-local, so a guard only
#: ever reads and writes the set belonging to the thread asking -- there is no
#: shared state to race over, and no other thread's entry to misread. Keyed by
#: the *underlying* lock rather than by the guard, because two stores handed
#: one lock share one critical section and must therefore share one answer to
#: "am I already inside it?". An id identifies its lock unambiguously for as
#: long as the entry exists: the guard holding the lock keeps it alive, so the
#: id cannot be handed to a different object while it is in the set.
_HELD = threading.local()


def _held_locks() -> set:
    """Return this thread's set of entered lock ids, creating it on first use."""
    try:
        return _HELD.locks
    except AttributeError:
        locks = _HELD.locks = set()
        return locks


class _ReentrancyGuard:
    """The store's lock, with same-thread re-entry turned into an error.

    Nothing in :class:`Store` re-enters; the callers that can are finalizers,
    and they arrive in the middle of whatever tier operation was running. A
    plain lock hangs on them with no diagnosis, and a reentrant one lets them
    read the half-applied state I4 exists to hide. Refusing is the third
    option and the only safe one: the re-entrant call never runs, so it cannot
    observe anything, and the operation underneath it is left alone.

    The entry is made *before* ``acquire`` and dropped *after* ``release``, so
    a thread counts as inside the lock for the whole of both calls. The
    ``Lock`` protocol of PLAN.md 3.6 admits a lock written in Python, whose
    ``acquire`` and ``release`` run Python -- and therefore allocate, and
    therefore may hand the cyclic collector a finalizer -- while the underlying
    lock is already held or still held. Bracketing the calls rather than
    sitting inside them is what covers those two windows.

    The bias is deliberate: between ``release`` returning and the entry being
    dropped the lock is free but still claimed, so a finalizer arriving in that
    sliver is refused although it could have been served. That is a visible
    error on a path that would otherwise have worked, against a deadlock on a
    path that cannot work -- and the window holds no call that can block.
    """

    def __init__(self, lock):
        self._lock = lock
        # Stable for this guard's life: ``self._lock`` keeps the lock alive.
        self._id = id(lock)

    def __enter__(self):
        held = _held_locks()
        if self._id in held:
            raise RuntimeError(
                "Store operation re-entered on the same thread. A stored "
                "value's __del__ called back into the store -- directly, or "
                "because the cyclic garbage collector ran that finalizer "
                "inside an unrelated store operation, possibly on another "
                "store sharing this lock. Re-entry cannot be served: it would "
                "read a half-applied operation. Keep the store out of the "
                "finalizer."
            )
        held.add(self._id)
        try:
            self._lock.acquire()
        except BaseException:
            held.discard(self._id)
            raise
        return self

    def __exit__(self, *exc_info):
        # ``held`` is bound before the release, so once the lock is free the
        # only step left is the discard -- nothing that can block, and nothing
        # that needs the lock.
        held = _held_locks()
        try:
            self._lock.release()
        finally:
            held.discard(self._id)


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

    I2 is the cache's: an entry carries a ``dirty`` tag, set by the write that
    put it there and cleared only by a caller that has since written it out.
    :meth:`_spill` reads that tag, which is what makes the first half of I5 --
    evicting a clean entry costs no I/O -- decidable at all. Every entry is
    dirty today, because the two callers that could produce a clean one are not
    here yet: ``flush`` (issue #1.4) and promotion on read (see :meth:`get`).

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

    Deferral reaches only the references this class drops. A value in a
    reference *cycle* outlives them: the collector frees it whenever it next
    runs, which may be inside any later locked operation, and no refcount-side
    strategy can reach that. Nor does deferral help a value whose
    ``__reduce__`` re-enters, since pickling runs under the lock by necessity
    -- I4 forbids releasing it around the write.

    What is left over is therefore reported rather than allowed or waited on:
    the lock is a :class:`_ReentrancyGuard`, which raises on same-thread
    re-entry. Measured, with a collection forced inside the lock: a bare
    ``Lock`` hangs there indefinitely, and the guard raises instead, leaving
    the operation underneath untouched. CPython prints the error from the
    finalizer and carries on.
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
        # A reentrant default would not make the one caller that re-enters on
        # its own -- a finalizer -- correct; it would only let it observe a
        # half-applied operation. How the deferral and the guard divide that
        # work between them is in the class docstring.
        self._lock = _ReentrancyGuard(threading.Lock() if lock is None else lock)
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

        A disk hit is *not* promoted into the cache. The clean/dirty tag makes
        a promoted entry free to evict, which removes the write amplification
        that used to be the objection -- but it does nothing about the other
        one: promotion changes *what is cached* as a side effect of reading,
        and two specs forbid exactly that.
        ``test_full_scan_does_not_evict_the_working_set`` requires a walk to
        leave the caller's cached keys intact, and
        ``test_eq_does_not_mutate_either_operand`` requires comparing two stores
        to leave both unchanged. Both walk every key through a smaller cache,
        so promotion evicts the caller's working set to hold keys the caller
        only passed over. A read path that streams without caching is issue
        #4.2; promotion can only follow it.
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

        A dirty victim is written *then* dropped, per I5: between the two the
        key is in both tiers, so a reader looking at either one still finds it.
        Dropping first would open a window in which the key is in neither, which
        is the gap ``test_no_key_is_ever_absent_from_both_tiers`` probes.

        A clean victim is dropped without being written, which is the other half
        of I5. Disk already holds that exact value, so the write would be a copy
        of what is there -- and a read-only pass over a store larger than its
        cache is made of such evictions, one per key. The tag is what tells the
        two cases apart; ``peek`` rather than ``get`` so measuring a victim does
        not promote it.

        Each evicted value goes to ``released``, the caller's list, so that the
        discard here cannot be the drop that runs its ``__del__``.
        """
        for key in self._cache.evict_candidates():
            value = self._cache.peek(key)
            released.append(value)
            if self._cache.is_dirty(key):
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
