from __future__ import annotations

import sys
from typing import Iterator


def estimate_size(value) -> int:
    """Approximate the resident bytes of ``value``.

    Approximate on purpose, and it has to be: an exact accounting of an
    arbitrary Python object is not achievable. A buffer shared by two cached
    values is counted twice, and ``sys.getsizeof`` reports a container's own
    footprint rather than that of what it holds, so a list of large strings
    measures as a list of pointers. The budget only has to be right to within a
    small factor to keep a cache bounded, which is all it is for.

    Three rules, in order:

    * anything with a callable ``memory_usage`` -- a pandas ``DataFrame`` or
      ``Series`` -- is measured with ``memory_usage(deep=True)``, which follows
      object columns to the strings they point at instead of counting one
      pointer each.
    * anything carrying an ``nbytes`` int -- a numpy array, a ``memoryview`` --
      reports its buffer directly.
    * everything else falls back to ``sys.getsizeof``.

    Duck-typed rather than imported. numpy and pandas are test dependencies,
    not runtime ones, and a store holding strings must not import either to
    measure them.
    """
    measure = getattr(value, "memory_usage", None)
    if callable(measure):
        # A DataFrame answers per column and needs summing; a Series answers
        # with the total already.
        used = measure(deep=True)
        total = getattr(used, "sum", None)
        return int(total() if callable(total) else used)

    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes

    return sys.getsizeof(value)


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

        Each value is measured once, when it is put, and the measurement is
        kept with it. Re-measuring the whole cache on every write would make a
        fill quadratic, and ``memory_usage(deep=True)`` is far too expensive to
        call per resident value per write. The cost of keeping it is that a
        value mutated in place after it was stored is measured as it was --
        acceptable for a figure already documented as approximate.
        """
        self.policy = policy
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.size_estimator = size_estimator or estimate_size
        #: key -> (value, dirty). One container rather than a parallel
        #: dirty set, so a key cannot be dropped from one and left in the
        #: other -- the phantom-state defect ``on_remove`` exists to avoid
        #: in the policies.
        self._values = {}

    def __iter__(self) -> Iterator:
        """Iterate the cached keys, so ``set(cache)`` is the resident key set."""
        return iter(self._values)

    def has(self, key) -> bool:
        """Return whether ``key`` is present in the cache."""
        return key in self._values

    def get(self, key):
        """Return ``key`` and record access, or raise ``KeyError`` if absent.

        Reading does not touch the tag: I6 says a read never marks an entry
        dirty, so a clean entry stays free to evict however often it is read.
        """
        value, _, _ = self._values[key]
        self.policy.on_access(key)
        return value

    def peek(self, key):
        """Return ``key`` without recording access, or raise ``KeyError``."""
        return self._values[key][0]

    def put(self, key, value, dirty: bool = True) -> None:
        """Store ``value`` under ``key`` and record dirty state.

        A rewrite is reported to the policy as an access, not an insertion.
        ``on_insert`` restarts a key's bookkeeping -- frequency back to one --
        so routing an update through it would discard the usage history of a key
        that never left the cache. That is the defect
        ``test_mfu_replacement_keeps_frequency_across_a_value_update`` pins
        against the strategies this class replaces.

        ``dirty=True`` is the right default because the ordinary caller is a
        write: ``Store.set`` puts a value the disk does not have yet. Only a
        caller that knows disk already holds this exact value may pass
        ``False``, and being wrong about that loses data -- the entry would be
        dropped at eviction without ever being written.
        """
        if key in self._values:
            self.policy.on_access(key)
        else:
            self.policy.on_insert(key)
        self._values[key] = (value, dirty, self.size_estimator(value))

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
        """Return the approximate resident byte size.

        Summed over measurements taken at ``put`` time, so this costs no
        estimator calls -- see :meth:`__init__` for why they are not retaken.
        """
        return sum(size for _, _, size in self._values.values())

    def is_dirty(self, key) -> bool:
        """Return whether ``key`` has unwritten cache changes.

        Raises ``KeyError`` for a key the cache does not hold, like ``peek``:
        "is it dirty?" has no answer for an absent key, and returning ``False``
        would read as "clean, safe to drop".
        """
        return self._values[key][1]

    def mark_clean(self, key) -> None:
        """Mark ``key`` as having no unwritten cache changes.

        For the caller that has just written the value out -- ``flush`` in
        issue #1.4. There is deliberately no ``mark_dirty``: an entry becomes
        dirty by being written through ``put``, which is the only way its value
        can diverge from disk.
        """
        value, _, size = self._values[key]
        self._values[key] = (value, False, size)

    def dirty_items(self) -> Iterator:
        """Iterate ``(key, value)`` for the entries disk does not have.

        What ``flush`` has to write (I7, issue #1.4). A generator over the live
        dict, so a caller that marks entries clean while walking it would be
        mutating what it is reading -- ``flush`` collects first for that reason.
        """
        return (
            (key, value)
            for key, (value, dirty, _) in self._values.items()
            if dirty
        )

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

        Both budgets, whichever binds first: enough keys to bring the item count
        inside ``max_items`` *and* the estimated bytes inside ``max_bytes``.
        Either left as ``None`` constrains nothing, so a cache given only
        ``max_items`` behaves exactly as it did before the byte budget existed.
        """
        surplus = (
            0 if self.max_items is None else len(self._values) - self.max_items
        )
        excess = (
            0 if self.max_bytes is None else self.nbytes() - self.max_bytes
        )
        if surplus <= 0 and excess <= 0:
            return iter(())

        if surplus == 1 and excess <= 0:
            # One write past the item bound sheds one key, which is the steady
            # state and nearly every call. ``victims()`` has to build the whole
            # ordering before it yields anything, so asking it for one key costs
            # the size of the cache and a fill becomes quadratic; ``victim()``
            # reads the ordering's endpoint instead. The two name the same key,
            # which ``test_victims_enumerates_in_eviction_order`` pins. Only
            # when bytes are inside budget: a byte surplus needs each victim's
            # size to know when enough has been shed, and ``victim()`` names one.
            return iter((self.policy.victim(),))

        # Walked eagerly rather than handed back lazily: the caller discards
        # what it is given while walking it, and a policy that enumerates its
        # live ordering rather than a snapshot would raise mid-eviction. The
        # stopping condition reads both budgets, so whichever is still breached
        # keeps the walk going.
        candidates = []
        freed = 0
        for key in self.policy.victims():
            candidates.append(key)
            freed += self._values[key][2]
            if len(candidates) >= surplus and freed >= excess:
                break
        return iter(candidates)
