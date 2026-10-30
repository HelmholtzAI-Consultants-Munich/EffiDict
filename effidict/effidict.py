from __future__ import annotations

from typing import Optional

from .cache import Cache
from .store import MISSING, Store


class EffiDict:
    def __init__(
        self,
        disk_backend,
        policy,
        *,
        max_in_memory: int,
        max_bytes: Optional[int] = None,
    ):
        """Create an EffiDict facade over a single ``Store``.

        The facade holds no state of its own. Every question about where a key
        lives goes to the store, which owns the backend, the cache and the lock.
        It used to own a backend reference and alias the strategy's dict as
        ``self.memory``, so two objects decided placement and the alias went
        stale the moment the strategy was replaced.

        ``policy`` is an ``EvictionPolicy`` -- pure bookkeeping, no constructor
        arguments -- so the bound that used to live on the strategy is now
        ``max_in_memory`` here, and it becomes the cache's item budget. It is
        required: a store with no bound never spills, which is the one thing
        this library exists to do.

        ``max_bytes`` is accepted as the approximate cache byte budget, but it is
        not enforced yet; see issue #1.4.
        """
        self._store = Store(
            backend=disk_backend,
            cache=Cache(policy=policy, max_items=max_in_memory, max_bytes=max_bytes),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Explicitly close the backend resources."""
        self._store.close()

    def __iter__(self):
        self._iter_keys = iter(self.keys())
        return self

    def __next__(self):
        return next(self._iter_keys)

    def __len__(self):
        return self._store.count()

    def items(self):
        for key in self.keys():
            yield (key, self[key])

    def values(self):
        for key in self.keys():
            yield self[key]

    def __contains__(self, key):
        return self._store.contains(key)

    def pop(self, key, default=MISSING):
        """Remove ``key`` and return its value, ``dict``-style.

        ``MISSING`` rather than ``None``: ``pop(key, None)`` has to return
        ``None`` while ``pop(key)`` raises, and the two are indistinguishable
        once the default is spelled ``None``.
        """
        return self._store.pop(key, default)

    def get(self, key, default=None):
        """Return ``key`` if present, otherwise ``default``."""
        raise NotImplementedError("see issue #6.1")

    def setdefault(self, key, default=None):
        """Return ``key`` if present, otherwise set and return ``default``."""
        raise NotImplementedError("see issue #6.1")

    def popitem(self):
        """Remove and return the **most recently inserted** key/value pair.

        LIFO, not arbitrary: ``dict.popitem`` has been last-in-first-out since
        3.7 and the differential oracle compares against ``dict``, so returning
        an arbitrary pair would fail conformance. Raises ``KeyError`` when empty.
        """
        raise NotImplementedError("see issue #6.1")

    def update(self, other=(), /, **kwargs):
        """Update this dictionary from another mapping or iterable.

        ``other`` is positional-only, as it is on ``dict.update``. Without the
        ``/`` the name is bindable, so ``update(other=5)`` would assign 5 to the
        mapping parameter instead of storing a key called ``other`` -- and
        ``update({"other": a}, other=b)`` would raise ``TypeError`` for a
        duplicate argument rather than letting the keyword win.
        """
        raise NotImplementedError("see issue #6.1")

    def iter_keys(self):
        """Iterate over keys without materializing a list."""
        raise NotImplementedError("see issue #4.2")

    def flush(self):
        """Write every dirty cached entry to storage (I7)."""
        self._store.flush()

    def clone(self, new_path):
        """Clone this dictionary to ``new_path``."""
        raise NotImplementedError("see issue #3.2")

    def destroy(self):
        """Delete the backing storage. Explicit, and idempotent.

        Separate from ``close``, which must release handles and leave the data
        in place. Today ``close`` destroys, which is why collecting a store --
        or a joblib worker finishing a task -- takes the data with it.

        Idempotent: a second call is a no-op, not ``FileNotFoundError``. That is
        what lets teardown paths call it without first checking.
        """
        raise NotImplementedError("see issue #3.1")

    def __reversed__(self):
        """Iterate keys newest-first, mirroring ``reversed(dict)``.

        Declared because its absence is not inert: with no ``__reversed__``,
        ``reversed()`` falls back to the sequence protocol and subscripts the
        store with integers, so the caller gets a ``TypeError`` from inside
        ``os.path.join`` or a ``KeyError`` from the backend rather than anything
        about reversal.
        """
        raise NotImplementedError("see issue #6.1")

    def __or__(self, other):
        """Return a new mapping with ``other`` merged over this one."""
        raise NotImplementedError("see issue #6.1")

    @classmethod
    def fromkeys(cls, keys, value=None):
        """Build a store whose keys are ``keys``, each mapped to ``value``.

        A classmethod, as on ``dict``, which leaves one question open: a store
        needs somewhere to live and ``dict.fromkeys`` takes no path. Issue 6.1
        settles that and may well make this an instance method borrowing the
        receiver's backend -- which is why the spec asserts the *contents* of
        the result and not this signature.
        """
        raise NotImplementedError("see issue #6.1")

    def clear(self):
        self._store.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __eq__(self, other):
        if not isinstance(other, EffiDict):
            return False

        if len(self) != len(other):
            return False

        for key, value in self.items():
            if key not in other or other[key] != value:
                return False
        return True

    def keys(self):
        """Return a list of all unique keys known to the dictionary (memory + disk)."""
        return list(self._store.iter_keys())

    def __getitem__(self, key):
        return self._store.get(key)

    def __setitem__(self, key, value):
        self._store.set(key, value)

    def __delitem__(self, key):
        self._store.delete(key)

    def load_from_dict(self, dictionary):
        self._store.load_from_dict(dictionary)
