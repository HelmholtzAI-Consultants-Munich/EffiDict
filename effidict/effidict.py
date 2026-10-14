from __future__ import annotations

from typing import Optional


class EffiDict:
    def __init__(self, disk_backend=None, replacement_strategy=None, *, max_bytes: Optional[int] = None):
        """Create an EffiDict facade.

        ``max_bytes`` is accepted as the future approximate cache byte budget,
        but it is not enforced yet; see issue #1.4.
        """
        self.disk_backend = disk_backend
        self.replacement_strategy = replacement_strategy
        self.max_bytes = max_bytes
        self.memory = replacement_strategy.memory

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Explicitly close the backend resources."""
        if self.disk_backend is not None:
            self.disk_backend.destroy()

    def __iter__(self):
        self._iter_keys = iter(self.keys())
        return self

    def __next__(self):
        return next(self._iter_keys)

    def __len__(self):
        return len(self.keys())

    def items(self):
        for key in self.keys():
            yield (key, self[key])

    def values(self):
        for key in self.keys():
            yield self[key]

    def __contains__(self, key):
        return key in self.memory or (self.disk_backend and key in self.disk_backend.keys())

    def pop(self, key, default=None):
        try:
            value = self.memory.pop(key)
        except KeyError:
            if key in self.keys():
                value = self[key]
                self.__delitem__(key)
            else:
                return default

        return value

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

    def update(self, other=(), **kwargs):
        """Update this dictionary from another mapping or iterable."""
        raise NotImplementedError("see issue #6.1")

    def iter_keys(self):
        """Iterate over keys without materializing a list."""
        raise NotImplementedError("see issue #4.2")

    def flush(self):
        """Flush dirty cached entries to storage."""
        raise NotImplementedError("see issue #1.4")

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

    @property
    def _store(self):
        """Future Store facade used by white-box tier-invariant specs."""
        raise NotImplementedError("see issue #1.2")

    def clear(self):
        all_keys = self.keys()
        self.memory.clear()
        if self.disk_backend:
            for key in all_keys:
                self.disk_backend.del_item(key)
        # Also clear strategy-specific caches like LFU/MFU counts
        if hasattr(self.replacement_strategy, 'clear'):
            self.replacement_strategy.clear()


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
        mem_keys = set(self.memory.keys())
        disk_keys = set(self.disk_backend.keys() if self.disk_backend is not None else [])
        return list(mem_keys.union(disk_keys))

    def __getitem__(self, key):
        return self.replacement_strategy.get(key)

    def __setitem__(self, key, value):
        self.replacement_strategy.put(key, value)

    def __delitem__(self, key):
        # Check existence first to raise KeyError properly
        if key not in self:
            raise KeyError(key)
        # The strategy handles its own memory (and secondary_memory for LFU/MFU)
        self.replacement_strategy.delete(key)
        # Ensure it's also removed from the disk backend
        self.disk_backend.del_item(key)

    def load_from_dict(self, dictionary):
        self.disk_backend.load_from_dict(dictionary)
