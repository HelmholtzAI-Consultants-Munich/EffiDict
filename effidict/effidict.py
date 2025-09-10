class EffiDict:
    def __init__(self, max_in_memory=100, disk_backend=None, replacement_strategy=None):
        self.max_in_memory = max_in_memory
        self.disk_backend = disk_backend
        self.replacement_strategy = replacement_strategy
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