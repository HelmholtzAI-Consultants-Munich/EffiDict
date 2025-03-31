class EffiDict:

    def __init__(self, max_in_memory=100, disk_backend=None, replacement_strategy=None):
        self.max_in_memory = max_in_memory
        self.disk_backend = disk_backend
        self.replacement_strategy = replacement_strategy

        self.memory = replacement_strategy.memory

    def __iter__(self):
        self._iter_keys = iter(self.keys())
        return self

    def __next__(self):
        """
        Return the next key in the cache.

        :return: The next key from the iterator.
        """
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
        return key in self.keys()

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
        self.memory.clear()
        for key in self.keys():
            self.__delitem__(key)

    def __del__(self):
        # wait:
            self.disk_backend.destroy()

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
        return self.memory.keys()+self.disk_backend.keys()

    def __getitem__(self, key):
        return self.replacement_strategy.get(key)

    def __setitem__(self, key, value):
        self.replacement_strategy.put(key, value)

    def __delitem__(self, key):
        if key in self.memory:
            del self.memory[key]
        self.disk_backend.del_item(key)

    def load_from_dict(self, dictionary):
        self.disk_backend.load_from_dict(dictionary)


