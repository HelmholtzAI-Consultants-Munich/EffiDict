from ._base import PickleDict, SqliteDict


class LRUPickleDict(PickleDict):
    """
    Implementation of a Least Recently Used (LRU) cache for PickleDict.
    """

    def __getitem__(self, key):
        """
        Get an item from the cache.

        If the item is in memory, it is returned directly. If not, it is loaded from disk,
        added back to the memory cache, and then returned.

        :param key: The key of the item to retrieve.
        :return: The value associated with the key if it exists, otherwise None.
        """
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        else:
            value = self._deserialize(key)
            if value is not None:
                self[key] = value  # Re-add it to memory, possibly evicting another item
            return value

    def __setitem__(self, key, value):
        """
        Set an item in the cache.

        If the cache exceeds its memory limit, the least recently used item is serialized
        and stored on disk.

        :param key: The key of the item to set.
        :param value: The value of the item to set.
        """
        self.memory[key] = value
        self.memory.move_to_end(key)
        if len(self.memory) > self.max_in_memory:
            oldest_key, oldest_value = self.memory.popitem(last=False)
            self._serialize(oldest_key, oldest_value)


class LRUSqliteDict(SqliteDict):
    """
    Implementation of a Least Recently Used (LRU) cache for SqliteDict.
    """

    def __getitem__(self, key):
        """
        Get an item from the cache.

        If the item is in memory, it is returned directly. If not, it is loaded from the database,
        added back to the memory cache, and then returned.

        :param key: The key of the item to retrieve.
        :return: The value associated with the key if it exists, otherwise None.
        """
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        else:
            value = self._deserialize(key)
            if value is not None:
                self[key] = value  # Re-add it to memory, possibly evicting another item
            return value

    def __setitem__(self, key, value):
        """
        Set an item in the cache.

        If the cache exceeds its memory limit, the oldest item is serialized to the database directly.

        :param key: The key of the item to set.
        :param value: The value of the item to set.
        """
        self.memory[key] = value
        self.memory.move_to_end(key)
        if len(self.memory) > self.max_in_memory:
            oldest_key, oldest_value = self.memory.popitem(last=False)
            self._serialize(oldest_key, oldest_value)
