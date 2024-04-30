import os
import time
from abc import abstractmethod
from collections import OrderedDict


class EffiDictBase:
    """
    Base class for classes using both the memory and the disk (we'll call them caches), it contains some shared functionalities.

    :param max_in_memory: The maximum number of items to keep in memory.
    :type max_in_memory: int
    :param storage_path: Path to the disk storage.
    :type storage_path: str
    """

    def __init__(self, max_in_memory=100, storage_path="cache"):
        self.max_in_memory = max_in_memory
        self.storage_path = storage_path
        # add a unique identifier to the storage path to avoid conflicts
        self.storage_path = self.storage_path + f"{int(time.time())}_{id(self)}"

        self.memory = OrderedDict()

    def __iter__(self):
        """
        Return an iterator over the keys of the cache.

        :return: An iterator object.
        """
        self._iter_keys = iter(self.keys())
        return self

    def __next__(self):
        """
        Return the next key in the cache.

        :return: The next key from the iterator.
        """
        return next(self._iter_keys)

    def __len__(self):
        """
        Return the number of items in the cache, both in memory and on disk.

        :return: The total number of items.
        """
        return len(self.keys())

    def items(self):
        """
        Get all key-value pairs in the cache as a generator.

        This method iterates over all keys in the cache and yields key-value pairs. It retrieves
        values from memory or deserializes them from disk as needed.

        :return: A generator that yields (key, value) tuples for each item in the cache.
        """
        for key in self.keys():
            yield (key, self[key])

    def values(self):
        """
        Get all values in the cache as a generator.

        This method iterates over all keys in the cache and yields values. It retrieves
        values from memory or deserializes them from disk as needed.

        :return: A generator that yields values for each item in the cache.
        """
        for key in self.keys():
            yield self[key]

    def __contains__(self, key):
        """
        Check if the key is in the cache.

        :param key: The key to check for.
        :return: True if the key is in the cache, otherwise False.
        """
        return key in self.keys()

    def pop(self, key, default=None):
        """
        Remove the specified key and return the corresponding value.
        If the key is not found, return the default value. If the key is in memory, it is removed from there;
        if it is only on disk, it should also be removed from there.

        :param key: Key for which the value needs to be popped.
        :type key: Any hashable
        :param default: Default value to return if the key is not found.
        :type default: Any
        :return: Value associated with the key, or the default if the key is not found.
        :rtype: Any
        """
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
        """
        Remove all items from the cache.
        """
        self.memory.clear()
        for key in self.keys():
            self.__delitem__(key)

    def __del__(self):
        self.destroy()

    def __eq__(self, other):
        if not isinstance(other, EffiDictBase):
            return False

        if len(self) != len(other):
            return False

        for key, value in self.items():
            if key not in other or other[key] != value:
                return False

        return True

    @abstractmethod
    def keys(self):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def __delitem__(self, key):
        pass

    @abstractmethod
    def _serialize(self, key, value):
        pass

    @abstractmethod
    def _deserialize(self, key):
        pass

    @abstractmethod
    def load_from_dict(self, dictionary):
        pass

    @abstractmethod
    def destroy(self):
        pass
