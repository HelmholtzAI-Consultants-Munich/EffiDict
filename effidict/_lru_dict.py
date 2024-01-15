import json
import os
import pickle
import sqlite3
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

    @abstractmethod
    def keys(self):
        pass


class LRUDict(EffiDictBase):
    """
    A class implementing a Least Recently Used (LRU) cache.

    This class manages a cache that stores a limited number of items in memory and
    the rest on disk as pickle files. It inherits from EffiDictBase and extends its
    functionality to include serialization and deserialization of cache items.

    :param max_in_memory: The maximum number of items to keep in memory.
    :type max_in_memory: int
    :param storage_path: The path to the directory where items will be stored on disk.
    :type storage_path: str
    """

    def __init__(self, max_in_memory=100, storage_path="cache"):
        """
        Initialize an LRUDict object.

        This class implements a Least Recently Used (LRU) cache which stores a limited
        number of items in memory and the rest on the disk at the specified storage path as pickle files.

        :param max_in_memory: The maximum number of items to keep in memory.
        :type max_in_memory: int
        :param storage_path: The path to the directory where items will be stored on disk.
        :type storage_path: str

        """
        super().__init__(max_in_memory, storage_path)
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

    def _serialize(self, key, value):
        """
        Serialize and store the value associated with the key to the disk.

        This method is used to store items that are evicted from the memory cache.

        :param key: The key of the item to serialize.
        :param value: The value of the item to serialize.
        """
        with open(os.path.join(self.storage_path, str(key)), "wb") as file:
            pickle.dump(value, file)

    def _deserialize(self, key):
        """
        Deserialize and return the value associated with the key from the disk.

        This method is used to retrieve items that are not currently in the memory cache.

        :param key: The key of the item to deserialize.
        :return: The deserialized value if the key exists on disk, otherwise None.
        """
        try:
            with open(os.path.join(self.storage_path, str(key)), "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None

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

    def __delitem__(self, key):
        """
        Delete an item from the cache.

        If the item is in memory, it is removed. If it's on disk, the file is deleted.

        :param key: The key of the item to delete.
        """
        if key in self.memory:
            del self.memory[key]
        else:
            path = os.path.join(self.storage_path, str(key))
            if os.path.exists(path):
                os.remove(path)

    def keys(self):
        """
        Get all keys in the cache, including those in memory and those serialized on disk.

        This method combines keys from the memory cache and keys of serialized files on disk.

        :return: A list of all keys in the cache.
        """
        memory_keys = set(self.memory.keys())
        serialized_keys = {
            filename
            for filename in os.listdir(self.storage_path)
            if os.path.isfile(os.path.join(self.storage_path, filename))
        }
        return list(memory_keys.union(serialized_keys))


class LRUDBDict(EffiDictBase):
    """
    A class implementing a Least Recently Used (LRU) cache with a SQLite backend.

    This class manages a cache that stores a limited number of items in memory and
    the rest in a SQLite database. It extends the functionality of EffiDictBase by
    adding database operations for serialization and deserialization of cache items.

    :param max_in_memory: The maximum number of items to keep in memory.
    :type max_in_memory: int
    :param storage_path: The file path to the SQLite database.
    :type storage_path: str
    :param batch_size: The number of items to batch for database operations.
    :type batch_size: int
    """

    def __init__(self, max_in_memory=100, storage_path="cache.db", batch_size=10):
        super().__init__(max_in_memory, storage_path)
        self.conn = sqlite3.connect(storage_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS data (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.batch_cache = []
        self.batch_size = batch_size

    def _serialize_batch(self):
        """
        Serialize and store a batch of items to the SQLite database.

        This method is invoked when the batch cache reaches its specified size,
        indicating it's time to persist the items to the database.
        """
        with self.conn:
            self.cursor.executemany(
                "REPLACE INTO data (key, value) VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in self.batch_cache],
            )
        self.batch_cache.clear()

    def _deserialize(self, key):
        """
        Deserialize and return the value associated with the key from the database.

        :param key: The key of the item to deserialize.
        :return: The deserialized value if the key exists in the database, otherwise raises KeyError.
        :raises: KeyError if the key is not found in the database.
        """
        with self.conn:
            self.cursor.execute("SELECT value FROM data WHERE key=?", (key,))
            result = self.cursor.fetchone()
        if result:
            return json.loads(result[0])
        raise KeyError(key)

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

        If the cache exceeds its memory limit, the item is added to the batch cache for later
        serialization to the database.

        :param key: The key of the item to set.
        :param value: The value of the item to set.
        """
        self.memory[key] = value
        self.memory.move_to_end(key)
        if len(self.memory) > self.max_in_memory:
            oldest_key, oldest_value = self.memory.popitem(last=False)
            self.batch_cache.append((oldest_key, oldest_value))
            if len(self.batch_cache) >= self.batch_size:
                self._serialize_batch()

    def __delitem__(self, key):
        """
        Delete an item from the cache.

        If the item is in memory, it is removed. Additionally, the item is removed from the database.

        :param key: The key of the item to delete.
        """
        if key in self.memory:
            del self.memory[key]
        self.cursor.execute("DELETE FROM data WHERE key=?", (key,))
        self.conn.commit()

    def keys(self):
        """
        Get all keys in the cache, including those in memory and those stored in the database.

        :return: A list of all keys in the cache.
        """
        memory_keys = set(self.memory.keys())
        self.cursor.execute("SELECT key FROM data")
        return list(memory_keys.union(self.cursor.fetchall()))
