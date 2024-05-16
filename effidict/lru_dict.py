import inspect
import json
import os
import pickle
import shutil
import sqlite3
import warnings

from ._base import EffiDictBase


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
        os.makedirs(self.storage_path)

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

    def load_from_dict(self, dictionary):
        """
        Load items from a dictionary into the cache.

        This method is used to populate the cache with items from a dictionary.

        :param dictionary: A dictionary containing items to load into the cache.
        """
        for key, value in dictionary.items():
            self[key] = value

    def destroy(self):
        """
        Destroy the cache and remove all serialized files on disk.
        """
        # Problem with joblib: don't delete the storage_path if called by joblib
        called_by_joblib = any(
            record.function == "_process_worker" for record in inspect.stack()
        )

        if not called_by_joblib:
            shutil.rmtree(self.storage_path)
        else:
            try:
                shutil.rmtree(self.storage_path)
            except:
                pass

        del self.memory


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
    """

    def __init__(self, max_in_memory=100, storage_path="cache"):
        super().__init__(max_in_memory, storage_path)
        self.conn = sqlite3.connect(self.storage_path + ".db")
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS data (key TEXT PRIMARY KEY, value TEXT)"
        )

    def _serialize(self, key, value):
        """
        Serialize and store a single item to the SQLite database.

        :param key: The key of the item to serialize.
        :param value: The value of the item to serialize.
        """
        with self.conn:
            self.cursor.execute(
                "REPLACE INTO data (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

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

        If the cache exceeds its memory limit, the oldest item is serialized to the database directly.

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
        db_keys = {row[0] for row in self.cursor.fetchall()}
        return list(memory_keys.union(db_keys))

    def load_from_dict(self, dictionary):
        """
        Load items from a dictionary into the cache.

        :param dictionary: A dictionary containing items to load into the cache.
        """
        with self.conn:

            items_to_insert = [
                (key, json.dumps(value)) for key, value in dictionary.items()
            ]
            self.cursor.executemany(
                "REPLACE INTO data (key, value) VALUES (?, ?)",
                items_to_insert,
            )

    def destroy(self):
        """
        Destroy the cache and remove the SQLite database file.
        """
        del self.memory
        self.conn.close()
        os.remove(self.storage_path + ".db")
