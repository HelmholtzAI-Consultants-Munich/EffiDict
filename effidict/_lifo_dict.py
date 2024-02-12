import json
import os
import pickle
import sqlite3

from ._base import EffiDictBase


class LIFODict(EffiDictBase):
    """
    A class implementing a Last In First Out (LIFO) cache. i.e., the last item to be inserted into the cache is the first one to be evicted.

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
        Initialize a LIFODict object.

        This class implements a Last In First Out (LIFO) cache which stores a limited
        number of items in memory and the rest on the disk at the specified storage path as pickle files.

        :param max_in_memory: The maximum number of items to keep in memory.
        :type max_in_memory: int
        :param storage_path: The path to the directory where items will be stored on disk.
        :type storage_path: str
        """
        super().__init__(max_in_memory, storage_path)
