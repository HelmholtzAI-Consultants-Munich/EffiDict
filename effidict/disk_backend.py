from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, Iterator
import sqlite3  
import json
import os
import pickle
import shutil
import ast 

try:
    import h5py
    import numpy as np
except Exception:
    h5py = None
    np = None

class DiskBackend(ABC):
    def __init__(self, storage_path):
        self.storage_path = storage_path + f"{int(time.time())}_{id(self)}"
        
    @abstractmethod
    def serialize(self, key, value):
        pass

    @abstractmethod
    def deserialize(self, key):
        pass

    @abstractmethod
    def del_item(self, key):
        pass

    @abstractmethod
    def keys(self):
        pass

    @abstractmethod
    def destroy(self):
        pass

    def load_from_dict(self, dictionary):
        for key, value in dictionary.items():
            self.serialize(key, value)

    def has(self, key) -> bool:
        """Return whether ``key`` exists in this backend."""
        raise NotImplementedError("see issue #2.2")

    def count(self) -> int:
        """Return the number of keys in this backend."""
        raise NotImplementedError("see issue #2.2")

    def iter_keys(self) -> Iterator:
        """Iterate over keys in this backend."""
        raise NotImplementedError("see issue #4.2")

    def read_many(self, keys) -> Dict:
        """Read multiple keys from this backend."""
        raise NotImplementedError("see issue #4.1")

    def write_many(self, items) -> None:
        """Write multiple items to this backend."""
        raise NotImplementedError("see issue #4.1")

    def delete_many(self, keys) -> None:
        """Delete multiple keys from this backend."""
        raise NotImplementedError("see issue #4.1")

    def compact(self) -> None:
        """Reclaim unused storage space."""
        raise NotImplementedError("see issue #7.2")

    @classmethod
    def create(cls, path, **kwargs):
        """Create a new backend at ``path``."""
        raise NotImplementedError("see issue #2.1")

    @classmethod
    def open(cls, path, mode: str = "r+", **kwargs):
        """Open a backend at ``path``.

        ``mode`` follows the ``create``/``open``/``open_or_create``
        distinction. The existing ``__init__`` behavior of deriving a unique
        path from the clock and ``id(self)`` is a bug fixed in issue #2.1.
        """
        raise NotImplementedError("see issue #2.1")

    @classmethod
    def open_or_create(cls, path, **kwargs):
        """Open an existing backend or create one at ``path``."""
        raise NotImplementedError("see issue #2.1")

    @classmethod
    def temporary(cls, prefix: str = "effidict-"):
        """Create a scratch backend at a unique path."""
        raise NotImplementedError("see issue #2.1")


class SqliteBackend(DiskBackend):
    def __init__(self, storage_path):
        super().__init__(storage_path)
        self.conn = sqlite3.connect(self.storage_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS data (key TEXT PRIMARY KEY, value TEXT)"
        )

    def serialize(self, key, value):
        json_value = json.dumps(value)
        self.cursor.execute(
            "REPLACE INTO data (key, value) VALUES (?, ?)", (key, json_value)
        )
        self.conn.commit()

    def deserialize(self, key):
        self.cursor.execute("SELECT value FROM data WHERE key=?", (key,))
        result = self.cursor.fetchone()
        if result:
            return json.loads(result[0])
        raise KeyError(key)
    
    def del_item(self, key):
        self.cursor.execute("DELETE FROM data WHERE key=?", (key,))
        self.conn.commit()

    def keys(self):
        self.cursor.execute("SELECT key FROM data")
        return [key[0] for key in self.cursor.fetchall()]
    
    def load_from_dict(self, dictionary):
        with self.conn:
            items_to_insert = [
                (key, json.dumps(value)) for key, value in dictionary.items()
            ]
            self.cursor.executemany(
                "REPLACE INTO data (key, value) VALUES (?, ?)",
                items_to_insert,
            )

    def destroy(self):
        self.conn.close()
        os.remove(self.storage_path)


class PickleBackend(DiskBackend):
    def __init__(self, storage_path):
        super().__init__(storage_path)
        os.makedirs(self.storage_path, exist_ok=True)

    def serialize(self, key, value):
        with open(os.path.join(self.storage_path, key), "wb") as f:
            pickle.dump(value, f)

    def deserialize(self, key):
        try:
            with open(os.path.join(self.storage_path, key), "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise KeyError(key)
        
    def del_item(self, key):
        try:
            os.remove(os.path.join(self.storage_path, key))
        except FileNotFoundError:
            pass

    def keys(self):
        return os.listdir(self.storage_path)

    def destroy(self):
        shutil.rmtree(self.storage_path)


class Hdf5Backend(DiskBackend):
    def __init__(self, storage_path):
        if h5py is None or np is None:
            raise ImportError("h5py and numpy are required for Hdf5Backend but are not installed.")
        super().__init__(storage_path)
        self.file = h5py.File(self.storage_path, "w")

    def serialize(self, key, value):
        """
        Store data and its Python type as an HDF5 attribute.
        """
        if key in self.file:
            del self.file[key] 

        type_name = type(value).__name__
        
        if value is None:
            ds = self.file.create_dataset(key, data=h5py.Empty("f"))
        elif isinstance(value, (dict, list, tuple)):
            ds = self.file.create_dataset(key, data=repr(value))
        else:
            ds = self.file.create_dataset(key, data=value)
            
        ds.attrs['python_type'] = type_name

    def deserialize(self, key):
        """
        Read data and use the 'python_type' attribute to cast it back correctly.
        """
        try:
            dataset = self.file[key]
            type_name = dataset.attrs.get('python_type')

            if type_name is None:
                raise TypeError(f"Dataset for key '{key}' is missing 'python_type' attribute.")

            if type_name == 'NoneType':
                return None
            elif type_name in ('dict', 'list', 'tuple'):
                str_data = dataset.asstr()[()]
                return ast.literal_eval(str_data)
            elif type_name == 'str':
                return dataset.asstr()[()]
            else: 
                return dataset[()]

        except KeyError:
            raise KeyError(key)
        
    def del_item(self, key):
        try:
            del self.file[key]
        except KeyError:
            pass

    def keys(self):
        return list(self.file.keys())
    
    def destroy(self):
        self.file.close()
        os.remove(self.storage_path)    

class JSONBackend(DiskBackend):
    def __init__(self, storage_path):
        super().__init__(storage_path)
        os.makedirs(self.storage_path, exist_ok=True)

    def _file_path(self, key):
        return os.path.join(self.storage_path, f"{key}.json")

    def serialize(self, key, value):
        with open(self._file_path(key), "w") as f:
            json.dump(value, f)

    def deserialize(self, key):
        try:
            with open(self._file_path(key), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise KeyError(key)
        
    def del_item(self, key):
        try:
            os.remove(self._file_path(key))
        except FileNotFoundError:
            pass

    def keys(self):
        return [os.path.splitext(f)[0] for f in os.listdir(self.storage_path)]
    
    def destroy(self):
        shutil.rmtree(self.storage_path)
