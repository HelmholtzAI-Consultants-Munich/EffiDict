import time
from abc import abstractmethod
import sqlite3  
import json
import os
import pickle
import h5py
import shutil

class DiskBackend:
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

    def serialize(self, key, value):
        with open(self.storage_path + key, "wb") as f:
            pickle.dump(value, f)

    def deserialize(self, key):
        try:
            with open(self.storage_path + key, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise KeyError(key)
        
    def del_item(self, key):
        os.remove(self.storage_path + key)

    def keys(self):
        return os.listdir(self.storage_path)

    def destroy(self):
        shutil.rmtree(self.storage_path)

    
class Hdf5Backend(DiskBackend):
    def __init__(self, storage_path):
        super().__init__(storage_path)
        self.file = h5py.File(self.storage_path, "w")

    def serialize(self, key, value):
        self.file.create_dataset(key, data=value)

    def deserialize(self, key):
        try:
            return self.file[key][()]
        except KeyError:
            raise KeyError(key)
        
    def del_item(self, key):
        del self.file[key]

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
        os.remove(self._file_path(key))

    def keys(self):
        return [os.path.splitext(f)[0] for f in os.listdir(self.storage_path)]
    
    def destroy(self):
        shutil.rmtree(self.storage_path)