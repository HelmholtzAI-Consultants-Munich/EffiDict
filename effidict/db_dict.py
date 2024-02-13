import json
import os
import pickle
import sqlite3


class DBDict:
    def __init__(self, storage_path):
        self.conn = sqlite3.connect(storage_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS data (key TEXT PRIMARY KEY, value TEXT)"
        )

    def __getitem__(self, key):
        self.cursor.execute("SELECT value FROM data WHERE key=?", (key,))
        result = self.cursor.fetchone()
        if result:
            return json.loads(result[0])
        raise KeyError(key)

    def __setitem__(self, key, value):
        json_value = json.dumps(value)
        self.cursor.execute(
            "REPLACE INTO data (key, value) VALUES (?, ?)", (key, json_value)
        )
        self.conn.commit()

    def __delitem__(self, key):
        self.cursor.execute("DELETE FROM data WHERE key=?", (key,))
        self.conn.commit()

    def __iter__(self):
        self.cursor.execute("SELECT key FROM data")
        return (key[0] for key in self.cursor.fetchall())

    def __len__(self):
        self.cursor.execute("SELECT COUNT(*) FROM data")
        return self.cursor.fetchone()[0]

    def keys(self):
        self.cursor.execute("SELECT key FROM data")
        return [key[0] for key in self.cursor.fetchall()]
