import os

import pytest

from effidict import LRUDBDict, LRUDict


@pytest.fixture
def lru_dict(tmp_path):
    """Fixture to create an LRUDict instance and clean up after tests."""
    cache = LRUDict(max_in_memory=2, storage_path=str(tmp_path / "lrudict_cache"))
    yield cache


def test_lrudict_set_and_get_item(lru_dict):
    lru_dict["key1"] = "value1"
    assert lru_dict["key1"] == "value1", "Value should be value1"


def test_lrudict_eviction_to_disk(lru_dict):
    lru_dict["key1"] = "value1"
    lru_dict["key2"] = "value2"
    lru_dict["key3"] = "value3"  # key1 should go to the disk
    assert (
        "key1" not in lru_dict.memory
    ), "key1 should be evicted to the disk, and not in memory"
    assert os.path.exists(
        os.path.join(lru_dict.storage_path, "key1")
    ), "key1 should be on the disk"


def test_lrudict_retrieval_from_disk(lru_dict):
    lru_dict["key1"] = "value1"
    lru_dict["key2"] = "value2"
    lru_dict["key3"] = "value3"  # key1 should go to the disk
    assert lru_dict["key1"] == "value1", "key1 should be brought back to memory"


def test_lrudict_delete_item(lru_dict):
    lru_dict["key1"] = "value1"
    del lru_dict["key1"]
    assert "key1" not in lru_dict.memory, "key1 should be removed from memory"
    assert not os.path.exists(
        os.path.join(lru_dict.storage_path, "key1")
    ), "key1 should be removed from the disk"


def test_lrudict_keys_method(lru_dict):
    lru_dict["key1"] = "value1"
    lru_dict["key2"] = "value2"
    lru_dict["key3"] = "value3"  # key1 should go to the database
    keys = lru_dict.keys()
    # Ensure keys from both memory and database are returned
    assert set(keys) == {
        "key1",
        "key2",
        "key3",
    }, "All keys should be returned, including those in the database"


def test_lrudict_load_from_dict(lru_dict):
    lru_dict.load_from_dict({"key1": "value1", "key2": "value2"})
    assert lru_dict["key1"] == "value1", "key1 should be in the cache"
    assert lru_dict["key2"] == "value2", "key2 should be in the cache"


####################################################################################################
### Test for LRUDBDict
####################################################################################################


@pytest.fixture
def lrudb_dict(tmp_path):
    """Fixture to create an LRUDBDict instance and clean up after tests."""
    db_path = str(tmp_path / "lrudbdict_cache.db")
    cache = LRUDBDict(max_in_memory=2, storage_path=db_path)
    yield cache
    cache.conn.close()


def test_lrudbdict_initialization(lrudb_dict):
    assert lrudb_dict.max_in_memory == 2
    # Ensure the SQLite table exists
    lrudb_dict.cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='data';"
    )
    assert lrudb_dict.cursor.fetchone() is not None


def test_lrudbdict_set_and_get_item(lrudb_dict):
    lrudb_dict["key1"] = "value1"
    assert lrudb_dict["key1"] == "value1"


def test_lrudbdict_eviction_to_db(lrudb_dict):
    lrudb_dict["key1"] = "value1"
    lrudb_dict["key2"] = "value2"
    lrudb_dict["key3"] = "value3"  # key1 should go to the database
    assert "key1" not in lrudb_dict.memory, "key1 should be evicted to the database"
    lrudb_dict.cursor.execute("SELECT value FROM data WHERE key=?", ("key1",))
    result = lrudb_dict.cursor.fetchone()
    assert result is not None
    assert (
        result[0] == '"value1"'
    ), "value1 should be serialized and stored in the database"


def test_lrudbdict_retrieval_from_db(lrudb_dict):
    lrudb_dict["key1"] = "value1"
    lrudb_dict["key2"] = "value2"
    lrudb_dict["key3"] = "value3"  # key1 should go to the database
    assert lrudb_dict["key1"] == "value1", "key1 should be brought back to memory"


def test_lrudbdict_delete_item(lrudb_dict):
    lrudb_dict["key1"] = "value1"
    del lrudb_dict["key1"]
    assert "key1" not in lrudb_dict.memory, "key1 should be removed from memory"
    lrudb_dict.cursor.execute("SELECT value FROM data WHERE key=?", ("key1",))
    result = lrudb_dict.cursor.fetchone()
    assert result is None, "key1 should be removed from the database"


def test_lrudbdict_keys_method(lrudb_dict):
    lrudb_dict["key1"] = "value1"
    lrudb_dict["key2"] = "value2"
    lrudb_dict["key3"] = "value3"  # key1 should go to the database
    keys = lrudb_dict.keys()
    # Ensure keys from both memory and database are returned
    assert set(keys) == {
        "key1",
        "key2",
        "key3",
    }, "All keys should be returned, including those in the database"


def test_lrudbdict_load_from_dict(lrudb_dict):
    lrudb_dict.load_from_dict(
        {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }
    )
    assert lrudb_dict["key1"] == "value1", "key1 should be in the cache"
    assert lrudb_dict["key2"] == "value2", "key2 should be in the cache"
