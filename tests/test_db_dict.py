import os

import pytest

from effidict import DBDict


@pytest.fixture
def db():
    """
    Create a DBDict instance for testing.
    """
    storage_path = "test_db.sqlite"
    _db = DBDict(storage_path)
    yield _db
    _db.conn.close()
    os.remove(storage_path)


def test_set_and_get_item(db):
    """
    Test setting and getting an item in the cache.
    """
    db["test_key"] = {"a": 1, "b": 2}
    assert db["test_key"] == {"a": 1, "b": 2}


def test_delete_item(db):
    """
    Test deleting an item from the cache.
    """
    db["test_key"] = {"a": 1}
    del db["test_key"]
    with pytest.raises(KeyError):
        _ = db["test_key"]


def test_iteration_and_length(db):
    """
    Test iterating over the cache and getting its length.
    """
    db["key1"] = {"a": 1}
    db["key2"] = {"b": 2}
    keys = list(db)
    assert len(keys) == 2 and "key1" in keys and "key2" in keys
    assert len(db) == 2


def test_keys_method(db):
    """
    Test the keys method.
    """
    db["key1"] = {"a": 1}
    db["key2"] = {"b": 2}
    keys = db.keys()
    assert len(keys) == 2 and "key1" in keys and "key2" in keys
