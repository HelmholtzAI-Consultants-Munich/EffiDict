import pytest
import tempfile
import shutil
import os
import h5py

from effidict import (
    EffiDict,
    SqliteBackend, PickleBackend, Hdf5Backend, JSONBackend,
    RandomReplacement, FIFOReplacement, LIFOReplacement,
    LRUReplacement, MRUReplacement, LFUReplacement, MFUReplacement
)

BACKEND_CLASSES = [SqliteBackend, PickleBackend, JSONBackend]
if h5py:
    BACKEND_CLASSES.append(Hdf5Backend)

STRATEGY_CLASSES = [
    RandomReplacement, FIFOReplacement, LIFOReplacement,
    LRUReplacement, MRUReplacement, LFUReplacement, MFUReplacement,
]

TEST_COMBINATIONS = [
    pytest.param(backend, strategy, id=f"{backend.__name__}-{strategy.__name__}")
    for backend in BACKEND_CLASSES
    for strategy in STRATEGY_CLASSES
]

@pytest.fixture
def temp_dir_path():
    """Create a temporary directory for tests to use."""
    test_dir = tempfile.mkdtemp()
    yield test_dir  
    shutil.rmtree(test_dir) 


class TestEffiDictCRUD:

    @pytest.mark.parametrize("backend_cls, strategy_cls", TEST_COMBINATIONS)
    def test_within_capacity(self, temp_dir_path, backend_cls, strategy_cls):
        """
        Tests CRUD operations when the number of items is less than max_in_memory.
        """
        max_items = 5
        storage_path = os.path.join(temp_dir_path, backend_cls.__name__)
        
        disk_backend = backend_cls(storage_path)
        replacement_strategy = strategy_cls(disk_backend=disk_backend, max_in_memory=max_items)
        
        with EffiDict(
            max_in_memory=max_items,
            disk_backend=disk_backend,
            replacement_strategy=replacement_strategy
        ) as effidict:
            # CREATE
            effidict['a'] = 1
            effidict['b'] = {'data': [2, 3]}
            assert len(effidict) == 2
            assert len(effidict.memory) == 2
            assert len(effidict.disk_backend.keys()) == 0

            # READ
            assert effidict['a'] == 1
            assert effidict['b'] == {'data': [2, 3]}
            assert 'a' in effidict

            # UPDATE
            effidict['a'] = 100
            assert effidict['a'] == 100

            # DELETE
            del effidict['a']
            assert len(effidict) == 1
            value = effidict.pop('b')
            assert value == {'data': [2, 3]}
            assert len(effidict) == 0

    @pytest.mark.parametrize("backend_cls, strategy_cls", TEST_COMBINATIONS)
    def test_disk_is_used(self, temp_dir_path, backend_cls, strategy_cls):
        """
        Tests CRUD operations when items exceed max_in_memory, forcing disk usage.
        """
        max_items = 3
        num_to_add = 5
        storage_path = os.path.join(temp_dir_path, backend_cls.__name__)
        
        disk_backend = backend_cls(storage_path)
        replacement_strategy = strategy_cls(disk_backend=disk_backend, max_in_memory=max_items)

        with EffiDict(
            max_in_memory=max_items,
            disk_backend=disk_backend,
            replacement_strategy=replacement_strategy
        ) as effidict:
            items = {f'key{i}': f'value{i}' for i in range(num_to_add)}
            for key, value in items.items():
                effidict[key] = value

            assert len(effidict) == num_to_add
            assert len(effidict.memory) == max_items
            assert len(effidict.disk_backend.keys()) == num_to_add - max_items

            # READ from disk
            disk_keys = set(effidict.disk_backend.keys())
            key_on_disk = list(disk_keys)[0]
            assert effidict[key_on_disk] == items[key_on_disk]
            
            # UPDATE on disk
            effidict[key_on_disk] = "updated_value"
            assert effidict[key_on_disk] == "updated_value"

            # DELETE from disk
            del effidict[key_on_disk]
            assert len(effidict) == num_to_add - 1
            assert key_on_disk not in effidict

            # Test deleting a non-existent key
            with pytest.raises(KeyError):
                del effidict['non_existent_key']