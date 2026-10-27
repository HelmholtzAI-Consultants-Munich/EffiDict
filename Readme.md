[![test](https://github.com/HelmholtzAI-Consultants-Munich/EffiDict/actions/workflows/test.yml/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/EffiDict/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/effidict/badge/?version=latest)](https://effidict.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/EffiDict/branch/main/graph/badge.svg)](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/EffiDict)


# EffiDict
EffiDict is a fast, dictionary-like cache with pluggable on-disk backends and multiple replacement strategies. It gives you a familiar dict API, automatic spilling to disk, and control over how entries are evicted from memory.

## Features
- Multiple eviction policies: LRU, MRU, FIFO, LIFO, LFU, MFU, Random
- Pluggable persistence backends:
  - SQLite (single file, JSON-encoded values)
  - JSON folder (one JSON file per key)
  - Pickle folder (one pickle file per key)
  - HDF5 file (requires optional dependencies; best for numeric arrays)
- Dict-like API: get/set, delete, `in`, `len`, `keys/items/values`, `pop`, `clear`, and iteration

## Installation

Install the core package:

```bash
pip install effidict
```

Optional dependencies for the HDF5 backend:

```bash
pip install effidict[hdf5]
```

For development (tests):

```bash
pip install -e .[dev]
```

## Quick start

Create an `EffiDict` using the SQLite backend and LRU replacement strategy:

```python
from effidict import EffiDict, SqliteBackend, LRUPolicy

# A unique suffix is appended to avoid collisions, so a simple prefix is fine
backend = SqliteBackend("cache.sqlite_")
store = EffiDict(disk_backend=backend, policy=LRUPolicy(), max_in_memory=2)

store["a"] = 1
store["b"] = 2
store["c"] = 3  # spills least-recently-used to disk

assert store["a"] == 1  # transparently loads from disk if evicted
```

Use as a context manager to ensure cleanup:

```python
from effidict import EffiDict, JSONBackend, MRUPolicy

backend = JSONBackend("cache_dir_")
with EffiDict(disk_backend=backend, policy=MRUPolicy(), max_in_memory=100) as d:
	d["k"] = "v"
	print(d["k"])  # -> "v"
# resources are cleaned up on exit
```

You can also call `store.close()` explicitly when done.

## Backends and strategies

Construct the backend and pick an eviction policy, then pass both to `EffiDict` with
the number of entries to keep in memory:

```python
from effidict import (
	EffiDict,
	SqliteBackend, JSONBackend, PickleBackend, Hdf5Backend,
	LRUPolicy, MRUPolicy, FIFOPolicy, LIFOPolicy,
	LFUPolicy, MFUPolicy, RandomPolicy,
)

backend = SqliteBackend("cache.sqlite_")                  # or JSONBackend("cache_dir_"), PickleBackend("cache_dir_"), Hdf5Backend("cache.h5_")
store = EffiDict(disk_backend=backend, policy=LRUPolicy(), max_in_memory=100)
```

Notes:
- SQLite and JSON backends store values as JSON; values must be JSON-serializable.
- Pickle backend can store any picklable Python object.
- HDF5 backend requires `h5py` and is best suited for numeric arrays; basic Python types are also supported.
- The `storage_path` you provide is used as a prefix; a unique suffix is appended automatically per instance.

## Cleanup and lifecycle

- Prefer `with EffiDict(...) as d:` or call `close()` to release disk resources promptly.
- Backends attempt to clean up files on `close()`/`destroy()`. Automatic cleanup during object destruction is best-effort only.

## License
Licensed under the MIT License.
