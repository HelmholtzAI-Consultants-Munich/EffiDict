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

# The path is used exactly as given, so the store is at "cache.sqlite"
backend = SqliteBackend.open_or_create("cache.sqlite")
store = EffiDict(disk_backend=backend, policy=LRUPolicy(), max_in_memory=2)

store["a"] = 1
store["b"] = 2
store["c"] = 3  # spills least-recently-used to disk

assert store["a"] == 1  # transparently loads from disk if evicted
```

Use as a context manager to ensure cleanup:

```python
from effidict import EffiDict, JSONBackend, MRUPolicy

backend = JSONBackend.open_or_create("cache_dir")
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

backend = SqliteBackend.open_or_create("cache.sqlite")    # or JSONBackend, PickleBackend, Hdf5Backend
store = EffiDict(disk_backend=backend, policy=LRUPolicy(), max_in_memory=100)
```

Four ways to get a backend, depending on what you expect to be there:

```python
from effidict import SqliteBackend

SqliteBackend.create("cache.sqlite")           # must not exist yet
SqliteBackend.open("cache.sqlite")             # must exist already
SqliteBackend.open_or_create("cache.sqlite")   # either
SqliteBackend.temporary()                      # a scratch store at a fresh path
```

`create` and `open` refuse the case they are not for, so a typo in a path is an
error rather than a silently empty store.

Notes:
- SQLite and JSON backends store values as JSON; values must be JSON-serializable.
- Pickle backend can store any picklable Python object.
- HDF5 backend requires `h5py` and is best suited for numeric arrays; basic Python types are also supported.
- The `storage_path` you provide is the storage location, used exactly as given. Two
  backends built on the same path address the same store; use `temporary()` for a scratch
  store at a path nothing else will be handed.

## Cleanup and lifecycle

- `close()` on an `EffiDict` currently **destroys** the storage rather than just releasing
  it, so a store you want to keep should not be closed. Separating the two is in progress;
  until then, use `backend.close()` to release handles and leave the data in place.
- Automatic cleanup during object destruction is best-effort only.

## License
Licensed under the MIT License.
