[![test](https://github.com/HelmholtzAI-Consultants-Munich/EffiDict/actions/workflows/test.yml/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/Effidict/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/effidict/badge/?version=latest)](https://effidict.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/EffiDict/branch/main/graph/badge.svg)](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/EffiDict)


# EffiDict
EffiDict is an efficient and fast Python package providing enhanced dictionary-like data structures with advanced caching capabilities. It's perfect for applications needing speedy retrieval and persistent key-value pair storage.

## Features
**LRU Caching:** Implements Least Recently Used caching for optimal data access.

**Persistent Storage:** Supports disk storage with SQLite.

**Versatile:** Adaptable for various data types.

## Installation
You can install EffiDict via pip:

```
pip install effidict
```

## Usage
Importing the package
```
from effidict import LRUDBDict, LRUDict, DBDict
```

Using `LRUDict` for persistent storage on `pickle` files
```
cache_dict = LRUDict(max_in_memory=100, storage_path="cache")
cache_dict['key'] = 'value'
```

Using `LRUDBDict` for persistent storage on `sqlite`
```
db_cache_dict = LRUDBDict(max_in_memory=100, storage_path="cache.db")
db_cache_dict['key'] = 'value'
```
Standard `DBDict` (`sqlite` only)
```
db_dict = DBDict(storage_path="cache.db")
db_dict['key'] = 'value'
```

## License
Licensed under the MIT License.
