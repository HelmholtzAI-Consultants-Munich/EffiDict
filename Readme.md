# EffiDict
EffiDict is an efficient and fast Python package providing enhanced dictionary-like data structures with advanced caching capabilities. It's perfect for applications needing speedy retrieval and persistent key-value pair storage.

# Features
**LRU Caching:** Implements Least Recently Used caching for optimal data access.
**Persistent Storage:** Supports disk storage with SQLite.
**Versatile:** Adaptable for various data types.

# Installation
You can install EffiDict via pip:

```
pip install effidict
```

# Usage
```
from effidict import LRUDBDict, LRUDict, DBDict

# Using LRUDict
cache_dict = LRUDict(max_in_memory=100, storage_path="cache")
cache_dict['key'] = 'value'

# Using LRUDBDict for persistent storage
db_cache_dict = LRUDBDict(max_in_memory=100, storage_path="cache.db", batch_size=10)
db_cache_dict['key'] = 'value'

# Standard DBDict
db_dict = DBDict(storage_path="cache.db")
db_dict['key'] = 'value'
```

# License
Licensed under the MIT License.