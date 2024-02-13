EffiDict
========

Introduction
------------

EffiDict is an efficient and fast Python package providing enhanced dictionary-like data structures with advanced caching capabilities. It's designed for applications requiring speedy retrieval and persistent storage of key-value pairs.

Features
--------

- **LRU Caching:** Implements Least Recently Used caching for optimal data retrieval efficiency.
- **Persistent Storage:** Offers support for disk-based storage using SQLite, ideal for long-term data retention.
- **Versatile:** Easily adaptable to various data types, enhancing flexibility across different use cases.

Getting Started
---------------

Installation
^^^^^^^^^^^^

You can install EffiDict via pip:

.. code-block:: shell

    pip install effidict

Usage Examples
^^^^^^^^^^^^^^

.. code-block:: python

    from effiDict import LRUDBDict, LRUDict, DBDict

    # LRUDict
    cache_dict = LRUDict(max_in_memory=100, storage_path="cache")
    cache_dict['key'] = 'value'

    # LRUDBDict
    db_cache_dict = LRUDBDict(max_in_memory=100, storage_path="cache.db")
    db_cache_dict['key'] = 'value'

    # DBDict
    db_dict = DBDict(storage_path="cache.db")
    db_dict['key'] = 'value'

API Documentation
-----------------

For detailed information on all classes, functions, and modules, refer to the API documentation.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   effidict

License
-------

EffiDict is released under the MIT License, offering wide flexibility for private, commercial, or academic use.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

