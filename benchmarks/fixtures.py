"""
Fixture utilities for benchmark data structure setup.

Provides functions and classes for creating and managing benchmark data structures
(dict, shelve, EffiDict) with a unified interface.
"""

import tempfile
from contextlib import contextmanager
from typing import Dict, Any, Union, Optional
from effidict import EffiDict


class DataStructureFactory:
    """Factory for creating benchmark data structures with unified interface."""

    @staticmethod
    def create(
        implementation: str,
        populate: bool = True,
        dataset: Dict[str, Any] = None,
        max_in_memory: Optional[int] = None,
        backend_name: str = None,
        strategy_name: str = None,
    ) -> Union[Dict[str, Any], str, EffiDict]:
        """
        Create a data structure for benchmarking.

        Args:
            implementation: 'dict', 'shelve', or 'effidict'
            populate: Whether to populate with dataset
            dataset: Dataset to populate with
            max_in_memory: Max items in memory (for EffiDict)
            backend_name: Backend name (for EffiDict)
            strategy_name: Strategy name (for EffiDict)

        Returns:
            Created data structure (dict, shelve path string, or EffiDict)
        """
        if implementation == "dict":
            return DataStructureFactory._create_dict(populate, dataset)
        elif implementation == "shelve":
            return DataStructureFactory._create_shelve(populate, dataset)
        elif implementation.startswith("effidict"):
            return DataStructureFactory._create_effidict(
                populate,
                dataset,
                max_in_memory,
                backend_name,
                strategy_name,
            )
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

    @staticmethod
    def _create_dict(populate: bool, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standard Python dict.

        Args:
            populate: Whether to populate the dict with the dataset
            dataset: Dataset to populate with (if populate is True)

        Returns:
            Python dictionary, optionally populated
        """
        d = {}
        if populate:
            for key, value in dataset.items():
                d[key] = value
        return d

    @staticmethod
    def _create_shelve(populate: bool, dataset: Dict[str, Any]) -> str:
        """
        Create a shelve database.

        Args:
            populate: Whether to populate the shelve with the dataset
            dataset: Dataset to populate with (if populate is True)

        Returns:
            Path to the created shelve database file
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            shelve_path = tmp.name
        if populate:
            with open_shelve(shelve_path, "c") as db:
                for key, value in dataset.items():
                    db[key] = value
        return shelve_path

    @staticmethod
    def _create_effidict(
        populate: bool, dataset: Dict[str, Any], max_in_memory: int, backend_name: str, strategy_name: str
    ) -> EffiDict:
        """
        Create an EffiDict instance.

        Args:
            populate: Whether to populate the EffiDict with the dataset
            dataset: Dataset to populate with (if populate is True)
            max_in_memory: Maximum number of items to keep in memory
            backend_name: Name of the backend ('sqlite', 'json', 'pickle', 'hdf5')
            strategy_name: Name of the replacement strategy
                ('lru', 'mru', 'fifo', 'lifo', 'lfu', 'mfu', 'random')

        Returns:
            Configured EffiDict instance

        Raises:
            ValueError: If backend_name or strategy_name is unknown
        """
        BackendClass = get_backend_class(backend_name)
        StrategyClass = get_strategy_class(strategy_name)
        if BackendClass is None:
            raise ValueError(f"Unknown backend: {backend_name}")
        if StrategyClass is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Create temporary storage
        suffix = f"_{BackendClass.__name__}_{StrategyClass.__name__}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            effidict_path = tmp.name

        backend = BackendClass(effidict_path)
        strategy = StrategyClass(disk_backend=backend, max_in_memory=max_in_memory)

        ed = EffiDict(max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy)
        if populate:
            for key, value in dataset.items():
                ed[key] = value
        return ed


@contextmanager
def open_shelve(filename: str, flag: str = "c"):
    """
    Open a shelve database using dbm.dumb backend (works on all platforms).

    Args:
        filename: Path to the shelve database file
        flag: Open flag ('c' for create, 'r' for read, 'w' for write)

    Yields:
        Shelve database object (context manager)
    """
    import shelve
    import dbm.dumb

    # Use dbm.dumb explicitly to avoid platform-specific issues
    db = dbm.dumb.open(filename, flag)
    shelf = shelve.Shelf(db)
    try:
        yield shelf
    finally:
        shelf.close()


def get_backend_class(backend_name: str):
    """
    Get backend class from name.

    Args:
        backend_name: Name of the backend ('sqlite', 'json', 'pickle', 'hdf5')

    Returns:
        Backend class corresponding to the name, or None if not found
    """
    from effidict import SqliteBackend, JSONBackend, PickleBackend, Hdf5Backend

    backend_map = {"sqlite": SqliteBackend, "json": JSONBackend, "pickle": PickleBackend, "hdf5": Hdf5Backend}

    return backend_map.get(backend_name.lower())


def get_strategy_class(strategy_name: str):
    """
    Get strategy class from name.

    Args:
        strategy_name: Name of the replacement strategy
            ('lru', 'mru', 'fifo', 'lifo', 'lfu', 'mfu', 'random')

    Returns:
        Strategy class corresponding to the name, or None if not found
    """
    from effidict import (
        LRUReplacement,
        MRUReplacement,
        FIFOReplacement,
        LIFOReplacement,
        LFUReplacement,
        MFUReplacement,
        RandomReplacement,
    )

    strategy_map = {
        "lru": LRUReplacement,
        "mru": MRUReplacement,
        "fifo": FIFOReplacement,
        "lifo": LIFOReplacement,
        "lfu": LFUReplacement,
        "mfu": MFUReplacement,
        "random": RandomReplacement,
    }

    return strategy_map.get(strategy_name.lower())


def get_max_in_memory_for_value_size(value_size: float) -> int:
    """
    Determine maximum in-memory items for EffiDict based on value size.

    Args:
        value_size: Value size in MB

    Returns:
        Maximum number of items to keep in memory
    """
    size_bytes = max(10, int(value_size * 1024 * 1024))
    if size_bytes < 1024:
        return 10000
    elif size_bytes < 1024 * 1024:  # 1MB
        return 100
    else:  # 10MB
        return 10
