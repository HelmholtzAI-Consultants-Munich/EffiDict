"""
Memory usage benchmarks.

Measures RAM usage during mixed operations (read, write, modify, delete)
across increasing dataset sizes, comparing EffiDict, dict, and shelve.
"""

import statistics
import gc
import time
from typing import List, Callable
from memory_profiler import memory_usage

from benchmark_framework import BenchmarkResult
from fixtures import (
    open_shelve,
    DataStructureFactory,
    get_max_in_memory_for_value_size,
)
from datasets import generate_dataset


def _create_structure_with_factory(
    implementation: str,
    populate: bool,
    dataset: dict = None,
    max_in_memory: int = None,
    backend_name: str = None,
    strategy_name: str = None,
):
    """
    Create a data structure using DataStructureFactory.

    Args:
        implementation: 'dict', 'shelve', or 'effidict'
        populate: Whether to populate with dataset
        dataset: Dataset to populate with
        max_in_memory: Max items in memory (for EffiDict)
        backend_name: Backend name (for EffiDict)
        strategy_name: Strategy name (for EffiDict)

    Returns:
        Created data structure
    """
    return DataStructureFactory.create(
        implementation=implementation,
        populate=populate,
        dataset=dataset,
        max_in_memory=max_in_memory,
        backend_name=backend_name,
        strategy_name=strategy_name,
    )


def _create_mixed_operations_function():
    """
    Create a function that performs a mix of read, write, modify, and delete operations.

    Returns:
        Tuple of (dict_func, shelve_func, effidict_func) that perform mixed operations
    """

    def dict_func(d, keys, dataset):
        """Perform mixed operations on a dict."""
        # Write some new items
        for i, (key, value) in enumerate(list(dataset.items())[: len(keys) // 4]):
            d[f"new_key_{i}"] = value

        # Read some items
        for key in keys[: len(keys) // 4]:
            if key in d:
                _ = d[key]

        # Modify some items
        for key in keys[len(keys) // 4 : len(keys) // 2]:
            if key in d:
                d[key] = {"modified": True, "original": d[key]}

        # Delete some items
        for key in keys[len(keys) // 2 : 3 * len(keys) // 4]:
            if key in d:
                del d[key]

    def shelve_func(shelve_path, keys, dataset):
        """Perform mixed operations on a shelve database."""
        with open_shelve(shelve_path, "w") as db:
            # Write some new items
            for i, (key, value) in enumerate(list(dataset.items())[: len(keys) // 4]):
                db[f"new_key_{i}"] = value

            # Read some items
            for key in keys[: len(keys) // 4]:
                if key in db:
                    _ = db[key]

            # Modify some items
            for key in keys[len(keys) // 4 : len(keys) // 2]:
                if key in db:
                    db[key] = {"modified": True, "original": db[key]}

            # Delete some items
            for key in keys[len(keys) // 2 : 3 * len(keys) // 4]:
                if key in db:
                    del db[key]

    def effidict_func(ed, keys, dataset):
        """Perform mixed operations on an EffiDict."""
        # Write some new items
        for i, (key, value) in enumerate(list(dataset.items())[: len(keys) // 4]):
            ed[f"new_key_{i}"] = value

        # Read some items
        for key in keys[: len(keys) // 4]:
            if key in ed:
                _ = ed[key]

        # Modify some items
        for key in keys[len(keys) // 4 : len(keys) // 2]:
            if key in ed:
                ed[key] = {"modified": True, "original": ed[key]}

        # Delete some items
        for key in keys[len(keys) // 2 : 3 * len(keys) // 4]:
            if key in ed:
                del ed[key]

    return dict_func, shelve_func, effidict_func


def measure_memory_with_repetitions(
    dataset_size: int,
    dataset_value_size: int,
    implementation: str,
    repetitions: int,
    benchmark_func: Callable,
    benchmark_kwargs: dict,
    populate: bool,
    dataset: dict = None,
    max_in_memory: int = None,
    backend_name: str = None,
    strategy_name: str = None,
) -> BenchmarkResult:
    """
    Measure memory usage for a given implementation with repetitions.

    Args:
        dataset_size: Size of the dataset
        dataset_value_size: Size of each value in MB
        implementation: Implementation name ('dict', 'shelve', 'effidict')
        repetitions: Number of times to repeat the measurement
        benchmark_func: Function to benchmark the data structure
        benchmark_kwargs: Keyword arguments to pass to benchmark_func
        populate: Whether to populate the structure initially
        dataset: Dataset to populate with
        max_in_memory: Max items in memory (for EffiDict)
        backend_name: Backend name (for EffiDict)
        strategy_name: Strategy name (for EffiDict)

    Returns:
        BenchmarkResult with memory information including statistics
    """
    memories = []

    for repetition in range(repetitions):
        structure = _create_structure_with_factory(
            implementation=implementation,
            populate=populate,
            dataset=dataset,
            max_in_memory=max_in_memory,
            backend_name=backend_name,
            strategy_name=strategy_name,
        )
        time.sleep(1)

        mem_samples = memory_usage((benchmark_func, (structure,), benchmark_kwargs), interval=0.01)
        memory_used = max(mem_samples)
        memories.append(memory_used)

        del structure
        gc.collect()

    result = BenchmarkResult(
        operation="memory",
        dataset_size=dataset_size,
        dataset_value_size=dataset_value_size,
        implementation=implementation,
        time_seconds=0.0,
        time_std=None,
        memory_mb=statistics.mean(memories),
        memory_std=statistics.stdev(memories) if len(memories) > 1 else 0.0,
        metadata={
            "repetitions": repetitions,
            "individual_memories": memories,
        },
    )
    return result


def benchmark_memory_usage(
    dataset_sizes: List[int],
    dataset_value_sizes: List[float],
    repetitions: int,
    effidict_backends: List[str],
    effidict_strategies: List[str],
) -> List[BenchmarkResult]:
    """
    Benchmark memory usage for mixed operations across different dataset sizes and value sizes.

    Args:
        dataset_sizes: List of dataset sizes to test
        dataset_value_sizes: List of value sizes in MB
        repetitions: Number of repetitions for each benchmark
        effidict_backends: List of backend names for EffiDict
        effidict_strategies: List of strategy names for EffiDict

    Returns:
        List of BenchmarkResult objects
    """
    dict_func, shelve_func, effidict_func = _create_mixed_operations_function()
    results = []

    for dataset_value_size in dataset_value_sizes:
        max_in_memory = get_max_in_memory_for_value_size(dataset_value_size)

        for dataset_size in dataset_sizes:
            print(
                f"Running memory benchmarks on {dataset_value_size} MB for {dataset_size} items for {repetitions} repetitions..."
            )

            dataset = generate_dataset(dataset_size, dataset_value_size)
            keys = list(dataset.keys())

            benchmark_kwargs = {"keys": keys, "dataset": dataset}

            #########################################################
            # Benchmark standard dict
            #########################################################

            result = measure_memory_with_repetitions(
                dataset_size=dataset_size,
                dataset_value_size=dataset_value_size,
                implementation="dict",
                benchmark_func=dict_func,
                benchmark_kwargs=benchmark_kwargs,
                populate=True,
                dataset=dataset,
                repetitions=repetitions,
            )
            results.append(result)

            #########################################################
            # Benchmark shelve
            #########################################################

            result = measure_memory_with_repetitions(
                dataset_size=dataset_size,
                dataset_value_size=dataset_value_size,
                implementation="shelve",
                benchmark_func=shelve_func,
                benchmark_kwargs=benchmark_kwargs,
                populate=True,
                dataset=dataset,
                repetitions=repetitions,
            )
            results.append(result)

            #########################################################
            # Benchmark EffiDict with different backend/strategy combinations
            #########################################################

            for backend_name in effidict_backends:
                for strategy_name in effidict_strategies:
                    implementation_name = f"effidict_{strategy_name}_{backend_name}"

                    result = measure_memory_with_repetitions(
                        dataset_size=dataset_size,
                        dataset_value_size=dataset_value_size,
                        implementation=implementation_name,
                        benchmark_func=effidict_func,
                        benchmark_kwargs=benchmark_kwargs,
                        populate=True,
                        dataset=dataset,
                        max_in_memory=max_in_memory,
                        backend_name=backend_name,
                        strategy_name=strategy_name,
                        repetitions=repetitions,
                    )
                    results.append(result)

    return results
