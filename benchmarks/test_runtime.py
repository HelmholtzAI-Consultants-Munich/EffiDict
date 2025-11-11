"""
Runtime benchmarks for read, write, delete, and modification operations.

Measures total time for operations across increasing dataset sizes,
comparing EffiDict, dict, and shelve.
"""

import statistics
import time
import gc
from typing import List, Callable


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


def measure_runtime_with_repetitions(
    operation: str,
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

    times = []

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

        start_time = time.perf_counter()
        benchmark_func(structure, **benchmark_kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

        del structure
        gc.collect()

    result = BenchmarkResult(
        operation=operation,
        dataset_size=dataset_size,
        dataset_value_size=dataset_value_size,
        implementation=implementation,
        time_seconds=statistics.mean(times),
        time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
        memory_mb=None,
        memory_std=None,
        metadata={
            "repetitions": repetitions,
            "individual_times": times,
        },
    )
    return result


def _create_operation_functions(operation: str):
    """
    Create operation-specific functions for dict, shelve, and effidict.

    Args:
        operation: Operation name ('write', 'read', 'delete', 'modify')

    Returns:
        Tuple of (dict_func, shelve_func, effidict_func)
    """
    if operation == "write":

        def dict_func(d, dataset):
            for key, value in dataset.items():
                d[key] = value

        def shelve_func(shelve_path, dataset):
            with open_shelve(shelve_path, "c") as db:
                for key, value in dataset.items():
                    db[key] = value

        def effidict_func(ed, dataset):
            for key, value in dataset.items():
                ed[key] = value

        return dict_func, shelve_func, effidict_func

    elif operation == "read":

        def dict_func(d, keys):
            for key in keys:
                _ = d[key]

        def shelve_func(shelve_path, keys):
            with open_shelve(shelve_path, "r") as db:
                for key in keys:
                    _ = db[key]

        def effidict_func(ed, keys):
            for key in keys:
                _ = ed[key]

        return dict_func, shelve_func, effidict_func

    elif operation == "delete":

        def dict_func(d, keys):
            for key in keys:
                if key in d:
                    del d[key]

        def shelve_func(shelve_path, keys):
            with open_shelve(shelve_path, "w") as db:
                for key in keys:
                    if key in db:
                        del db[key]

        def effidict_func(ed, keys):
            for key in keys:
                if key in ed:
                    del ed[key]

        return dict_func, shelve_func, effidict_func

    elif operation == "modify":

        def dict_func(d, keys):
            for key in keys:
                if key in d:
                    d[key] = {"modified": True, "original": d[key]}

        def shelve_func(shelve_path, keys):
            with open_shelve(shelve_path, "w") as db:
                for key in keys:
                    if key in db:
                        db[key] = {"modified": True, "original": db[key]}

        def effidict_func(ed, keys):
            for key in keys:
                if key in ed:
                    ed[key] = {"modified": True, "original": ed[key]}

        return dict_func, shelve_func, effidict_func

    else:
        raise ValueError(f"Unknown operation: {operation}")


def benchmark_operations(
    operation: str,
    dataset_sizes: List[int],
    dataset_value_sizes: List[int],
    repetitions: int,
    effidict_backends: List[str],
    effidict_strategies: List[str],
) -> List[BenchmarkResult]:
    """
    Generic function to benchmark any operation (write, read, delete, modify).

    Args:
        operation: Operation name ('write', 'read', 'delete', 'modify')
        dataset_sizes: List of dataset sizes to test
        dataset_value_sizes: List of value sizes in MB
        repetitions: Number of repetitions for each benchmark
        effidict_backends: List of backend names for EffiDict
        effidict_strategies: List of strategy names for EffiDict

    Returns:
        List of BenchmarkResult objects
    """
    dict_func, shelve_func, effidict_func = _create_operation_functions(operation)
    results = []

    # Determine if operation needs dataset or keys
    needs_keys = operation != "write"

    for dataset_value_size in dataset_value_sizes:
        max_in_memory = get_max_in_memory_for_value_size(dataset_value_size)

        for dataset_size in dataset_sizes:
            print(
                f"Running {operation} operation benchmarks on {dataset_value_size} MB for {dataset_size} items for {repetitions} repetitions..."
            )

            dataset = generate_dataset(dataset_size, dataset_value_size)

            # Prepare benchmark kwargs and populate flag based on operation type
            if needs_keys:
                keys = list(dataset.keys())
                benchmark_kwargs = {"keys": keys}
                populate = True
            else:
                benchmark_kwargs = {"dataset": dataset}
                populate = False

            #########################################################
            # Benchmark standard dict
            #########################################################

            result = measure_runtime_with_repetitions(
                operation=operation,
                dataset_size=dataset_size,
                dataset_value_size=dataset_value_size,
                implementation="dict",
                benchmark_func=dict_func,
                benchmark_kwargs=benchmark_kwargs,
                populate=populate,
                dataset=dataset if populate else None,
                repetitions=repetitions,
            )
            results.append(result)

            #########################################################
            # Benchmark shelve
            #########################################################

            result = measure_runtime_with_repetitions(
                operation=operation,
                dataset_size=dataset_size,
                dataset_value_size=dataset_value_size,
                implementation="shelve",
                benchmark_func=shelve_func,
                benchmark_kwargs=benchmark_kwargs,
                populate=populate,
                dataset=dataset if populate else None,
                repetitions=repetitions,
            )
            results.append(result)

            #########################################################
            # Benchmark EffiDict with different backend/strategy combinations
            #########################################################

            for backend_name in effidict_backends:
                for strategy_name in effidict_strategies:
                    implementation_name = f"effidict_{strategy_name}_{backend_name}"

                    result = measure_runtime_with_repetitions(
                        operation=operation,
                        dataset_size=dataset_size,
                        dataset_value_size=dataset_value_size,
                        implementation=implementation_name,
                        benchmark_func=effidict_func,
                        benchmark_kwargs=benchmark_kwargs,
                        populate=populate,
                        dataset=dataset if populate else None,
                        max_in_memory=max_in_memory,
                        backend_name=backend_name,
                        strategy_name=strategy_name,
                        repetitions=repetitions,
                    )
                    results.append(result)

    return results
