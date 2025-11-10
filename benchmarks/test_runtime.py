"""
Runtime benchmarks for read, write, delete, and modification operations.

Measures total time for operations across increasing dataset sizes,
comparing EffiDict, dict, and shelve.
"""

import tempfile
import os
from typing import Dict, Any, List
from pathlib import Path
import sys

from benchmark_framework import BenchmarkRunner, BenchmarkResult, open_shelve
from datasets import generate_dataset
from config import DEFAULT_CONFIG, get_backend_class, get_strategy_class, get_max_in_memory_for_value_size


def benchmark_write_operations(
    runner: BenchmarkRunner,
    dataset_sizes: List[int],
    value_type: str = "small_uniform",
    value_size: Any = None,
    config: Any = None,
    repetitions: int = 1,
) -> List[BenchmarkResult]:
    """
    Benchmark write operations (insertions) for different dataset sizes.

    Args:
        runner: Benchmark runner instance
        dataset_sizes: List of dataset sizes to test
        value_type: Type of values to generate
        value_size: Size parameter for values
        config: Benchmark configuration (uses DEFAULT_CONFIG if None)

    Returns:
        List of benchmark results
    """
    if config is None:
        config = DEFAULT_CONFIG

    results = []

    for dataset_size in dataset_sizes:
        # Generate dataset
        dataset = generate_dataset(dataset_size, value_type, value_size)

        # Determine max_in_memory based on value type and size
        max_in_memory = get_max_in_memory_for_value_size(value_type, value_size)

        # Benchmark standard dict
        def write_dict():
            d = {}
            for key, value in dataset.items():
                d[key] = value
            return d

        result = runner.run_benchmark("write", dataset_size, "dict", write_dict, repetitions=repetitions)
        results.append(result)

        # Benchmark shelve
        with tempfile.NamedTemporaryFile(delete=False, suffix="") as tmp:
            shelve_path = tmp.name

        try:

            def write_shelve():
                with open_shelve(shelve_path, "c") as db:
                    for key, value in dataset.items():
                        db[key] = value
                return db

            result = runner.run_benchmark(
                "write", dataset_size, "shelve", write_shelve, repetitions=repetitions
            )
            results.append(result)
        finally:
            if os.path.exists(shelve_path):
                os.remove(shelve_path)

        # Benchmark EffiDict with different backend/strategy combinations
        from effidict import EffiDict

        for backend_name in config.backends:
            for strategy_name in config.strategies:
                BackendClass = get_backend_class(backend_name)
                StrategyClass = get_strategy_class(strategy_name)

                if BackendClass is None or StrategyClass is None:
                    continue

                # Create temporary storage
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{backend_name}") as tmp:
                    effidict_path = tmp.name

                try:
                    backend = BackendClass(effidict_path)
                    strategy = StrategyClass(disk_backend=backend, max_in_memory=max_in_memory)

                    def write_effidict():
                        ed = EffiDict(
                            max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy
                        )
                        for key, value in dataset.items():
                            ed[key] = value
                        return ed

                    implementation_name = f"effidict_{strategy_name}_{backend_name}"
                    result = runner.run_benchmark(
                        "write", dataset_size, implementation_name, write_effidict, repetitions=repetitions
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error benchmarking {strategy_name}/{backend_name} for write: {e}")
                finally:
                    if os.path.exists(effidict_path):
                        if backend_name == "sqlite" or backend_name == "hdf5":
                            try:
                                os.remove(effidict_path)
                            except:
                                pass
                        else:
                            import shutil

                            try:
                                shutil.rmtree(effidict_path)
                            except:
                                pass
                    if backend:
                        try:
                            backend.destroy()
                        except:
                            pass

    return results


def benchmark_read_operations(
    runner: BenchmarkRunner,
    dataset_sizes: List[int],
    value_type: str = "small_uniform",
    value_size: Any = None,
    config: Any = None,
    repetitions: int = 1,
) -> List[BenchmarkResult]:
    """
    Benchmark read operations (accesses) for different dataset sizes.

    Args:
        runner: Benchmark runner instance
        dataset_sizes: List of dataset sizes to test
        value_type: Type of values to generate
        value_size: Size parameter for values
        config: Benchmark configuration (uses DEFAULT_CONFIG if None)

    Returns:
        List of benchmark results
    """
    if config is None:
        config = DEFAULT_CONFIG

    results = []

    for dataset_size in dataset_sizes:
        # Generate dataset
        dataset = generate_dataset(dataset_size, value_type, value_size)
        keys = list(dataset.keys())

        # Benchmark standard dict
        d = dict(dataset)

        # Determine max_in_memory based on value type and size
        max_in_memory = get_max_in_memory_for_value_size(value_type, value_size)

        def read_dict():
            for key in keys:
                _ = d[key]

        result = runner.run_benchmark("read", dataset_size, "dict", read_dict)
        results.append(result)

        # Benchmark shelve
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            shelve_path = tmp.name

        try:
            # Pre-populate shelve
            with open_shelve(shelve_path, "c") as db:
                for key, value in dataset.items():
                    db[key] = value

            def read_shelve():
                with open_shelve(shelve_path, "r") as db:
                    for key in keys:
                        _ = db[key]

            result = runner.run_benchmark(
                "read", dataset_size, "shelve", read_shelve, repetitions=repetitions
            )
            results.append(result)
        finally:
            if os.path.exists(shelve_path):
                os.remove(shelve_path)

        # Benchmark EffiDict with different backend/strategy combinations
        from effidict import EffiDict

        for backend_name in config.backends:
            for strategy_name in config.strategies:
                BackendClass = get_backend_class(backend_name)
                StrategyClass = get_strategy_class(strategy_name)

                if BackendClass is None or StrategyClass is None:
                    continue

                # Create temporary storage
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{backend_name}") as tmp:
                    effidict_path = tmp.name

                try:
                    backend = BackendClass(effidict_path)
                    strategy = StrategyClass(disk_backend=backend, max_in_memory=max_in_memory)

                    # Pre-populate EffiDict
                    ed = EffiDict(
                        max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy
                    )
                    for key, value in dataset.items():
                        ed[key] = value

                    def read_effidict():
                        ed = EffiDict(
                            max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy
                        )
                        for key in keys:
                            _ = ed[key]

                    implementation_name = f"effidict_{strategy_name}_{backend_name}"
                    result = runner.run_benchmark(
                        "read", dataset_size, implementation_name, read_effidict, repetitions=repetitions
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error benchmarking {strategy_name}/{backend_name} for read: {e}")
                finally:
                    if os.path.exists(effidict_path):
                        if backend_name == "sqlite" or backend_name == "hdf5":
                            try:
                                os.remove(effidict_path)
                            except:
                                pass
                        else:
                            import shutil

                            try:
                                shutil.rmtree(effidict_path)
                            except:
                                pass
                    if backend:
                        try:
                            backend.destroy()
                        except:
                            pass

    return results


def benchmark_delete_operations(
    runner: BenchmarkRunner,
    dataset_sizes: List[int],
    value_type: str = "small_uniform",
    value_size: Any = None,
    config: Any = None,
    repetitions: int = 1,
) -> List[BenchmarkResult]:
    """
    Benchmark delete operations for different dataset sizes.

    Args:
        runner: Benchmark runner instance
        dataset_sizes: List of dataset sizes to test
        value_type: Type of values to generate
        value_size: Size parameter for values
        config: Benchmark configuration (uses DEFAULT_CONFIG if None)

    Returns:
        List of benchmark results
    """
    if config is None:
        config = DEFAULT_CONFIG

    repetitions = getattr(config, "repetitions", 1)
    results = []

    for dataset_size in dataset_sizes:
        # Generate dataset
        dataset = generate_dataset(dataset_size, value_type, value_size)
        keys = list(dataset.keys())

        # Benchmark standard dict
        d = dict(dataset)

        # Determine max_in_memory based on value type and size
        max_in_memory = get_max_in_memory_for_value_size(value_type, value_size)

        def delete_dict():
            for key in keys:
                if key in d:
                    del d[key]

        result = runner.run_benchmark("delete", dataset_size, "dict", delete_dict, repetitions=repetitions)
        results.append(result)

        # Benchmark shelve
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            shelve_path = tmp.name

        try:
            # Pre-populate shelve
            with open_shelve(shelve_path, "c") as db:
                for key, value in dataset.items():
                    db[key] = value

            def delete_shelve():
                with open_shelve(shelve_path, "w") as db:
                    for key in keys:
                        if key in db:
                            del db[key]

            result = runner.run_benchmark(
                "delete", dataset_size, "shelve", delete_shelve, repetitions=repetitions
            )
            results.append(result)
        finally:
            if os.path.exists(shelve_path):
                os.remove(shelve_path)

        # Benchmark EffiDict with different backend/strategy combinations
        from effidict import EffiDict

        for backend_name in config.backends:
            for strategy_name in config.strategies:
                BackendClass = get_backend_class(backend_name)
                StrategyClass = get_strategy_class(strategy_name)

                if BackendClass is None or StrategyClass is None:
                    continue

                # Create temporary storage
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{backend_name}") as tmp:
                    effidict_path = tmp.name

                try:
                    backend = BackendClass(effidict_path)
                    strategy = StrategyClass(disk_backend=backend, max_in_memory=max_in_memory)

                    # Pre-populate EffiDict
                    ed = EffiDict(
                        max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy
                    )
                    for key, value in dataset.items():
                        ed[key] = value

                    def delete_effidict():
                        ed = EffiDict(
                            max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy
                        )
                        for key in keys:
                            if key in ed:
                                del ed[key]

                    implementation_name = f"effidict_{strategy_name}_{backend_name}"
                    result = runner.run_benchmark(
                        "delete", dataset_size, implementation_name, delete_effidict, repetitions=repetitions
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error benchmarking {strategy_name}/{backend_name} for delete: {e}")
                finally:
                    if os.path.exists(effidict_path):
                        if backend_name == "sqlite" or backend_name == "hdf5":
                            try:
                                os.remove(effidict_path)
                            except:
                                pass
                        else:
                            import shutil

                            try:
                                shutil.rmtree(effidict_path)
                            except:
                                pass
                    if backend:
                        try:
                            backend.destroy()
                        except:
                            pass

    return results


def benchmark_modify_operations(
    runner: BenchmarkRunner,
    dataset_sizes: List[int],
    value_type: str = "small_uniform",
    value_size: Any = None,
    config: Any = None,
    repetitions: int = 1,
) -> List[BenchmarkResult]:
    """
    Benchmark modification operations (updates) for different dataset sizes.

    Args:
        runner: Benchmark runner instance
        dataset_sizes: List of dataset sizes to test
        value_type: Type of values to generate
        value_size: Size parameter for values
        config: Benchmark configuration (uses DEFAULT_CONFIG if None)

    Returns:
        List of benchmark results
    """
    if config is None:
        config = DEFAULT_CONFIG

    repetitions = getattr(config, "repetitions", 1)
    results = []

    for dataset_size in dataset_sizes:
        # Generate dataset
        dataset = generate_dataset(dataset_size, value_type, value_size)
        keys = list(dataset.keys())

        # Benchmark standard dict
        d = dict(dataset)

        # Determine max_in_memory based on value type and size
        max_in_memory = get_max_in_memory_for_value_size(value_type, value_size)

        def modify_dict():
            for key in keys:
                if key in d:
                    d[key] = {"modified": True, "original": d[key]}

        result = runner.run_benchmark("modify", dataset_size, "dict", modify_dict, repetitions=repetitions)
        results.append(result)

        # Benchmark shelve
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            shelve_path = tmp.name

        try:
            # Pre-populate shelve
            with open_shelve(shelve_path, "c") as db:
                for key, value in dataset.items():
                    db[key] = value

            def modify_shelve():
                with open_shelve(shelve_path, "w") as db:
                    for key in keys:
                        if key in db:
                            db[key] = {"modified": True, "original": db[key]}

            result = runner.run_benchmark(
                "modify", dataset_size, "shelve", modify_shelve, repetitions=repetitions
            )
            results.append(result)
        finally:
            if os.path.exists(shelve_path):
                os.remove(shelve_path)

        # Benchmark EffiDict with different backend/strategy combinations
        from effidict import EffiDict

        for backend_name in config.backends:
            for strategy_name in config.strategies:
                BackendClass = get_backend_class(backend_name)
                StrategyClass = get_strategy_class(strategy_name)

                if BackendClass is None or StrategyClass is None:
                    continue

                # Create temporary storage
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{backend_name}") as tmp:
                    effidict_path = tmp.name

                try:
                    backend = BackendClass(effidict_path)
                    strategy = StrategyClass(disk_backend=backend, max_in_memory=max_in_memory)

                    # Pre-populate EffiDict
                    ed = EffiDict(
                        max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy
                    )
                    for key, value in dataset.items():
                        ed[key] = value

                    def modify_effidict():
                        ed = EffiDict(
                            max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy
                        )
                        for key in keys:
                            if key in ed:
                                ed[key] = {"modified": True, "original": ed[key]}

                    implementation_name = f"effidict_{strategy_name}_{backend_name}"
                    result = runner.run_benchmark(
                        "modify", dataset_size, implementation_name, modify_effidict, repetitions=repetitions
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error benchmarking {strategy_name}/{backend_name} for modify: {e}")
                finally:
                    if os.path.exists(effidict_path):
                        if backend_name == "sqlite" or backend_name == "hdf5":
                            try:
                                os.remove(effidict_path)
                            except:
                                pass
                        else:
                            import shutil

                            try:
                                shutil.rmtree(effidict_path)
                            except:
                                pass
                    if backend:
                        try:
                            backend.destroy()
                        except:
                            pass

    return results


def run_all_runtime_benchmarks(config: Any = None, output_dir: str = "benchmark_results") -> BenchmarkRunner:
    """
    Run all runtime benchmarks.

    Args:
        config: Benchmark configuration (uses DEFAULT_CONFIG if None)
        output_dir: Output directory for results

    Returns:
        BenchmarkRunner with all results
    """
    if config is None:
        config = DEFAULT_CONFIG

    runner = BenchmarkRunner(output_dir=output_dir)

    # Run benchmarks for each operation type
    print("Running write operation benchmarks...")
    benchmark_write_operations(runner, config.dataset_sizes, config=config)

    print("Running read operation benchmarks...")
    benchmark_read_operations(runner, config.dataset_sizes, config=config)

    print("Running delete operation benchmarks...")
    benchmark_delete_operations(runner, config.dataset_sizes, config=config)

    print("Running modify operation benchmarks...")
    benchmark_modify_operations(runner, config.dataset_sizes, config=config)

    # Save results
    runner.save_results("runtime_benchmarks")

    return runner
