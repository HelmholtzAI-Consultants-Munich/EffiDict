"""
Memory usage benchmarks.

Measures RAM/swap usage across increasing dataset sizes,
comparing EffiDict, dict, and shelve.
"""

import tempfile
import os
from typing import Dict, Any, List
import gc
import sys


from benchmark_framework import BenchmarkRunner, BenchmarkResult, MemoryTracker, open_shelve
from datasets import generate_dataset
from config import DEFAULT_CONFIG, get_backend_class, get_strategy_class, get_max_in_memory_for_value_size


def measure_memory_usage(
    runner: BenchmarkRunner,
    dataset_size: int,
    implementation: str,
    dataset: Dict[str, Any],
    backend_name: str = None,
    strategy_name: str = None,
    max_in_memory: int = 100,
    repetitions: int = 1,
) -> BenchmarkResult:
    """
    Measure memory usage for a given implementation and dataset.

    Args:
        runner: Benchmark runner instance
        dataset_size: Size of the dataset
        implementation: Implementation name ('dict', 'shelve', 'effidict')
        dataset: Dataset to store
        backend_name: Backend name for EffiDict (optional)
        strategy_name: Strategy name for EffiDict (optional)
        max_in_memory: Maximum items in memory for EffiDict (default: 100)

    Returns:
        BenchmarkResult with memory information
    """
    memory_tracker = MemoryTracker()
    memory_tracker.start()

    if implementation == "dict":
        d = dict(dataset)
        memory_mb = memory_tracker.stop()
        del d
        gc.collect()

    elif implementation == "shelve":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            shelve_path = tmp.name

        try:
            with open_shelve(shelve_path, "c") as db:
                for key, value in dataset.items():
                    db[key] = value
            memory_mb = memory_tracker.stop()
        finally:
            if os.path.exists(shelve_path):
                os.remove(shelve_path)

    elif implementation.startswith("effidict"):
        from effidict import EffiDict

        BackendClass = get_backend_class(backend_name)
        StrategyClass = get_strategy_class(strategy_name)

        # Create temporary storage
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{backend_name}") as tmp:
            effidict_path = tmp.name

        try:
            backend = BackendClass(effidict_path)
            strategy = StrategyClass(disk_backend=backend, max_in_memory=max_in_memory)

            ed = EffiDict(max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy)
            for key, value in dataset.items():
                ed[key] = value
            memory_mb = memory_tracker.stop()
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

    else:
        raise ValueError(f"Invalid implementation: {implementation}")

    # If repetitions > 1, we need to run multiple times and aggregate
    if repetitions > 1:
        import statistics

        memories = [memory_mb]

        # Run additional repetitions
        for _ in range(repetitions - 1):
            memory_tracker = MemoryTracker()
            memory_tracker.start()

            if implementation == "dict":
                d = dict(dataset)
                mem = memory_tracker.stop()
                del d
                gc.collect()
            elif implementation == "shelve":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                    shelve_path = tmp.name
                try:
                    with open_shelve(shelve_path, "c") as db:
                        for key, value in dataset.items():
                            db[key] = value
                    mem = memory_tracker.stop()
                finally:
                    if os.path.exists(shelve_path):
                        os.remove(shelve_path)
            elif implementation.startswith("effidict"):
                from effidict import EffiDict

                BackendClass = get_backend_class(backend_name)
                StrategyClass = get_strategy_class(strategy_name)
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{backend_name}") as tmp:
                    effidict_path = tmp.name
                try:
                    backend = BackendClass(effidict_path)
                    strategy = StrategyClass(disk_backend=backend, max_in_memory=max_in_memory)
                    ed = EffiDict(
                        max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy
                    )
                    for key, value in dataset.items():
                        ed[key] = value
                    mem = memory_tracker.stop()
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

            memories.append(mem)

        # Calculate statistics
        mean_memory = statistics.mean(memories)
        memory_std = statistics.stdev(memories) if len(memories) > 1 else 0.0

        result = BenchmarkResult(
            operation="memory",
            dataset_size=dataset_size,
            implementation=implementation,
            time_seconds=0.0,
            memory_mb=mean_memory,
            memory_std=memory_std,
            metadata={"repetitions": repetitions, "individual_memories": memories},
        )
    else:
        result = BenchmarkResult(
            operation="memory",
            dataset_size=dataset_size,
            implementation=implementation,
            time_seconds=0.0,
            memory_mb=memory_mb,
        )

    runner.results.append(result)
    return result


def benchmark_memory_usage(
    runner: BenchmarkRunner,
    dataset_sizes: List[int],
    value_type: str = "small_uniform",
    value_size: Any = None,
    config: Any = None,
) -> List[BenchmarkResult]:
    """
    Benchmark memory usage for different dataset sizes.

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

        # Determine max_in_memory based on value type and size
        max_in_memory = get_max_in_memory_for_value_size(value_type, value_size)

        # Force garbage collection before measuring
        gc.collect()

        # Measure memory for dict
        result = measure_memory_usage(runner, dataset_size, "dict", dataset, repetitions=repetitions)
        results.append(result)
        gc.collect()

        # Measure memory for shelve
        result = measure_memory_usage(runner, dataset_size, "shelve", dataset, repetitions=repetitions)
        results.append(result)
        gc.collect()

        # Measure memory for EffiDict with different backend/strategy combinations
        for backend_name in config.backends:
            for strategy_name in config.strategies:
                BackendClass = get_backend_class(backend_name)
                StrategyClass = get_strategy_class(strategy_name)

                if BackendClass is None or StrategyClass is None:
                    continue

                gc.collect()
                implementation_name = f"effidict_{strategy_name}_{backend_name}"
                result = measure_memory_usage(
                    runner,
                    dataset_size,
                    implementation_name,
                    dataset,
                    backend_name,
                    strategy_name,
                    max_in_memory,
                    repetitions=repetitions,
                )
                results.append(result)
                gc.collect()

    return results


def run_memory_benchmarks(config: Any = None, output_dir: str = "benchmark_results") -> BenchmarkRunner:
    """
    Run all memory benchmarks.

    Args:
        config: Benchmark configuration (uses DEFAULT_CONFIG if None)
        output_dir: Output directory for results

    Returns:
        BenchmarkRunner with all results
    """
    if config is None:
        config = DEFAULT_CONFIG

    runner = BenchmarkRunner(output_dir=output_dir)

    print("Running memory usage benchmarks...")
    benchmark_memory_usage(runner, config.dataset_sizes, config=config)

    # Save results
    runner.save_results("memory_benchmarks")

    return runner
