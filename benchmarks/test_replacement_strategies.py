"""
Replacement strategy comparison benchmarks.

Compares all 7 replacement strategies under different access patterns
and dataset types.
"""

import tempfile
import os
from typing import List, Dict, Any
from pathlib import Path
import sys


from benchmark_framework import BenchmarkRunner, BenchmarkResult
from datasets import generate_dataset, generate_oligo_dataset
from workloads import generate_access_pattern
from config import DEFAULT_CONFIG, get_backend_class, get_strategy_class, get_max_in_memory_for_value_size


def benchmark_strategy(
    runner: BenchmarkRunner,
    strategy_name: str,
    backend_name: str,
    dataset: Dict[str, Any],
    access_pattern: str = "uniform",
    max_in_memory: int = 100,
    **kwargs,
) -> BenchmarkResult:
    """
    Benchmark a specific strategy/backend combination.

    Args:
        runner: Benchmark runner instance
        strategy_name: Name of replacement strategy
        backend_name: Name of backend
        dataset: Dataset to use
        access_pattern: Access pattern ('uniform', 'zipf', 'sequential_circular')
        max_in_memory: Maximum items in memory
        **kwargs: Additional parameters for access pattern

    Returns:
        BenchmarkResult
    """
    from effidict import EffiDict

    BackendClass = get_backend_class(backend_name)
    StrategyClass = get_strategy_class(strategy_name)

    if BackendClass is None or StrategyClass is None:
        return None

    # Create temporary storage
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{backend_name}") as tmp:
        storage_path = tmp.name

    try:
        backend = BackendClass(storage_path)
        strategy = StrategyClass(disk_backend=backend, max_in_memory=max_in_memory)

        # Pre-populate dataset
        ed = EffiDict(max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy)
        for key, value in dataset.items():
            ed[key] = value

        # Generate access pattern
        keys = list(dataset.keys())
        access_sequence = list(generate_access_pattern(keys, len(keys) * 2, pattern=access_pattern, **kwargs))

        # Benchmark read operations
        def read_operations():
            ed = EffiDict(max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy)
            for key in access_sequence:
                if key in ed:
                    _ = ed[key]

        result = runner.run_benchmark(
            f"read_{access_pattern}",
            len(dataset),
            f"effidict_{strategy_name}_{backend_name}",
            read_operations,
        )

        return result

    finally:
        if os.path.exists(storage_path):
            if backend_name == "sqlite" or backend_name == "hdf5":
                try:
                    os.remove(storage_path)
                except:
                    pass
            else:
                import shutil

                try:
                    shutil.rmtree(storage_path)
                except:
                    pass
        if backend:
            try:
                backend.destroy()
            except:
                pass


def compare_strategies(
    runner: BenchmarkRunner,
    dataset_sizes: List[int],
    value_type: str = "small_uniform",
    value_size: Any = None,
    access_patterns: List[str] = None,
    strategies: List[str] = None,
    backends: List[str] = None,
    config: Any = None,
) -> List[BenchmarkResult]:
    """
    Compare all strategies under different conditions.

    Args:
        runner: Benchmark runner instance
        dataset_sizes: List of dataset sizes to test
        value_type: Type of values to generate
        value_size: Size parameter for values
        access_patterns: List of access patterns to test
        strategies: List of strategies to test
        backends: List of backends to test
        config: Benchmark configuration (uses DEFAULT_CONFIG if None)

    Returns:
        List of benchmark results
    """
    if config is None:
        config = DEFAULT_CONFIG

    if access_patterns is None:
        access_patterns = ["uniform", "zipf", "sequential_circular"]

    if strategies is None:
        strategies = ["lru", "mru", "fifo", "lifo", "lfu", "mfu", "random"]

    if backends is None:
        backends = ["sqlite", "json", "pickle"]

    results = []

    for dataset_size in dataset_sizes:
        # Generate dataset
        if value_type == "oligo":
            dataset = generate_oligo_dataset(
                num_regions=dataset_size, sequences_per_region=int(value_size) if value_size else 100
            )
        else:
            dataset = generate_dataset(dataset_size, value_type, value_size)

        # Determine max_in_memory based on value type and size
        max_in_memory = get_max_in_memory_for_value_size(value_type, value_size)

        for access_pattern in access_patterns:
            for strategy in strategies:
                for backend in backends:
                    try:
                        result = benchmark_strategy(
                            runner,
                            strategy,
                            backend,
                            dataset,
                            access_pattern=access_pattern,
                            max_in_memory=max_in_memory,
                        )
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"Error benchmarking {strategy}/{backend} with {access_pattern}: {e}")
                        continue

    return results


def run_strategy_benchmarks(config: Any = None, output_dir: str = "benchmark_results") -> BenchmarkRunner:
    """
    Run all replacement strategy benchmarks.

    Args:
        config: Benchmark configuration (uses DEFAULT_CONFIG if None)
        output_dir: Output directory for results

    Returns:
        BenchmarkRunner with all results
    """
    if config is None:
        config = DEFAULT_CONFIG

    runner = BenchmarkRunner(output_dir=output_dir)

    print("Running replacement strategy comparison benchmarks...")

    # Test with small uniform objects
    print("Testing with small uniform objects...")
    compare_strategies(
        runner,
        config.dataset_sizes[:3],  # Use smaller sizes for strategy comparison
        value_type="small_uniform",
        value_size=100,
        config=config,
    )

    # Test with large objects
    print("Testing with large objects...")
    compare_strategies(
        runner,
        config.dataset_sizes[:2],  # Use smaller sizes for large objects
        value_type="large",
        value_size=1.0,
        config=config,
    )

    # Test with oligo dataset and sequential circular access
    print("Testing with oligo dataset (sequential circular access)...")
    compare_strategies(
        runner,
        config.num_regions[:2],  # Use smaller number of regions
        value_type="oligo",
        value_size=100,
        access_patterns=["sequential_circular"],
        config=config,
    )

    # Save results
    runner.save_results("strategy_benchmarks")

    return runner
