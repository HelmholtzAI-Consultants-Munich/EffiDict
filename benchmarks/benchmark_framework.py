"""
Core benchmarking framework for EffiDict comparisons.

Provides utilities for measuring runtime, memory usage, and comparing
EffiDict against standard Python dict and shelve.
"""

import time
import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from collections import defaultdict

from memory_profiler import memory_usage


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    operation: str
    dataset_size: int
    implementation: str
    time_seconds: float
    memory_mb: Optional[float] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    # Statistics for repeated measurements
    time_std: Optional[float] = None  # Standard deviation of time
    memory_std: Optional[float] = None  # Standard deviation of memory

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/CSV export."""
        result = asdict(self)
        if result["metadata"] is None:
            result["metadata"] = {}
        return result


class Timer:
    """Simple timer context manager."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed is None:
            raise RuntimeError("Timer not stopped yet")
        return self.elapsed


class MemoryTracker:
    """Track memory usage during operations."""

    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None
        self._memory_samples = []

    def start(self):
        """Record initial memory usage."""
        # memory_usage(proc=-1) returns current process memory usage in MB as a list
        mem_usage = memory_usage(proc=-1)
        self.initial_memory = mem_usage[0]
        self._memory_samples = [self.initial_memory]

    def stop(self) -> float:
        """Record final memory and return peak usage in MB."""
        # memory_usage(proc=-1) returns current process memory usage in MB as a list
        mem_usage = memory_usage(proc=-1)
        self.final_memory = mem_usage[0]
        self._memory_samples.append(self.final_memory)
        self.peak_memory = max(self._memory_samples) if self._memory_samples else self.final_memory

        return self.peak_memory - self.initial_memory

    def get_peak_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_memory


class BenchmarkRunner:
    """Main benchmark runner that orchestrates benchmarks."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def run_benchmark(
        self,
        operation: str,
        dataset_size: int,
        implementation: str,
        func: Callable,
        repetitions: int = 1,
        *args,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Run a benchmark operation, optionally with repetitions.

        Args:
            operation: Name of the operation (e.g., 'write', 'read', 'delete')
            dataset_size: Size of the dataset
            implementation: Name of implementation ('dict', 'shelve', 'effidict')
            func: Function to benchmark
            repetitions: Number of times to repeat the benchmark (default: 1)
            *args, **kwargs: Arguments to pass to func

        Returns:
            BenchmarkResult with timing and memory information (aggregated if repetitions > 1)
        """
        if repetitions > 1:
            return self.run_benchmark_repeated(
                operation, dataset_size, implementation, func, repetitions, *args, **kwargs
            )
        memory_tracker = MemoryTracker()
        memory_tracker.start()

        timer = Timer()
        with timer:
            # memory_usage with a function tuple calls the function and tracks memory
            # interval=0.1 samples every 0.01 seconds
            mem_samples, result = memory_usage((func, args, kwargs), interval=0.01, retval=True)

        # Calculate peak memory from samples
        peak_memory = max(mem_samples)
        memory_mb = max(0.0, peak_memory - memory_tracker.initial_memory)

        benchmark_result = BenchmarkResult(
            operation=operation,
            dataset_size=dataset_size,
            implementation=implementation,
            time_seconds=timer.elapsed_seconds(),
            memory_mb=memory_mb,
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def run_benchmark_repeated(
        self,
        operation: str,
        dataset_size: int,
        implementation: str,
        func: Callable,
        repetitions: int = 3,
        *args,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Run a benchmark operation multiple times and aggregate results.

        Args:
            operation: Name of the operation (e.g., 'write', 'read', 'delete')
            dataset_size: Size of the dataset
            implementation: Name of implementation ('dict', 'shelve', 'effidict')
            func: Function to benchmark
            repetitions: Number of times to repeat the benchmark
            *args, **kwargs: Arguments to pass to func

        Returns:
            BenchmarkResult with aggregated timing and memory information (mean ± std)
        """
        import statistics

        times = []
        memories = []

        for _ in range(repetitions):
            result = self.run_benchmark(operation, dataset_size, implementation, func, *args, **kwargs)
            times.append(result.time_seconds)
            if result.memory_mb is not None:
                memories.append(result.memory_mb)

        # Remove the individual results (keep only aggregated)
        self.results = self.results[:-repetitions]

        # Calculate statistics
        mean_time = statistics.mean(times)
        time_std = statistics.stdev(times) if len(times) > 1 else 0.0

        mean_memory = statistics.mean(memories) if memories else None
        memory_std = statistics.stdev(memories) if len(memories) > 1 else None

        # Create aggregated result
        aggregated_result = BenchmarkResult(
            operation=operation,
            dataset_size=dataset_size,
            implementation=implementation,
            time_seconds=mean_time,
            memory_mb=mean_memory,
            time_std=time_std,
            memory_std=memory_std,
            metadata={"repetitions": repetitions, "individual_times": times, "individual_memories": memories},
        )

        self.results.append(aggregated_result)
        return aggregated_result

    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON and CSV files."""
        if filename is None:
            filename = "benchmark_results"

        # Save as JSON
        json_path = self.output_dir / f"{filename}.json"
        results_dict = [r.to_dict() for r in self.results]
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        # Save as CSV
        csv_path = self.output_dir / f"{filename}.csv"
        if self.results:
            fieldnames = list(self.results[0].to_dict().keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    row = result.to_dict()
                    # Convert None values to empty strings for CSV (or "N/A" if preferred)
                    row = {k: (v if v is not None else "") for k, v in row.items()}
                    writer.writerow(row)

    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()

    def get_results_by_operation(self, operation: str) -> List[BenchmarkResult]:
        """Get all results for a specific operation."""
        return [r for r in self.results if r.operation == operation]

    def get_results_by_implementation(self, implementation: str) -> List[BenchmarkResult]:
        """Get all results for a specific implementation."""
        return [r for r in self.results if r.implementation == implementation]

    def aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results by operation and implementation."""
        aggregated = defaultdict(lambda: defaultdict(list))

        for result in self.results:
            aggregated[result.operation][result.implementation].append(result)

        return dict(aggregated)


def compare_implementations(
    results: List[BenchmarkResult], metric: str = "time_seconds"
) -> Dict[str, Dict[str, float]]:
    """
    Compare implementations across different dataset sizes.

    Args:
        results: List of benchmark results
        metric: Metric to compare ('time_seconds' or 'memory_mb')

    Returns:
        Dictionary mapping dataset_size -> implementation -> metric_value
    """
    comparison = defaultdict(dict)

    for result in results:
        value = getattr(result, metric)
        comparison[result.dataset_size][result.implementation] = value

    return dict(comparison)


@contextmanager
def temp_storage(storage_path: str):
    """Context manager for temporary storage cleanup."""
    try:
        yield storage_path
    finally:
        # Cleanup logic can be added here if needed
        pass


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
