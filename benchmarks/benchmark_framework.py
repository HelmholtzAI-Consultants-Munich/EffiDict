"""
Core benchmarking framework for EffiDict comparisons.

Provides utilities for measuring runtime, memory usage, and comparing
EffiDict against standard Python dict and shelve.
"""

import time
import json
import csv
import os
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    operation: str
    dataset_size: int
    implementation: str
    time_seconds: float
    memory_mb: Optional[float] = None  # RAM usage in MB
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


class BenchmarkRunner:
    """Main benchmark runner that orchestrates benchmarks."""

    def __init__(self, results: List[BenchmarkResult], output_dir: str = "benchmark_results"):

        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
