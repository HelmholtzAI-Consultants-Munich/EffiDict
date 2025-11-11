"""
Core benchmarking framework for EffiDict comparisons.

Provides utilities for measuring runtime, memory usage, and comparing
EffiDict against standard Python dict and shelve.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sys

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# from test_memory import run_memory_benchmarks
from config import DEFAULT_CONFIG, BenchmarkConfig

# Add benchmarks directory to path for script execution
benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
if benchmarks_dir not in sys.path:
    sys.path.insert(0, benchmarks_dir)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


@dataclass
class BenchmarkResult:
    """
    Represents the result of a single benchmark run.

    Attributes:
        operation: Type of operation benchmarked (e.g., 'read', 'write', 'delete', 'modify')
        dataset_size: Number of entries in the dataset
        dataset_value_size: Size of each value in MB
        implementation: Name of the implementation tested (e.g., 'dict', 'shelve', 'effidict_lru_sqlite')
        time_seconds: Mean execution time in seconds
        time_std: Standard deviation of execution time (optional)
        memory_mb: Memory usage in MB (optional)
        memory_std: Standard deviation of memory usage (optional)
        cache_hits: Number of cache hits (optional, for EffiDict)
        cache_misses: Number of cache misses (optional, for EffiDict)
        metadata: Additional metadata dictionary (optional)
    """

    operation: str
    dataset_size: int
    dataset_value_size: int
    implementation: str
    time_seconds: float
    time_std: Optional[float]
    memory_mb: Optional[float]
    memory_std: Optional[float]
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON/CSV export.

        Returns:
            Dictionary representation of the benchmark result
        """
        result = asdict(self)
        if result["metadata"] is None:
            result["metadata"] = {}
        return result


class BenchmarkRunner:
    """
    Main class for running and managing benchmark suites.

    Handles execution of benchmarks, result collection, visualization,
    and export of results to various formats.
    """

    def __init__(self, output_dir: str = "benchmark_results") -> None:
        """
        Initialize the benchmark runner.

        Args:
            output_dir: Directory path where benchmark results will be saved
        """
        self.results: List[BenchmarkResult] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all_benchmarks(
        self,
        config: Optional[BenchmarkConfig],
        skip_runtime_write: bool,
        skip_runtime_read: bool,
        skip_runtime_delete: bool,
        skip_runtime_modify: bool,
        skip_memory: bool,
    ):
        """
        Run all configured benchmark suites.

        Args:
            config: Benchmark configuration object containing test parameters
            skip_runtime_write: If True, skip runtime write benchmarks
            skip_runtime_read: If True, skip runtime read benchmarks
            skip_runtime_delete: If True, skip runtime delete benchmarks
            skip_runtime_modify: If True, skip runtime modify benchmarks
            skip_memory: If True, skip memory benchmarks
            skip_use_case: If True, skip use case benchmarks
        """

        # Run runtime benchmarks
        if not skip_runtime_write:
            print("=" * 60, "\nRunning Runtime Write Benchmarks\n", "=" * 60, sep="")
            from test_runtime import benchmark_operations  # Import here to avoid circular dependency

            runtime_results = benchmark_operations(
                operation="write",
                dataset_sizes=config.dataset_sizes,
                dataset_value_sizes=config.dataset_value_sizes,
                repetitions=config.repetitions,
                effidict_backends=config.effidict_backends,
                effidict_strategies=config.effidict_strategies,
            )
            self.results.extend(runtime_results)

        if not skip_runtime_read:
            print("=" * 60, "\nRunning Runtime Read Benchmarks\n", "=" * 60, sep="")
            from test_runtime import benchmark_operations  # Import here to avoid circular dependency

            runtime_results = benchmark_operations(
                operation="read",
                dataset_sizes=config.dataset_sizes,
                dataset_value_sizes=config.dataset_value_sizes,
                repetitions=config.repetitions,
                effidict_backends=config.effidict_backends,
                effidict_strategies=config.effidict_strategies,
            )
            self.results.extend(runtime_results)

        if not skip_runtime_delete:
            print("=" * 60, "\nRunning Runtime Delete Benchmarks\n", "=" * 60, sep="")
            from test_runtime import benchmark_operations  # Import here to avoid circular dependency

            runtime_results = benchmark_operations(
                operation="delete",
                dataset_sizes=config.dataset_sizes,
                dataset_value_sizes=config.dataset_value_sizes,
                repetitions=config.repetitions,
                effidict_backends=config.effidict_backends,
                effidict_strategies=config.effidict_strategies,
            )
            self.results.extend(runtime_results)

        if not skip_runtime_modify:
            print("=" * 60, "\nRunning Runtime Modify Benchmarks\n", "=" * 60, sep="")
            from test_runtime import benchmark_operations  # Import here to avoid circular dependency

            runtime_results = benchmark_operations(
                operation="modify",
                dataset_sizes=config.dataset_sizes,
                dataset_value_sizes=config.dataset_value_sizes,
                repetitions=config.repetitions,
                effidict_backends=config.effidict_backends,
                effidict_strategies=config.effidict_strategies,
            )
            self.results.extend(runtime_results)

        # Run memory benchmarks
        if not skip_memory:
            print("=" * 60, "\nRunning Memory Benchmarks\n", "=" * 60, sep="")
            from test_memory import benchmark_memory_usage  # Import here to avoid circular dependency

            memory_results = benchmark_memory_usage(
                config.dataset_sizes,
                config.dataset_value_sizes,
                config.repetitions,
                config.effidict_backends,
                config.effidict_strategies,
            )
            self.results.extend(memory_results)

        # Generate visualizations
        print("=" * 60, "\nGenerating Visualizations\n", "=" * 60, sep="")

        # Create comparison table
        self.create_comparison_table()

        # Save all results
        self.save_results()

        # Generate all plots
        self._generate_all_plots(config)

        print("=" * 60, f"\nBenchmarking Complete!\nResults saved to: {self.output_dir}\n", "=" * 60, sep="")

    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create a comparison table from benchmark results.

        Returns:
            DataFrame containing operation, dataset_size, implementation,
            time_seconds, and memory_mb columns

        Note:
            The table is also saved as CSV to the output directory
        """
        output_path = self.output_dir / "comparison_table.csv"
        df = pd.DataFrame(
            [
                {
                    "operation": r.operation,
                    "dataset_size": r.dataset_size,
                    "implementation": r.implementation,
                    "time_seconds": r.time_seconds,
                    "memory_mb": r.memory_mb,
                }
                for r in self.results
            ]
        )
        df.to_csv(output_path, index=False)
        print(f"Saved comparison table to {output_path}")
        return df

    def save_results(self):
        """Save results as JSON and CSV files."""
        base_path = self.output_dir / "benchmark_results"

        # Save as JSON
        with open(base_path.with_suffix(".json"), "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

        # Save as CSV
        if self.results:
            with open(base_path.with_suffix(".csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.results[0].to_dict().keys()))
                writer.writeheader()
                writer.writerows(
                    {k: (v if v is not None else "") for k, v in r.to_dict().items()} for r in self.results
                )

    def plot_runtime_operations(
        self,
        operation: str,
        dataset_value_size: int,
        output_path: Path,
        title: Optional[str] = None,
    ):
        """
        Plot runtime operations for a specific operation and value size.

        Args:
            operation: Operation type to plot (e.g., 'read', 'write', 'delete', 'modify')
            dataset_value_size: Value size in MB to filter results
            output_path: Path where the plot image will be saved
            title: Optional custom title for the plot. If None, uses default format
        """
        results = [
            r for r in self.results if r.operation == operation and r.dataset_value_size == dataset_value_size
        ]

        if not results:
            print(f"No results found for operation: {operation} and dataset value size: {dataset_value_size}")
            return

        # Create DataFrames
        df = pd.DataFrame(
            {
                "implementation": [r.implementation for r in results],
                "dataset_size": [r.dataset_size for r in results],
                "time_seconds": [r.time_seconds for r in results],
                "time_std": [r.time_std for r in results],
            }
        )

        pivot_df = df.pivot_table(
            index="dataset_size", columns="implementation", values="time_seconds", aggfunc="mean"
        )
        pivot_std = df.pivot_table(
            index="dataset_size", columns="implementation", values="time_std", aggfunc="mean"
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for implementation in pivot_df.columns:
            x, y = pivot_df.index, pivot_df[implementation]
            plot_kwargs = {"marker": "o", "label": implementation, "linewidth": 2}
            if implementation in pivot_std.columns:
                ax.errorbar(x, y, yerr=pivot_std[implementation], **plot_kwargs, capsize=3, capthick=1)
            else:
                ax.plot(x, y, **plot_kwargs)

        ax.set_xlabel("Number of entries", fontsize=12)
        ax.set_ylabel("Time [s]", fontsize=12)
        ax.set_title(title or f"{operation.capitalize()} Operations Runtime", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {output_path}")

    def plot_memory_usage(
        self,
        dataset_value_size: int,
        output_path: Path,
        title: Optional[str] = None,
    ):
        """
        Plot memory usage for a specific value size.

        Args:
            dataset_value_size: Value size in MB to filter results
            output_path: Path where the plot image will be saved
            title: Optional custom title for the plot. If None, uses default format
        """
        # Filter results for memory operation and value size
        results = [
            r for r in self.results if r.operation == "memory" and r.dataset_value_size == dataset_value_size
        ]

        if not results:
            print(f"No memory results found for dataset value size: {dataset_value_size}")
            return

        # Create DataFrame
        df = pd.DataFrame(
            {
                "implementation": [r.implementation for r in results],
                "dataset_size": [r.dataset_size for r in results],
                "memory_mb": [r.memory_mb for r in results],
                "memory_std": [r.memory_std for r in results],
            }
        )

        pivot_df = df.pivot_table(
            index="dataset_size", columns="implementation", values="memory_mb", aggfunc="mean"
        )
        pivot_std = df.pivot_table(
            index="dataset_size", columns="implementation", values="memory_std", aggfunc="mean"
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for implementation in pivot_df.columns:
            x, y = pivot_df.index, pivot_df[implementation]
            plot_kwargs = {"marker": "o", "label": implementation, "linewidth": 2}
            if implementation in pivot_std.columns:
                ax.errorbar(x, y, yerr=pivot_std[implementation], **plot_kwargs, capsize=3, capthick=1)
            else:
                ax.plot(x, y, **plot_kwargs)

        ax.set_xlabel("Number of entries", fontsize=12)
        ax.set_ylabel("RAM/swap usage (MB)", fontsize=12)
        ax.set_title(title or f"Memory Usage ({dataset_value_size} MB values)", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {output_path}")

    def _generate_all_plots(self, config: BenchmarkConfig):
        """
        Generate all runtime operation plots and memory usage plots for each value size.

        Args:
            config: Benchmark configuration containing dataset_value_sizes to plot
        """
        # Generate runtime operation plots
        for operation in ["read", "write", "delete", "modify"]:
            for dataset_value_size in config.dataset_value_sizes:
                output_path = self.output_dir / f"{operation}_operations_{dataset_value_size}_mb.png"
                self.plot_runtime_operations(operation, dataset_value_size, output_path=output_path)

        # Generate memory usage plots
        for dataset_value_size in config.dataset_value_sizes:
            output_path = self.output_dir / f"memory_usage_{dataset_value_size}_mb.png"
            self.plot_memory_usage(dataset_value_size, output_path=output_path)


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run EffiDict benchmarks", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument("--skip-runtime-write", action="store_true", help="Skip runtime write benchmarks")
    parser.add_argument("--skip-runtime-read", action="store_true", help="Skip runtime read benchmarks")
    parser.add_argument("--skip-runtime-delete", action="store_true", help="Skip runtime delete benchmarks")
    parser.add_argument("--skip-runtime-modify", action="store_true", help="Skip runtime modify benchmarks")
    parser.add_argument("--skip-memory", action="store_true", help="Skip memory benchmarks")

    args = parser.parse_args()
    runner = BenchmarkRunner(output_dir=args.output_dir)
    runner.run_all_benchmarks(
        config=DEFAULT_CONFIG,
        skip_runtime_write=args.skip_runtime_write,
        skip_runtime_read=args.skip_runtime_read,
        skip_runtime_delete=args.skip_runtime_delete,
        skip_runtime_modify=args.skip_runtime_modify,
        skip_memory=args.skip_memory,
    )


if __name__ == "__main__":
    main()
