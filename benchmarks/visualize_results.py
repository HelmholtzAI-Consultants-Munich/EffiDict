"""
Visualization utilities for benchmark results.

Creates plots for runtime and memory usage comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Handle imports for both package and direct script execution
if __name__ == "__main__" or not __package__:
    # Running as script or not as package - use absolute imports
    from benchmark_framework import BenchmarkResult, BenchmarkRunner
else:
    # Running as package - use relative imports
    from .benchmark_framework import BenchmarkResult, BenchmarkRunner

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def plot_runtime_operations(
    results: List[BenchmarkResult],
    operation: str,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
):
    """
    Plot runtime for a specific operation type.

    Args:
        results: List of benchmark results
        operation: Operation type ('read', 'write', 'delete', 'modify')
        output_path: Path to save the plot
        title: Plot title
    """
    # Filter results for the operation
    op_results = [r for r in results if r.operation == operation]

    if not op_results:
        print(f"No results found for operation: {operation}")
        return

    # Create DataFrame
    data = {
        "dataset_size": [r.dataset_size for r in op_results],
        "time_seconds": [r.time_seconds for r in op_results],
        "implementation": [r.implementation for r in op_results],
    }
    df = pd.DataFrame(data)

    # Group by implementation and dataset size
    pivot_df = df.pivot_table(
        index="dataset_size", columns="implementation", values="time_seconds", aggfunc="mean"
    )

    # Get standard deviations if available
    pivot_std = None
    if any(r.time_std is not None for r in op_results):
        df_std = pd.DataFrame(
            {
                "dataset_size": [r.dataset_size for r in op_results],
                "time_std": [r.time_std if r.time_std is not None else 0.0 for r in op_results],
                "implementation": [r.implementation for r in op_results],
            }
        )
        pivot_std = df_std.pivot_table(
            index="dataset_size", columns="implementation", values="time_std", aggfunc="mean"
        )

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for implementation in pivot_df.columns:
        x = pivot_df.index
        y = pivot_df[implementation]
        if pivot_std is not None and implementation in pivot_std.columns:
            yerr = pivot_std[implementation]
            ax.errorbar(x, y, yerr=yerr, marker="o", label=implementation, linewidth=2, capsize=3, capthick=1)
        else:
            ax.plot(x, y, marker="o", label=implementation, linewidth=2)

    ax.set_xlabel("Number of entries", fontsize=12)
    ax.set_ylabel("Time [s]", fontsize=12)
    ax.set_title(title or f"{operation.capitalize()} Operations Runtime", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_memory_usage(
    results: List[BenchmarkResult], output_path: Optional[Path] = None, title: Optional[str] = None
):
    """
    Plot memory usage across dataset sizes.

    Args:
        results: List of benchmark results
        output_path: Path to save the plot
        title: Plot title
    """
    # Filter results for memory operation
    memory_results = [r for r in results if r.operation == "memory"]

    if not memory_results:
        print("No memory results found")
        return

    # Create DataFrame
    pseudo_count = 1
    data = {
        "dataset_size": [r.dataset_size for r in memory_results],
        "memory_mb": [r.memory_mb + pseudo_count for r in memory_results],
        "implementation": [r.implementation for r in memory_results],
    }
    df = pd.DataFrame(data)

    # Group by implementation and dataset size
    pivot_df = df.pivot_table(
        index="dataset_size", columns="implementation", values="memory_mb", aggfunc="mean"
    )

    # Get standard deviations if available
    pivot_std = None
    if any(r.memory_std is not None for r in memory_results):
        df_std = pd.DataFrame(
            {
                "dataset_size": [r.dataset_size for r in memory_results],
                "memory_std": [r.memory_std if r.memory_std is not None else 0.0 for r in memory_results],
                "implementation": [r.implementation for r in memory_results],
            }
        )
        pivot_std = df_std.pivot_table(
            index="dataset_size", columns="implementation", values="memory_std", aggfunc="mean"
        )

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for implementation in pivot_df.columns:
        x = pivot_df.index
        y = pivot_df[implementation]
        if pivot_std is not None and implementation in pivot_std.columns:
            yerr = pivot_std[implementation]
            ax.errorbar(x, y, yerr=yerr, marker="o", label=implementation, linewidth=2, capsize=3, capthick=1)
        else:
            ax.plot(x, y, marker="o", label=implementation, linewidth=2)

    ax.set_xlabel("Number of entries", fontsize=12)
    ax.set_ylabel("RAM/swap usage (MB)", fontsize=12)
    ax.set_title(title or "Memory Usage", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_all_operations(runner: BenchmarkRunner, output_dir: Path = Path("benchmark_results")):
    """
    Create all plots for runtime and memory operations.

    Args:
        runner: BenchmarkRunner with results
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = runner.results

    # Plot each operation type
    for operation in ["read", "write", "delete", "modify"]:
        output_path = output_dir / f"{operation}_operations.png"
        plot_runtime_operations(results, operation, output_path=output_path)

    # Plot memory usage
    output_path = output_dir / "memory_usage.png"
    plot_memory_usage(results, output_path=output_path)


def create_comparison_table(runner: BenchmarkRunner, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Create a comparison table of all results.

    Args:
        runner: BenchmarkRunner with results
        output_path: Path to save CSV table

    Returns:
        DataFrame with comparison results
    """
    results = runner.results

    data = []
    for result in results:
        data.append(
            {
                "operation": result.operation,
                "dataset_size": result.dataset_size,
                "implementation": result.implementation,
                "time_seconds": result.time_seconds,
                "memory_mb": result.memory_mb,
            }
        )

    df = pd.DataFrame(data)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved comparison table to {output_path}")

    return df
