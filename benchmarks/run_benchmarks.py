"""
Main benchmark runner.

Orchestrates all benchmarks and generates reports.
"""

import argparse
from pathlib import Path
from typing import Optional
import sys
import os

# Add benchmarks directory to path for script execution
benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
if benchmarks_dir not in sys.path:
    sys.path.insert(0, benchmarks_dir)

from benchmark_framework import BenchmarkRunner
from test_runtime import run_all_runtime_benchmarks
from test_memory import run_memory_benchmarks
from test_use_cases import run_use_case_benchmarks
from visualize_results import plot_all_operations, create_comparison_table
from config import DEFAULT_CONFIG, BenchmarkConfig


def run_all_benchmarks(
    config: Optional[BenchmarkConfig] = None,
    output_dir: str = "benchmark_results",
    skip_runtime: bool = False,
    skip_memory: bool = False,
    skip_use_case: bool = False,
) -> BenchmarkRunner:
    """
    Run all benchmarks.

    Args:
        config: Benchmark configuration (uses DEFAULT_CONFIG if None)
        output_dir: Output directory for results
        skip_runtime: Skip runtime benchmarks
        skip_memory: Skip memory benchmarks
        skip_use_case: Skip use case benchmarks

    Returns:
        Combined BenchmarkRunner with all results
    """
    if config is None:
        config = DEFAULT_CONFIG

    combined_runner = BenchmarkRunner(output_dir=output_dir)

    # Run runtime benchmarks
    if not skip_runtime:
        print("=" * 60)
        print("Running Runtime Benchmarks")
        print("=" * 60)
        runtime_runner = run_all_runtime_benchmarks(config, output_dir)
        combined_runner.results.extend(runtime_runner.results)

    # Run memory benchmarks
    if not skip_memory:
        print("=" * 60)
        print("Running Memory Benchmarks")
        print("=" * 60)
        memory_runner = run_memory_benchmarks(config, output_dir)
        combined_runner.results.extend(memory_runner.results)

    # Run use case benchmarks
    if not skip_use_case:
        print("=" * 60)
        print("Running Use Case Benchmarks")
        print("=" * 60)
        use_case_runner = run_use_case_benchmarks(config, output_dir)
        combined_runner.results.extend(use_case_runner.results)

    # Generate visualizations
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    # Create comparison table
    comparison_path = Path(output_dir) / "comparison_table.csv"
    create_comparison_table(combined_runner, comparison_path)

    # Save all results
    combined_runner.save_results("all_benchmarks")

    print("=" * 60)
    print("Benchmarking Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    plot_all_operations(combined_runner, Path(output_dir))

    return combined_runner


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run EffiDict benchmarks", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )

    parser.add_argument("--skip-runtime", action="store_true", help="Skip runtime benchmarks")
    parser.add_argument("--skip-memory", action="store_true", help="Skip memory benchmarks")
    parser.add_argument("--skip-use-case", action="store_true", help="Skip use case benchmarks")
    args = parser.parse_args()

    # Create custom config if dataset sizes specified
    config = DEFAULT_CONFIG

    # Run benchmarks
    run_all_benchmarks(
        config=config,
        output_dir=args.output_dir,
        skip_runtime=args.skip_runtime,
        skip_memory=args.skip_memory,
        skip_use_case=args.skip_use_case,
    )


if __name__ == "__main__":
    main()
