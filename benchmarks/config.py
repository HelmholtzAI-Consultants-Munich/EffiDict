"""
Configuration management for benchmarks.

Centralized configuration for all benchmark parameters.
"""

from typing import List
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmarks.

    Attributes:
        dataset_sizes: List of dataset sizes (number of entries) to test
        dataset_value_sizes: List of value sizes in MB to test
        zipf_skewness: List of Zipf distribution skewness parameters for access patterns
        workload_sizes: List of workload sizes for use case benchmarks
        effidict_backends: List of backend names to test ('sqlite', 'json', 'pickle', 'hdf5')
        effidict_strategies: List of replacement strategy names to test
            ('lru', 'mru', 'fifo', 'lifo', 'lfu', 'mfu', 'random')
        repetitions: Number of times to repeat each benchmark for statistical significance
        output_dir: Directory path where benchmark results will be saved

    Note:
        All list attributes default to None and are set to default values in __post_init__
        if not provided during initialization.
    """

    # Dataset sizes for main plots
    dataset_sizes: List[int] = None

    # Value sizes
    dataset_value_sizes: List[float] = None  # MB

    # Access patterns
    zipf_skewness: List[float] = None

    # Workload sizes
    workload_sizes: List[int] = None

    # Backend/strategy combinations
    effidict_backends: List[str] = None
    effidict_strategies: List[str] = None

    # Repetitions for statistical significance
    repetitions: int = None

    # Output directories
    output_dir: str = "benchmark_results"

    def __post_init__(self):
        """Set default values if not provided."""
        if self.dataset_sizes is None:
            self.dataset_sizes = [
                1000,
                2000,
                # 5000,
                # 10000,
                # 20000,
                # 50000,
                # 100000,
                # 2000000,
                # 5000000,
                # 1000000,
            ]

        if self.dataset_value_sizes is None:
            self.dataset_value_sizes = [
                0.001,
                # 0.01,
                # 0.1,
                # 1.0,
                10.0,
                # 100.0,
                # 500.0,
                # 1024.0,
            ]

        if self.zipf_skewness is None:
            self.zipf_skewness = [
                1.0,
                1.5,
                2.0,
            ]

        if self.workload_sizes is None:
            self.workload_sizes = [
                1000,
                10000,
                100000,
                1000000,
            ]

        if self.effidict_backends is None:
            self.effidict_backends = [
                "sqlite",
                # "json",
                "pickle",
                # "hdf5",
            ]

        if self.effidict_strategies is None:
            self.effidict_strategies = [
                "lru",
                # "mru",
                "fifo",
                # "lifo",
                # "lfu",
                # "mfu",
                "random",
            ]

        if self.repetitions is None:
            self.repetitions = 5


# Default configuration
DEFAULT_CONFIG = BenchmarkConfig()
