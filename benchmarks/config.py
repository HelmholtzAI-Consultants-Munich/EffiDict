"""
Configuration management for benchmarks.

Centralized configuration for all benchmark parameters.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""

    # Dataset sizes for main plots
    dataset_sizes: List[int] = None

    # Memory limits for EffiDict
    memory_limits: List[int] = None
    max_in_memory: int = None  # Maximum items in memory for EffiDict

    # Value sizes
    small_value_sizes: List[int] = None  # bytes
    large_value_sizes: List[float] = None  # MB

    # Access patterns
    zipf_skewness: List[float] = None

    # Workload sizes
    workload_sizes: List[int] = None

    # Oligo Designer Toolsuite parameters
    num_regions: List[int] = None
    sequences_per_region: List[int] = None

    # Output directories
    output_dir: str = "benchmark_results"

    # Backend/strategy combinations
    backends: List[str] = None
    strategies: List[str] = None

    # Repetitions for statistical significance
    repetitions: int = 10  # Number of times to repeat each benchmark

    def __post_init__(self):
        """Set default values if not provided."""
        if self.dataset_sizes is None:
            self.dataset_sizes = [
                1000,
                # 2000,
                # 5000,
                # 10000,
                # 20000,
                # 50000,
                # 100000,
                # 2000000,
                # 5000000,
                # 1000000,
            ]

        if self.memory_limits is None:
            self.memory_limits = [10]  # [10, 100, 1000, 10000]

        if self.max_in_memory is None:
            self.max_in_memory = 100

        if self.small_value_sizes is None:
            self.small_value_sizes = [100]  # [100, 1024, 10240]  # 100 bytes, 1KB, 10KB

        if self.large_value_sizes is None:
            self.large_value_sizes = [
                0.1
            ]  # [0.1, 1.0, 10.0, 100.0, 500.0, 1024.0]  # 100KB, 1MB, 10MB, 100MB, 500MB, 1GB

        if self.zipf_skewness is None:
            self.zipf_skewness = [1.0]  # [1.0, 1.5, 2.0]

        if self.workload_sizes is None:
            self.workload_sizes = [1000]  # [1000, 10000, 100000, 1000000]

        if self.num_regions is None:
            self.num_regions = [10]  # [10, 100, 1000, 10000]

        if self.sequences_per_region is None:
            self.sequences_per_region = [10]  # [10, 100, 1000]

        if self.backends is None:
            self.backends = ["sqlite"]  # ["sqlite", "json", "pickle", "hdf5"]

        if self.strategies is None:
            self.strategies = ["lru"]  # ["lru", "mru", "fifo", "lifo", "lfu", "mfu", "random"]


# Default configuration
DEFAULT_CONFIG = BenchmarkConfig()


def get_backend_class(backend_name: str):
    """Get backend class from name."""
    from effidict import SqliteBackend, JSONBackend, PickleBackend, Hdf5Backend

    backend_map = {"sqlite": SqliteBackend, "json": JSONBackend, "pickle": PickleBackend, "hdf5": Hdf5Backend}

    return backend_map.get(backend_name.lower())


def get_strategy_class(strategy_name: str):
    """Get strategy class from name."""
    from effidict import (
        LRUReplacement,
        MRUReplacement,
        FIFOReplacement,
        LIFOReplacement,
        LFUReplacement,
        MFUReplacement,
        RandomReplacement,
    )

    strategy_map = {
        "lru": LRUReplacement,
        "mru": MRUReplacement,
        "fifo": FIFOReplacement,
        "lifo": LIFOReplacement,
        "lfu": LFUReplacement,
        "mfu": MFUReplacement,
        "random": RandomReplacement,
    }

    return strategy_map.get(strategy_name.lower())


def get_max_in_memory_for_value_size(
    value_type: str, value_size: Any = None, sequences_per_region: Any = None
) -> int:
    """
    Determine max_in_memory based on value type and size.

    Args:
        value_type: Type of values ('small_uniform', 'large', 'oligo')
        value_size: Size parameter (bytes for small_uniform, MB for large)

    Returns:
        Appropriate max_in_memory value:
        - For small_uniform: 10000
        - For large values <= 100MB: 100
        - For large values > 100MB: 10
        - For oligo: 100 (default)
    """
    if value_type == "small_uniform":
        return 10000
    elif value_type == "large":
        if value_size is None:
            return 100  # Default for large
        # value_size is in MB
        if value_size <= 100.0:
            return 100
        else:
            return 10
    elif value_type == "oligo":
        if sequences_per_region <= 10000:
            return 100  # Default for oligo
        else:
            return 10
    else:
        return 100  # Default fallback
