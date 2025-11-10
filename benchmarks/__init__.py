"""
Benchmarking suite for EffiDict.

Provides comprehensive benchmarking tools for comparing EffiDict
against standard Python dict and shelve.
"""

from .benchmark_framework import BenchmarkRunner, BenchmarkResult, Timer, MemoryTracker
from .config import DEFAULT_CONFIG, BenchmarkConfig
from .datasets import (
    generate_dataset,
    generate_oligo_dataset,
    generate_small_uniform_object,
    generate_large_object,
)
from .workloads import (
    generate_access_pattern,
    generate_uniform_access_pattern,
    generate_zipf_access_pattern,
    generate_sequential_circular_access_pattern,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "Timer",
    "MemoryTracker",
    "DEFAULT_CONFIG",
    "BenchmarkConfig",
    "generate_dataset",
    "generate_oligo_dataset",
    "generate_small_uniform_object",
    "generate_large_object",
    "generate_access_pattern",
    "generate_uniform_access_pattern",
    "generate_zipf_access_pattern",
    "generate_sequential_circular_access_pattern",
]
