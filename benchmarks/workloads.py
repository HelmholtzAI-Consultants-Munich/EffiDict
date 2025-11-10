"""
Workload pattern generators for benchmarking.

Implements different access patterns: uniform distribution, Zipf distribution,
and sequential circular access (for oligo designer toolsuite).
"""

import random
from typing import List, Iterator, Optional
from collections import deque

from scipy.stats import zipf


def generate_uniform_access_pattern(keys: List[str], num_operations: int) -> Iterator[str]:
    """
    Generate uniform distribution access pattern.

    All keys are equally likely to be accessed.

    Args:
        keys: List of available keys
        num_operations: Number of operations to generate

    Yields:
        Key to access
    """
    for _ in range(num_operations):
        yield random.choice(keys)


def generate_zipf_access_pattern(keys: List[str], num_operations: int, s: float = 1.0) -> Iterator[str]:
    """
    Generate Zipf distribution access pattern.

    Small subset of "hot" keys get most accesses.

    Args:
        keys: List of available keys
        num_operations: Number of operations to generate
        s: Zipf skewness parameter (higher = more skewed)

    Yields:
        Key to access
    """
    if not keys:
        return

    n = len(keys)

    # Use scipy for accurate Zipf distribution
    zipf_dist = zipf(s)
    # Generate indices according to Zipf distribution
    indices = zipf_dist.rvs(num_operations) - 1  # scipy uses 1-based
    indices = indices % n  # Wrap around if needed
    for idx in indices:
        yield keys[idx]


def generate_sequential_circular_access_pattern(keys: List[str], num_operations: int) -> Iterator[str]:
    """
    Generate sequential circular access pattern.

    Keys are accessed in order, wrapping around: key_1, key_2, ..., key_N, key_1, ...
    Simulates oligo designer toolsuite access pattern.

    Args:
        keys: List of available keys
        num_operations: Number of operations to generate

    Yields:
        Key to access
    """
    if not keys:
        return

    # Use deque for efficient circular iteration
    key_queue = deque(keys)

    for _ in range(num_operations):
        key = key_queue[0]
        key_queue.rotate(-1)  # Move to next key
        yield key


def generate_access_pattern(
    keys: List[str], num_operations: int, pattern: str = "uniform", **kwargs
) -> Iterator[str]:
    """
    Generate access pattern based on pattern type.

    Args:
        keys: List of available keys
        num_operations: Number of operations to generate
        pattern: Pattern type ('uniform', 'zipf', 'sequential_circular')
        **kwargs: Additional parameters for specific patterns
            - s: Zipf skewness parameter (default: 1.0)

    Yields:
        Key to access
    """
    if pattern == "uniform":
        return generate_uniform_access_pattern(keys, num_operations)
    elif pattern == "zipf":
        s = kwargs.get("s", 1.0)
        return generate_zipf_access_pattern(keys, num_operations, s)
    elif pattern == "sequential_circular":
        return generate_sequential_circular_access_pattern(keys, num_operations)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def generate_operation_sequence(
    keys: List[str],
    num_operations: int,
    read_ratio: float = 0.5,
    write_ratio: float = 0.3,
    delete_ratio: float = 0.1,
    modify_ratio: float = 0.1,
    pattern: str = "uniform",
    **kwargs,
) -> Iterator[tuple]:
    """
    Generate a sequence of operations (read, write, delete, modify).

    Args:
        keys: List of available keys
        num_operations: Number of operations to generate
        read_ratio: Ratio of read operations
        write_ratio: Ratio of write operations
        delete_ratio: Ratio of delete operations
        modify_ratio: Ratio of modify operations
        pattern: Access pattern for reads ('uniform', 'zipf', 'sequential_circular')
        **kwargs: Additional parameters for access pattern

    Yields:
        Tuple of (operation_type, key)
        operation_type: 'read', 'write', 'delete', 'modify'
    """
    # Normalize ratios
    total = read_ratio + write_ratio + delete_ratio + modify_ratio
    read_ratio /= total
    write_ratio /= total
    delete_ratio /= total
    modify_ratio /= total

    # Generate access pattern for reads
    access_pattern = list(generate_access_pattern(keys, num_operations, pattern, **kwargs))

    # Generate operation types
    operations = []
    for i in range(num_operations):
        rand = random.random()
        if rand < read_ratio:
            operations.append("read")
        elif rand < read_ratio + write_ratio:
            operations.append("write")
        elif rand < read_ratio + write_ratio + delete_ratio:
            operations.append("delete")
        else:
            operations.append("modify")

    # Pair operations with keys
    for op, key in zip(operations, access_pattern):
        yield (op, key)
