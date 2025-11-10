"""
Dataset generators for benchmarking.

Generates test data with different value sizes and structures,
including oligo designer toolsuite nested dictionary structures.
"""

import random
import string
import numpy as np
from typing import Dict, Any, List, Optional


def generate_random_string(length: int) -> str:
    """Generate a random string of given length."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_small_uniform_object(size_bytes: int = 100) -> Dict[str, Any]:
    """
    Generate a small uniform object (baseline comparison).

    Args:
        size_bytes: Target size in bytes

    Returns:
        Dictionary with small uniform data
    """
    num_keys = max(1, size_bytes // 20)  # Rough estimate
    obj = {}
    for i in range(num_keys):
        key = f"key{i}"
        value = f"value{i}"
        obj[key] = value

    return obj


def generate_large_object(size_mb: float = 1.0) -> Dict[str, Any]:
    """
    Generate a large object (EffiDict's strength).

    Args:
        size_mb: Target size in megabytes

    Returns:
        Dictionary with large data (NumPy array or large nested dict)
    """
    # Option 1: Large NumPy array
    if size_mb >= 0.1:  # For larger sizes, use arrays
        size_bytes = int(size_mb * 1024 * 1024)
        # Rough estimate: 8 bytes per float64
        num_elements = size_bytes // 8
        array = np.random.rand(num_elements)
        return {"data": array.tolist(), "type": "array"}

    # Option 2: Large nested dictionary
    size_bytes = int(size_mb * 1024 * 1024)
    obj = {}
    current_size = 0
    i = 0

    while current_size < size_bytes:
        key = f"key{i}"
        # Each value is roughly 100 bytes
        value = generate_random_string(100)
        obj[key] = value
        current_size += len(key) + len(value) + 50  # Rough estimate
        i += 1

    return obj


def generate_dna_sequence(length: int = 25) -> str:
    """Generate a random DNA sequence."""
    bases = "ACGT"
    return "".join(random.choices(bases, k=length))


def generate_oligo_metadata(region_id: str, sequence: str, index: int) -> Dict[str, Any]:
    """
    Generate metadata for an oligo sequence.

    Args:
        region_id: ID of the region
        sequence: DNA sequence
        index: Index of the sequence within the region

    Returns:
        Dictionary with oligo metadata
    """
    return {
        "chromosome": random.randint(1, 22),
        "start": random.randint(1, 1000000),
        "end": random.randint(1, 1000000),
        "strand": random.choice(["+", "-"]),
        "length": len(sequence),
        "region_id": region_id,
        "sequence_index": index,
        "additional_information": f"info_{index}",
    }


def generate_oligo_dataset(
    num_regions: int = 100, sequences_per_region: int = 100, sequence_length: int = 25
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Generate oligo designer toolsuite dataset.

    Structure: region_id -> {DNA_sequence -> metadata}

    Args:
        num_regions: Number of regions
        sequences_per_region: Number of DNA sequences per region
        sequence_length: Length of each DNA sequence

    Returns:
        Nested dictionary structure matching oligo designer toolsuite format
    """
    dataset = {}

    for region_idx in range(num_regions):
        region_id = f"region_{region_idx}"
        region_data = {}

        for seq_idx in range(sequences_per_region):
            sequence = generate_dna_sequence(sequence_length)
            metadata = generate_oligo_metadata(region_id, sequence, seq_idx)
            region_data[sequence] = metadata

        dataset[region_id] = region_data

    return dataset


def generate_dataset(
    num_entries: int, value_type: str = "small_uniform", value_size: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate a dataset with specified number of entries.

    Args:
        num_entries: Number of key-value pairs
        value_type: Type of values ('small_uniform', 'large', 'oligo')
        value_size: Size parameter (bytes for small_uniform, MB for large)

    Returns:
        Dictionary with generated data
    """
    dataset = {}

    if value_type == "small_uniform":
        size_bytes = value_size or 100
        for i in range(num_entries):
            key = f"key_{i}"
            dataset[key] = generate_small_uniform_object(size_bytes)

    elif value_type == "large":
        size_mb = value_size or 1.0
        for i in range(num_entries):
            key = f"key_{i}"
            dataset[key] = generate_large_object(size_mb)

    elif value_type == "oligo":
        # For oligo, we use the specific structure
        # num_entries represents number of regions
        sequences_per_region = int(value_size) if value_size else 100
        return generate_oligo_dataset(num_regions=num_entries, sequences_per_region=sequences_per_region)

    else:
        raise ValueError(f"Unknown value_type: {value_type}")

    return dataset


def estimate_dataset_size(dataset: Dict[str, Any]) -> int:
    """
    Estimate the size of a dataset in bytes.

    Args:
        dataset: Dictionary to estimate

    Returns:
        Estimated size in bytes
    """
    import sys

    def get_size(obj):
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum(get_size(k) + get_size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            size += sum(get_size(item) for item in obj)
        elif isinstance(obj, str):
            size += len(obj)
        return size

    return get_size(dataset)
