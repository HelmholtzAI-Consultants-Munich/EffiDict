"""
Dataset generators for benchmarking.

Generates test data with different value sizes and structures,
including oligo designer toolsuite nested dictionary structures.
"""

import sys
import random
import string
import numpy as np
from typing import Dict, Any, Optional

########################################################
# Standard objects
########################################################


def generate_random_string(length: int) -> str:
    """
    Generate a random string of specified length.

    Args:
        length: Length of the string to generate

    Returns:
        Random string containing letters and digits
    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def estimate_object_size(obj: Any) -> int:
    """
    Estimate the size of an object in bytes.

    Recursively calculates the size including nested structures.

    Args:
        obj: Object to estimate size for (dict, list, tuple, str, etc.)

    Returns:
        Estimated size in bytes
    """

    def get_size(obj):
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum(get_size(k) + get_size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            size += sum(get_size(item) for item in obj)
        elif isinstance(obj, str):
            size += len(obj)
        return size

    return get_size(obj)


def generate_small_object(size_bytes: int) -> Dict[str, Any]:
    """
    Generate a small uniform dictionary object of approximately specified size.

    Args:
        size_bytes: Target size in bytes (approximate)

    Returns:
        Dictionary with string keys and values
    """
    num_keys = max(1, size_bytes // 20)  # Rough estimate
    obj = {}
    for i in range(num_keys):
        key = f"key{i}"
        value = f"value{i}"
        obj[key] = value

    return obj


def generate_large_object(size_bytes: int) -> Dict[str, Any]:
    """
    Generate a large object of approximately specified size.

    Randomly chooses between a NumPy array or nested dictionary structure.

    Args:
        size_bytes: Target size in bytes (approximate)

    Returns:
        Either a NumPy array or dictionary with string keys and values
    """
    # Randomly choose between Option 1 (NumPy array) or Option 2 (nested dictionary)
    use_array = random.choice([True, False])
    if use_array:
        # Option 1: Large NumPy array
        # Rough estimate: 8 bytes per float64
        num_elements = size_bytes // 8
        obj = np.random.rand(num_elements)
    else:
        # Option 2: Large nested dictionary
        obj = {}
        i = 0
        while estimate_object_size(obj) < size_bytes:
            key = f"key{i}"
            value = generate_random_string(random.randint(10, 10000))
            obj[key] = value
            i += 1

        return obj


def generate_dataset(num_entries: int, value_size: Optional[float]) -> Dict[str, Any]:
    """
    Generate a dataset with specified number of entries and value size.

    Args:
        num_entries: Number of key-value pairs in the dataset
        value_size: Size of each value in MB. If None, uses a default small size

    Returns:
        Dictionary with keys 'key_0', 'key_1', etc., and values of approximately
        the specified size (small objects for <1KB, large objects for >=1KB)
    """
    dataset = {}
    size_bytes = max(10, int(value_size * 1024 * 1024))
    if size_bytes < 1024:
        for i in range(num_entries):
            key = f"key_{i}"
            dataset[key] = generate_small_object(size_bytes)
    else:
        for i in range(num_entries):
            key = f"key_{i}"
            dataset[key] = generate_large_object(size_bytes)

    return dataset


########################################################
# Oligo designer toolsuite objects
########################################################


# def generate_dna_sequence(length: int = 25) -> str:
#     """Generate a random DNA sequence."""
#     bases = "ACGT"
#     return "".join(random.choices(bases, k=length))


# def generate_oligo_metadata(region_id: str, sequence: str, index: int) -> Dict[str, Any]:
#     """
#     Generate metadata for an oligo sequence.

#     Args:
#         region_id: ID of the region
#         sequence: DNA sequence
#         index: Index of the sequence within the region

#     Returns:
#         Dictionary with oligo metadata
#     """
#     return {
#         "chromosome": random.randint(1, 22),
#         "start": random.randint(1, 1000000),
#         "end": random.randint(1, 1000000),
#         "strand": random.choice(["+", "-"]),
#         "length": len(sequence),
#         "region_id": region_id,
#         "sequence_index": index,
#         "additional_information": f"info_{index}",
#     }


# def generate_dataset_oligo(
#     num_regions: int = 100, sequences_per_region: int = 100, sequence_length: int = 25
# ) -> Dict[str, Dict[str, Dict[str, Any]]]:
#     """
#     Generate oligo designer toolsuite dataset.

#     Structure: region_id -> {DNA_sequence -> metadata}

#     Args:
#         num_regions: Number of regions
#         sequences_per_region: Number of DNA sequences per region
#         sequence_length: Length of each DNA sequence

#     Returns:
#         Nested dictionary structure matching oligo designer toolsuite format
#     """
#     dataset = {}

#     for region_idx in range(num_regions):
#         region_id = f"region_{region_idx}"
#         region_data = {}

#         for seq_idx in range(sequences_per_region):
#             sequence = generate_dna_sequence(sequence_length)
#             metadata = generate_oligo_metadata(region_id, sequence, seq_idx)
#             region_data[sequence] = metadata

#         dataset[region_id] = region_data

#     return dataset
