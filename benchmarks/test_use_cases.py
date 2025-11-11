# """
# Use case scenario benchmarks.

# Real-world scenarios emphasizing large values, including the
# oligo designer toolsuite use case with nested dictionary structures
# and sequential circular access patterns.
# """

# import tempfile
# import os
# import numpy as np
# from typing import List, Dict, Any
# from pathlib import Path
# import sys


# from benchmark_framework import BenchmarkRunner, BenchmarkResult, open_shelve
# from datasets import generate_dataset, generate_oligo_dataset
# from workloads import generate_sequential_circular_access_pattern, generate_zipf_access_pattern
# from config import DEFAULT_CONFIG, get_max_in_memory_for_value_size


# def benchmark_ml_feature_cache(
#     runner: BenchmarkRunner, dataset_sizes: List[int], array_size_mb: float = 10.0
# ) -> List[BenchmarkResult]:
#     """
#     Benchmark machine learning feature cache use case.

#     Large NumPy arrays (1MB-100MB per key) with Zipf access pattern.

#     Args:
#         runner: Benchmark runner instance
#         dataset_sizes: List of dataset sizes to test
#         array_size_mb: Size of each array in MB

#     Returns:
#         List of benchmark results
#     """
#     from effidict import EffiDict, Hdf5Backend, LRUReplacement

#     results = []

#     for dataset_size in dataset_sizes:
#         # Generate dataset with large NumPy arrays
#         print(f"Generating dataset with {dataset_size} keys")
#         dataset = {}
#         for i in range(dataset_size):
#             key = f"feature_{i}"
#             # Create large NumPy array
#             size_elements = int(array_size_mb * 1024 * 1024 / 8)  # float64 = 8 bytes
#             array = np.random.rand(size_elements)
#             dataset[key] = array

#         keys = list(dataset.keys())

#         print(f"Generated {len(keys)} keys")

#         # Generate Zipf access pattern
#         access_sequence = list(generate_zipf_access_pattern(keys, len(keys) * 10, s=1.5))

#         print(f"Generated access sequence with {len(access_sequence)} operations")

#         # Benchmark dict (baseline - will use lots of RAM)
#         def read_dict():
#             d = dict(dataset)
#             for key in access_sequence[:100]:  # Limit for memory reasons
#                 if key in d:
#                     _ = d[key]

#         print(f"Benchmarking dict with {dataset_size} keys")

#         result = runner.run_benchmark("ml_feature_cache", dataset_size, "dict", read_dict)
#         results.append(result)

#         print(f"Benchmarked dict with {dataset_size} keys")

#         # Benchmark shelve
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
#             shelve_path = tmp.name

#         print(f"Benchmarking shelve with {dataset_size} keys")
#         try:
#             # Pre-populate shelve
#             with open_shelve(shelve_path, "c") as db:
#                 for key, value in dataset.items():
#                     db[key] = value

#             def read_shelve():
#                 with open_shelve(shelve_path, "r") as db:
#                     for key in access_sequence[:100]:
#                         if key in db:
#                             _ = db[key]

#             result = runner.run_benchmark("ml_feature_cache", dataset_size, "shelve", read_shelve)
#             results.append(result)
#         finally:
#             if os.path.exists(shelve_path):
#                 os.remove(shelve_path)

#         print(f"Benchmarked shelve with {dataset_size} keys")

#         print(f"Benchmarking EffiDict with HDF5 backend with {dataset_size} keys")

#         # Benchmark EffiDict with HDF5 backend (best for arrays)
#         # Determine max_in_memory based on array size
#         max_in_memory = get_max_in_memory_for_value_size("large", array_size_mb)

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
#             hdf5_path = tmp.name

#         try:
#             backend = Hdf5Backend(hdf5_path)
#             strategy = LRUReplacement(disk_backend=backend, max_in_memory=max_in_memory)

#             # Pre-populate EffiDict
#             ed = EffiDict(max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy)
#             for key, value in dataset.items():
#                 ed[key] = value

#             def read_effidict():
#                 ed = EffiDict(
#                     max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy
#                 )
#                 for key in access_sequence[:100]:
#                     if key in ed:
#                         _ = ed[key]

#             result = runner.run_benchmark(
#                 "ml_feature_cache", dataset_size, "effidict_hdf5_lru", read_effidict
#             )
#             results.append(result)
#         finally:
#             if os.path.exists(hdf5_path):
#                 try:
#                     os.remove(hdf5_path)
#                 except:
#                     pass
#             if backend:
#                 try:
#                     backend.destroy()
#                 except:
#                     pass

#     return results


# def benchmark_oligo_designer_toolsuite(
#     runner: BenchmarkRunner, num_regions: List[int], sequences_per_region: int = 100
# ) -> List[BenchmarkResult]:
#     """
#     Benchmark oligo designer toolsuite use case.

#     Nested dictionary structure: region_id -> {DNA_sequence -> metadata}
#     Sequential circular access pattern.

#     Args:
#         runner: Benchmark runner instance
#         num_regions: List of number of regions to test
#         sequences_per_region: Number of sequences per region

#     Returns:
#         List of benchmark results
#     """
#     from effidict import EffiDict, SqliteBackend, LRUReplacement

#     results = []

#     for n_regions in num_regions:
#         # Generate oligo dataset
#         dataset = generate_oligo_dataset(n_regions, sequences_per_region)
#         region_keys = list(dataset.keys())

#         # Generate sequential circular access pattern
#         access_sequence = list(generate_sequential_circular_access_pattern(region_keys, len(region_keys) * 5))

#         # Benchmark dict
#         def read_dict():
#             d = dict(dataset)
#             for key in access_sequence:
#                 if key in d:
#                     _ = d[key]

#         result = runner.run_benchmark("oligo_designer", n_regions, "dict", read_dict)
#         results.append(result)

#         # Benchmark shelve
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
#             shelve_path = tmp.name

#         try:
#             # Pre-populate shelve
#             with open_shelve(shelve_path, "c") as db:
#                 for key, value in dataset.items():
#                     db[key] = value

#             def read_shelve():
#                 with open_shelve(shelve_path, "r") as db:
#                     for key in access_sequence:
#                         if key in db:
#                             _ = db[key]

#             result = runner.run_benchmark("oligo_designer", n_regions, "shelve", read_shelve)
#             results.append(result)
#         finally:
#             if os.path.exists(shelve_path):
#                 os.remove(shelve_path)

#         # Benchmark EffiDict with SQLite backend
#         # Determine max_in_memory for oligo dataset
#         max_in_memory = get_max_in_memory_for_value_size("oligo", None, sequences_per_region)

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite") as tmp:
#             sqlite_path = tmp.name

#         try:
#             backend = SqliteBackend(sqlite_path)
#             strategy = LRUReplacement(disk_backend=backend, max_in_memory=max_in_memory)

#             # Pre-populate EffiDict
#             ed = EffiDict(max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy)
#             for key, value in dataset.items():
#                 ed[key] = value

#             def read_effidict():
#                 ed = EffiDict(
#                     max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy
#                 )
#                 for key in access_sequence:
#                     if key in ed:
#                         _ = ed[key]

#             result = runner.run_benchmark("oligo_designer", n_regions, "effidict_sqlite_lru", read_effidict)
#             results.append(result)
#         finally:
#             if os.path.exists(sqlite_path):
#                 try:
#                     os.remove(sqlite_path)
#                 except:
#                     pass
#             if backend:
#                 try:
#                     backend.destroy()
#                 except:
#                     pass

#     return results


# def run_use_case_benchmarks(config: Any = None, output_dir: str = "benchmark_results") -> BenchmarkRunner:
#     """
#     Run all use case scenario benchmarks.

#     Args:
#         config: Benchmark configuration (uses DEFAULT_CONFIG if None)
#         output_dir: Output directory for results

#     Returns:
#         BenchmarkRunner with all results
#     """
#     if config is None:
#         config = DEFAULT_CONFIG

#     runner = BenchmarkRunner(output_dir=output_dir)

#     print("Running use case scenario benchmarks...")

#     # Machine learning feature cache
#     print("Testing ML feature cache use case...")
#     benchmark_ml_feature_cache(runner, config.dataset_sizes[:2], array_size_mb=10.0)

#     # Oligo designer toolsuite
#     print("Testing oligo designer toolsuite use case...")
#     benchmark_oligo_designer_toolsuite(runner, config.num_regions[:3], sequences_per_region=100)

#     # Save results
#     runner.save_results("use_case_benchmarks")

#     return runner
