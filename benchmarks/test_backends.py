# """
# Backend comparison benchmarks.

# Compares all 4 backends (SQLite, JSON, Pickle, HDF5) for serialization
# speed, disk I/O performance, and storage efficiency.
# """

# import tempfile
# import os
# from typing import List, Dict, Any
# from pathlib import Path
# import sys


# from benchmark_framework import BenchmarkRunner, BenchmarkResult, open_shelve
# from datasets import generate_dataset
# from config import DEFAULT_CONFIG, get_backend_class, get_max_in_memory_for_value_size


# def benchmark_backend_serialization(
#     runner: BenchmarkRunner, backend_name: str, dataset: Dict[str, Any], value_size_mb: float = 1.0
# ) -> BenchmarkResult:
#     """
#     Benchmark serialization/deserialization speed for a backend.

#     Args:
#         runner: Benchmark runner instance
#         backend_name: Name of backend
#         dataset: Dataset to serialize
#         value_size_mb: Size of values in MB

#     Returns:
#         BenchmarkResult
#     """
#     from effidict import EffiDict, LRUReplacement

#     BackendClass = get_backend_class(backend_name)
#     if BackendClass is None:
#         return None

#     # Determine max_in_memory based on value size
#     max_in_memory = get_max_in_memory_for_value_size("large", value_size_mb)

#     # Create temporary storage
#     with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{backend_name}") as tmp:
#         storage_path = tmp.name

#     try:
#         backend = BackendClass(storage_path)
#         strategy = LRUReplacement(disk_backend=backend, max_in_memory=max_in_memory)

#         # Benchmark serialization (write)
#         def serialize_operations():
#             ed = EffiDict(max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy)
#             for key, value in dataset.items():
#                 ed[key] = value

#         result = runner.run_benchmark(
#             "serialize", len(dataset), f"effidict_{backend_name}", serialize_operations
#         )
#         result.metadata = {"value_size_mb": value_size_mb}

#         # Benchmark deserialization (read)
#         def deserialize_operations():
#             ed = EffiDict(max_in_memory=max_in_memory, disk_backend=backend, replacement_strategy=strategy)
#             for key in dataset.keys():
#                 if key in ed:
#                     _ = ed[key]

#         result2 = runner.run_benchmark(
#             "deserialize", len(dataset), f"effidict_{backend_name}", deserialize_operations
#         )
#         result2.metadata = {"value_size_mb": value_size_mb}

#         return result, result2

#     finally:
#         if os.path.exists(storage_path):
#             if backend_name == "sqlite" or backend_name == "hdf5":
#                 try:
#                     os.remove(storage_path)
#                 except:
#                     pass
#             else:
#                 import shutil

#                 try:
#                     shutil.rmtree(storage_path)
#                 except:
#                     pass
#         if backend:
#             try:
#                 backend.destroy()
#             except:
#                 pass


# def compare_backends(
#     runner: BenchmarkRunner, dataset_sizes: List[int], value_sizes: List[float] = None
# ) -> List[BenchmarkResult]:
#     """
#     Compare all backends.

#     Args:
#         runner: Benchmark runner instance
#         dataset_sizes: List of dataset sizes to test
#         value_sizes: List of value sizes in MB

#     Returns:
#         List of benchmark results
#     """
#     if value_sizes is None:
#         value_sizes = [0.1, 1.0, 10.0]  # 100KB, 1MB, 10MB

#     backends = ["sqlite", "json", "pickle"]
#     try:
#         from effidict import Hdf5Backend

#         backends.append("hdf5")
#     except:
#         pass

#     results = []

#     for dataset_size in dataset_sizes:
#         for value_size_mb in value_sizes:
#             # Generate dataset with large values
#             dataset = generate_dataset(dataset_size, value_size_mb)

#             for backend_name in backends:
#                 try:
#                     result = benchmark_backend_serialization(runner, backend_name, dataset, value_size_mb)
#                     if result:
#                         if isinstance(result, tuple):
#                             results.extend(result)
#                         else:
#                             results.append(result)
#                 except Exception as e:
#                     print(f"Error benchmarking {backend_name} with {value_size_mb}MB values: {e}")
#                     continue

#     return results


# def compare_with_shelve(
#     runner: BenchmarkRunner, dataset_sizes: List[int], value_sizes: List[float] = None
# ) -> List[BenchmarkResult]:
#     """
#     Compare EffiDict backends with shelve.

#     Args:
#         runner: Benchmark runner instance
#         dataset_sizes: List of dataset sizes to test
#         value_sizes: List of value sizes in MB

#     Returns:
#         List of benchmark results
#     """
#     if value_sizes is None:
#         value_sizes = [0.1, 1.0, 10.0]

#     results = []

#     for dataset_size in dataset_sizes:
#         for value_size_mb in value_sizes:
#             # Generate dataset
#             dataset = generate_dataset(dataset_size, value_size_mb)

#             # Benchmark shelve
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
#                 shelve_path = tmp.name

#             try:

#                 def shelve_operations():
#                     with open_shelve(shelve_path, "c") as db:
#                         for key, value in dataset.items():
#                             db[key] = value
#                     with open_shelve(shelve_path, "r") as db:
#                         for key in dataset.keys():
#                             if key in db:
#                                 _ = db[key]

#                 result = runner.run_benchmark(
#                     "serialize_deserialize", dataset_size, "shelve", shelve_operations
#                 )
#                 result.metadata = {"value_size_mb": value_size_mb}
#                 results.append(result)
#             finally:
#                 if os.path.exists(shelve_path):
#                     os.remove(shelve_path)

#     return results


# def run_backend_benchmarks(config: Any = None, output_dir: str = "benchmark_results") -> BenchmarkRunner:
#     """
#     Run all backend comparison benchmarks.

#     Args:
#         config: Benchmark configuration (uses DEFAULT_CONFIG if None)
#         output_dir: Output directory for results

#     Returns:
#         BenchmarkRunner with all results
#     """
#     if config is None:
#         config = DEFAULT_CONFIG

#     runner = BenchmarkRunner(output_dir=output_dir)

#     print("Running backend comparison benchmarks...")

#     # Compare backends
#     compare_backends(runner, config.dataset_sizes[:3], config.large_value_sizes[:3])

#     # Compare with shelve
#     compare_with_shelve(runner, config.dataset_sizes[:3], config.large_value_sizes[:3])

#     # Save results
#     runner.save_results("backend_benchmarks")

#     return runner
